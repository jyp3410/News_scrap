# 정보보안 뉴스 스크랩

국내외 정보보안 뉴스를 수집·분류·요약해 SQLite에 저장하고, Streamlit 앱과 FAISS 기반 RAG 챗봇으로 보여준다. 요구사항은 `PRD.md`, 개발 단계별 프롬프트는 `PROMPTS.md` 참고.

## 설치
```
pip install -r requirements.txt
copy .env.example .env     # OPENAI_API_KEY 입력
```

## 실행
```
python scripts/run_pipeline.py              # 수집 → 분류/요약 → 임베딩 (재실행해도 중복 없음)
python scripts/run_pipeline.py --skip-llm   # 수집·저장만 (OpenAI 호출 없음)
python scripts/run_pipeline.py --full       # 수집 기준 시간을 무시하고 피드가 주는 기사를 전부 대상으로 (장애 복구용)
python scripts/run_pipeline.py --limit 20   # 이번에 분류/요약할 기사 수 상한 (비용 제어)
python scripts/run_pipeline.py --revalidate --dry-run  # 기존 기사의 보안 관련성 재판정 미리보기 (DB 변경 없음)
python scripts/run_pipeline.py --revalidate            # 위 결과를 적용 (무관 기사 숨김 + 인덱스 재구축)
python scripts/rebuild_index.py --check     # FAISS ↔ DB 정합성 검증
python scripts/rebuild_index.py             # FAISS 전체 재구축 (전체 재임베딩, API 비용 발생)
python scripts/remove_source.py "뉴스사"     # 저장된 특정 뉴스사 기사 삭제 미리보기 (--apply로 실제 삭제)
python scripts/calibrate_threshold.py       # 챗봇 검색 유사도 기준(RAG_MIN_SCORE) 재측정
streamlit run app/main.py                   # 앱 (뉴스 목록 / 챗봇 / 대시보드)
python -m pytest tests                      # 테스트 (단일: pytest tests/test_stage2.py::test_enrich_pending_is_idempotent_and_keeps_failures)
```

## 설정
- `config/sources.yaml`: 수집 소스 (RSS URL, 국가/언어, 시간대 오프셋, 키워드 필터 여부)
- `config/categories.yaml`: 중분류와 키워드
- 환경변수(전체 목록은 `.env.example`): `OPENAI_API_KEY`, `CHAT_MODEL`(기본 gpt-6-astra), `EMBEDDING_MODEL`(기본 text-embedding-3-small), `RAG_MIN_SCORE`(기본 0.27), `PIPELINE_MAX_ENRICH`

## 불필요한 기사 거르기
- **보안 무관 기사 검증**: 분류·요약 호출 안에서 LLM이 "주제가 정보보안인가"를 함께 판정한다 (추가 호출 없음).
  확신도 0.7 이상(`RELEVANCE_REJECT_CONF`)으로 무관하다고 판단한 기사만 **숨기며**, 삭제하지는 않는다
  (목록·챗봇·대시보드·인덱스에서 빠지고, DB에는 `relevant=0`과 사유가 남는다). 애매하면 노출한다.
  되돌리려면 `UPDATE articles SET relevant=1 WHERE id='...'` 후 `rebuild_index.py`.
- **수집하지 않을 뉴스사**: `config/sources.yaml`의 `exclude_sources: ["뉴스사 이름"]`에 적거나, 소스에 `enabled: false`를 둔다.
  이미 저장된 기사는 그대로 남는다. 이름에 오타가 있으면 경고 로그가 남는다.
- 일반 IT 매체는 `keyword_filter: true`로 제목·요약에 보안 키워드가 있는 기사만 1차로 받을 수 있다.

## 자동 수집 (GitHub Actions)
`.github/workflows/collect.yml`이 **하루 4번(한국시간 00·06·12·18시)** 파이프라인을 실행하고 `data/` 변경분을 커밋한다.
저장소 Settings → Secrets에 `OPENAI_API_KEY`를 등록해야 한다.

## 수집 기준 시간
스케줄은 6시간마다지만, 지연·실패로 기사를 놓치지 않도록 소스마다 **실행 시각 − 6시간 30분** 이후 발행된 기사를 대상으로 한다
(`config/sources.yaml`의 `collection`). 직전 성공 시각 − 30분이 더 이르면 그쪽을 쓰므로, 실행이 밀리거나 한 번 실패해도
빠지는 구간이 없다. 소스별 마지막 성공 시각은 기사를 DB에 저장한 **뒤에** `source_state` 테이블에 기록하고, 실패한 소스는
기록을 앞당기지 않는다. 처음 수집하는 소스는 제한 없이 전부 가져오며, `run_pipeline.py --full`로 언제든 제한을 끌 수 있다.
중복은 이전과 같이 URL·제목으로 걸러진다.

## 데이터와 Git 운영
- `data/`(news.db, faiss.index, id_map.json)는 **GitHub Actions가 갱신하는 것을 원칙**으로 한다.
  로컬에서 수집한 데이터를 push하면 CI가 만든 바이너리와 충돌한다. 로컬에서는 시험만 하고 `data/`는 커밋하지 않거나,
  push 전에 `git fetch && git checkout origin/<브랜치> -- data` 로 CI 데이터를 받아 둔다.
- 데이터 파일은 6시간마다 커밋돼 저장소가 커진다 (커밋당 약 2MB, 하루 약 8MB). 장기 운영 시 Git LFS 도입이나 이력 정리를 검토한다.
- 워크플로는 파이프라인이 실패해도 그때까지 저장된 결과를 커밋하고, 잡은 실패로 표시한다.

## 알려진 사항
- robots.txt가 막는 소스(예: Dark Reading)는 수집에서 제외한다 (`exclude_sources`). HTML 스크래핑은 미지원(`method: rss`만 가능).
- 제목 중복 판정은 정밀도를 우선한다: 서로 다른 기사를 잃는 것보다 다른 매체의 같은 기사가 일부 남는 편이 낫다.
- gpt-6-astra는 `max_completion_tokens`만 지원하고 temperature는 기본값만 허용한다.
