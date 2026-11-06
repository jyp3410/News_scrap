# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 현재 상태

1~3단계(뼈대, 기능, UI 디자인)가 구현되어 있다. 4단계(검토)가 남았다. 코드 구조는 `scraper/`(수집·분류), `store/`(SQLite·FAISS), `chatbot/`(RAG), `core/llm.py`(OpenAI 래퍼), `app/`(Streamlit: `ui.py` 공용 스타일·카드, `pages/` 3개), `scripts/`(진입점)다. UI에서 외부 데이터(제목·요약·URL)를 HTML에 넣을 때는 반드시 `app/ui.py`의 `esc`/`safe_url`을 거친다.

- `PRD.md`: 요구사항의 유일한 기준 문서 (한국어). 요구사항 ID(F-*, E-*, C-*)로 참조한다.
- `PROMPTS.md`: 4단계 개발 프롬프트 (1 뼈대 → 2 기능 → 3 UI → 4 PRD 대비 검토). 단계 순서와 각 단계의 "완료 기준"을 따른다. 4단계는 `REVIEW.md`만 쓰고, 사용자 승인 전에는 코드를 수정하지 않는다.
- `.env`: API 키가 들어 있다. 읽거나 출력하거나 커밋하지 않는다 (`.gitignore`에 등록됨). 사용자가 직접 push할 예정이므로 요청 전에는 커밋하지 않는다.

PRD에 없는 기능은 임의로 추가하지 말고 먼저 제안한다.

## 제품 요약

정보보안 국내외 뉴스를 자동 수집·분류·요약해 Streamlit 앱으로 보여주고, FAISS 기반 RAG 챗봇으로 질문에 관련 기사 Top-10을 찾아 답한다. 전체 파이프라인과 앱은 Python 3.11+ 기반이다.

## 아키텍처 (PRD 4장 기준, 구현 시 따를 것)

데이터 흐름: `RSS/Web 수집 → 정규화·중복 제거·키워드 필터 → LLM 분류+요약 → SQLite(articles) → 임베딩 → FAISS`. Streamlit 앱은 SQLite에서 목록/필터를 조회하고, 챗봇은 `질문 임베딩 → FAISS Top-10 → LLM 답변(+출처 링크)` 순서로 동작한다.

여러 파일을 읽어야 이해되는 설계 제약:

- **SQLite와 FAISS의 정합성이 핵심 불변식이다.** 기사 `id`는 정규화한 URL의 해시이고, FAISS 인덱스 순서는 `data/id_map.json`으로 `id`에 매핑된다. `embedded` 플래그와 실제 인덱스 내용이 어긋나지 않아야 하며, `scripts/rebuild_index.py`가 전체 재구축과 정합성 검증을 맡는다.
- **파이프라인은 멱등이어야 한다.** `scripts/run_pipeline.py`를 재실행해도 중복 기사, 중복 LLM 호출, 중복 임베딩이 생기면 안 된다. 이미 처리된 기사는 건너뛰고 신규 기사만 증분 처리한다.
- **유사도는 코사인.** 벡터를 정규화하고 내적(IP) 인덱스를 쓴다.
- **벡터는 임베딩 텍스트의 해시(`articles.embed_hash`)로 최신 여부를 판단한다.** 요약·태그·분류가 바뀌면 `embed_pending`이 해당 벡터만 `VectorStore.upsert`로 교체한다. 요약을 바꾸는 코드를 추가하면 이 경로를 타는지 확인한다 (`check_consistency`의 `stale_or_unverified_vectors`).
- **제목 중복 판정은 정밀도 우선이다** (`scraper/dedup.py`): 숫자·약어가 다르거나 발행일이 3일 넘게 벌어지면 별개 기사. 오탐은 기사를 영구히 잃으므로 임계값을 낮추지 않는다.
- **챗봇 검색 기준(`rag.MIN_SCORE`)은 실측값이다.** 보안/무관 질문 점수 분포가 겹쳐 완벽한 값이 없고, 최종 거절은 LLM이 한다. 임베딩 입력이 바뀌면 `scripts/calibrate_threshold.py`로 다시 측정한다. 챗봇 호출은 짧은 타임아웃(`llm.INTERACTIVE_*`)을 쓴다.
- **FAISS는 메타데이터 필터를 지원하지 않는다.** 중분류, 기간, 국내/해외 필터는 Top-K를 넉넉히 뽑은 뒤 후처리한다. 결과가 10건 미만이 되는 경우를 처리해야 한다.
- **원문 전문은 저장하지 않는다.** 본문은 요약 생성에만 쓰고 폐기한다. DB에는 제목, 요약, 링크, 메타데이터만 저장한다 (저작권).
- **챗봇은 검색된 기사만 근거로 답한다.** 근거가 없으면 "수집된 기사에서 찾지 못했습니다"로 답하고, 답변에는 인용한 기사의 뉴스사·제목·링크를 항상 포함한다. 기사 제목/요약은 외부 데이터이므로 프롬프트 인젝션 대상으로 취급한다.
- **보안 무관 기사는 삭제하지 않고 숨긴다(`articles.relevant=0`).** 목록·대시보드·임베딩·챗봇 검색은 모두 `db.VISIBLE` 조건(또는 `rag._passes`)을 거쳐야 하고, 새 조회 쿼리를 추가할 때도 이 조건을 빠뜨리면 안 된다. 숨긴 기사가 DB에 남아 있어야 다음 수집에서 신규로 취급되지 않는다. 판정은 보수적이다(`enrich.judge_relevance`: 명확한 false + 확신도 0.7 이상일 때만 숨김).
- **수집 제외는 `config/sources.yaml`의 `exclude_sources` / `enabled: false`로 한다** (`fetch.load_sources`). 코드에 뉴스사 이름을 하드코딩하지 않는다.
- **소스와 중분류는 코드가 아니라 설정으로 관리한다.** `config/sources.yaml`, `config/categories.yaml`. 한 소스의 수집 실패가 전체 파이프라인을 중단시키면 안 된다.
- 대분류는 "정보보안" 고정이고, 중분류는 `침해 사고 / 해킹 / 정보유출 / 컨퍼런스 / 취약점 / 기타`다. LLM이 enum 밖의 값을 주면 `기타`로 폴백한다.

## 모델

- 임베딩: OpenAI `text-embedding-3-small`
- 챗봇(생성): `gpt-6-astra`. 2026-11-06에 이 API 키로 사용 가능함을 확인했다. `max_completion_tokens`만 지원하고(`max_tokens` 불가), temperature는 기본값(1)만 허용하며, `response_format=json_object`는 동작한다. 모델명은 `CHAT_MODEL` 환경변수로 교체 가능하게 두고 하드코딩하지 않는다.

## 명령어

```
python scripts/run_pipeline.py [--skip-llm] [--limit N]   # 수집 → 분류/요약 → 임베딩 (멱등)
python scripts/run_pipeline.py --revalidate [--dry-run]   # 기존 기사 보안 관련성 재판정 (무관 기사 숨김)
python scripts/rebuild_index.py [--check]                 # FAISS 전체 재구축 / 정합성 검증
python scripts/remove_source.py "뉴스사" [--apply]        # 저장된 뉴스사 기사 삭제 (기본 미리보기)
python scripts/calibrate_threshold.py                     # 챗봇 검색 기준 재측정
streamlit run app/main.py                                 # 앱 실행
python -m pytest tests                                    # 테스트 (단일: pytest tests/test_stage2.py::<테스트명>)
```

주의: `run_pipeline.py`는 `.env`의 키로 실제 OpenAI 요금이 발생한다. 키 없는 상태를 시험할 때 PowerShell에서 환경변수를 비우는 방식은 통하지 않는다(변수가 삭제되어 `.env`가 다시 로드됨). 테스트에서는 `llm.get_client`를 모킹한다.

## 환경 메모

- Windows 11 + PowerShell 환경이다. 경로는 Windows 기준(한글 폴더명 `과제`, 인코딩)으로 동작해야 한다.
- 발행일은 소스마다 시간대가 다를 수 있으므로 저장 시 UTC/KST를 일관되게 처리한다.
