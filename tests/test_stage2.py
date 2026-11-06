import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import rag  # noqa: E402
from core import llm  # noqa: E402
from scraper import enrich  # noqa: E402
from store import db, vector  # noqa: E402

MINORS = ["침해 사고", "해킹", "정보유출", "컨퍼런스", "취약점", "기타"]


def _art(i, minor="해킹", country="KR", pub="2026-11-05T03:00:00+00:00", summary="요약"):
    return {"id": f"id{i}", "source": "S", "title": f"title {i}", "url": f"https://x.com/{i}",
            "published_at": pub, "collected_at": "2026-11-06T00:00:00+00:00", "country": country,
            "language": "ko", "category_minor": minor, "summary": summary}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


# ---- 분류/요약 검증 ----
def test_validate_item_enum_fallback_and_cve():
    v = enrich.validate_item(
        {"category": "랜섬", "confidence": 0.9, "summary": "s", "tags": ["a"]}, MINORS, "see cve-2026-12345")
    assert v["minor"] == "기타" and v["confidence"] == 0.0
    assert "CVE-2026-12345" in v["tags"]


def test_validate_item_requires_summary():
    assert enrich.validate_item({"category": "해킹", "summary": "  "}, MINORS, "") is None


def test_enrich_pending_is_idempotent_and_keeps_failures(conn, monkeypatch):
    for i in range(3):
        a = _art(i, summary=None)
        a["category_minor"] = None
        db.insert_articles(conn, [a])
    calls = []

    def fake_chat_json(messages, **kw):
        calls.append(1)
        body = messages[-1]["content"]
        ids = [i for i in ("id0", "id1", "id2") if f'id="{i}"' in body]
        # id2는 항상 요약 누락 → 대기 상태 유지되어야 함
        return {"items": [{"id": i, "category": "해킹", "confidence": 0.8, "reason": "r",
                           "tags": [], "summary": "요약"} for i in ids if i != "id2"]}

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(enrich.content, "get_text", lambda url, snippet="": "본문")
    s1 = enrich.enrich_pending(conn)
    assert (s1["pending"], s1["done"], s1["failed"], s1["rejected"]) == (3, 2, 1, 0)
    n = len(calls)
    s2 = enrich.enrich_pending(conn)          # 처리된 2건은 재호출 대상이 아님
    assert s2["pending"] == 1 and len(calls) > n
    assert conn.execute("SELECT COUNT(*) FROM articles WHERE summary IS NULL").fetchone()[0] == 1


# ---- 벡터 저장소 ----
def fake_embed(texts):
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(rng.normal(size=16))
    return np.asarray(out, dtype="float32")


def test_embed_pending_incremental_and_consistent(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(5)])
    store = vector.VectorStore(tmp_path)
    assert vector.embed_pending(conn, store, fake_embed)["added"] == 5
    assert vector.embed_pending(conn, store, fake_embed)["added"] == 0     # 재실행 시 추가 없음
    db.insert_articles(conn, [_art(9)])
    assert vector.embed_pending(conn, store, fake_embed)["added"] == 1
    reloaded = vector.VectorStore(tmp_path)
    assert reloaded.total == 6
    rep = vector.check_consistency(conn, reloaded)
    assert not rep["flagged_but_not_in_index"] and not rep["index_ids_missing_in_db"]


def test_flag_lost_after_crash_is_recovered_without_duplicates(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(3)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    conn.execute("UPDATE articles SET embedded=0")        # 인덱스 저장 후 플래그 갱신 전 크래시 상황
    conn.commit()
    stats = vector.embed_pending(conn, vector.VectorStore(tmp_path), fake_embed)
    assert stats["added"] == 0 and stats["replaced"] == 3           # 중복 추가 없이 벡터만 교체
    assert vector.VectorStore(tmp_path).total == 3


def test_search_returns_self_first_with_cosine_score(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(5)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    row = db.get_by_ids(conn, ["id3"])["id3"]
    hits = store.search(fake_embed([vector.embedding_text(row)])[0], 3)
    assert hits[0][0] == "id3" and hits[0][1] == pytest.approx(1.0, abs=1e-4)


def test_corrupt_index_detected(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(2)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    (tmp_path / "id_map.json").write_text('["id0"]', encoding="utf-8")
    with pytest.raises(vector.IndexCorrupt):
        vector.VectorStore(tmp_path)


# ---- 검색 필터 ----
def _setup_store(conn, tmp_path, arts):
    db.insert_articles(conn, arts)
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    return store


def test_retrieve_filters_widen_search(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    arts = [_art(i, minor="해킹") for i in range(60)] + [_art(100 + i, minor="취약점") for i in range(3)]
    store = _setup_store(conn, tmp_path, arts)
    got = rag.retrieve("q", rag.Filters(minors=["취약점"]), store, conn, k=10, embed_fn=fake_embed)
    assert {a["category_minor"] for a in got} == {"취약점"} and len(got) == 3   # 넓혀서라도 3건 모두 찾음


def test_retrieve_date_filter_boundaries_kst(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    arts = [
        _art(1, pub="2026-11-04T14:59:59+00:00"),   # KST 11-04 23:59:59 → 제외(11-05 시작 전)
        _art(2, pub="2026-11-04T15:00:00+00:00"),   # KST 11-05 00:00:00 → 포함
        _art(3, pub="2026-11-05T14:59:59+00:00"),   # KST 11-05 23:59:59 → 포함
        _art(4, pub="2026-11-05T15:00:00+00:00"),   # KST 11-06 00:00:00 → 제외
        _art(5, pub=None),                          # 발행일 없음 → collected_at(11-06 00:00Z=KST 09시) 사용 → 제외
    ]
    store = _setup_store(conn, tmp_path, arts)
    f = rag.Filters(d_from=date(2026, 11, 5), d_to=date(2026, 11, 5))
    got = {a["id"] for a in rag.retrieve("q", f, store, conn, k=10, embed_fn=fake_embed)}
    assert got == {"id2", "id3"}
    rows, total = db.query_articles(conn, d_from=date(2026, 11, 5), d_to=date(2026, 11, 5))
    assert {r["id"] for r in rows} == {"id2", "id3"} and total == 2


def test_min_score_blocks_irrelevant(conn, tmp_path, monkeypatch):
    store = _setup_store(conn, tmp_path, [_art(i) for i in range(5)])
    monkeypatch.setattr(rag, "MIN_SCORE", 2.0)         # 코사인은 최대 1 → 아무것도 통과 못함
    assert rag.retrieve("q", rag.Filters(), store, conn, embed_fn=fake_embed) == []


def test_answer_not_found_without_llm_call(conn, tmp_path, monkeypatch):
    store = _setup_store(conn, tmp_path, [_art(1)])
    monkeypatch.setattr(rag, "MIN_SCORE", 2.0)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("LLM 호출되면 안 됨"))
    r = rag.answer("오늘 날씨", store=store, conn=conn, embed_fn=fake_embed)
    assert r.not_found and r.answer == rag.NOT_FOUND


def test_answer_extracts_valid_citations_only(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    store = _setup_store(conn, tmp_path, [_art(i) for i in range(3)])
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "내용입니다 [1][3][99]")
    r = rag.answer("q", store=store, conn=conn, embed_fn=fake_embed)
    assert r.cited == [1, 3] and len(r.sources) == 3 and not r.not_found


def test_query_keyword_escapes_wildcards(conn):
    db.insert_articles(conn, [_art(1, summary="100% 안전"), _art(2, summary="평범")])
    rows, total = db.query_articles(conn, keyword="%")
    assert total == 1 and rows[0]["id"] == "id1"


def test_vector_store_works_in_non_ascii_directory(conn, tmp_path):
    """faiss C++ I/O는 Windows 한글 경로에서 실패한다 → 회귀 방지."""
    d = tmp_path / "한글 폴더"
    db.insert_articles(conn, [_art(i) for i in range(3)])
    store = vector.VectorStore(d)
    vector.embed_pending(conn, store, fake_embed)
    assert vector.VectorStore(d).total == 3


def test_missing_api_key_gives_clear_config_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(llm.ConfigError, match="OPENAI_API_KEY"):
        llm.get_client()


def test_shorten_cuts_at_sentence_boundary_without_ellipsis():
    long = "첫 문장은 짧습니다. " + "두 번째 문장은 상당히 길어서 상한을 넘기게 됩니다 " * 6 + "끝났습니다."
    out = enrich.shorten(long, 40)
    assert out == "첫 문장은 짧습니다." and "…" not in out and "..." not in out
    assert enrich.shorten("짧은 요약입니다.", 40) == "짧은 요약입니다."
    only = "한 문장이 너무 길어서 자를 곳이 없는 요약 " * 5 + "입니다."
    assert enrich.shorten(only, 20) == " ".join(only.split())      # 문장 중간을 자르지 않음


def test_resummarize_replaces_only_on_success(conn, monkeypatch):
    a = _art(1, summary="x" * 200)
    b = _art(2, summary="y" * 200)
    c = _art(3, summary="짧은 요약")
    db.insert_articles(conn, [a, b, c])

    def fake(messages, **kw):
        body = messages[-1]["content"]
        # id2는 실패(누락) → 기존 요약 유지되어야 함
        return {"items": [{"id": i, "category": "해킹", "summary": "새 요약", "confidence": .9}
                          for i in ("id1", "id2", "id3") if f'id="{i}"' in body and i != "id2"]}

    monkeypatch.setattr(llm, "chat_json", fake)
    monkeypatch.setattr(enrich.content, "get_text", lambda url, snippet="": "본문")
    st = enrich.enrich_pending(conn, resummarize=True)
    assert st["pending"] == 2 and st["done"] == 1 and st["failed"] == 1     # 짧은 요약(id3)은 대상 아님
    got = {r["id"]: r["summary"] for r in conn.execute("SELECT id, summary FROM articles")}
    assert got["id1"] == "새 요약" and got["id2"] == "y" * 200 and got["id3"] == "짧은 요약"


def test_tags_included_in_embedding_text_and_chat_context():
    row = {**_art(1), "tags": '["CVE-2026-1111", "OpenSSH"]'}
    assert "CVE-2026-1111" in vector.embedding_text(row)
    assert "CVE-2026-1111" in rag._articles_block([row])
    assert "태그" not in vector.embedding_text({**_art(1), "tags": None})      # 태그 없음/깨진 값도 안전
    assert "태그" not in vector.embedding_text({**_art(1), "tags": "not json"})


# ---- R-10: 요약이 바뀐 뒤 벡터가 오래된 채 남는 문제 ----
def test_changed_summary_is_detected_and_vector_replaced(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(4)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    assert vector.check_consistency(conn, store)["stale_or_unverified_vectors"] == []
    before = store.index.reconstruct_n(0, store.total).copy()
    conn.execute("UPDATE articles SET summary='완전히 다른 새 요약' WHERE id='id2'")     # 요약 재생성을 흉내
    conn.commit()
    rep = vector.check_consistency(conn, store)
    assert rep["stale_or_unverified_vectors"] == ["id2"]                                # 검출됨
    stats = vector.embed_pending(conn, store, fake_embed)
    assert stats == {"pending": 1, "added": 0, "replaced": 1}                          # 1건만 교체, 총 개수 그대로
    assert store.total == 4
    after = store.index.reconstruct_n(0, store.total)
    pos = store.ids.index("id2")
    assert not np.allclose(before[pos], after[pos])                                     # 바뀐 행만 달라짐
    assert all(np.allclose(before[i], after[i]) for i in range(4) if i != pos)
    assert vector.check_consistency(conn, store)["stale_or_unverified_vectors"] == []
    assert vector.embed_pending(conn, store, fake_embed)["pending"] == 0               # 재실행하면 할 일 없음


def test_legacy_rows_without_hash_are_reembedded_once(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(2)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    conn.execute("UPDATE articles SET embed_hash=NULL")       # 해시 도입 전에 만든 데이터
    conn.commit()
    assert len(vector.check_consistency(conn, store)["stale_or_unverified_vectors"]) == 2
    assert vector.embed_pending(conn, store, fake_embed)["replaced"] == 2
    assert vector.embed_pending(conn, store, fake_embed)["pending"] == 0


def test_upsert_adds_and_replaces_in_one_call(tmp_path):
    store = vector.VectorStore(tmp_path)
    v = lambda *x: np.asarray([x], dtype="float32")        # noqa: E731
    assert store.upsert(["a", "b"], np.vstack([v(1, 0, 0), v(0, 1, 0)])) == (2, 0)
    assert store.upsert(["b", "c"], np.vstack([v(0, 0, 1), v(1, 1, 0)])) == (1, 1)   # b 교체, c 추가
    assert store.ids == ["a", "b", "c"]
    assert store.search(v(0, 0, 1)[0], 1)[0][0] == "b"                              # b의 벡터가 새 값
    assert store.upsert(["a", "a"], np.vstack([v(0, 1, 0), v(1, 0, 0)])) == (0, 1)   # 같은 id 반복은 마지막 것
    assert store.search(v(1, 0, 0)[0], 1)[0][0] == "a"


def test_rebuild_records_hashes(conn, tmp_path):
    db.insert_articles(conn, [_art(i) for i in range(3)])
    vector.rebuild(conn, tmp_path, fake_embed)
    assert vector.check_consistency(conn, vector.VectorStore(tmp_path))["stale_or_unverified_vectors"] == []


# ---- R-12 / R-13: 챗봇 응답 처리 ----
def test_is_not_found_handles_prefixed_refusal_but_not_long_answers():
    assert rag.is_not_found("수집된 기사에서 찾지 못했습니다.")
    assert rag.is_not_found("죄송하지만 수집된 기사에서 찾지 못했습니다.")
    assert not rag.is_not_found("가" * 200 + " 수집된 기사에서 찾지 못했습니다.")
    assert not rag.is_not_found("랜섬웨어 사례는 다음과 같습니다 [1]")


def test_sanitize_answer_strips_foreign_links_and_images_only():
    src = [{"url": "https://ok.example/a?utm_source=x"}]
    t = ("근거 [1][2] ![사진](http://evil.example/x.png?d=1) [여기](http://evil.example/p) "
         "[원문](https://ok.example/a) 직접 http://evil.example/z, 그리고 https://ok.example/a/ 입니다")
    out = rag.sanitize_answer(t, src)
    assert "evil.example" not in out
    assert "[1][2]" in out                                          # 인용 표기는 보존
    assert "사진" in out and "여기" in out                           # 대체 텍스트·링크 문구는 남김
    assert "[원문](https://ok.example/a)" in out and "https://ok.example/a/" in out   # 검색된 기사 링크는 허용
    assert "링크 제거됨" in out


def test_answer_applies_sanitizer_and_ignores_it_for_not_found(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    store = _setup_store(conn, tmp_path, [_art(1)])
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "답변 [1] ![x](http://evil.example/p.png) 끝")
    r = rag.answer("q", store=store, conn=conn, embed_fn=fake_embed)
    assert "evil.example" not in r.answer and r.cited == [1]


def test_chat_uses_short_interactive_timeouts(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    store = _setup_store(conn, tmp_path, [_art(1)])
    seen = {}

    def fake_chat(*a, **k):
        seen.update(k)
        return "답변 [1]"
    monkeypatch.setattr(llm, "chat", fake_chat)
    rag.answer("q", store=store, conn=conn, embed_fn=fake_embed)
    assert seen["timeout"] == llm.INTERACTIVE_TIMEOUT and seen["attempts"] == llm.INTERACTIVE_ATTEMPTS
    assert llm.INTERACTIVE_TIMEOUT * llm.INTERACTIVE_ATTEMPTS <= 90        # 장애 시 최악 대기 상한


def test_default_min_score_is_inside_measured_band():
    """실측(보안 질문 top1 최저 0.29, 무관 질문 최고 0.36) 안에서 관련 질문을 막지 않는 값이어야 한다."""
    import os
    if "RAG_MIN_SCORE" not in os.environ:
        assert 0.25 <= rag.MIN_SCORE <= 0.29
