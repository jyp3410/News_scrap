import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import rag  # noqa: E402
from core import llm  # noqa: E402
from scraper import enrich, fetch  # noqa: E402
from store import db, vector  # noqa: E402


def _art(i, **kw):
    base = {"id": f"id{i}", "source": f"S{i % 2}", "title": f"title {i}", "url": f"https://x.com/{i}",
            "published_at": "2026-11-05T03:00:00+00:00", "collected_at": "2026-11-06T00:00:00+00:00",
            "country": "KR", "language": "ko", "category_minor": "해킹", "summary": "요약"}
    return {**base, **kw}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def fake_embed(texts):
    return np.asarray([np.random.default_rng(abs(hash(t)) % (2**32)).normal(size=16) for t in texts], dtype="float32")


# ---- 판정 로직: 보수적이어야 한다 ----
@pytest.mark.parametrize("item,expected", [
    ({"relevant": False, "relevance_confidence": 0.95, "relevance_reason": "반도체"}, 0),   # 확실히 무관
    ({"relevant": "false", "relevance_confidence": 0.9}, 0),                                # 문자열 불리언
    ({"relevant": False, "relevance_confidence": 0.5}, 1),                                  # 확신도 낮음 → 유지
    ({"relevant": False}, 1),                                                               # 확신도 없음 → 유지
    ({"relevant": False, "relevance_confidence": "높음"}, 1),                               # 형식 오류 → 유지
    ({"relevant": True, "relevance_confidence": 0.99}, 1),
    ({}, 1),                                                                                # 필드 누락 → 유지
    ({"relevant": None, "relevance_confidence": 0.99}, 1),
    ({"relevant": "아니오", "relevance_confidence": 0.99}, 1),                              # 모르는 값 → 유지
])
def test_judge_relevance_is_conservative(item, expected):
    assert enrich.judge_relevance(item)[0] == expected


def test_judge_relevance_returns_reason_only_when_rejected():
    assert enrich.judge_relevance({"relevant": False, "relevance_confidence": 1, "relevance_reason": "사유"}) == (0, "사유")
    assert enrich.judge_relevance({"relevant": True}) == (1, None)


# ---- 수집 파이프라인: 무관 기사는 보관하되 숨긴다 ----
def _fake_chat_json(irrelevant_ids=(), low_conf_ids=()):
    def f(messages, **kw):
        body = messages[-1]["content"]
        items = []
        for i in range(10):
            if f'id="id{i}"' not in body:
                continue
            it = {"id": f"id{i}", "category": "기타", "confidence": 0.9, "reason": "r", "tags": [], "summary": "요약",
                  "relevant": True, "relevance_confidence": 0.9, "relevance_reason": ""}
            if f"id{i}" in irrelevant_ids:
                it.update(relevant=False, relevance_confidence=0.95, relevance_reason="보안과 무관한 반도체 기사")
            if f"id{i}" in low_conf_ids:
                it.update(relevant=False, relevance_confidence=0.4)
            items.append(it)
        return {"items": items}
    return f


def test_enrich_hides_irrelevant_but_keeps_row_and_skips_embedding(conn, tmp_path, monkeypatch):
    for i in range(4):
        db.insert_articles(conn, [_art(i, summary=None, category_minor=None)])
    monkeypatch.setattr(llm, "chat_json", _fake_chat_json(irrelevant_ids={"id1"}, low_conf_ids={"id2"}))
    monkeypatch.setattr(enrich.content, "get_text", lambda url, snippet="": "본문")
    st = enrich.enrich_pending(conn)
    assert st["done"] == 4 and st["rejected"] == 1 and st["rejected_by_source"] == {"S1": 1}
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 4           # 삭제하지 않음
    row = conn.execute("SELECT relevant, relevance_reason FROM articles WHERE id='id1'").fetchone()
    assert row["relevant"] == 0 and "반도체" in row["relevance_reason"]
    assert conn.execute("SELECT relevant FROM articles WHERE id='id2'").fetchone()[0] == 1   # 낮은 확신도는 유지
    rows, total = db.query_articles(conn)
    assert total == 3 and "id1" not in {r["id"] for r in rows}                       # 목록에서 숨김
    assert "id1" not in {r["id"] for r in db.pending_embedding(conn)}               # 임베딩 대상 제외
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    assert "id1" not in store.ids and store.total == 3
    # 재실행해도 숨긴 기사를 다시 LLM에 보내지 않는다
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: pytest.fail("재호출되면 안 됨"))
    assert enrich.enrich_pending(conn)["pending"] == 0


def test_hidden_article_is_not_refetched_as_new(conn):
    """숨긴 기사도 DB에 남으므로 다음 수집에서 신규로 취급되지 않는다 (재판정 비용 방지)."""
    db.insert_articles(conn, [_art(1)])
    db.set_relevance(conn, "id1", 0, "무관")
    assert "id1" in db.existing_ids(conn)


# ---- 기존 기사 재검증 ----
def _fake_revalidate(irrelevant=(), skip=()):
    def f(messages, **kw):
        body = messages[-1]["content"]
        items = []
        for i in range(10):
            if f'id="id{i}"' in body and f"id{i}" not in skip:
                bad = f"id{i}" in irrelevant
                items.append({"id": f"id{i}", "relevant": not bad, "relevance_confidence": 0.9,
                              "relevance_reason": "무관" if bad else ""})
        return {"items": items}
    return f


def test_revalidate_dry_run_changes_nothing(conn, monkeypatch):
    db.insert_articles(conn, [_art(i) for i in range(3)])
    monkeypatch.setattr(llm, "chat_json", _fake_revalidate(irrelevant={"id1"}))
    st = enrich.revalidate_relevance(conn, dry_run=True)
    assert st["rejected"] == 1 and st["kept"] == 2 and st["rejected_items"][0]["title"] == "title 1"
    assert conn.execute("SELECT COUNT(*) FROM articles WHERE relevant IS NOT NULL").fetchone()[0] == 0


def test_revalidate_apply_hides_and_is_idempotent(conn, monkeypatch):
    db.insert_articles(conn, [_art(i) for i in range(4)])
    monkeypatch.setattr(llm, "chat_json", _fake_revalidate(irrelevant={"id1"}, skip={"id3"}))
    st = enrich.revalidate_relevance(conn)
    assert (st["checked"], st["rejected"], st["missing"]) == (3, 1, 1)
    assert conn.execute("SELECT relevant FROM articles WHERE id='id3'").fetchone()[0] is None   # 응답 누락 → 다음에 재시도
    monkeypatch.setattr(llm, "chat_json", _fake_revalidate())
    st2 = enrich.revalidate_relevance(conn)
    assert st2["checked"] == 1                                   # 미판정이던 id3만 처리
    assert conn.execute("SELECT relevant FROM articles WHERE id='id1'").fetchone()[0] == 0   # 이미 숨긴 건 그대로


def test_revalidate_uses_title_and_summary_not_body(conn, monkeypatch):
    seen = {}
    db.insert_articles(conn, [_art(1, summary="핵심 요약문")])

    def f(messages, **kw):
        seen["body"] = messages[-1]["content"]
        return {"items": []}
    monkeypatch.setattr(llm, "chat_json", f)
    enrich.revalidate_relevance(conn)
    assert "핵심 요약문" in seen["body"] and "title 1" in seen["body"]


# ---- 숨긴 기사는 검색·인덱스에서도 빠진다 ----
def test_retrieve_excludes_hidden_even_if_vector_remains(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "MIN_SCORE", -1.0)
    db.insert_articles(conn, [_art(i) for i in range(5)])
    store = vector.VectorStore(tmp_path)
    vector.embed_pending(conn, store, fake_embed)
    db.set_relevance(conn, "id2", 0, "무관")                       # 인덱스에는 아직 벡터가 남아 있음
    got = {a["id"] for a in rag.retrieve("q", rag.Filters(), store, conn, k=10, embed_fn=fake_embed)}
    assert "id2" not in got and len(got) == 4
    rep = vector.check_consistency(conn, store)
    assert rep["in_index_but_hidden"] == ["id2"]                  # 정합성 검사가 잔존 벡터를 알려줌
    vector.rebuild(conn, tmp_path, fake_embed)
    store2 = vector.VectorStore(tmp_path)
    assert "id2" not in store2.ids and store2.total == 4
    assert vector.check_consistency(conn, store2)["in_index_but_hidden"] == []


def test_dashboard_and_sources_exclude_hidden(conn):
    db.insert_articles(conn, [_art(1, source="A"), _art(2, source="B")])
    db.set_relevance(conn, "id2", 0, "무관")
    assert db.distinct_sources(conn) == ["A"] and db.hidden_count(conn) == 1
    assert [r["id"] for r in db.list_articles(conn)] == ["id1"]


def test_legacy_db_gets_new_columns(tmp_path):
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL UNIQUE,"
              " published_at TEXT, collected_at TEXT NOT NULL, country TEXT, language TEXT, category_major TEXT NOT NULL DEFAULT '정보보안',"
              " category_minor TEXT, tags TEXT, summary TEXT, embedded INTEGER NOT NULL DEFAULT 0)")
    c.execute("INSERT INTO articles (id,source,title,url,collected_at,summary) VALUES ('a','S','t','https://x/a','2026-01-01T00:00:00+00:00','s')")
    c.commit()
    c.close()
    conn = db.connect(p)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    assert {"relevant", "relevance_reason"} <= cols
    assert db.query_articles(conn)[1] == 1                          # 기존 기사는 미판정(NULL)이라 계속 보인다
    assert len(db.pending_revalidate(conn)) == 1
    conn.close()


# ---- 제외 뉴스사 설정 ----
CFG = {"exclude_sources": ["  dark READING ", "데이터넷"],
       "sources": [{"name": "Dark Reading", "url": "u1"}, {"name": "데이터넷", "url": "u2"},
                   {"name": "데일리시큐", "url": "u3"}, {"name": "끈 소스", "url": "u4", "enabled": False},
                   {"name": "The Record", "url": "u5", "enabled": True}]}


def test_exclude_sources_and_enabled_flag():
    assert [s["name"] for s in fetch.load_sources(CFG)] == ["데일리시큐", "The Record"]


def test_exclude_sources_empty_or_missing_keeps_everything():
    assert len(fetch.load_sources({"sources": CFG["sources"][:3]})) == 3
    assert len(fetch.load_sources({"exclude_sources": None, "sources": CFG["sources"][:3]})) == 3


def test_exclude_sources_typo_is_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="scraper.fetch"):
        out = fetch.load_sources({"exclude_sources": ["데일리 시큐"], "sources": CFG["sources"][:3]})
    assert len(out) == 3 and "오타" in caplog.text


def test_excluded_sources_are_not_fetched(monkeypatch):
    monkeypatch.setattr(fetch, "load_yaml", lambda name: {"exclude_sources": ["The Hacker News"],
                                                          "sources": [{"name": "The Hacker News", "url": "u"},
                                                                      {"name": "Krebs on Security", "url": "u2"}]}
                        if name == "sources.yaml" else {"security_keywords": {}})
    called = []
    monkeypatch.setattr(fetch, "fetch_source", lambda s, k=None, since=None: called.append(s["name"]) or [])
    monkeypatch.setattr(fetch, "REQUEST_DELAY_SEC", 0)
    _, report = fetch.fetch_all()
    assert called == ["Krebs on Security"] and list(report) == ["Krebs on Security"]


def test_shipped_sources_yaml_is_valid():
    cfg = fetch.load_yaml("sources.yaml")
    assert isinstance(cfg.get("exclude_sources"), list)
    assert len(fetch.load_sources()) >= 1


# ---- R-06: 요약 속 '자료의 한계' 문장 제거 ----
@pytest.mark.parametrize("meta", [
    "본문이 없어 구체적인 공격 방식이나 피해 규모는 알 수 없습니다.",
    "구체적인 논점이나 권고 사항은 제목만으로 확인할 수 없습니다.",
    "구체적인 위협 유형이나 확산 경로는 제목에 제시되지 않았다.",
    "제공된 정보는 제목뿐이라 자세한 내용은 확인하기 어렵다.",
])
def test_strip_meta_removes_limitation_sentences(meta):
    out = enrich.strip_meta("APT29가 워터링홀 공격으로 모바일 사용자를 노렸다. " + meta, "제목")
    assert out == "APT29가 워터링홀 공격으로 모바일 사용자를 노렸다."


def test_strip_meta_falls_back_to_title_and_keeps_legit_phrases():
    assert enrich.strip_meta("본문이 없어 확인할 수 없습니다.", "Why Caribbean Threats Cascade") == "Why Caribbean Threats Cascade"
    legit = "행안부는 원문 다운로드를 중단했다. 악용 사례는 확인되지 않았다. 피해 규모는 공개되지 않았다."
    assert enrich.strip_meta(legit, "t") == legit            # 기사 내용인 표현은 그대로


def test_validate_item_never_returns_meta_summary():
    v = enrich.validate_item({"category": "기타", "summary": "본문이 없어 알 수 없습니다.", "relevant": True},
                             enrich.minors(), "", "원래 제목")
    assert v["summary"] == "원래 제목"


# ---- R-08: method 검증 ----
def test_unsupported_method_fails_loudly_not_silently():
    with pytest.raises(ValueError, match="지원하지 않는 수집 방식"):
        fetch.fetch_source({"name": "X", "url": "https://x.example/", "method": "html"})


def test_unsupported_method_is_isolated_per_source(monkeypatch):
    monkeypatch.setattr(fetch, "load_yaml", lambda name: {"sources": [
        {"name": "H", "url": "u", "method": "html"}, {"name": "R", "url": "u2"}]} if name == "sources.yaml"
        else {"security_keywords": {}})
    monkeypatch.setattr(fetch.http, "get", lambda url: (_ for _ in ()).throw(RuntimeError("네트워크 호출 금지")) if url == "u" else
                        type("R", (), {"content": b"<rss><channel><item><title>t</title><link>http://x/1</link></item></channel></rss>"})())
    monkeypatch.setattr(fetch, "REQUEST_DELAY_SEC", 0)
    arts, report = fetch.fetch_all()
    assert report["H"]["ok"] is False and "지원하지 않는" in report["H"]["error"]
    assert report["R"]["ok"] is True and len(arts) == 1
