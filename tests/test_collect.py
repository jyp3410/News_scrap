"""수집·HTTP·파이프라인·워크플로 테스트 (REVIEW R-17)."""
import importlib.util
import sys
from pathlib import Path

import feedparser
import numpy as np
import pytest
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from core import llm  # noqa: E402
from scraper import content, enrich, fetch, http  # noqa: E402
from store import db, vector  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────── 발행일 파싱 ─────────────
def _entry(pub):
    xml = f'<rss version="2.0"><channel><item><title>t</title><link>http://x/1</link><pubDate>{pub}</pubDate></item></channel></rss>'
    return feedparser.parse(xml).entries[0]


@pytest.mark.parametrize("raw,offset,expected", [
    ("Fri, 6 Nov 2026 12:14:57 +0900", 0, "2026-11-06T03:14:57+00:00"),     # 숫자 오프셋은 소스 기본값보다 우선
    ("Thu, 05 Nov 2026 22:08:00 GMT", 9, "2026-11-05T22:08:00+00:00"),       # GMT는 UTC
    ("2026-11-06 09:13:59", 9, "2026-11-06T00:13:59+00:00"),                 # 표기 없음 → 소스 시간대(KST)
    ("2026-11-06 09:13:59", 0, "2026-11-06T09:13:59+00:00"),                 # 표기 없음 → 소스 시간대(UTC)
    ("2026-11-06T09:13:59+09:00", 0, "2026-11-06T00:13:59+00:00"),
    ("2026-11-06T00:13:59Z", 9, "2026-11-06T00:13:59+00:00"),
    ("Fri, 06 Nov 2026 09:13:59 KST", 0, "2026-11-06T00:13:59+00:00"),       # R-11: feedparser가 모르는 약어 보정
    ("Fri, 06 Nov 2026 09:13:59 JST", 0, "2026-11-06T00:13:59+00:00"),
    ("Fri, 06 Nov 2026 09:13:59 EST", 9, "2026-11-06T14:13:59+00:00"),       # feedparser가 아는 약어는 그대로
    ("Fri, 06 Nov 2026 09:13:59 PST", 9, "2026-11-06T17:13:59+00:00"),
    ("2026-11-06", 9, "2026-11-05T15:00:00+00:00"),                           # 날짜만 → 자정(소스 시간대)
])
def test_parse_published(raw, offset, expected):
    assert fetch._parse_published(_entry(raw), offset) == expected


def test_parse_published_returns_none_without_date():
    entry = feedparser.parse('<rss version="2.0"><channel><item><title>t</title><link>http://x/1</link></item></channel></rss>').entries[0]
    assert fetch._parse_published(entry, 9) is None


# ───────────── 피드 파싱 ─────────────
RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>랜섬웨어 &lt;b&gt;공격&lt;/b&gt; 발생</title><link>https://n.example/a?utm_source=x</link>
  <description>짧은 요약</description><content:encoded><![CDATA[<p>본문 <b>전체</b> 내용입니다</p>]]></content:encoded>
  <pubDate>Fri, 06 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>링크 없는 항목</title><description>x</description></item>
<item><title></title><link>https://n.example/empty</link></item>
<item><title>신제품 출시 소식</title><link>https://n.example/c</link><description>일반 IT 소식입니다</description></item>
<item><title>취약점 패치 공개</title><link>https://n.example/d</link><description>보안 업데이트</description></item>
</channel></rss>"""


class _Resp:
    def __init__(self, content=b"", status=200, text="", headers=None):
        self.content, self.status_code, self.text, self.headers = content, status, text, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_fetch_source_parses_cleans_and_skips_bad_items(monkeypatch):
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(RSS.encode("utf-8")))
    arts = fetch.fetch_source({"name": "N", "url": "u", "country": "KR", "language": "ko", "utc_offset": 9})
    assert [a["url"] for a in arts] == ["https://n.example/a", "https://n.example/c", "https://n.example/d"]  # 링크·제목 없는 항목 제외
    a = arts[0]
    assert a["title"] == "랜섬웨어 공격 발생" and "<" not in a["title"]            # HTML 제거
    assert a["snippet"] == "본문 전체 내용입니다"                                    # content:encoded 우선, 태그 제거
    assert a["country"] == "KR" and a["language"] == "ko" and a["published_at"] == "2026-11-06T09:00:00+00:00"
    assert a["id"] == fetch.article_id("https://n.example/a")                        # 추적 파라미터를 뗀 URL로 id 생성


def test_fetch_source_keyword_filter_only_when_enabled(monkeypatch):
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(RSS.encode("utf-8")))
    kw = {"ko": ["랜섬웨어", "취약점", "보안"], "en": []}
    src = {"name": "N", "url": "u", "keyword_filter": True}
    titles = [a["title"] for a in fetch.fetch_source(src, kw)]
    assert titles == ["랜섬웨어 공격 발생", "취약점 패치 공개"]                        # '신제품 출시 소식' 제외
    assert len(fetch.fetch_source({"name": "N", "url": "u"}, kw)) == 3               # 필터 꺼짐 → 전부


def test_fetch_source_rejects_non_rss_response(monkeypatch):
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(b"<html><body>not a feed</body></html>"))
    with pytest.raises(ValueError, match="RSS"):
        fetch.fetch_source({"name": "N", "url": "u"})


def test_snippet_is_capped(monkeypatch):
    big = RSS.replace("짧은 요약", "가" * 5000).replace("<![CDATA[<p>본문 <b>전체</b> 내용입니다</p>]]>", "")
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(big.encode("utf-8")))
    assert len(fetch.fetch_source({"name": "N", "url": "u"})[0]["snippet"]) <= 2000


# ───────────── HTTP: 재시도 · robots · 간격 ─────────────
@pytest.fixture
def net(monkeypatch):
    """requests.get을 응답 큐로 대체하고 대기·상태를 초기화한다."""
    calls = []
    queue = []

    def fake_get(url, **kw):
        calls.append(url)
        r = queue.pop(0) if queue else _Resp(b"ok", 200, "ok")
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(http.requests, "get", fake_get)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    monkeypatch.setattr(http, "HOST_MIN_INTERVAL", 0)
    http._robots_cache.clear()
    http._host_last.clear()
    return type("Net", (), {"calls": calls, "queue": queue})


def test_get_retries_server_errors_then_succeeds(net):
    net.queue[:] = [_Resp(status=404, text=""), _Resp(status=503), _Resp(status=502), _Resp(b"ok", 200)]   # 첫 응답은 robots.txt(404=허용)
    assert http.get("https://a.example/feed").content == b"ok"
    assert net.calls == ["https://a.example/robots.txt"] + ["https://a.example/feed"] * 3


def test_get_does_not_retry_client_errors(net):
    net.queue[:] = [_Resp(status=404), _Resp(status=404)]
    with pytest.raises(requests.HTTPError):
        http.get("https://a.example/gone")
    assert net.calls.count("https://a.example/gone") == 1


def test_get_gives_up_after_max_attempts(net):
    net.queue[:] = [_Resp(status=404)] + [_Resp(status=500)] * 5
    with pytest.raises(requests.HTTPError):
        http.get("https://a.example/down")
    assert net.calls.count("https://a.example/down") == http.MAX_ATTEMPTS


def test_robots_disallow_blocks_without_fetching_the_page(net):
    robots = "User-agent: *\nDisallow: /private\n"
    net.queue[:] = [_Resp(status=200, text=robots, headers={"content-type": "text/plain"})]
    with pytest.raises(http.RobotsDisallowed):
        http.get("https://a.example/private/rss.xml")
    assert net.calls == ["https://a.example/robots.txt"]                    # 본문 요청은 하지 않음
    assert http.allowed("https://a.example/public")                          # 캐시된 규칙: 다른 경로는 허용


def test_robots_403_means_disallow_all_and_404_means_allow(net):
    net.queue[:] = [_Resp(status=403)]
    assert not http.allowed("https://blocked.example/x")
    net.queue[:] = [_Resp(status=404)]
    assert http.allowed("https://open.example/x")


def test_robots_fetch_failure_is_treated_as_allowed(net):
    net.queue[:] = [requests.ConnectionError("down")]
    assert http.allowed("https://flaky.example/x")


def test_host_throttle_enforces_min_interval(monkeypatch):
    sleeps, clock = [], [100.0]
    monkeypatch.setattr(http, "HOST_MIN_INTERVAL", 1.0)
    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(http.time, "sleep", lambda s: (sleeps.append(round(s, 2)), clock.__setitem__(0, clock[0] + s)))
    http._host_last.clear()
    http._throttle("https://a.example/1")            # 첫 요청: 대기 없음
    clock[0] += 0.3
    http._throttle("https://a.example/2")            # 0.3초 뒤: 0.7초 더 기다림
    http._throttle("https://b.example/1")            # 다른 호스트: 대기 없음
    assert sleeps == [0.7]


def test_content_get_text_uses_snippet_when_long_and_falls_back_on_failure(monkeypatch):
    monkeypatch.setattr(content.http, "get", lambda url: pytest.fail("긴 스니펫이면 요청하지 않는다"))
    assert content.get_text("https://x/1", "가" * 400) == "가" * 400
    monkeypatch.setattr(content.http, "get", lambda url: (_ for _ in ()).throw(requests.ConnectionError("x")))
    assert content.get_text("https://x/2", "짧은 스니펫") == "짧은 스니펫"       # 실패해도 예외 없이 스니펫으로
    monkeypatch.setattr(content.http, "get", lambda url: (_ for _ in ()).throw(http.RobotsDisallowed("no")))
    assert content.get_text("https://x/3", "") == ""


# ───────────── 파이프라인 통합 ─────────────
def _fake_chat_json(messages, **kw):
    body = messages[-1]["content"]
    ids = [w.split('"')[1] for w in body.split("<article id=")[1:]]
    return {"items": [{"id": i, "category": "해킹", "confidence": 0.9, "reason": "r", "tags": [],
                       "summary": "요약", "relevant": True, "relevance_confidence": 0.9} for i in ids]}


def test_pipeline_is_idempotent_end_to_end(tmp_path, monkeypatch):
    rp = _load_script("run_pipeline")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(rp, "VectorStore", lambda: vector.VectorStore(tmp_path))
    arts = [{"id": fetch.article_id(f"https://x.example/{i}"), "source": "S", "title": f"서로 다른 제목 {i} 번째 기사",
             "url": f"https://x.example/{i}", "published_at": "2026-11-05T01:00:00+00:00",
             "collected_at": "2026-11-06T00:00:00+00:00", "country": "KR", "language": "ko", "snippet": "본문" * 200}
            for i in range(7)]
    monkeypatch.setattr(rp, "fetch_all", lambda **kw: ([dict(a) for a in arts], {"S": {"ok": True, "count": 7, "fetched_at": datetime.now(timezone.utc)}}))
    llm_calls = []
    monkeypatch.setattr(llm, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(llm, "chat_json", lambda m, **k: (llm_calls.append(1), _fake_chat_json(m))[1])
    monkeypatch.setattr(llm, "embed", lambda texts, **k: np.asarray([[float(i == j) for j in range(8)] for i, _ in enumerate(texts)], dtype="float32"))
    monkeypatch.setattr(enrich.content, "get_text", lambda url, snippet="": snippet)

    r1 = rp.run()
    assert r1["new"] == 7 and r1["enrich"]["done"] == 7 and r1["embed"]["added"] == 7
    first_calls = len(llm_calls)
    ids_before = list(vector.VectorStore(tmp_path).ids)

    r2 = rp.run()                                                          # 같은 기사가 다시 수집돼도
    assert r2["new"] == 0 and r2["enrich"]["pending"] == 0 and r2["embed"] == {"pending": 0, "added": 0, "replaced": 0}
    assert len(llm_calls) == first_calls                                   # LLM 재호출 없음
    assert list(vector.VectorStore(tmp_path).ids) == ids_before            # 인덱스 불변
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 7
    assert vector.check_consistency(conn, vector.VectorStore(tmp_path))["stale_or_unverified_vectors"] == []
    conn.close()


def test_pipeline_skip_llm_does_not_need_api_key(tmp_path, monkeypatch):
    rp = _load_script("run_pipeline")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(rp, "fetch_all", lambda **kw: ([{"id": "a", "source": "S", "title": "제목", "url": "https://x/a",
                                                    "collected_at": "2026-11-06T00:00:00+00:00"}], {}))
    monkeypatch.setattr(llm, "get_client", lambda *a, **k: pytest.fail("키를 요구하면 안 됨"))
    assert rp.run(skip_llm=True)["new"] == 1


def test_pipeline_without_api_key_keeps_collected_articles(tmp_path, monkeypatch):
    rp = _load_script("run_pipeline")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(rp, "fetch_all", lambda **kw: ([{"id": "a", "source": "S", "title": "제목", "url": "https://x/a",
                                                    "collected_at": "2026-11-06T00:00:00+00:00"}], {}))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(llm.ConfigError, match="OPENAI_API_KEY"):
        rp.run()
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1   # 수집 결과는 보존
    conn.close()


# ───────────── 뉴스사 삭제 스크립트 ─────────────
def test_remove_source_dry_run_then_apply(tmp_path):
    rs = _load_script("remove_source")
    conn = db.connect(tmp_path / "t.db")
    arts = [{"id": f"i{i}", "source": "A" if i < 3 else "B", "title": f"t{i}", "url": f"https://x/{i}",
             "collected_at": "2026-11-06T00:00:00+00:00"} for i in range(5)]
    db.insert_articles(conn, arts)
    db.set_relevance(conn, "i0", 0, "무관")
    assert rs.remove_source(conn, "A") == {"source": "A", "found": 3, "of_which_hidden": 1, "deleted": 0}
    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 5          # 미리보기는 아무것도 지우지 않음
    assert rs.remove_source(conn, "A", apply=True)["deleted"] == 3
    assert [r[0] for r in conn.execute("SELECT DISTINCT source FROM articles")] == ["B"]
    assert rs.remove_source(conn, "없는 뉴스사")["found"] == 0
    assert rs.remove_source(conn, "a")["found"] == 0                                  # 정확히 일치할 때만 (오삭제 방지)
    conn.close()


# ───────────── 워크플로 / 저장소 설정 ─────────────
@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "collect.yml").read_text(encoding="utf-8"))


def _steps(wf):
    return {s.get("name") or s.get("uses") or s.get("run"): s for s in wf["jobs"]["collect"]["steps"]}


def test_workflow_commits_partial_results_even_if_pipeline_fails(workflow):
    steps = _steps(workflow)
    pipeline, commit = steps["Run pipeline"], steps["Commit data"]
    assert pipeline.get("continue-on-error") is True and pipeline.get("id") == "pipeline"
    assert commit.get("if") == "always()"                                            # R-03: 실패해도 커밋 단계 실행
    final = steps["Fail the job if the pipeline failed"]
    assert "steps.pipeline.outcome == 'failure'" in final["if"]                      # 그래도 잡은 실패로 표시
    order = list(steps)
    assert order.index("Run pipeline") < order.index("Commit data") < order.index("Fail the job if the pipeline failed")


def test_workflow_push_retries_and_prefers_ci_data_on_conflict(workflow):
    script = _steps(workflow)["Commit data"]["run"]
    assert "git push" in script and "pull --rebase -X theirs" in script and "rebase --abort" in script
    assert "git add data/news.db data/faiss.index data/id_map.json" in script        # 데이터 파일만 커밋


def test_workflow_basics(workflow):
    on = workflow.get("on") or workflow.get(True)                                    # PyYAML이 'on'을 True로 읽는다
    assert "workflow_dispatch" in on
    minute, hour, dom, month, dow = on["schedule"][0]["cron"].split()
    assert minute == "0" and (dom, month, dow) == ("*", "*", "*")
    utc_hours = sorted(int(h) for h in hour.split(","))
    kst_hours = sorted((h + 9) % 24 for h in utc_hours)
    assert kst_hours == [0, 6, 12, 18]                                              # 하루 4번, 한국시간 자정부터 6시간마다
    assert len({(b - a) % 24 for a, b in zip(kst_hours, kst_hours[1:] + kst_hours[:1])}) == 1   # 간격이 일정
    assert workflow["permissions"]["contents"] == "write"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "${{ secrets.OPENAI_API_KEY }}" in _steps(workflow)["Run pipeline"]["env"]["OPENAI_API_KEY"]


def test_repo_settings_protect_secrets_and_data():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gi and "*.tmp" in gi
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "data/news.db binary" in ga and "data/faiss.index binary" in ga
    assert "OPENAI_API_KEY=" in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "sk-" not in (ROOT / ".env.example").read_text(encoding="utf-8")


# ───────────── UI 라벨 (R-20) ─────────────
def test_timeline_day_header_says_displayed_count():
    import ui
    h = ui.timeline([{"source": "S", "title": "t", "summary": "s", "url": "https://x/1", "category_minor": "해킹",
                      "country": "KR", "published_at": "2026-11-05T03:00:00+00:00", "collected_at": "x"}])
    assert "표시 1건" in h


# ───────────── 행사 검색 소스 (SECON · ISEC · RSA …) ─────────────
from datetime import datetime, timedelta, timezone  # noqa: E402


def _bing_item(title, real_url, outlet, pub, summary="요약"):
    link = "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=abc&url=" + real_url.replace(":", "%3a").replace("/", "%2f") + "&c=1&mkt=ko-kr"
    return (f"<item><title>{title}</title><link>{link}</link><description>{summary}</description>"
            f"<pubDate>{pub}</pubDate><News:Source>{outlet}</News:Source></item>")


def _bing_feed(items):
    return ('<?xml version="1.0" encoding="utf-8"?><rss version="2.0" xmlns:News="https://www.bing.com/news/search">'
            "<channel><title>q</title>" + "".join(items) + "</channel></rss>").encode("utf-8")


def _rfc(dt):
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


NOW = datetime.now(timezone.utc)
EV_SRC = {"name": "행사검색: SECON", "url": "https://www.bing.com/news/search?q=x&format=rss", "kind": "event_search",
          "utc_offset": 0, "max_age_days": 180, "limit": 15}


def test_event_search_unwraps_real_url_and_uses_outlet_name(monkeypatch):
    feed = _bing_feed([_bing_item("SECON 2026 개막", "https://www.etnews.com/20260319000369", "Etnews.com", _rfc(NOW - timedelta(days=30))),
                       _bing_item("RSA 2026 trends", "https://www.itworld.co.kr/article/1", "ITWorld", _rfc(NOW - timedelta(days=5)))])
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(feed))
    arts = fetch.fetch_source(EV_SRC)
    assert [a["url"] for a in arts] == ["https://www.etnews.com/20260319000369", "https://www.itworld.co.kr/article/1"]   # 원문 주소
    assert [a["source"] for a in arts] == ["Etnews.com", "ITWorld"]                       # 검색어가 아니라 실제 매체명
    assert arts[0]["id"] == fetch.article_id("https://www.etnews.com/20260319000369")     # 기존 기사와 같은 id → 중복 제거됨
    assert (arts[0]["country"], arts[0]["language"]) == ("KR", "ko") and (arts[1]["country"], arts[1]["language"]) == ("GLOBAL", "en")


def test_event_search_drops_old_undated_and_wrongly_dated_items(monkeypatch):
    feed = _bing_feed([
        _bing_item("최근 행사 소식", "https://a.example/1", "A", _rfc(NOW - timedelta(days=10))),
        _bing_item("너무 오래된 기사", "https://a.example/2", "A", _rfc(NOW - timedelta(days=400))),
        _bing_item("2017년으로 잘못 표기된 기사", "https://a.example/3", "A", "Sat, 25 Mar 2017 17:00:00 GMT"),
        "<item><title>날짜 없는 기사</title><link>http://www.bing.com/news/apiclick.aspx?url=https%3a%2f%2fa.example%2f4</link></item>",
    ])
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(feed))
    assert [a["title"] for a in fetch.fetch_source(EV_SRC)] == ["최근 행사 소식"]


def test_event_search_respects_per_query_limit_and_falls_back_to_host(monkeypatch):
    items = [_bing_item(f"기사 {i}", f"https://www.host.example/{i}", "", _rfc(NOW - timedelta(days=1))) for i in range(10)]
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(_bing_feed(items)))
    arts = fetch.fetch_source({**EV_SRC, "limit": 3})
    assert len(arts) == 3 and arts[0]["source"] == "host.example"                         # 매체명이 없으면 호스트


def test_bing_real_url_leaves_normal_links_alone():
    assert fetch._bing_real_url("https://www.etnews.com/1") == "https://www.etnews.com/1"
    assert fetch._bing_real_url("http://www.bing.com/news/apiclick.aspx?url=javascript%3aalert(1)") != "javascript:alert(1)"   # http(s)만 허용


def test_event_search_expansion_toggle_and_exclusion():
    cfg = {"sources": [{"name": "S", "url": "u"}], "exclude_sources": [],
           "event_search": {"enabled": True, "queries": ["SECON 보안", "RSA Conference"], "max_age_days": 90, "per_query_limit": 5}}
    names = [s["name"] for s in fetch.load_sources(cfg)]
    assert names == ["S", "행사검색: SECON 보안", "행사검색: RSA Conference"]
    ev = [s for s in fetch.load_sources(cfg) if s.get("kind") == "event_search"]
    assert "bing.com/news/search" in ev[0]["url"] and "format=rss" in ev[0]["url"] and ev[0]["max_age_days"] == 90 and ev[0]["limit"] == 5
    assert "SECON" in ev[0]["url"] and " " not in ev[0]["url"]                                 # 검색어가 URL 인코딩됨
    assert [s["name"] for s in fetch.load_sources({**cfg, "event_search": {**cfg["event_search"], "enabled": False}})] == ["S"]
    assert [s["name"] for s in fetch.load_sources({"sources": cfg["sources"]})] == ["S"]       # 설정이 없으면 기존 동작 그대로
    ex = fetch.load_sources({**cfg, "exclude_sources": ["행사검색: RSA Conference"]})
    assert [s["name"] for s in ex] == ["S", "행사검색: SECON 보안"]                              # 검색어 단위로도 끌 수 있음


def test_event_search_outlets_follow_exclude_list(monkeypatch):
    feed = _bing_feed([_bing_item("행사 소식 하나", "https://a.example/1", "광고성 매체", _rfc(NOW - timedelta(days=1))),
                       _bing_item("행사 소식 둘", "https://b.example/1", "전자신문", _rfc(NOW - timedelta(days=1)))])
    monkeypatch.setattr(fetch, "load_yaml", lambda name: {"exclude_sources": [" 광고성 매체 "], "sources": [],
                                                          "event_search": {"enabled": True, "queries": ["q"]}}
                        if name == "sources.yaml" else {"security_keywords": {}})
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(feed))
    monkeypatch.setattr(fetch, "REQUEST_DELAY_SEC", 0)
    arts, report = fetch.fetch_all()
    assert [a["source"] for a in arts] == ["전자신문"] and report["행사검색: q"]["count"] == 1   # 원치 않는 매체는 검색으로도 안 들어옴


def test_shipped_event_search_config_covers_requested_events():
    cfg = fetch.load_yaml("sources.yaml")["event_search"]
    qs = " ".join(cfg["queries"]).upper()
    for event in ("SECON", "ISEC", "RSAC"):      # RSA Conference는 보도에서 RSAC로 불린다
        assert event in qs
    assert cfg["enabled"] is True and 30 <= cfg["max_age_days"] <= 365


def test_conference_prompt_names_the_events():
    assert all(name in enrich.SYSTEM for name in ("SECON", "ISEC", "RSA Conference", "Black Hat"))
    assert "SECON" in enrich.RELEVANCE_RULES


def test_guess_language_by_script():
    assert fetch._guess_language("SECON 2026 개막") == "ko"
    assert fetch._guess_language("DEF CON CTF世界一までの20年") == "ja"
    assert fetch._guess_language("韩国战队获亚军") == "zh"
    assert fetch._guess_language("Black Hat Europe 2026 keynotes") == "en"
    assert fetch._guess_language("") == "en"


# ───────────── 수집 기준 시간 (실행 시각 − 6h30m, 직전 성공 − 30m 중 더 이른 쪽) ─────────────
T0 = datetime(2026, 11, 6, 6, 0, tzinfo=timezone.utc)      # 실행 시각 (KST 15:00)
H = lambda h: timedelta(hours=h)                            # noqa: E731


def test_compute_since_first_run_has_no_limit():
    assert fetch.compute_since(T0, None) is None                                   # 처음 수집하는 소스는 전부


def test_compute_since_normal_cadence_is_run_time_minus_6h30m():
    assert fetch.compute_since(T0, T0 - H(6)) == T0 - H(6.5)                       # 정확히 6시간 간격이면 두 기준이 같다
    assert fetch.compute_since(T0, T0 - H(1)) == T0 - H(6.5)                       # 수동으로 자주 돌려도 6h30m보다 좁아지지 않는다


def test_compute_since_widens_after_a_delay_or_failure():
    assert fetch.compute_since(T0, T0 - H(12)) == T0 - H(12.5)                     # 한 회차를 건너뛰었으면 그만큼 거슬러 올라간다
    assert fetch.compute_since(T0, T0 - H(30)) == T0 - H(30.5)                     # 하루 넘게 멈췄어도 빠지는 구간이 없다


def test_compute_since_uses_configured_values():
    assert fetch.compute_since(T0, T0 - H(6), lookback_hours=8, overlap_minutes=60) == T0 - H(8)
    assert fetch.window_settings({"collection": {"lookback_hours": 8, "overlap_minutes": 15}}) == (8.0, 15.0)
    assert fetch.window_settings({}) == (6.5, 30.0)


def test_shipped_window_config_matches_6h_schedule():
    lookback, overlap = fetch.window_settings()
    assert (lookback, overlap) == (6.5, 30.0)


FEED = """<rss version="2.0"><channel>
<item><title>오래된 기사</title><link>https://x/old</link><pubDate>{old}</pubDate></item>
<item><title>경계 직전 기사</title><link>https://x/edge</link><pubDate>{edge}</pubDate></item>
<item><title>최근 기사</title><link>https://x/new</link><pubDate>{new}</pubDate></item>
<item><title>날짜 없는 기사</title><link>https://x/nodate</link></item>
</channel></rss>"""


def _since_feed(now):
    return FEED.format(old=_rfc(now - H(10)), edge=_rfc(now - H(6.4)), new=_rfc(now - H(1))).encode("utf-8")


def test_fetch_source_since_filters_old_items_but_keeps_undated(monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(_since_feed(now)))
    got = [a["title"] for a in fetch.fetch_source({"name": "N", "url": "u"}, since=now - H(6.5))]
    assert got == ["경계 직전 기사", "최근 기사", "날짜 없는 기사"]                      # 6h24m 전 기사는 포함, 10시간 전은 제외, 날짜 모르면 유지
    assert len(fetch.fetch_source({"name": "N", "url": "u"})) == 4                      # since가 없으면 전부


def test_since_does_not_apply_to_event_search(monkeypatch):
    now = datetime.now(timezone.utc)
    feed = _bing_feed([_bing_item("지난 행사 기사", "https://a.example/1", "A", _rfc(now - H(24 * 90)))])
    monkeypatch.setattr(fetch.http, "get", lambda url: _Resp(feed))
    assert len(fetch.fetch_source(EV_SRC, since=now - H(6.5))) == 1                    # 행사 검색은 max_age_days만 본다


def test_fetch_all_applies_per_source_state_and_reports_fetch_time(monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    seen = {}
    monkeypatch.setattr(fetch, "load_yaml", lambda name: {"sources": [], "collection": {}} if name == "sources.yaml" else {"security_keywords": {}})
    monkeypatch.setattr(fetch, "fetch_source", lambda s, k=None, since=None: (seen.setdefault(s["name"], since), [])[1])
    monkeypatch.setattr(fetch, "REQUEST_DELAY_SEC", 0)
    srcs = [{"name": "신규"}, {"name": "정상"}, {"name": "밀림"}]
    states = {"정상": now - H(6), "밀림": now - H(18)}
    _, report = fetch.fetch_all(srcs, states=states, now=now)
    assert seen["신규"] is None                                                         # 기록 없는 소스: 제한 없음
    assert seen["정상"] == now - H(6.5)
    assert seen["밀림"] == now - H(18.5)                                                # 18시간 전에 성공했으면 거기서 30분 더 거슬러
    assert all(isinstance(r["fetched_at"], datetime) for r in report.values())
    seen.clear()
    fetch.fetch_all(srcs, states=None, now=now)                                         # states를 안 주면(--full) 전부 제한 없음
    assert seen == {"신규": None, "정상": None, "밀림": None}


# ───────────── 파이프라인: 성공 시각 기록 ─────────────
def _pipeline_env(tmp_path, monkeypatch, report, articles):
    rp = _load_script("run_pipeline")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    seen = {}
    monkeypatch.setattr(rp, "fetch_all", lambda **kw: (seen.update(kw) or (articles, report)))
    return rp, seen


def _art(i):
    return {"id": f"id{i}", "source": "S", "title": f"서로 다른 제목 {i} 번째", "url": f"https://x/{i}",
            "published_at": "2026-11-06T01:00:00+00:00", "collected_at": "2026-11-06T02:00:00+00:00"}


def test_pipeline_records_state_only_for_ok_sources_after_saving(tmp_path, monkeypatch):
    t_ok, t_bad = datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc), datetime(2026, 11, 6, 2, 1, tzinfo=timezone.utc)
    report = {"좋은소스": {"ok": True, "count": 1, "fetched_at": t_ok}, "죽은소스": {"ok": False, "error": "x"}}
    rp, seen = _pipeline_env(tmp_path, monkeypatch, report, [_art(1)])
    conn = db.connect()
    rp.collect(conn)
    assert db.get_source_states(conn) == {"좋은소스": t_ok}                              # 실패한 소스는 기록하지 않아 다음에 그 구간을 다시 본다
    assert seen["states"] == {}                                                         # 첫 실행: 기록 없음 → 제한 없이 수집
    rp.collect(conn)
    assert seen["states"] == {"좋은소스": t_ok}                                          # 다음 실행은 기록을 넘겨 기준 시간을 적용
    rp.collect(conn, full=True)
    assert seen["states"] is None                                                       # --full: 기준 시간 무시
    conn.close()


def test_pipeline_does_not_advance_state_if_saving_fails(tmp_path, monkeypatch):
    t = datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc)
    rp, _ = _pipeline_env(tmp_path, monkeypatch, {"S": {"ok": True, "count": 1, "fetched_at": t}}, [_art(1)])
    conn = db.connect()
    monkeypatch.setattr(db, "insert_articles", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("디스크 오류")))
    with pytest.raises(RuntimeError):
        rp.collect(conn)
    assert db.get_source_states(conn) == {}                                             # 저장 실패 → 상태 불변 → 기사가 영영 빠지지 않음
    conn.close()


def test_source_state_upsert(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    a, b = datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc), datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
    db.set_source_ok(conn, "S", a)
    db.set_source_ok(conn, "S", b)
    assert db.get_source_states(conn) == {"S": b}
    conn.close()
