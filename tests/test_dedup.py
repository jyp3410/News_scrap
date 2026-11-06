import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.dedup import article_id, dedup, normalize_url  # noqa: E402
from store import db  # noqa: E402


def test_normalize_strips_tracking_and_fragment():
    a = normalize_url("HTTPS://Example.com/a/?utm_source=x&b=1&fbclid=z#top")
    assert a == "https://example.com/a?b=1"


def test_same_article_same_id():
    assert article_id("https://x.com/a/?utm_medium=m") == article_id("https://x.com/a")


def test_keep_meaningful_query():
    assert article_id("https://x.com/v?idxno=1") != article_id("https://x.com/v?idxno=2")


def _a(url, title):
    return {"id": article_id(url), "title": title, "url": url}


def test_dedup_batch_and_known():
    batch = [
        _a("https://x.com/1", "랜섬웨어 공격으로 병원 마비"),
        _a("https://x.com/1?utm_source=a", "랜섬웨어 공격으로 병원 마비"),
        _a("https://y.com/2", "랜섬웨어 공격으로 병원 마비!"),  # 제목 유사
        _a("https://z.com/3", "완전히 다른 기사 제목입니다"),
    ]
    out = dedup(batch, set(), [])
    assert [a["url"] for a in out] == ["https://x.com/1", "https://z.com/3"]
    assert dedup(batch, {article_id("https://z.com/3")}, ["랜섬웨어 공격으로 병원 마비"]) == []


def test_db_insert_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    art = {**_a("https://x.com/1", "t"), "source": "S", "collected_at": "2026-01-01T00:00:00+00:00"}
    assert db.insert_articles(conn, [art]) == 1
    assert db.insert_articles(conn, [art]) == 0
    conn.close()


# ---- R-01: 서로 다른 기사를 같은 기사로 판단해 영구히 버리던 오탐 ----
from scraper.dedup import is_similar_title  # noqa: E402

D1, D2, D_LATER = "2026-11-05T01:00:00+00:00", "2026-11-05T09:00:00+00:00", "2026-11-20T01:00:00+00:00"


@pytest.mark.parametrize("a,b", [
    ("CVE-2026-1234 allows remote code execution in FooApp", "CVE-2026-1235 allows remote code execution in FooApp"),
    ("KT 해킹 사고, 피해 고객 1만명", "SKT 해킹 사고, 피해 고객 1만명"),
    ("랜섬웨어 공격으로 A병원 진료 중단", "랜섬웨어 공격으로 B병원 진료 중단"),
    ("랜섬웨어 공격으로 서울병원 진료 중단", "랜섬웨어 공격으로 부산병원 진료 중단"),
    ("Microsoft Patch Tuesday: 63 flaws fixed", "Microsoft Patch Tuesday: 59 flaws fixed"),
])
def test_different_stories_are_not_merged(a, b):
    assert not is_similar_title(a, b, D1, D1)


def test_same_title_weeks_apart_is_not_duplicate_but_same_day_is():
    assert not is_similar_title("Weekly Recap", "Weekly Recap", D1, D_LATER)    # 주간 정리글
    assert is_similar_title("Weekly Recap", "Weekly Recap", D1, D2)
    assert is_similar_title("Hackers exploit WinRAR flaw", "Hackers exploit WinRAR flaw")   # 날짜 모르면 제목만


def test_real_duplicates_are_still_caught():
    assert is_similar_title("OpenSSH 취약점 패치 공개", "OpenSSH 취약점 패치 공개!", D1, D2)


def test_dedup_keeps_lookalike_articles_and_known_titles_accept_tuples():
    batch = [
        {"id": "1", "title": "랜섬웨어 공격으로 A병원 진료 중단", "published_at": D1},
        {"id": "2", "title": "랜섬웨어 공격으로 B병원 진료 중단", "published_at": D1},
        {"id": "3", "title": "Weekly Recap", "published_at": D_LATER},
    ]
    out = dedup(batch, set(), [("Weekly Recap", D1)])          # 2주 전 같은 제목이 저장돼 있어도 새 기사로 인정
    assert [a["id"] for a in out] == ["1", "2", "3"]
    assert dedup(batch[:1], set(), ["랜섬웨어 공격으로 A병원 진료 중단"]) == []   # 문자열 목록도 계속 지원


def test_timezone_mixed_dates_do_not_crash():
    assert is_similar_title("같은 제목의 기사입니다", "같은 제목의 기사입니다", "2026-11-05T01:00:00+00:00", "2026-11-05T01:00:00")
