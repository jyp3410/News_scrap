import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import ui  # noqa: E402


def _a(**kw):
    base = {"source": "S", "title": "t", "summary": "s", "url": "https://x.com/a", "category_minor": "해킹",
            "country": "KR", "published_at": "2026-11-05T15:30:00+00:00", "collected_at": "2026-11-06T00:00:00+00:00",
            "tags": '["CVE-2026-1", "OpenSSH"]'}
    return {**base, **kw}


def test_row_escapes_external_html():
    h = ui.row(_a(title='<script>alert(1)</script>', summary='<img src=x onerror=alert(1)>', source='"><b>',
                  tags='["<i>x</i>"]'))
    assert "<script>" not in h and "<img" not in h and "<b>" not in h and "<i>" not in h
    assert "&lt;script&gt;" in h


def test_javascript_url_is_blocked():
    h = ui.row(_a(url="javascript:alert(1)"))
    assert "javascript:" not in h and 'href="#"' in h


def test_row_has_text_labels_and_kst_time():
    h = ui.row(_a())
    assert "해킹" in h and "국내" in h and "00:30" in h    # UTC 15:30 → KST 다음날 00:30
    assert "원문 보기" in h and "CVE-2026-1" in h


def test_bad_tags_json_does_not_break():
    assert "tag" not in ui.row(_a(tags="not json"))
    assert ui.row(_a(tags=None))


def test_unknown_or_missing_category_does_not_break():
    assert "미분류" in ui.badge(None)
    assert ui.badge("없는분류")


def test_timeline_groups_by_kst_day_newest_first_and_labels_today():
    items = [  # 최신순으로 들어온다고 가정
        _a(url="https://x.com/1", published_at="2026-11-06T02:00:00+00:00"),   # KST 11-06 11:00
        _a(url="https://x.com/2", published_at="2026-11-05T20:00:00+00:00"),   # KST 11-06 05:00
        _a(url="https://x.com/3", published_at="2026-11-05T10:00:00+00:00"),   # KST 11-05 19:00
        _a(url="https://x.com/4", published_at="2026-11-03T10:00:00+00:00"),   # KST 11-03 19:00
    ]
    h = ui.timeline(items, today=date(2026, 11, 6))
    assert h.count('class="day"') == 3
    assert h.index("11월 6일 (금)") < h.index("11월 5일 (목)") < h.index("11월 3일 (화)")
    assert "오늘" in h and "어제" in h
    assert h.index("11:00") < h.index("05:00")       # 같은 날 안에서도 최신이 위


def test_timeline_has_no_blank_lines_that_break_markdown():
    assert "\n\n" not in ui.timeline([_a(), _a(published_at=None)])


def test_period_range_presets_are_inclusive_kst_days():
    t = date(2026, 11, 6)
    assert ui.period_range("전체 기간", today=t) == (None, None)
    assert ui.period_range("오늘", today=t) == (t, t)
    assert ui.period_range("최근 7일", today=t) == (date(2026, 10, 31), t)
    assert ui.period_range("직접 선택", (date(2026, 11, 1), t), today=t) == (date(2026, 11, 1), t)
    assert ui.period_range("직접 선택", None, today=t) == (None, None)
