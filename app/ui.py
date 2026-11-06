"""공용 UI 헬퍼 (스타일, 배지, 타임라인 목록, 가로 필터). 외부 데이터는 모두 HTML 이스케이프해서 넣는다."""
import html
import json
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit

import streamlit as st

from store import db

# 색은 테두리/배경 틴트에만 쓰고 글자는 항상 텍스트 라벨로 구분한다 (색에만 의존하지 않음).
# 글자색은 상속하므로 라이트/다크 어느 쪽에서도 읽힌다.
CATEGORY_COLORS = {
    "침해 사고": "#dc2626",
    "해킹": "#ea580c",
    "정보유출": "#9333ea",
    "컨퍼런스": "#2563eb",
    "취약점": "#ca8a04",
    "기타": "#6b7280",
}
REGION_LABEL = {"KR": "국내", "GLOBAL": "해외"}
WEEKDAYS = "월화수목금토일"
PERIODS = ["전체 기간", "오늘", "최근 3일", "최근 7일", "최근 30일", "직접 선택"]

CSS = """
<style>
.block-container{padding-top:2.2rem;max-width:1180px}
.page-sub{opacity:.65;font-size:.92rem;margin:-.4rem 0 1rem}
.day{display:flex;align-items:center;gap:10px;margin:30px 0 2px;padding-bottom:10px;
  border-bottom:2px solid rgba(128,128,128,.4)}
.day .d{font-size:1.2rem;font-weight:800;letter-spacing:-.01em}
.day .chip{font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:999px;background:#2563eb;color:#fff}
.day .n{margin-left:auto;opacity:.6;font-size:.85rem}
.row{display:grid;grid-template-columns:62px minmax(0,1fr);gap:16px;padding:16px 10px;
  border-bottom:1px solid rgba(128,128,128,.18);border-radius:10px;transition:background .15s}
.row:hover{background:rgba(128,128,128,.08)}
.row .t{font-variant-numeric:tabular-nums;font-size:.92rem;font-weight:700;opacity:.6;padding-top:3px}
.row .meta{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;font-size:.82rem}
.badge{--c:#6b7280;display:inline-block;padding:1px 9px;border-radius:999px;font-weight:700;font-size:.76rem;
  border:1px solid var(--c);background:rgba(128,128,128,.15);background:color-mix(in srgb,var(--c) 18%,transparent)}
.src{font-weight:700}
.dim{opacity:.65}
.row .title{display:block;margin:7px 0 5px;font-size:1.1rem;font-weight:700;line-height:1.42;color:inherit;
  text-decoration:none;overflow-wrap:anywhere}
.row .title:hover{text-decoration:underline;text-underline-offset:3px}
.row .sum{margin:0;font-size:.94rem;line-height:1.65;opacity:.85;overflow-wrap:anywhere}
.row .foot{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:9px}
.tag{font-size:.74rem;padding:0 8px;border-radius:6px;border:1px solid rgba(128,128,128,.4);opacity:.8}
.row .more{margin-left:auto;font-size:.85rem;font-weight:700;text-decoration:none;white-space:nowrap}
.row .more:hover{text-decoration:underline;text-underline-offset:3px}
.srclist{list-style:none;padding:0;margin:0}
.srclist li{padding:8px 0;border-bottom:1px solid rgba(128,128,128,.2);font-size:.9rem;overflow-wrap:anywhere}
.st-key-newsgrid [data-testid="stHorizontalBlock"],.st-key-chatgrid [data-testid="stHorizontalBlock"]{align-items:flex-start}
[data-testid="stColumn"]:has(.st-key-filters){position:sticky;top:3.6rem;align-self:flex-start}
.filter-title{font-weight:800;font-size:1.05rem;margin:0 0 .2rem}
@media (max-width:640px){
  .st-key-newsgrid [data-testid="stHorizontalBlock"],.st-key-chatgrid [data-testid="stHorizontalBlock"]{flex-direction:column-reverse}
  [data-testid="stColumn"]:has(.st-key-filters){position:static}
  .row{grid-template-columns:1fr;gap:2px;padding:14px 4px}
  .row .t{padding-top:0}
  .row .title{font-size:1.03rem}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def safe_url(url: str | None) -> str:
    """http/https 이외 스킴(javascript: 등)은 막는다."""
    try:
        return url if url and urlsplit(url).scheme in ("http", "https") else "#"
    except ValueError:
        return "#"


def kst_dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone(db.KST)
    except ValueError:
        return None


def kst(iso: str | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    dt = kst_dt(iso)
    return dt.strftime(fmt) if dt else ""


def badge(minor: str | None) -> str:
    name = minor or "미분류"
    color = CATEGORY_COLORS.get(name, "#6b7280")
    return f'<span class="badge" style="--c:{color}">{esc(name)}</span>'


def _tags(raw) -> list[str]:
    try:
        tags = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return []
    return [str(t) for t in tags if str(t).strip()] if isinstance(tags, list) else []


def row(a: dict) -> str:
    url = esc(safe_url(a.get("url")))
    dt = kst_dt(a.get("published_at") or a.get("collected_at"))
    region = REGION_LABEL.get(a.get("country"), "")
    tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in _tags(a.get("tags"))[:3])
    return (
        '<article class="row">'
        f'<div class="t">{dt.strftime("%H:%M") if dt else "--:--"}</div>'
        "<div>"
        f'<div class="meta">{badge(a.get("category_minor"))}'
        f'<span class="src">{esc(a.get("source"))}</span><span class="dim">{esc(region)}</span></div>'
        f'<a class="title" href="{url}" target="_blank" rel="noopener noreferrer">{esc(a.get("title"))}</a>'
        f'<p class="sum">{esc(a.get("summary"))}</p>'
        f'<div class="foot">{tags}'
        f'<a class="more" href="{url}" target="_blank" rel="noopener noreferrer">원문 보기 ↗</a></div>'
        "</div></article>"
    )


def _day_label(d: date, today: date) -> tuple[str, str]:
    label = f"{d.month}월 {d.day}일 ({WEEKDAYS[d.weekday()]})"
    if d == today:
        return label, "오늘"
    if d == today - timedelta(days=1):
        return label, "어제"
    return label, ""


def timeline(items: list[dict], today: date | None = None) -> str:
    """최신순으로 들어온 기사를 KST 날짜별로 묶어 타임라인 HTML로 만든다."""
    today = today or datetime.now(db.KST).date()
    parts = []
    groups: list[list] = []   # [날짜, [기사...]]
    for a in items:
        dt = kst_dt(a.get("published_at") or a.get("collected_at"))
        d = dt.date() if dt else None
        if not groups or groups[-1][0] != d:
            groups.append([d, []])
        groups[-1][1].append(a)
    for d, arts in groups:
        if d is None:
            parts.append('<div class="day"><span class="d">날짜 미상</span>'
                         f'<span class="n">표시 {len(arts)}건</span></div>')
        else:
            label, chip = _day_label(d, today)
            chip_html = f'<span class="chip">{chip}</span>' if chip else ""
            parts.append(f'<div class="day"><span class="d">{label}</span>{chip_html}'
                         f'<span class="n">표시 {len(arts)}건</span></div>')
        parts.extend(row(a) for a in arts)
    return "".join(parts)


def source_list(items: list[dict], cited: list[int] | None = None) -> str:
    cited = cited or []
    lis = []
    for n, a in enumerate(items, 1):
        tag = "인용됨" if n in cited else "참고"
        lis.append(
            f"<li><b>[{n}]</b> {badge(a.get('category_minor'))} "
            f"<span class='dim'>{esc(a.get('source'))} · {tag}</span><br>"
            f"<a href=\"{esc(safe_url(a.get('url')))}\" target=\"_blank\" rel=\"noopener noreferrer\">"
            f"{esc(a.get('title'))}</a></li>"
        )
    return '<ul class="srclist">' + "".join(lis) + "</ul>"


def period_range(choice: str, custom: tuple | None = None, today: date | None = None):
    """기간 프리셋 → (시작일, 종료일) KST 기준, 양끝 포함. '전체 기간'은 (None, None)."""
    today = today or datetime.now(db.KST).date()
    days = {"오늘": 0, "최근 3일": 2, "최근 7일": 6, "최근 30일": 29}
    if choice in days:
        return today - timedelta(days=days[choice]), today
    if choice == "직접 선택" and custom and len(custom) == 2:
        return custom[0], custom[1]
    return None, None


def period_picker(key: str):
    """가로 필터 줄에서 쓰는 기간 선택. (시작일, 종료일) 반환."""
    choice = st.selectbox("기간 (KST)", PERIODS, key=f"{key}_period")
    custom = None
    if choice == "직접 선택":
        today = datetime.now(db.KST).date()
        custom = st.date_input("날짜 범위", value=(today - timedelta(days=6), today), key=f"{key}_range")
        if not (isinstance(custom, (tuple, list)) and len(custom) == 2):
            st.caption("시작일과 종료일을 모두 선택하세요.")
            custom = None
    return period_range(choice, custom)


def region_picker(key: str) -> str | None:
    """'KR' | 'GLOBAL' | None(전체)."""
    st.session_state.setdefault(f"{key}_region", "전체")   # default= 와 세션 상태 초기화를 함께 쓰면 경고가 난다
    sel = st.segmented_control("지역", ["전체", "국내", "해외"], key=f"{key}_region")
    return {"국내": "KR", "해외": "GLOBAL"}.get(sel)


def two_pane(name: str):
    """왼쪽 본문 + 오른쪽 필터 패널. (main, side) 컬럼을 돌려준다. 좁은 화면에서는 필터가 위로 올라온다."""
    with st.container(key=f"{name}grid"):
        return st.columns([3.4, 1.2], gap="large")


def reset_filters(prefix: str, text_keys=(), multi_keys=()) -> None:
    """'초기화' 버튼 콜백: 필터 위젯 상태를 기본값으로 되돌린다."""
    for k in text_keys:
        st.session_state[k] = ""
    for k in multi_keys:
        st.session_state[k] = []
    st.session_state[f"{prefix}_period"] = PERIODS[0]
    st.session_state[f"{prefix}_region"] = "전체"
    st.session_state.pop(f"{prefix}_range", None)
