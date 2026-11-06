"""뉴스 목록: 날짜별 타임라인(최신순) + 오른쪽 필터 패널 + '더 보기' 페이지네이션."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from scraper.fetch import load_yaml  # noqa: E402
from store import db  # noqa: E402

PAGE_SIZE = 30

ui.inject_css()
st.title("정보보안 뉴스")
st.markdown('<p class="page-sub">국내외 보안 기사를 최신순으로 모아 요약해 보여줍니다. 제목을 누르면 원문으로 이동합니다.</p>',
            unsafe_allow_html=True)


def _db_version() -> int:
    try:
        return db.DB_PATH.stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(ttl=300, show_spinner=False)
def load_articles(version, minors, country, d_from, d_to, sources, keyword, limit):
    conn = db.connect()
    try:
        rows, total = db.query_articles(
            conn, minors=list(minors) or None, country=country, d_from=d_from, d_to=d_to,
            sources=list(sources) or None, keyword=keyword or None, limit=limit)
        pending = conn.execute("SELECT COUNT(*) FROM articles WHERE summary IS NULL").fetchone()[0]
        return [dict(r) for r in rows], total, pending, db.hidden_count(conn)
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def source_options(version):
    conn = db.connect()
    try:
        return db.distinct_sources(conn)
    finally:
        conn.close()


ver = _db_version()
minor_options = list(load_yaml("categories.yaml")["minor"].keys())

main, side = ui.two_pane("news")

# ── 오른쪽 필터 패널 ──
with side:
    with st.container(key="filters", border=True):
        st.markdown('<p class="filter-title">필터</p>', unsafe_allow_html=True)
        keyword = st.text_input("검색", placeholder="제목·요약·태그에서 검색", key="news_kw").strip()
        sources = st.multiselect("뉴스사", source_options(ver), placeholder="전체", key="news_src")
        d_from, d_to = ui.period_picker("news")
        country = ui.region_picker("news")
        minors = st.pills("중분류", minor_options, selection_mode="multi", key="news_minors") or []
        st.button("필터 초기화", width="stretch", on_click=ui.reset_filters,
                  args=("news", ("news_kw",), ("news_src", "news_minors")))

# 필터가 바뀌면 '더 보기' 상태를 처음으로 되돌린다.
signature = (tuple(minors), country, d_from, d_to, tuple(sources), keyword)
if st.session_state.get("filter_sig") != signature:
    st.session_state.filter_sig = signature
    st.session_state.limit = PAGE_SIZE
limit = st.session_state.setdefault("limit", PAGE_SIZE)

items, total, pending, hidden = load_articles(
    ver, tuple(minors), country, d_from, d_to, tuple(sources), keyword, limit)

# ── 왼쪽 기사 목록 ──
with main:
    if total == 0:
        if signature == ((), None, None, None, (), "") and pending == 0:
            st.info("아직 저장된 기사가 없습니다. 터미널에서 `python scripts/run_pipeline.py`를 실행해 기사를 수집하세요.")
        else:
            st.info("조건에 맞는 기사가 없습니다. 필터를 줄이거나 다른 키워드를 시도해 보세요.")
    else:
        note = (f" · 요약 대기 {pending}건" if pending else "") + \
               (f" · 보안 무관 {hidden}건 제외됨" if hidden else "")
        st.caption(f"조건에 맞는 기사 {total}건 중 최신 {len(items)}건 표시 · 시간은 KST{note}")
        st.markdown(ui.timeline(items), unsafe_allow_html=True)
        if len(items) < total:
            if st.button(f"이전 기사 더 보기 ({total - len(items)}건 남음)", width="stretch"):
                st.session_state.limit = limit + PAGE_SIZE
                st.rerun()
