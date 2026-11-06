"""대시보드: 수집 현황 요약 (읽기 전용)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from scraper.fetch import load_yaml  # noqa: E402
from store import db  # noqa: E402

ui.inject_css()
st.title("수집 현황")


@st.cache_data(ttl=300, show_spinner=False)
def load_stats(version):
    conn = db.connect()
    try:
        df = pd.read_sql_query(
            "SELECT source, country, category_minor, summary IS NOT NULL AS done, "
            f"published_at, collected_at FROM articles WHERE {db.VISIBLE}", conn)
        hidden = db.hidden_count(conn)
    finally:
        conn.close()
    return df, hidden


try:
    version = db.DB_PATH.stat().st_mtime_ns
except OSError:
    version = 0
df, hidden = load_stats(version)

if df.empty:
    st.info("아직 저장된 기사가 없습니다. `python scripts/run_pipeline.py`를 먼저 실행하세요.")
    st.stop()

now_kst = datetime.now(timezone.utc).astimezone(db.KST)
today_start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
collected = pd.to_datetime(df["collected_at"], utc=True, format="ISO8601")
eff = pd.to_datetime(df["published_at"].fillna(df["collected_at"]), utc=True, format="ISO8601")

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 기사", f"{len(df):,}건")
c2.metric("오늘 수집 (KST)", f"{int((collected >= today_start).sum()):,}건")
c3.metric("요약 대기", f"{int((df['done'] == 0).sum()):,}건")
c4.metric("마지막 수집 (KST)", ui.kst(collected.max().isoformat(), "%m-%d %H:%M") or "-")

st.subheader("중분류별 기사 수")
order = list(load_yaml("categories.yaml")["minor"].keys())
by_minor = df[df["done"] == 1]["category_minor"].value_counts().reindex(order, fill_value=0)
st.bar_chart(by_minor, horizontal=True)

left, right = st.columns(2)
with left:
    st.subheader("최근 14일 기사 수")
    days = eff.dt.tz_convert(db.KST).dt.date
    since = (now_kst - timedelta(days=13)).date()
    daily = days[days >= since].value_counts().sort_index()
    daily = daily.reindex([since + timedelta(days=i) for i in range(14)], fill_value=0)
    daily.index = [d.strftime("%m-%d") for d in daily.index]
    st.bar_chart(daily)
with right:
    st.subheader("뉴스사별 기사 수")
    st.bar_chart(df["source"].value_counts(), horizontal=True)

region = df["country"].map(ui.REGION_LABEL).fillna("기타").value_counts()
st.caption("지역 구성: " + " · ".join(f"{k} {v}건" for k, v in region.items())
           + (f" · 보안 무관으로 제외된 기사 {hidden}건 (집계에서 빠짐)" if hidden else ""))
