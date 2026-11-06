"""Streamlit 진입점: 페이지 내비게이션만 담당한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="정보보안 뉴스", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

nav = st.navigation([
    st.Page("pages/1_뉴스목록.py", title="뉴스 목록", default=True),
    st.Page("pages/2_챗봇.py", title="챗봇"),
    st.Page("pages/3_대시보드.py", title="대시보드"),
], position="top")
nav.run()
