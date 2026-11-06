"""RAG 챗봇: 왼쪽 대화 + 오른쪽 필터 패널, 예시 질문, 접이식 참고 기사 목록."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

import ui  # noqa: E402
from chatbot import rag  # noqa: E402
from core import llm  # noqa: E402
from scraper.fetch import load_yaml  # noqa: E402
from store.vector import IndexCorrupt  # noqa: E402

EXAMPLES = [
    "최근 랜섬웨어 침해 사고 사례 알려줘",
    "이번 주 주요 취약점(CVE) 정리해줘",
    "국내 개인정보 유출 사고 있었어?",
    "다가오는 보안 컨퍼런스 소식 알려줘",
]

ui.inject_css()
st.title("보안 뉴스 챗봇")
st.markdown('<p class="page-sub">수집된 기사 중 질문과 가장 관련 있는 기사 10건을 찾아 답변합니다. '
            "답변에는 근거 기사 링크가 함께 표시됩니다.</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role", "content", "sources", "cited", "not_found"}

main, side = ui.two_pane("chat")

# ── 오른쪽 필터 패널 ──
with side:
    with st.container(key="filters", border=True):
        st.markdown('<p class="filter-title">검색 필터</p>', unsafe_allow_html=True)
        minors = st.multiselect("중분류", list(load_yaml("categories.yaml")["minor"].keys()),
                                placeholder="전체", key="chat_minors")
        country = ui.region_picker("chat")
        d_from, d_to = ui.period_picker("chat")
        st.button("필터 초기화", width="stretch", on_click=ui.reset_filters,
                  args=("chat", (), ("chat_minors",)))
        if st.button("대화 초기화", width="stretch"):
            st.session_state.messages = []
            st.rerun()


def render_sources(msg: dict):
    if msg.get("not_found") or not msg.get("sources"):
        return
    with st.expander(f"참고 기사 {len(msg['sources'])}건 보기"):
        st.markdown(ui.source_list(msg["sources"], msg.get("cited", [])), unsafe_allow_html=True)


def render_message(m: dict):
    with st.chat_message(m["role"]):
        if m.get("not_found"):
            st.info(m["content"])
        else:
            st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m)


clicked = None
question = None
with main:
    for m in st.session_state.messages:
        render_message(m)

    # 예시 질문: 대화가 비어 있을 때만 노출
    examples_slot = st.empty()
    if not st.session_state.messages:
        with examples_slot.container():
            st.markdown("**이런 질문을 해보세요**")
            cols = st.columns(2)
            for i, ex in enumerate(EXAMPLES):
                if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                    clicked = ex

typed = st.chat_input("질문을 입력하세요 (예: 최근 랜섬웨어 침해 사고 사례 알려줘)")
question = clicked or typed

if question:
    with main:
        examples_slot.empty()  # 질문 처리 중에는 예시 버튼을 숨긴다
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("관련 기사를 찾고 답변을 작성하는 중..."):
                    res = rag.answer(question, history, rag.Filters(minors or None, country, d_from, d_to))
                msg = {"role": "assistant", "content": res.answer, "sources": res.sources,
                       "cited": res.cited, "not_found": res.not_found}
                if res.not_found:
                    st.info(res.answer)
                else:
                    st.markdown(res.answer)
                render_sources(msg)
                st.session_state.messages.append(msg)
            except llm.ConfigError as e:
                st.error(str(e))
            except IndexCorrupt as e:
                st.error(str(e))
            except llm.LLMError as e:
                st.error(f"답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. ({e})")
