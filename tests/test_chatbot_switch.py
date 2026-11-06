"""챗봇 스위치(CHATBOT_ENABLED) / 키 없음 처리: 꺼져 있으면 안내만 보이고 OpenAI를 호출하지 않는다."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from chatbot import rag  # noqa: E402
from core import llm  # noqa: E402

PAGE = str(ROOT / "app" / "pages" / "2_챗봇.py")
NOTICE = "챗봇은 현재 이용할 수 없습니다"


# ───────────── 스위치 해석 ─────────────
@pytest.mark.parametrize("value", [None, "", "   ", "true", "True", "TRUE", " true ", "1", "yes", "on", "Y"])
def test_switch_on_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CHATBOT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CHATBOT_ENABLED", value)
    assert llm.chatbot_switch_on() is True


@pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off", "disabled", "flase", "아니오", "nope"])
def test_switch_off_values_including_typos_fail_closed(monkeypatch, value):
    """'false'뿐 아니라 오타·알 수 없는 값도 꺼짐으로 본다 (끄려던 설정이 오타 때문에 켜진 채 요금을 쓰지 않도록)."""
    monkeypatch.setenv("CHATBOT_ENABLED", value)
    assert llm.chatbot_switch_on() is False


def test_status_reasons(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CHATBOT_ENABLED", raising=False)
    assert llm.chatbot_status() == (True, None)
    monkeypatch.setenv("CHATBOT_ENABLED", "false")
    assert llm.chatbot_status() == (False, "switch_off")                      # 키가 있어도 스위치가 우선
    monkeypatch.delenv("CHATBOT_ENABLED")
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    assert llm.chatbot_status() == (False, "no_key")
    monkeypatch.setenv("CHATBOT_ENABLED", "false")
    assert llm.chatbot_status() == (False, "switch_off")                      # 둘 다 해당하면 스위치 사유


# ───────────── 로직 계층: 페이지를 거치지 않아도 막힌다 ─────────────
def test_rag_answer_refuses_without_calling_openai(monkeypatch):
    monkeypatch.setenv("CHATBOT_ENABLED", "false")
    monkeypatch.setattr(llm, "embed", lambda *a, **k: pytest.fail("임베딩 호출 금지"))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("채팅 호출 금지"))
    with pytest.raises(llm.ChatbotDisabled):
        rag.answer("랜섬웨어 사례 알려줘")


# ───────────── 페이지: 방문자에게 보이는 것 ─────────────
def _assert_disabled_page(at):
    assert not at.exception
    assert any(NOTICE in i.value for i in at.info)                      # 안내가 보이고
    assert len(at.chat_input) == 0                                       # 입력창이 없고
    assert len(at.button) == 0                                           # 예시 질문 버튼도 없고
    assert len(at.multiselect) == 0 and len(at.selectbox) == 0           # 필터 패널도 만들지 않는다
    shown = " ".join([i.value for i in at.info] + [e.value for e in at.error] + [m.value for m in at.markdown])
    assert "OPENAI_API_KEY" not in shown and ".env" not in shown        # 개발자용 문구가 방문자에게 노출되지 않는다


@pytest.fixture
def no_openai(monkeypatch):
    monkeypatch.setattr(llm, "embed", lambda *a, **k: pytest.fail("OpenAI 임베딩이 호출됨"))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("OpenAI 채팅이 호출됨"))
    monkeypatch.setattr(llm, "get_client", lambda *a, **k: pytest.fail("OpenAI 클라이언트가 만들어짐"))


def test_page_when_switch_is_off(monkeypatch, no_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHATBOT_ENABLED", "false")
    _assert_disabled_page(AppTest.from_file(PAGE, default_timeout=60).run())


def test_page_when_key_is_missing(monkeypatch, no_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("CHATBOT_ENABLED", raising=False)
    _assert_disabled_page(AppTest.from_file(PAGE, default_timeout=60).run())


def test_page_still_works_when_enabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")                      # 렌더링만 확인하므로 실제 호출은 일어나지 않는다
    monkeypatch.delenv("CHATBOT_ENABLED", raising=False)
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert not any(NOTICE in i.value for i in at.info)
    assert len(at.chat_input) == 1                                       # 입력창
    assert len([b for b in at.button if b.label.startswith(("최근", "이번", "국내", "다가오는"))]) == 4   # 예시 질문 4개


def test_page_disabled_state_is_logged_for_the_owner(monkeypatch, no_openai, caplog):
    import logging
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHATBOT_ENABLED", "false")
    with caplog.at_level(logging.WARNING, logger="app.chatbot"):
        AppTest.from_file(PAGE, default_timeout=60).run()
    assert "switch_off" in caplog.text                                   # 사유는 서버 로그에만 남는다


def test_chat_page_does_not_use_st_stop():
    """st.stop()을 쓰면 상단 메뉴(st.navigation)가 옛 사이드바 메뉴로 바뀌는 문제가 있었다 → 본문을 함수로 나눠 호출한다."""
    import ast
    tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "stop" and getattr(n.func.value, "id", "") == "st"]
    assert not calls
