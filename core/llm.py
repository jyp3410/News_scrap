"""OpenAI 호출 공통 래퍼. 모델명은 환경변수(CHAT_MODEL, EMBEDDING_MODEL)로 교체 가능.

gpt-6-astra 제약(2026-11-06 확인): max_tokens 대신 max_completion_tokens 사용,
temperature는 기본값(1)만 허용, response_format=json_object 사용 가능.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

log = logging.getLogger("core.llm")
EMBED_BATCH = 100
MAX_ATTEMPTS = 3


class ConfigError(RuntimeError):
    """설정(API 키 등) 문제. 사용자가 고쳐야 한다."""


class LLMError(RuntimeError):
    """OpenAI 호출/응답 처리 실패."""


class ChatbotDisabled(RuntimeError):
    """챗봇이 꺼져 있다 (CHATBOT_ENABLED 스위치). 방문자에게는 친절한 안내만 보여 주고 OpenAI는 호출하지 않는다."""


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def chatbot_switch_on() -> bool:
    """CHATBOT_ENABLED 스위치. 설정하지 않으면 켜짐.

    값을 설정했는데 true 계열이 아니면 꺼짐으로 본다 ('false', '0', 'off', 오타 'flase' 등).
    꺼서 비용을 막으려는 설정이 오타 때문에 켜진 채로 남는 일을 피하기 위해 실패 시 닫는 쪽으로 판정한다.
    """
    raw = os.getenv("CHATBOT_ENABLED")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in _TRUE_VALUES


def chatbot_status() -> tuple[bool, str | None]:
    """(사용 가능 여부, 불가 사유). 사유: 'switch_off'(스위치) 또는 'no_key'(OpenAI 키 없음)."""
    if not chatbot_switch_on():
        return False, "switch_off"
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return False, "no_key"
    return True, None


def chat_model() -> str:
    return os.getenv("CHAT_MODEL") or "gpt-6-astra"


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"


DEFAULT_TIMEOUT = 90          # 배치 파이프라인용
# 사용자가 기다리는 챗봇 요청용: 장애 시 최악의 대기를 약 1분으로 제한 (30초 × 2회 + 백오프)
INTERACTIVE_TIMEOUT = 30
INTERACTIVE_ATTEMPTS = 2


def get_client(timeout: float | None = None):
    import openai
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트 루트의 .env 파일에 "
            "OPENAI_API_KEY=... 를 추가하세요 (.env.example 참고)."
        )
    return openai.OpenAI(api_key=key, timeout=timeout or DEFAULT_TIMEOUT, max_retries=0)


def _call(fn, attempts: int | None = None):
    import openai
    attempts = attempts or MAX_ATTEMPTS
    transient = (openai.RateLimitError, openai.APIConnectionError,
                 openai.APITimeoutError, openai.InternalServerError)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except openai.AuthenticationError as e:
            raise ConfigError("OpenAI 인증에 실패했습니다. OPENAI_API_KEY를 확인하세요.") from e
        except transient as e:
            if attempt == attempts:
                raise LLMError(f"OpenAI 서버/네트워크 오류로 {attempts}회 시도 후 실패: {e}") from e
            wait = 2 ** attempt
            log.warning("OpenAI 일시 오류(%s), %ds 후 재시도 (%d/%d)", type(e).__name__, wait, attempt, attempts)
            time.sleep(wait)
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI 요청 실패: {e}") from e


def chat(messages: list[dict], json_mode: bool = False, max_completion_tokens: int = 6000,
         timeout: float | None = None, attempts: int | None = None) -> str:
    """timeout/attempts를 생략하면 배치용 기본값(90초, 3회). 챗봇은 INTERACTIVE_* 를 넘긴다."""
    client = get_client(timeout)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _call(lambda: client.chat.completions.create(
        model=chat_model(), messages=messages,
        max_completion_tokens=max_completion_tokens, **kwargs), attempts)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length" or not text:
        raise LLMError("모델 응답이 잘렸거나 비어 있습니다 (finish_reason=%s)" % choice.finish_reason)
    return text


def chat_json(messages: list[dict], **kw) -> dict:
    text = chat(messages, json_mode=True, **kw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 파싱 실패: {text[:200]}") from e
    if not isinstance(data, dict):
        raise LLMError("JSON 객체가 아닌 응답")
    return data


def embed(texts: list[str], timeout: float | None = None, attempts: int | None = None) -> np.ndarray:
    """텍스트 목록 → float32 (n, dim) 배열. 정규화는 호출자(vector.py)에서 한다."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    client = get_client(timeout)
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t.replace("\n", " ")[:8000] or " " for t in texts[i:i + EMBED_BATCH]]
        resp = _call(lambda: client.embeddings.create(model=embedding_model(), input=batch), attempts)
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return np.asarray(out, dtype="float32")
