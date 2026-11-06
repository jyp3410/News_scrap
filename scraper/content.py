"""요약용 본문 확보. 본문은 메모리에서만 쓰고 DB에는 저장하지 않는다 (PRD 3.3, 저작권)."""
import logging

import trafilatura

from scraper import http

log = logging.getLogger("scraper.content")
MIN_SNIPPET = 300      # RSS 요약이 이보다 짧으면 본문을 가져온다
MAX_CHARS = 3000       # LLM 입력 상한


def get_text(url: str, snippet: str = "") -> str:
    snippet = (snippet or "").strip()
    if len(snippet) >= MIN_SNIPPET:
        return snippet[:MAX_CHARS]
    try:
        html = http.get(url).text
        body = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        if len(body) > len(snippet):
            return body[:MAX_CHARS]
    except http.RobotsDisallowed:
        log.info("robots.txt 불허, 본문 생략: %s", url)
    except Exception as e:
        log.warning("본문 수집 실패(%s): %s", url, e)
    return snippet[:MAX_CHARS]
