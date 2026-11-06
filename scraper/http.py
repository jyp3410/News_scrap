"""robots.txt 준수 + 재시도 HTTP (PRD F-6, F-7)."""
import logging
import threading
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger("scraper.http")

USER_AGENT = "SecNewsBot/0.1 (personal news scraper)"
TIMEOUT = 20
MAX_ATTEMPTS = 3
HOST_MIN_INTERVAL = 1.0   # 같은 호스트에 대한 요청 사이 최소 간격(초). 병렬 스레드가 몰려도 지켜진다
_robots_cache: dict[str, RobotFileParser | None] = {}
_host_locks: dict[str, threading.Lock] = {}
_host_last: dict[str, float] = {}
_registry_lock = threading.Lock()


class RobotsDisallowed(Exception):
    pass


def _throttle(url: str) -> None:
    """같은 호스트로 가는 요청을 직렬화하고 최소 간격을 둔다 (PRD F-6)."""
    host = urlsplit(url).netloc.lower()
    with _registry_lock:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        wait = HOST_MIN_INTERVAL - (time.monotonic() - _host_last.get(host, float("-inf")))
        if wait > 0:
            time.sleep(wait)
        _host_last[host] = time.monotonic()


def _robots_for(url: str) -> RobotFileParser | None:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base in _robots_cache:
        return _robots_cache[base]
    rp = RobotFileParser()
    try:
        _throttle(base)
        r = requests.get(base + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if r.status_code == 200 and "html" not in r.headers.get("content-type", "").lower():
            rp.parse(r.text.splitlines())
        elif r.status_code in (401, 403):
            rp.disallow_all = True
        else:  # 404 등: robots.txt 없음 → 허용
            rp.parse([])
    except requests.RequestException as e:
        log.warning("robots.txt 확인 실패(%s): %s — 허용으로 간주", base, e)
        rp = None
    _robots_cache[base] = rp
    return rp


def allowed(url: str) -> bool:
    rp = _robots_for(url)
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


def get(url: str, check_robots: bool = True) -> requests.Response:
    if check_robots and not allowed(url):
        raise RobotsDisallowed(f"robots.txt가 수집을 허용하지 않음: {url}")
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _throttle(url)
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and 400 <= status < 500 and status != 429:
                break  # 재시도해도 소용없는 클라이언트 오류
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    raise last
