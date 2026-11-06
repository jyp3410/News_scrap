"""RSS 수집 (PRD F-1, F-2, F-6 일부, F-7)."""
import calendar
import html
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import feedparser
import yaml

from scraper import http
from scraper.dedup import article_id, normalize_url

log = logging.getLogger("scraper.fetch")

ROOT = Path(__file__).resolve().parent.parent
REQUEST_DELAY_SEC = 1.0  # 소스 간 요청 간격

_TZ_SUFFIX = re.compile(r"(Z|GMT|UTC?|[A-Z]{3,4}|[+-]\d{2}:?\d{2})\s*$")
_TZ_ABBR = re.compile(r"\b([A-Z]{2,5})\s*$")
_FEEDPARSER_ZONES = {"GMT", "UT", "UTC", "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT"}  # feedparser가 올바로 해석
_ABBR_OFFSET = {"KST": 9, "JST": 9, "CET": 1, "CEST": 2, "BST": 1, "IST": 5.5, "SGT": 8, "HKT": 8, "AEST": 10}


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _key(name: str) -> str:
    return str(name).strip().casefold()


EVENT_KIND = "event_search"
BING_NEWS_RSS = "https://www.bing.com/news/search?q={q}&format=rss&setlang=ko"


def event_sources(cfg: dict) -> list[dict]:
    """`event_search` 설정을 검색 기반 소스 목록으로 펼친다.

    뉴스 RSS는 최신 며칠치만 주므로 SECON·ISEC·RSA 같은 행사는 그 주에 기사가 없으면 잡히지 않는다.
    행사 이름으로 검색한 뉴스(Bing News RSS)를 함께 모아 이를 보완한다.
    """
    ev = cfg.get("event_search") or {}
    if not ev.get("enabled", False):
        return []
    return [{
        "name": f"행사검색: {q}", "url": BING_NEWS_RSS.format(q=quote(str(q))), "method": "rss",
        "kind": EVENT_KIND, "utc_offset": 0,
        "max_age_days": ev.get("max_age_days", 180), "limit": ev.get("per_query_limit", 15),
    } for q in (ev.get("queries") or [])]


def excluded_names(cfg: dict | None = None) -> set[str]:
    cfg = cfg if cfg is not None else load_yaml("sources.yaml")
    return {_key(n) for n in (cfg.get("exclude_sources") or [])}


def load_sources(config: dict | None = None) -> list[dict]:
    """수집 대상 소스. `enabled: false`이거나 `exclude_sources`에 이름이 있는 소스는 건너뛴다."""
    cfg = config if config is not None else load_yaml("sources.yaml")
    excluded = excluded_names(cfg)
    all_sources = list(cfg["sources"]) + event_sources(cfg)
    known = {_key(s["name"]) for s in all_sources}
    for typo in sorted(excluded - known):   # 오타로 제외가 안 먹히는 일을 막는다
        log.warning("exclude_sources에 있는 '%s'는 sources에 없는 이름입니다 (오타 확인)", typo)
    active = []
    for s in all_sources:
        if s.get("enabled", True) is False or _key(s["name"]) in excluded:
            log.info("수집 제외 소스: %s", s["name"])
            continue
        active.append(s)
    return active


def _strip_html(text: str) -> str:
    # 태그를 공백으로 바꾼 뒤 연속 공백을 한 칸으로 줄인다 (인라인 태그가 있어도 '공격  발생'처럼 벌어지지 않게)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def _parse_published(entry, utc_offset: int) -> str | None:
    """발행일을 UTC ISO8601로. 원문에 시간대 표기가 없으면 소스의 utc_offset을 가정한다."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    raw = entry.get("published") or entry.get("updated") or ""
    dt = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    abbr = _TZ_ABBR.search(raw)
    if abbr and abbr.group(1) not in _FEEDPARSER_ZONES:
        # feedparser가 모르는 약어(KST 등)는 UTC로 잘못 해석한다 → 알려진 오프셋으로 보정,
        # 표에 없는 약어는 소스의 utc_offset을 가정한다.
        dt = dt - timedelta(hours=_ABBR_OFFSET.get(abbr.group(1), utc_offset))
    elif not _TZ_SUFFIX.search(raw):
        # 시간대 표기가 없으면 feedparser는 UTC로 취급하므로 현지 시각으로 되돌린다.
        dt = dt - timedelta(hours=utc_offset)
    return dt.isoformat()


def _matches_keywords(text: str, keywords: dict) -> bool:
    low = text.lower()
    return any(k.lower() in low for lang in ("ko", "en") for k in keywords.get(lang, []))


SUPPORTED_METHODS = {"rss"}
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")


def _guess_language(text: str) -> str:
    """제목의 문자로 언어를 추정한다 (검색 결과에는 언어 정보가 없다)."""
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _HAN.search(text):
        return "zh"
    return "en"


def _bing_real_url(link: str) -> str:
    """Bing 뉴스 RSS의 리다이렉트 링크(apiclick.aspx?...&url=원문)에서 원문 주소를 꺼낸다."""
    if "bing.com" in urlsplit(link).netloc:
        real = parse_qs(urlsplit(link).query).get("url", [""])[0]
        if real.startswith(("http://", "https://")):
            return real
    return link


def _outlet_name(entry, url: str) -> str:
    name = str(entry.get("news_source") or "").strip()
    return name or urlsplit(url).netloc.lower().removeprefix("www.")


DEFAULT_LOOKBACK_HOURS = 6.5     # 실행 시각 − 6시간 30분 (6시간 주기 + 30분 여유)
DEFAULT_OVERLAP_MINUTES = 30


def window_settings(cfg: dict | None = None) -> tuple[float, float]:
    cfg = cfg if cfg is not None else load_yaml("sources.yaml")
    w = cfg.get("collection") or {}
    return float(w.get("lookback_hours", DEFAULT_LOOKBACK_HOURS)), float(w.get("overlap_minutes", DEFAULT_OVERLAP_MINUTES))


def compute_since(now: datetime, last_ok: datetime | None, lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
                  overlap_minutes: float = DEFAULT_OVERLAP_MINUTES) -> datetime | None:
    """이 소스에서 이 시각 이후에 발행된 기사만 가져온다. None이면 제한 없음.

    - 처음 수집하는 소스(기록 없음): 제한 없음 (피드가 주는 만큼 전부)
    - 그 외: (실행 시각 − lookback)와 (직전 성공 시각 − overlap) 중 **더 이른 쪽**.
      실행이 지연되거나 한 번 실패해도 직전 성공 이후 구간이 빠지지 않고, 정상일 때는 lookback보다 좁아지지 않는다.
    """
    if last_ok is None:
        return None
    return min(now - timedelta(hours=lookback_hours), last_ok - timedelta(minutes=overlap_minutes))


def fetch_source(src: dict, security_keywords: dict | None = None, since: datetime | None = None) -> list[dict]:
    method = str(src.get("method", "rss")).lower()
    if method not in SUPPORTED_METHODS:   # 설정이 지원하지 않는 방식을 암시하면 조용히 실패하지 않고 알린다
        raise ValueError(f"지원하지 않는 수집 방식 '{method}' (현재 지원: {', '.join(sorted(SUPPORTED_METHODS))})")
    resp = http.get(src["url"])  # robots.txt 확인 + 최대 3회 재시도
    feed = feedparser.parse(resp.content)
    if not feed.entries:
        raise ValueError(f"RSS 항목이 없음 (RSS가 아니거나 형식 오류): {src['url']}")

    collected_at = datetime.now(timezone.utc).isoformat()
    is_event = src.get("kind") == EVENT_KIND
    max_age = timedelta(days=src["max_age_days"]) if is_event and src.get("max_age_days") else None
    out = []
    for e in feed.entries:
        title = _strip_html(e.get("title", ""))
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        if is_event:
            link = _bing_real_url(link)
        snippet = _strip_html(
            (e.get("content") or [{}])[0].get("value") or e.get("summary", "")
        )
        if src.get("keyword_filter") and security_keywords:
            blob = f"{title} {snippet}"
            if not _matches_keywords(blob, security_keywords):
                continue
        published = _parse_published(e, src.get("utc_offset", 0))
        # 수집 기준 시간(since)은 일반 소스에만 적용한다. 행사 검색은 지난 행사까지 모으는 것이 목적이라 max_age_days를 쓴다.
        if since is not None and not is_event and published and datetime.fromisoformat(published) < since:
            continue
        if max_age is not None:
            # 검색 결과는 오래된 기사와 발행일이 틀린 기사가 섞여 있다. 날짜를 확인할 수 없거나 너무 오래되면 버린다.
            if not published or datetime.fromisoformat(published) < datetime.now(timezone.utc) - max_age:
                continue
        if is_event:   # 뉴스사는 검색 결과가 알려 주는 실제 매체명, 국내/해외는 제목의 한글 여부로 구분
            source_name = _outlet_name(e, link)
            language = _guess_language(title)
            country = "KR" if language == "ko" else "GLOBAL"
        else:
            source_name, country, language = src["name"], src.get("country"), src.get("language")
        out.append({
            "id": article_id(link),
            "source": source_name,
            "title": title,
            "url": normalize_url(link),
            "published_at": published,
            "collected_at": collected_at,
            "country": country,
            "language": language,
            "snippet": snippet[:2000],  # DB에 저장하지 않는 임시 요약용 텍스트
        })
        if is_event and src.get("limit") and len(out) >= src["limit"]:
            break
    return out


def fetch_all(sources: list[dict] | None = None, states: dict[str, datetime] | None = None,
              now: datetime | None = None) -> tuple[list[dict], dict]:
    """모든 소스를 수집. 한 소스가 실패해도 계속 진행하고, (기사, 소스별 결과)를 반환.

    states: 소스별 마지막 성공 시각. 주면 수집 기준 시간(compute_since)을 적용하고, 생략하면 제한 없이 전부 가져온다.
    report[이름]["fetched_at"]은 그 소스를 요청하기 직전 시각으로, 기사를 저장한 뒤 상태로 기록하는 값이다.
    """
    sources = sources if sources is not None else load_sources()
    keywords = load_yaml("categories.yaml").get("security_keywords")
    excluded = excluded_names()
    lookback, overlap = window_settings()
    now = now or datetime.now(timezone.utc)
    articles, report = [], {}
    for i, src in enumerate(sources):
        if i:
            time.sleep(REQUEST_DELAY_SEC)
        since = compute_since(now, (states or {}).get(src["name"]), lookback, overlap) if states is not None else None
        requested_at = datetime.now(timezone.utc)
        try:
            items = fetch_source(src, keywords, since)
            if src.get("kind") == EVENT_KIND and excluded:   # 검색 결과의 실제 매체명도 제외 설정을 따른다
                items = [a for a in items if _key(a["source"]) not in excluded]
            articles.extend(items)
            report[src["name"]] = {"ok": True, "count": len(items), "fetched_at": requested_at,
                                   "since": since.isoformat() if since else None}
            log.info("%s: %d건%s", src["name"], len(items), f" (기준 {since.astimezone(timezone(timedelta(hours=9))):%m-%d %H:%M} KST 이후)" if since else "")
        except Exception as exc:  # 소스별 격리
            report[src["name"]] = {"ok": False, "error": str(exc)}
            log.warning("%s 수집 실패: %s", src["name"], exc)
    return articles, report
