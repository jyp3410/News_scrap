"""URL 정규화와 중복 제거 (PRD F-4)."""
import hashlib
import re
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {
    "fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src",
    "source", "feature", "spm", "cmpid", "ocid",
}
TITLE_SIMILARITY = 0.92


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((
        parts.scheme.lower(), parts.netloc.lower(), path,
        urlencode(sorted(query)), "",
    ))


def article_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()


MIN_WORD_JACCARD = 0.75   # 제목 단어 집합의 겹침 비율 하한
SAME_STORY_DAYS = 3       # 발행일이 이보다 벌어지면 제목이 같아도 별개 기사로 본다 (주간 정리글 등)

_DIGITS = re.compile(r"\d[\d.,]*\d|\d")
_ACRONYM = re.compile(r"(?<![A-Za-z])[A-Z]{1,5}(?![A-Za-z])")   # KT, SKT, A, B, AI, CVE ...


def _norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", title.lower()).strip()


def _identifiers(title: str) -> tuple[frozenset, frozenset]:
    """제목의 숫자(CVE 번호·버전·건수)와 대문자 약어(KT/SKT, A/B병원). 한 글자만 달라도 다른 기사다."""
    return frozenset(_DIGITS.findall(title)), frozenset(_ACRONYM.findall(title))


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(_norm_title(a).split()), set(_norm_title(b).split())
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def _parse_dt(value):
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def is_similar_title(a: str, b: str, a_date: str | None = None, b_date: str | None = None) -> bool:
    """같은 기사로 볼 만큼 제목이 비슷한가. 서로 다른 기사를 영구히 잃지 않도록 보수적으로 판정한다.

    모두 만족해야 중복: ① 숫자·약어가 같음 ② 문자열 유사도 ≥ 0.92 ③ 단어 겹침 ≥ 0.75
    ④ 발행일을 알 때 3일 이내.
    """
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if _identifiers(a) != _identifiers(b):
        return False
    if SequenceMatcher(None, na, nb).ratio() < TITLE_SIMILARITY or _jaccard(a, b) < MIN_WORD_JACCARD:
        return False
    da, db_ = _parse_dt(a_date), _parse_dt(b_date)
    if da and db_:
        try:
            if abs((da - db_).total_seconds()) > SAME_STORY_DAYS * 86400:
                return False
        except TypeError:   # tz-aware와 naive가 섞인 경우: 날짜 비교는 포기하고 제목만 본다
            pass
    return True


def _as_pair(item) -> tuple[str, str | None]:
    return (item, None) if isinstance(item, str) else (item[0], item[1])


def dedup(articles: list[dict], known_ids: set[str], known_titles: list) -> list[dict]:
    """이미 저장된 id/제목과, 이번 배치 내부의 중복을 제거한다.

    known_titles: 제목 문자열 또는 (제목, 발행일) 튜플의 목록.
    """
    seen_ids = set(known_ids)
    titles = [_as_pair(t) for t in known_titles]
    out = []
    for a in articles:
        aid = a["id"]
        if aid in seen_ids:
            continue
        pub = a.get("published_at")
        if any(is_similar_title(a["title"], t, pub, d) for t, d in titles):
            continue
        seen_ids.add(aid)
        titles.append((a["title"], pub))
        out.append(a)
    return out
