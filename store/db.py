"""SQLite 저장소 (PRD 3.4). 기사 id는 정규화한 URL의 해시."""
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "news.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    title          TEXT NOT NULL,
    url            TEXT NOT NULL UNIQUE,
    published_at   TEXT,              -- ISO8601 UTC (예: 2026-11-06T03:14:00+00:00)
    collected_at   TEXT NOT NULL,     -- ISO8601 UTC
    country        TEXT,
    language       TEXT,
    category_major TEXT NOT NULL DEFAULT '정보보안',
    category_minor TEXT,
    tags           TEXT,              -- JSON 배열 문자열
    summary        TEXT,              -- NULL이면 아직 분류/요약 전(대기 상태)
    embedded       INTEGER NOT NULL DEFAULT 0,
    category_confidence REAL,         -- LLM 분류 신뢰도 0~1
    category_reason     TEXT,         -- 분류 근거 한 줄
    relevant            INTEGER,      -- 보안 관련성: NULL=미판정, 1=관련, 0=무관(숨김, 삭제하지 않음)
    relevance_reason    TEXT,         -- 무관 판정 사유
    embed_hash          TEXT          -- 마지막으로 임베딩한 텍스트의 해시 (요약이 바뀌면 벡터 재생성 대상)
);
CREATE TABLE IF NOT EXISTS source_state (
    name       TEXT PRIMARY KEY,    -- 소스 이름
    last_ok_at TEXT NOT NULL        -- 이 소스를 마지막으로 성공적으로 수집·저장한 시각의 '요청 직전' (UTC ISO8601)
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_minor ON articles(category_minor);
"""

COLUMNS = [
    "id", "source", "title", "url", "published_at", "collected_at",
    "country", "language", "category_major", "category_minor", "tags",
    "summary", "embedded", "category_confidence", "category_reason",
    "relevant", "relevance_reason", "embed_hash",
]
_MIGRATIONS = {  # 이전 단계에서 만든 DB와 호환
    "category_confidence": "ALTER TABLE articles ADD COLUMN category_confidence REAL",
    "category_reason": "ALTER TABLE articles ADD COLUMN category_reason TEXT",
    "relevant": "ALTER TABLE articles ADD COLUMN relevant INTEGER",
    "relevance_reason": "ALTER TABLE articles ADD COLUMN relevance_reason TEXT",
    "embed_hash": "ALTER TABLE articles ADD COLUMN embed_hash TEXT",
}
KST = timezone(timedelta(hours=9))
EFFECTIVE_DATE = "COALESCE(published_at, collected_at)"
# 화면·검색·인덱스에 노출되는 기사 조건 (보안 무관으로 판정된 기사는 숨긴다)
VISIBLE = "COALESCE(relevant, 1) = 1"


def connect(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    for col, ddl in _MIGRATIONS.items():
        if col not in have:
            conn.execute(ddl)
    conn.commit()
    return conn


def insert_articles(conn: sqlite3.Connection, articles: list[dict]) -> int:
    """중복(id 또는 url)은 무시하고 실제로 추가된 건수를 반환."""
    before = conn.total_changes
    for a in articles:
        row = {c: a.get(c) for c in COLUMNS}
        row["category_major"] = row["category_major"] or "정보보안"
        row["embedded"] = row["embedded"] or 0
        if isinstance(row["tags"], (list, tuple)):
            row["tags"] = json.dumps(row["tags"], ensure_ascii=False)
        conn.execute(
            f"INSERT OR IGNORE INTO articles ({','.join(COLUMNS)}) "
            f"VALUES ({','.join('?' * len(COLUMNS))})",
            [row[c] for c in COLUMNS],
        )
    conn.commit()
    return conn.total_changes - before


def recent_titles(conn: sqlite3.Connection, limit: int = 1000) -> list[str]:
    """최근 저장된 기사의 (제목, 발행일). 제목 중복 판정에 쓴다."""
    cur = conn.execute(
        "SELECT title, published_at FROM articles ORDER BY collected_at DESC LIMIT ?", (limit,)
    )
    return [(r["title"], r["published_at"]) for r in cur.fetchall()]


def existing_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["id"] for r in conn.execute("SELECT id FROM articles")}


def list_articles(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM articles WHERE {VISIBLE} ORDER BY {EFFECTIVE_DATE} DESC LIMIT ?", (limit,)
    ).fetchall()


# ---------- 2단계: 분류/요약/임베딩 상태 관리 ----------

def date_bounds(d_from: date | None, d_to: date | None) -> tuple[str | None, str | None]:
    """KST 기준 날짜 범위(양끝 포함) → UTC ISO 경계 [start, end)."""
    start = end = None
    if d_from:
        start = datetime(d_from.year, d_from.month, d_from.day, tzinfo=KST).astimezone(timezone.utc).isoformat()
    if d_to:
        nxt = d_to + timedelta(days=1)
        end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=KST).astimezone(timezone.utc).isoformat()
    return start, end


def pending_enrichment(conn, limit: int | None = None) -> list[sqlite3.Row]:
    sql = f"SELECT * FROM articles WHERE summary IS NULL ORDER BY {EFFECTIVE_DATE} DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def pending_resummarize(conn, max_len: int, limit: int | None = None) -> list[sqlite3.Row]:
    """요약이 max_len보다 긴 기사(재요약 대상)."""
    sql = (f"SELECT * FROM articles WHERE summary IS NOT NULL AND length(summary) > ? AND {VISIBLE} "
           f"ORDER BY {EFFECTIVE_DATE} DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, (max_len,)).fetchall()


def save_enrichment(conn, article_id: str, minor: str, confidence: float | None,
                    reason: str | None, tags: list[str], summary: str,
                    relevant: int = 1, relevance_reason: str | None = None) -> None:
    """relevant=0(보안 무관)이면 기사는 보관하되 화면·검색에서 숨기고 임베딩 대상에서도 뺀다."""
    conn.execute(
        "UPDATE articles SET category_minor=?, category_confidence=?, category_reason=?, "
        "tags=?, summary=?, embedded=0, relevant=?, relevance_reason=? WHERE id=?",
        (minor, confidence, reason, json.dumps(tags, ensure_ascii=False), summary,
         relevant, relevance_reason, article_id),
    )
    conn.commit()


def pending_revalidate(conn, limit: int | None = None) -> list[sqlite3.Row]:
    """보안 관련성을 아직 판정하지 않은 기존 기사 (relevant IS NULL, 요약 있음)."""
    sql = (f"SELECT * FROM articles WHERE summary IS NOT NULL AND relevant IS NULL "
           f"ORDER BY {EFFECTIVE_DATE} DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def set_relevance(conn, article_id: str, relevant: int, reason: str | None = None) -> None:
    # 숨김 처리 시 embedded도 0으로 되돌려, 인덱스 재구축 대상에서 빠지게 한다.
    conn.execute(
        "UPDATE articles SET relevant=?, relevance_reason=?, embedded=CASE WHEN ?=0 THEN 0 ELSE embedded END "
        "WHERE id=?", (relevant, reason, relevant, article_id))
    conn.commit()


def hidden_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM articles WHERE relevant = 0").fetchone()[0]


def pending_embedding(conn) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM articles WHERE summary IS NOT NULL AND embedded=0 AND {VISIBLE} ORDER BY collected_at"
    ).fetchall()


def mark_embedded(conn, ids: list[str], value: int = 1, hashes: dict[str, str] | None = None) -> None:
    """임베딩 완료 표시. hashes(id→텍스트 해시)를 주면 함께 저장해 이후 변경을 감지한다."""
    if hashes is None:
        conn.executemany("UPDATE articles SET embedded=? WHERE id=?", [(value, i) for i in ids])
    else:
        conn.executemany("UPDATE articles SET embedded=?, embed_hash=? WHERE id=?",
                         [(value, hashes.get(i), i) for i in ids])
    conn.commit()


def get_by_ids(conn, ids: list[str]) -> dict[str, sqlite3.Row]:
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = f"SELECT * FROM articles WHERE id IN ({','.join('?' * len(chunk))})"
        out.update({r["id"]: r for r in conn.execute(q, chunk)})
    return out


def query_articles(conn, minors=None, country=None, d_from=None, d_to=None,
                   sources=None, keyword=None, limit=200, offset=0) -> tuple[list[sqlite3.Row], int]:
    """뉴스 목록 필터 조회. (행, 전체 건수) 반환. country: 'KR' 또는 'GLOBAL'."""
    where, params = ["summary IS NOT NULL", VISIBLE], []
    if minors:
        where.append(f"category_minor IN ({','.join('?' * len(minors))})")
        params += list(minors)
    if country:
        where.append("country = ?")
        params.append(country)
    if sources:
        where.append(f"source IN ({','.join('?' * len(sources))})")
        params += list(sources)
    start, end = date_bounds(d_from, d_to)
    if start:
        where.append(f"{EFFECTIVE_DATE} >= ?")
        params.append(start)
    if end:
        where.append(f"{EFFECTIVE_DATE} < ?")
        params.append(end)
    if keyword:
        esc = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{esc}%"
        where.append("(title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')")
        params += [like, like, like]
    clause = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM articles WHERE {clause}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM articles WHERE {clause} ORDER BY {EFFECTIVE_DATE} DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return rows, total


def distinct_sources(conn) -> list[str]:
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT source FROM articles WHERE {VISIBLE} ORDER BY source")]


# ---------- 수집 기준 시간: 소스별 마지막 성공 시각 ----------

def get_source_states(conn) -> dict[str, datetime]:
    out = {}
    for r in conn.execute("SELECT name, last_ok_at FROM source_state"):
        try:
            out[r["name"]] = datetime.fromisoformat(r["last_ok_at"])
        except ValueError:
            continue
    return out


def set_source_ok(conn, name: str, at: datetime) -> None:
    """기사를 DB에 저장한 '뒤에' 호출해야 한다 (저장 전에 앞당기면 크래시 시 기사가 영영 빠진다)."""
    conn.execute(
        "INSERT INTO source_state (name, last_ok_at) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET last_ok_at = excluded.last_ok_at", (name, at.isoformat()))
    conn.commit()
