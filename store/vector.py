"""FAISS 벡터 저장소 (PRD 3.5). 코사인 유사도 = L2 정규화 + 내적(IndexFlatIP).

data/faiss.index 의 i번째 벡터 ↔ data/id_map.json 의 i번째 기사 id.
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

from store import db

log = logging.getLogger("store.vector")
DATA_DIR = db.ROOT / "data"
EMBED_BATCH = 100


class IndexCorrupt(RuntimeError):
    pass


def tag_list(raw) -> list[str]:
    """DB의 tags(JSON 문자열)를 리스트로. 깨진 값/None은 빈 리스트."""
    try:
        tags = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return []
    return [str(t) for t in tags if str(t).strip()] if isinstance(tags, list) else []


def embedding_text(row) -> str:
    # 요약이 짧아 CVE ID 등이 빠지므로 태그도 임베딩에 포함한다
    tags = ", ".join(tag_list(row["tags"]))
    tag_part = f"\n태그: {tags}" if tags else ""
    return f"{row['title']}\n{row['summary']}{tag_part}\n분류: {row['category_minor']} / 출처: {row['source']}"


def text_hash(row) -> str:
    """임베딩에 쓴 텍스트의 해시. 요약·태그·분류가 바뀌면 달라져 벡터가 오래됐음을 알 수 있다."""
    return hashlib.sha1(embedding_text(row).encode("utf-8")).hexdigest()


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.ascontiguousarray(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


class VectorStore:
    def __init__(self, directory: Path | str = DATA_DIR):
        self.dir = Path(directory)
        self.index_path = self.dir / "faiss.index"
        self.map_path = self.dir / "id_map.json"
        self.index = None
        self.ids: list[str] = []
        self.load()

    def load(self):
        if self.index_path.exists() and self.map_path.exists():
            # faiss.read_index는 Windows에서 한글 경로를 열지 못하므로 Python으로 읽는다
            self.index = faiss.deserialize_index(np.frombuffer(self.index_path.read_bytes(), dtype="uint8"))
            self.ids = json.loads(self.map_path.read_text(encoding="utf-8"))
            if self.index.ntotal != len(self.ids):
                raise IndexCorrupt(
                    f"인덱스({self.index.ntotal})와 id_map({len(self.ids)}) 개수가 다릅니다. "
                    "python scripts/rebuild_index.py 로 재구축하세요."
                )
        else:
            self.index, self.ids = None, []

    @property
    def total(self) -> int:
        return len(self.ids)

    def __contains__(self, article_id: str) -> bool:
        return article_id in set(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> int:
        """이미 있는 id는 건너뛴다(중복 추가 방지). 추가된 개수를 반환."""
        known = set(self.ids)
        keep, seen = [], set()
        for i, a in enumerate(ids):
            if a not in known and a not in seen:
                keep.append(i)
                seen.add(a)
        if not keep:
            return 0
        vecs = _normalize(np.asarray(vectors, dtype="float32")[keep])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.ids.extend(ids[i] for i in keep)
        return len(keep)

    def upsert(self, ids: list[str], vectors: np.ndarray) -> tuple[int, int]:
        """없는 id는 추가하고, 있는 id는 벡터를 교체한다. (추가 수, 교체 수) 반환.

        교체는 인덱스에 이미 있는 벡터를 꺼내 해당 행만 바꿔 다시 만든다 (API 호출 없음).
        """
        uniq: dict[str, int] = {}
        for j, a in enumerate(ids):
            uniq[a] = j                                   # 같은 id가 반복되면 마지막 것을 쓴다
        if not uniq:
            return 0, 0
        order = list(uniq)
        vecs = _normalize(np.asarray(vectors, dtype="float32")[[uniq[a] for a in order]])
        pos = {a: i for i, a in enumerate(self.ids)}
        replace = [(pos[a], k) for k, a in enumerate(order) if a in pos]
        new_k = [k for k, a in enumerate(order) if a not in pos]
        if replace:
            current = self.index.reconstruct_n(0, self.index.ntotal)
            for i, k in replace:
                current[i] = vecs[k]
            rebuilt = faiss.IndexFlatIP(current.shape[1])
            rebuilt.add(current)
            self.index = rebuilt
        if new_k:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs[new_k])
            self.ids.extend(order[k] for k in new_k)
        return len(new_k), len(replace)

    def save(self):
        """임시 파일에 쓴 뒤 교체한다 (중간에 끊겨도 기존 파일이 깨지지 않게)."""
        if self.index is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_i, tmp_m = self.dir / "faiss.index.tmp", self.dir / "id_map.json.tmp"
        tmp_i.write_bytes(faiss.serialize_index(self.index).tobytes())  # 한글 경로 대응
        tmp_m.write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_i, self.index_path)
        os.replace(tmp_m, self.map_path)

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.index is None or self.total == 0:
            return []
        q = _normalize(np.asarray(qvec, dtype="float32").reshape(1, -1))
        k = min(k, self.total)
        scores, idx = self.index.search(q, k)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def stale_rows(conn) -> list:
    """임베딩 표시는 돼 있지만 텍스트가 바뀌었거나(해시 불일치) 해시를 모르는(NULL, 구버전 데이터) 기사."""
    rows = conn.execute(
        f"SELECT * FROM articles WHERE summary IS NOT NULL AND embedded=1 AND {db.VISIBLE}").fetchall()
    return [r for r in rows if r["embed_hash"] != text_hash(r)]


def embed_pending(conn, store: VectorStore, embed_fn=None) -> dict:
    """임베딩이 필요한 기사만 처리한다: ① 아직 임베딩 안 된 기사 ② 텍스트가 바뀌어 벡터가 오래된 기사.

    이미 인덱스에 있는 id는 중복 추가하지 않고 벡터만 교체한다 (저장 중 크래시 복구도 같은 경로).
    """
    from core import llm
    embed_fn = embed_fn or llm.embed
    todo = db.pending_embedding(conn) + stale_rows(conn)
    stats = {"pending": len(todo), "added": 0, "replaced": 0}
    for i in range(0, len(todo), EMBED_BATCH):
        chunk = todo[i:i + EMBED_BATCH]
        vecs = embed_fn([embedding_text(r) for r in chunk])
        ids = [r["id"] for r in chunk]
        added, replaced = store.upsert(ids, vecs)
        stats["added"] += added
        stats["replaced"] += replaced
        store.save()                                                        # 인덱스를 먼저 저장하고
        db.mark_embedded(conn, ids, hashes={r["id"]: text_hash(r) for r in chunk})  # 그 다음 DB 표시
        log.info("임베딩 진행 %d/%d", min(i + EMBED_BATCH, len(todo)), len(todo))
    return stats


def check_consistency(conn, store: VectorStore) -> dict:
    db_ids = db.existing_ids(conn)
    idx_ids = set(store.ids)
    flagged = {r["id"] for r in conn.execute("SELECT id FROM articles WHERE embedded=1")}
    hidden = {r["id"] for r in conn.execute("SELECT id FROM articles WHERE relevant = 0")}
    return {
        "index_total": store.total,
        "duplicate_ids_in_map": store.total - len(idx_ids),
        "index_ids_missing_in_db": sorted(idx_ids - db_ids),
        "flagged_but_not_in_index": sorted(flagged - idx_ids),
        "in_index_but_flag_0": sorted((idx_ids & db_ids) - flagged - hidden),
        "in_index_but_hidden": sorted(idx_ids & hidden),   # 숨긴 기사의 벡터가 남음 → rebuild로 정리
        "stale_or_unverified_vectors": sorted(r["id"] for r in stale_rows(conn)),   # 텍스트가 바뀐 뒤 재임베딩 안 됨
    }


def rebuild(conn, directory=DATA_DIR, embed_fn=None) -> dict:
    """요약이 있는 모든 (숨기지 않은) 기사를 다시 임베딩해 인덱스를 처음부터 구축."""
    from core import llm
    embed_fn = embed_fn or llm.embed
    rows = conn.execute(
        f"SELECT * FROM articles WHERE summary IS NOT NULL AND {db.VISIBLE} ORDER BY collected_at").fetchall()
    store = VectorStore.__new__(VectorStore)
    store.dir = Path(directory)
    store.index_path, store.map_path = store.dir / "faiss.index", store.dir / "id_map.json"
    store.index, store.ids = None, []
    for i in range(0, len(rows), EMBED_BATCH):
        chunk = rows[i:i + EMBED_BATCH]
        store.add([r["id"] for r in chunk], embed_fn([embedding_text(r) for r in chunk]))
    store.save()
    conn.execute("UPDATE articles SET embedded=0, embed_hash=NULL")
    conn.commit()
    db.mark_embedded(conn, store.ids, hashes={r["id"]: text_hash(r) for r in rows})
    return {"indexed": store.total}
