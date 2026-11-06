"""FAISS 인덱스 정합성 검증 / 전체 재구축.

  python scripts/rebuild_index.py --check    # 검증만 (API 호출 없음)
  python scripts/rebuild_index.py            # 전체 재임베딩 후 재구축
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402
from store import db  # noqa: E402
from store.vector import IndexCorrupt, VectorStore, check_consistency, rebuild  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="검증만 수행")
args = ap.parse_args()

conn = db.connect()
try:
    if args.check:
        try:
            report = check_consistency(conn, VectorStore())
        except IndexCorrupt as e:
            print(f"[불일치] {e}")
            sys.exit(1)
        for k, v in report.items():
            print(f"{k}: {len(v) if isinstance(v, list) else v}")
        bad = (report["duplicate_ids_in_map"] or report["index_ids_missing_in_db"]
               or report["flagged_but_not_in_index"] or report["in_index_but_hidden"]
               or report["stale_or_unverified_vectors"])
        print("정합성 문제 있음 → rebuild 권장" if bad else "정합성 OK")
        sys.exit(1 if bad else 0)
    print(rebuild(conn))
except llm.ConfigError as e:
    print(f"[설정 오류] {e}")
    sys.exit(2)
finally:
    conn.close()
