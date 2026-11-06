"""특정 뉴스사의 저장된 기사를 DB에서 삭제하고 FAISS 인덱스를 재구축한다.

  python scripts/remove_source.py "Dark Reading"            # 미리보기 (아무것도 바꾸지 않음)
  python scripts/remove_source.py "Dark Reading" --apply    # 실제 삭제 (되돌릴 수 없으니 먼저 data/를 백업할 것)

수집을 막으려면 config/sources.yaml의 exclude_sources에도 이름을 추가한다 (이 스크립트는 설정을 바꾸지 않는다).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402
from store import db  # noqa: E402
from store.vector import rebuild  # noqa: E402


def remove_source(conn, name: str, apply: bool = False) -> dict:
    """name과 정확히 일치하는 source의 기사를 삭제. apply=False면 건수만 센다."""
    total = conn.execute("SELECT COUNT(*) FROM articles WHERE source = ?", (name,)).fetchone()[0]
    hidden = conn.execute("SELECT COUNT(*) FROM articles WHERE source = ? AND relevant = 0", (name,)).fetchone()[0]
    if apply and total:
        conn.execute("DELETE FROM articles WHERE source = ?", (name,))
        conn.commit()
    return {"source": name, "found": total, "of_which_hidden": hidden, "deleted": total if apply else 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="sources.yaml의 name과 같은 뉴스사 이름 (정확히 일치)")
    ap.add_argument("--apply", action="store_true", help="실제로 삭제한다")
    args = ap.parse_args()
    conn = db.connect()
    try:
        res = remove_source(conn, args.name, args.apply)
        if res["found"] == 0:
            known = [r[0] for r in conn.execute("SELECT DISTINCT source FROM articles ORDER BY source")]
            print(f"'{args.name}' 기사가 없습니다. 저장된 뉴스사: {known}")
            sys.exit(1)
        if not args.apply:
            print(f"[미리보기] '{args.name}' 기사 {res['found']}건(숨김 {res['of_which_hidden']}건 포함)이 삭제됩니다. "
                  "실제로 삭제하려면 --apply를 붙이세요.")
            sys.exit(0)
        print(f"삭제: {res['deleted']}건")
        try:
            print("인덱스 재구축:", rebuild(conn))
        except llm.ConfigError as e:
            print(f"[설정 오류] 기사는 삭제됐지만 인덱스는 재구축하지 못했습니다: {e}\n"
                  "키를 설정한 뒤 python scripts/rebuild_index.py 를 실행하세요.")
            sys.exit(2)
    finally:
        conn.close()
