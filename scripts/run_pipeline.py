"""수집 → 중복 제거 → 저장 → 분류/요약 → 임베딩. 재실행해도 중복 저장/호출이 없다(멱등).

옵션
  --skip-llm     수집·저장만 수행 (OpenAI 호출 없음)
  --resummarize  요약이 너무 긴 기존 기사를 다시 요약하고 인덱스를 재구축 (수집은 하지 않음)
  --revalidate   보안 관련성을 아직 판정하지 않은 기존 기사를 재판정 (--dry-run이면 결과만 출력, 수집은 하지 않음)
  --full         수집 기준 시간(실행 시각 − 6시간 30분)을 무시하고 피드가 주는 기사를 전부 대상으로 한다 (장애 복구·초기 수집용)
  --limit N      이번 실행에서 분류/요약할 기사 수 상한 (비용 제어, 기본 PIPELINE_MAX_ENRICH 또는 300)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm  # noqa: E402
from scraper.dedup import dedup  # noqa: E402
from scraper.enrich import enrich_pending, revalidate_relevance  # noqa: E402
from scraper.fetch import fetch_all  # noqa: E402
from store import db  # noqa: E402
from store.vector import VectorStore, embed_pending, rebuild  # noqa: E402


def collect(conn, full: bool = False) -> tuple[dict, dict, dict]:
    """수집 → 중복 제거 → 저장. 소스별 '마지막 성공 시각'은 저장이 끝난 뒤에만 기록한다.

    full=False면 소스별 수집 기준 시간(scraper.fetch.compute_since)을 적용한다: 실행 시각 − 6시간 30분과
    직전 성공 시각 − 30분 중 더 이른 쪽 이후에 발행된 기사만 본다. 처음 수집하는 소스는 전부 가져온다.
    """
    states = None if full else db.get_source_states(conn)
    articles, report = fetch_all(states=states)
    fresh = dedup(articles, db.existing_ids(conn), db.recent_titles(conn))
    inserted = db.insert_articles(conn, fresh)
    for name, r in report.items():
        if r["ok"]:                      # 실패한 소스는 상태를 앞당기지 않아 다음 실행이 그 구간을 다시 본다
            db.set_source_ok(conn, name, r["fetched_at"])
    snippets = {a["id"]: a.get("snippet", "") for a in fresh}
    return {"fetched": len(articles), "new": inserted}, report, snippets


def run_resummarize(limit: int | None = None) -> dict:
    """긴 요약만 다시 생성. 요약 텍스트가 바뀌면 임베딩도 달라지므로 인덱스를 처음부터 재구축한다.

    (embed_pending은 이미 인덱스에 있는 id를 건너뛰므로, 요약 수정 후에는 rebuild가 필요하다.)
    """
    conn = db.connect()
    try:
        llm.get_client()
        res = {"enrich": enrich_pending(conn, limit=limit, resummarize=True)}
        if res["enrich"]["done"]:
            res["rebuild"] = rebuild(conn)
        return res
    finally:
        conn.close()


def run_revalidate(limit: int | None = None, dry_run: bool = False) -> dict:
    """보안 관련성을 아직 판정하지 않은 기존 기사를 재판정한다 (수집은 하지 않음).

    무관 판정은 삭제가 아니라 숨김이다. 실제로 숨긴 기사가 있으면 인덱스를 다시 구축해 벡터도 정리한다.
    """
    conn = db.connect()
    try:
        llm.get_client()
        res = {"revalidate": revalidate_relevance(conn, limit, dry_run)}
        if not dry_run and res["revalidate"]["rejected"]:
            res["rebuild"] = rebuild(conn)
        return res
    finally:
        conn.close()


def run(skip_llm: bool = False, limit: int | None = None, full: bool = False) -> dict:
    conn = db.connect()
    try:
        stats, report, snippets = collect(conn, full)
        result = {**stats, "sources": report}
        if skip_llm:
            return result
        llm.get_client()  # 키가 없으면 여기서 ConfigError (수집 결과는 이미 저장됨)
        result["enrich"] = enrich_pending(conn, snippets, limit)
        result["embed"] = embed_pending(conn, VectorStore())
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--revalidate", action="store_true")
    ap.add_argument("--full", action="store_true", help="수집 기준 시간을 무시하고 전부 대상으로 한다")
    ap.add_argument("--dry-run", action="store_true", help="--revalidate와 함께: 결과만 보여주고 DB는 바꾸지 않음")
    ap.add_argument("--limit", type=int, default=int(os.getenv("PIPELINE_MAX_ENRICH", "300")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.resummarize:
            print(run_resummarize(args.limit))
            sys.exit(0)
        if args.revalidate:
            res = run_revalidate(None, args.dry_run)
            rv = res["revalidate"]
            print(f"재검증: {rv['checked']}건 판정 / 유지 {rv['kept']} / 무관 {rv['rejected']}"
                  f"{' (미적용: dry-run)' if rv['dry_run'] else ''} / 판정 실패 {rv['missing']}")
            for src, n in sorted(rv["rejected_by_source"].items(), key=lambda x: -x[1]):
                print(f"  - {src}: {n}/{rv['total_by_source'].get(src, 0)}건 무관")
            for it in rv["rejected_items"]:
                print(f"    · [{it['source']}] {it['title'][:60]}  ({it['reason']})")
            if "rebuild" in res:
                print(f"인덱스 재구축: {res['rebuild']}")
            sys.exit(0)
        res = run(args.skip_llm, args.limit, args.full)
    except llm.ConfigError as e:
        print(f"[설정 오류] {e}\n수집 결과는 DB에 저장되었습니다. 키를 설정한 뒤 다시 실행하면 이어서 처리합니다.")
        sys.exit(2)
    except llm.LLMError as e:
        print(f"[OpenAI 오류] {e}\n이미 처리된 기사는 보존되며 재실행하면 이어서 처리합니다.")
        sys.exit(3)
    failed = [n for n, r in res["sources"].items() if not r["ok"]]
    print(f"수집 {res['fetched']}건 / 신규 저장 {res['new']}건 / 실패 소스 {len(failed)}개")
    for n in failed:
        print(f"  - {n}: {res['sources'][n]['error']}")
    if "enrich" in res:
        en = res["enrich"]
        print(f"분류/요약: 대기 {en['pending']} / 완료 {en['done']} / 실패 {en['failed']} / 보안 무관으로 제외 {en['rejected']}")
        for src, n in en["rejected_by_source"].items():
            print(f"  - 무관 제외: {src} {n}건")
        print(f"임베딩: {res['embed']}")
