"""
Backfill cabinet sensor stats for all existing Supabase runs.

Scans a directory of XLSM cabinet exports and matches each file to a run
in Supabase by the Serial field in the file's Settings sheet.  Runs that
already have cabinet stats are skipped unless --force is used.

Usage:
    python backfill_cabinet.py --cabinet-dir cabinet_exports/
    python backfill_cabinet.py --cabinet-dir cabinet_exports/ --dry-run
    python backfill_cabinet.py --cabinet-dir cabinet_exports/ --force
    python backfill_cabinet.py --cabinet-dir cabinet_exports/ --serial r0054
"""

import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill cabinet sensor stats for existing Supabase runs."
    )
    parser.add_argument("--cabinet-dir", required=True, metavar="DIR",
                        help="Directory containing cabinet XLSM exports.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would happen without writing to Supabase.")
    parser.add_argument("--force",    action="store_true",
                        help="Re-process runs that already have cabinet stats.")
    parser.add_argument("--serial",   metavar="SERIAL", default=None,
                        help="Only process this run serial (e.g. r0054).")
    args = parser.parse_args()

    from fetch_cabinet import find_cabinet_files, read_cabinet_xlsm, aggregate_run_stats
    from supabase_utils import get_client, upsert_cabinet_stats, fetch_cabinet_stats

    client = get_client()

    # ── Fetch all runs ─────────────────────────────────────────────────────────
    runs = client.table("runs").select("id, stack_id, tab_name, date_start").execute().data
    if not runs:
        logger.error("No runs found in Supabase.")
        sys.exit(1)
    logger.info(f"Found {len(runs)} runs in Supabase")

    # ── Find which runs already have cabinet stats ─────────────────────────────
    if not args.force:
        existing_df = fetch_cabinet_stats(client)
        already_done: set[str] = set(existing_df["run_id"].tolist()) if not existing_df.empty else set()
        logger.info(f"{len(already_done)} runs already have cabinet stats (use --force to re-process)")
    else:
        already_done = set()

    # ── Filter by serial if requested ─────────────────────────────────────────
    if args.serial:
        runs = [r for r in runs if (r.get("stack_id") or r.get("tab_name", "")) == args.serial]
        if not runs:
            logger.error(f"No run found with serial {args.serial!r}")
            sys.exit(1)

    # ── Process ────────────────────────────────────────────────────────────────
    done = skipped = no_file = failed = 0

    for run in runs:
        run_uuid = run["id"]
        serial   = (run.get("stack_id") or run.get("tab_name") or "").strip()

        if not serial:
            logger.debug(f"Run {run_uuid} has no serial, skipping")
            no_file += 1
            continue

        if run_uuid in already_done:
            skipped += 1
            continue

        files = find_cabinet_files(args.cabinet_dir, serial)
        if not files:
            logger.debug(f"No cabinet file for serial {serial!r}")
            no_file += 1
            continue

        xlsm_path = files[-1]  # most data points

        try:
            cab_serial, start_dt, df_cab = read_cabinet_xlsm(xlsm_path)

            run_start_raw = run.get("date_start")
            if run_start_raw:
                run_start = pd.to_datetime(run_start_raw, dayfirst=True, errors="coerce")
                if pd.isna(run_start):
                    run_start = start_dt
            else:
                run_start = start_dt

            stats = aggregate_run_stats(df_cab, start_dt=run_start)
            if not stats:
                logger.warning(f"No stats computed for {serial} ({xlsm_path.name})")
                no_file += 1
                continue

            window_start = df_cab["Time"].min()
            window_end   = df_cab["Time"].max()
            n_pts        = len(df_cab)

            if args.dry_run:
                print(f"  [dry] {serial}  {n_pts} pts  →  {len(stats)} stats  ({xlsm_path.name})")
            else:
                upsert_cabinet_stats(
                    client, run_uuid, cab_serial, stats,
                    window_start, window_end, n_pts,
                )
                print(f"  OK  {serial}  {n_pts} pts  →  {len(stats)} stats")

            done += 1

        except Exception as exc:
            logger.error(f"  FAILED  {serial}: {exc}")
            failed += 1

    verb = "would be processed" if args.dry_run else "processed"
    print(
        f"\n{done} run(s) {verb}  |  "
        f"{skipped} already done  |  "
        f"{no_file} no matching file  |  "
        f"{failed} failed"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
