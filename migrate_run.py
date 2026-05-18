"""
Migrate a completed experiment from the live Google Sheet into Supabase.

Run this when an experiment finishes and you are ready to free up its slot
in the live sheet for a new experiment.

Usage:
    python migrate_run.py                           # migrate all new runs
    python migrate_run.py --tab r0066               # migrate only one tab
    python migrate_run.py --dry-run                 # preview without writing
    python migrate_run.py --tab r0066 --yes         # skip confirmation prompt

    # Also ingest cabinet sensor stats from XLSM exports:
    python migrate_run.py --cabinet-dir cabinet_exports/
    python migrate_run.py --tab r0066 --cabinet-dir cabinet_exports/ --yes

Cabinet XLSM files are matched to runs by the Serial field in their Settings
sheet (e.g. Serial = r0066).  Place all XLSM exports in the cabinet directory
before running.  If no matching file is found the run is still migrated; the
cabinet stats can be added later with backfill_cabinet.py.

After a successful migration the script prints the source_key of each inserted
run so you can verify it appeared in Supabase before clearing the live sheet.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _fetch_live_sheet(live_sheet_id: str):
    """Read the live Google Sheet and return a combined DataFrame."""
    import warnings
    warnings.filterwarnings("ignore")

    import config
    import gspread
    from fetch_sheets import _parse_tab, _coerce_numerics, _fix_time_hours
    from fetch_sheets import _trim_low_start_efficiency, _parse_datetime
    from fetch_sheets import _derive_time_from_datetime, _merge_duplicate_columns, _clean_data

    from fetch_sheets import _get_credentials
    creds  = _get_credentials()
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(live_sheet_id)

    import pandas as pd
    all_dfs = []
    for ws in sheet.worksheets():
        if ws.title in config.SKIP_TABS:
            continue
        run_dfs = _parse_tab(ws)
        all_dfs.extend(run_dfs)

    if not all_dfs:
        raise RuntimeError("No valid runs found in the live sheet.")

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    combined = _coerce_numerics(combined)
    combined = _fix_time_hours(combined)
    combined = _trim_low_start_efficiency(combined)
    combined = _parse_datetime(combined)
    combined = _derive_time_from_datetime(combined)
    combined = _merge_duplicate_columns(combined)
    combined = _clean_data(combined)
    return combined


def _existing_source_keys(client) -> set[str]:
    """Return all source_keys already in the Supabase `runs` table."""
    result = client.table("runs").select("source_key").execute()
    return {row["source_key"] for row in result.data}


def _ingest_cabinet_stats(sb_client, run_uuid: str, group, source_key: str, cabinet_dir: str) -> None:
    """Find the matching cabinet XLSM and upsert aggregated stats for this run."""
    import pandas as pd
    from fetch_cabinet import find_cabinet_files, read_cabinet_xlsm, aggregate_run_stats
    from supabase_utils import upsert_cabinet_stats

    import numpy as np

    def _get_meta(col: str):
        full = f"_meta_{col}"
        if full in group.columns:
            v = group[full].iloc[0]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return str(v).strip() or None
        return None

    serial = _get_meta("stack_id") or _get_meta("tab_name") or ""
    if not serial:
        logger.debug(f"No serial for {source_key}, skipping cabinet stats")
        return

    files = find_cabinet_files(cabinet_dir, serial)
    if not files:
        logger.debug(f"No cabinet file found for serial {serial!r}")
        return

    # Use the file with the most data points (last after sort by density)
    xlsm_path = files[-1]
    try:
        cab_serial, start_dt, df_cab = read_cabinet_xlsm(xlsm_path)

        # Use run date_start as window anchor if available, else file's StartTime
        run_start_raw = _get_meta("date_start")
        if run_start_raw:
            run_start = pd.to_datetime(run_start_raw, dayfirst=True, errors="coerce")
        else:
            run_start = start_dt

        stats = aggregate_run_stats(df_cab, start_dt=run_start)
        if not stats:
            logger.warning(f"No cabinet stats computed for {source_key}")
            return

        window_start = df_cab["Time"].min()
        window_end   = df_cab["Time"].max()

        upsert_cabinet_stats(
            sb_client, run_uuid, cab_serial, stats,
            window_start, window_end, len(df_cab),
        )
        print(f"  CAB  {source_key}: {len(df_cab)} sensor pts → {len(stats)} stats")

    except Exception as exc:
        logger.warning(f"Cabinet stats failed for {source_key}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a completed run from the live Google Sheet into Supabase."
    )
    parser.add_argument("--tab",     metavar="TAB_NAME", help="Only migrate runs from this sheet tab.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    parser.add_argument("--yes",     action="store_true", help="Skip confirmation prompt.")
    parser.add_argument(
        "--cabinet-dir",
        metavar="DIR",
        default=None,
        help="Directory containing cabinet XLSM exports. If provided, per-run sensor "
             "stats are aggregated and stored in Supabase alongside the run.",
    )
    args = parser.parse_args()

    import config
    live_sheet_id = getattr(config, "LIVE_SHEET_ID", "")
    if not live_sheet_id:
        logger.error(
            "LIVE_SHEET_ID is not set in config.py.\n"
            "Add:  LIVE_SHEET_ID = '<your-live-sheet-id>'"
        )
        sys.exit(1)

    logger.info("Fetching data from the live Google Sheet...")
    df = _fetch_live_sheet(live_sheet_id)

    if args.tab:
        df = df[df["_meta_tab_name"] == args.tab]
        if df.empty:
            logger.error(f"No runs found in tab '{args.tab}'. Available tabs: "
                         + ", ".join(df["_meta_tab_name"].unique().tolist()))
            sys.exit(1)

    from supabase_utils import get_client, insert_run, make_source_key

    if not args.dry_run:
        sb_client = get_client()
        already   = _existing_source_keys(sb_client)
    else:
        sb_client = None
        already   = set()

    # Identify which runs are new (not yet in Supabase)
    new_groups = []
    for run_id in df["_run_id"].unique():
        group    = df[df["_run_id"] == run_id]
        tab_name = group["_meta_tab_name"].iloc[0]
        is_inf   = bool(group["_meta_informal"].iloc[0]) if "_meta_informal" in group.columns else False

        def _get(col):
            full = f"_meta_{col}"
            if full in group.columns:
                v = group[full].iloc[0]
                import numpy as np
                return None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip() or None
            return None

        sk = make_source_key(tab_name, _get("stack_id"), _get("date_start"), is_inf, run_id_hint=run_id)
        if sk not in already:
            new_groups.append((group, tab_name, sk))

    if not new_groups:
        logger.info("No new runs found — everything is already in Supabase.")
        return

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Found {len(new_groups)} run(s) to migrate:\n")
    for _, tab, sk in new_groups:
        n_rows = len(_)
        print(f"  [{tab}]  {sk}  ({n_rows} measurements)")

    if not args.dry_run and not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            logger.info("Aborted.")
            return

    succeeded = failed = 0
    for group, tab_name, sk in new_groups:
        try:
            run_uuid = insert_run(sb_client, group, tab_name, dry_run=args.dry_run)
            succeeded += 1
            if not args.dry_run:
                print(f"  OK  {sk}")
        except Exception as exc:
            logger.error(f"  FAILED  {sk}: {exc}")
            failed += 1
            continue

        # ── Cabinet stats ──────────────────────────────────────────────────────
        if args.cabinet_dir and not args.dry_run and run_uuid:
            _ingest_cabinet_stats(sb_client, run_uuid, group, sk, args.cabinet_dir)

    verb = "would be migrated" if args.dry_run else "migrated"
    print(f"\n{succeeded} run(s) {verb}" + (f", {failed} failed." if failed else "."))

    if not args.dry_run and succeeded:
        print("\nYou can now clear the migrated slot(s) in the live sheet for the next experiment.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
