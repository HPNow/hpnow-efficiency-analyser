"""
Migrate a completed experiment from the live Google Sheet into Supabase.

Run this when an experiment finishes and you are ready to free up its slot
in the live sheet for a new experiment.

Usage:
    python migrate_run.py                    # migrate all new runs in the live sheet
    python migrate_run.py --tab r0066        # migrate only a specific test station tab
    python migrate_run.py --dry-run          # preview without writing
    python migrate_run.py --tab r0066 --yes  # skip the confirmation prompt

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
    from google.oauth2.service_account import Credentials
    from fetch_sheets import _parse_tab, _coerce_numerics, _fix_time_hours
    from fetch_sheets import _trim_low_start_efficiency, _parse_datetime
    from fetch_sheets import _merge_duplicate_columns, _clean_data

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file(config.CREDENTIALS_FILE, scopes=SCOPES)
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
    combined = _merge_duplicate_columns(combined)
    combined = _clean_data(combined)
    return combined


def _existing_source_keys(client) -> set[str]:
    """Return all source_keys already in the Supabase `runs` table."""
    result = client.table("runs").select("source_key").execute()
    return {row["source_key"] for row in result.data}


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a completed run from the live Google Sheet into Supabase."
    )
    parser.add_argument("--tab",     metavar="TAB_NAME", help="Only migrate runs from this sheet tab.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    parser.add_argument("--yes",     action="store_true", help="Skip confirmation prompt.")
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
            insert_run(sb_client, group, tab_name, dry_run=args.dry_run)
            succeeded += 1
            if not args.dry_run:
                print(f"  OK  {sk}")
        except Exception as exc:
            logger.error(f"  FAILED  {sk}: {exc}")
            failed += 1

    verb = "would be migrated" if args.dry_run else "migrated"
    print(f"\n{succeeded} run(s) {verb}" + (f", {failed} failed." if failed else "."))

    if not args.dry_run and succeeded:
        print("\nYou can now clear the migrated slot(s) in the live sheet for the next experiment.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
