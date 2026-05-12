"""
One-time migration: read every run from the storage Google Sheet and insert into Supabase.

Usage:
    python migrate_historical.py            # migrate everything
    python migrate_historical.py --dry-run  # preview without writing

The script is idempotent — re-running it will upsert existing runs without
creating duplicates (deduplication is keyed on source_key in the `runs` table).
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Migrate the storage Google Sheet into Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to the database.")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    logger.info("Fetching data from the storage Google Sheet...")
    from fetch_sheets import fetch_all_tabs
    df = fetch_all_tabs()

    run_ids = df["_run_id"].unique()
    logger.info(f"Found {len(run_ids)} runs across {df['_meta_tab_name'].nunique()} tabs.")

    if not args.dry_run:
        from supabase_utils import get_client, insert_run
        client = get_client()
        logger.info("Connected to Supabase.")
    else:
        logger.info("DRY RUN — no data will be written.")
        client = None

    from supabase_utils import insert_run

    succeeded = 0
    failed    = 0

    for run_id in run_ids:
        group    = df[df["_run_id"] == run_id]
        tab_name = group["_meta_tab_name"].iloc[0]
        try:
            insert_run(client, group, tab_name, dry_run=args.dry_run)
            succeeded += 1
        except Exception as exc:
            logger.error(f"Failed to insert run {run_id}: {exc}")
            failed += 1

    logger.info(
        f"\nDone. {succeeded} runs {'would be ' if args.dry_run else ''}migrated"
        + (f", {failed} failed." if failed else ".")
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
