"""
Backfill missing stack_ids for historical runs in Supabase.

Runs in Supabase that have a date_start but no stack_id get a new source_key
like  formal::<tab_name>::<date_start>  instead of the natural
formal::<stack_id>::<date_start>  key.  This script re-reads the historic
Google Sheet (using the fixed _extract_run_metadata parser) and updates each
matching Supabase row with the correct stack_id and source_key.

Usage:
    python backfill_stack_ids.py            # interactive confirm before writing
    python backfill_stack_ids.py --yes      # skip confirmation
    python backfill_stack_ids.py --dry-run  # preview only
"""

import argparse
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes",     action="store_true")
    args = parser.parse_args()

    import config
    import gspread
    import fetch_sheets
    from supabase_utils import get_client, make_source_key

    logger.info("Connecting to Supabase...")
    sb = get_client()

    result = sb.table("runs").select("id,tab_name,date_start,source_key,stack_id").execute()
    needs_backfill = [
        r for r in result.data
        if not r.get("stack_id") and r.get("date_start")
    ]
    logger.info(f"{len(needs_backfill)} Supabase runs have null stack_id with a date_start.")

    if not needs_backfill:
        logger.info("Nothing to do.")
        return

    # Build lookup: (tab_name, date_start) -> supabase row
    sb_lookup: dict[tuple, dict] = {}
    for r in needs_backfill:
        key = (r["tab_name"], r["date_start"])
        sb_lookup[key] = r

    logger.info("Parsing historic Google Sheet for stack_ids...")
    creds  = fetch_sheets._get_credentials()
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(config.SHEET_ID)

    updates: list[dict] = []
    for ws in sheet.worksheets():
        if ws.title in config.SKIP_TABS:
            continue
        run_dfs = fetch_sheets._parse_tab(ws)
        for df in run_dfs:
            if "_meta_informal" in df.columns and df["_meta_informal"].any():
                continue
            stack_id   = df["_meta_stack_id"].iloc[0]  if "_meta_stack_id"  in df.columns else None
            date_start = df["_meta_date_start"].iloc[0] if "_meta_date_start" in df.columns else None
            tab_name   = df["_meta_tab_name"].iloc[0]   if "_meta_tab_name"   in df.columns else None

            if not stack_id or not date_start or not tab_name:
                continue
            stack_id   = str(stack_id).strip()   or None
            date_start = str(date_start).strip() or None
            if not stack_id or not date_start:
                continue

            key = (tab_name, date_start)
            if key not in sb_lookup:
                continue

            sb_row = sb_lookup[key]
            new_sk = make_source_key(tab_name, stack_id, date_start, is_informal=False)
            updates.append({
                "id":             sb_row["id"],
                "stack_id":       stack_id,
                "source_key":     new_sk,
                "old_source_key": sb_row["source_key"],
            })
            # Remove from lookup so we don't match twice
            del sb_lookup[key]

    if not updates:
        logger.info("No matches found in the historic sheet — nothing to update.")
        return

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Found {len(updates)} runs to update:\n")
    for u in updates:
        print(f"  {u['old_source_key']}")
        print(f"    → stack_id={u['stack_id']}  source_key={u['source_key']}")

    unmatched = len(needs_backfill) - len(updates)
    if unmatched:
        print(f"\n  ({unmatched} runs with null stack_id had no match in the historic sheet "
              f"— they may be informal, on a skipped tab, or have no date recorded.)")

    if args.dry_run:
        return

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            logger.info("Aborted.")
            return

    succeeded = failed = 0
    for u in updates:
        try:
            sb.table("runs").update({
                "stack_id":   u["stack_id"],
                "source_key": u["source_key"],
            }).eq("id", u["id"]).execute()
            logger.info(f"  Updated {u['old_source_key']} → {u['source_key']}")
            succeeded += 1
        except Exception as exc:
            logger.error(f"  FAILED {u['old_source_key']}: {exc}")
            failed += 1

    print(f"\n{succeeded} runs updated" + (f", {failed} failed." if failed else "."))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
