"""
Seed the dev Supabase project with a sample of runs from production.

Run this once after creating the dev project and running schema.sql on it:

    python seed_dev_db.py              # copy 15 most-recent runs
    python seed_dev_db.py --runs 30    # copy more
    python seed_dev_db.py --clear      # wipe dev DB first, then seed
    python seed_dev_db.py --clear --runs 5   # small clean slate for quick tests

Requirements:
  - SUPABASE_URL / SUPABASE_KEY must point at the DEV project (set in start_dev.bat
    or exported in your shell before running this script)
  - SUPABASE_PROD_URL / SUPABASE_PROD_KEY must point at the PROD project
    (either set as env vars or stored in .streamlit/secrets.toml)
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _prod_client():
    """Connect to the production Supabase using PROD-specific env vars or secrets.toml."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_PROD_URL", "")
    key = os.environ.get("SUPABASE_PROD_KEY", "")

    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "supabase" in st.secrets:
                url = url or st.secrets["supabase"]["url"]
                key = key or st.secrets["supabase"]["service_key"]
        except Exception:
            pass

    if not url or not key:
        logger.error(
            "Production credentials not found.\n"
            "Set SUPABASE_PROD_URL and SUPABASE_PROD_KEY env vars,\n"
            "or keep production credentials in .streamlit/secrets.toml."
        )
        sys.exit(1)

    return create_client(url, key)


def _dev_client():
    """Connect to the dev Supabase using the SUPABASE_URL / SUPABASE_KEY env vars."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        logger.error(
            "Dev credentials not found.\n"
            "Set SUPABASE_URL and SUPABASE_KEY env vars (see start_dev.bat)."
        )
        sys.exit(1)

    if "localhost" not in url and os.environ.get("HPNOW_ENV") != "dev":
        logger.warning(
            "SUPABASE_URL does not look like a dev project and HPNOW_ENV != 'dev'.\n"
            "Are you sure you want to write here? Press Ctrl+C to abort, or Enter to continue."
        )
        input()

    return create_client(url, key)


def main():
    parser = argparse.ArgumentParser(
        description="Seed the dev Supabase with a sample of prod runs."
    )
    parser.add_argument("--runs",  type=int, default=15,
                        help="Number of most-recent runs to copy (default: 15).")
    parser.add_argument("--clear", action="store_true",
                        help="Delete all existing data in the dev DB before seeding.")
    args = parser.parse_args()

    prod = _prod_client()
    dev  = _dev_client()

    # ── Optionally wipe dev ────────────────────────────────────────────────────
    if args.clear:
        logger.info("Clearing dev DB (measurements → runs → cabinet_stats)…")
        dev.table("measurements").delete().neq("id", 0).execute()
        try:
            dev.table("cabinet_stats").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        except Exception:
            pass
        dev.table("runs").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("Dev DB cleared.")

    # ── Fetch N most-recent prod runs ──────────────────────────────────────────
    logger.info(f"Fetching {args.runs} most-recent runs from prod…")
    runs_data = (
        prod.table("runs")
        .select("*")
        .order("migrated_at", desc=True)
        .limit(args.runs)
        .execute()
        .data
    )
    if not runs_data:
        logger.error("No runs found in prod.")
        sys.exit(1)
    logger.info(f"Fetched {len(runs_data)} runs.")

    # ── Upsert runs into dev ───────────────────────────────────────────────────
    for run in runs_data:
        dev.table("runs").upsert(run, on_conflict="source_key").execute()

    run_uuids = [r["id"] for r in runs_data]
    logger.info(f"Upserted {len(runs_data)} runs into dev.")

    # ── Fetch and upsert measurements in chunks ────────────────────────────────
    logger.info("Fetching measurements…")
    chunk, offset = 1000, 0
    total_meas = 0
    while True:
        batch = (
            prod.table("measurements")
            .select("*")
            .in_("run_id", run_uuids)
            .range(offset, offset + chunk - 1)
            .execute()
            .data
        )
        if not batch:
            break
        # Remove auto-generated ids so dev assigns its own bigserial ids
        for row in batch:
            row.pop("id", None)
        dev.table("measurements").insert(batch).execute()
        total_meas += len(batch)
        offset += chunk
        if len(batch) < chunk:
            break
    logger.info(f"Copied {total_meas} measurements.")

    # ── Copy cabinet_stats if they exist ──────────────────────────────────────
    try:
        cab_data = (
            prod.table("cabinet_stats")
            .select("*")
            .in_("run_id", run_uuids)
            .execute()
            .data
        )
        if cab_data:
            for row in cab_data:
                row.pop("id", None)
                dev.table("cabinet_stats").upsert(row, on_conflict="run_id").execute()
            logger.info(f"Copied {len(cab_data)} cabinet_stats rows.")
    except Exception as exc:
        logger.warning(f"Could not copy cabinet_stats (table may not exist yet): {exc}")

    print(
        f"\nDone.  {len(runs_data)} runs · {total_meas} measurements "
        f"copied to dev Supabase.\n"
        f"Start the dev app with:  start_dev.bat"
    )


if __name__ == "__main__":
    main()
