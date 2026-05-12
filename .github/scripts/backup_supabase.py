"""
Export runs and measurements tables from Supabase to CSV.
Reads SUPABASE_URL and SUPABASE_KEY from environment variables.
Run by the weekly GitHub Actions backup workflow.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from supabase import create_client

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_KEY", "")

if not url or not key:
    print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set.")
    sys.exit(1)

client = create_client(url, key)


def fetch_all(table: str, page_size: int = 1000) -> list[dict]:
    rows, offset = [], 0
    while True:
        batch = (
            client.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        rows.extend(batch)
        print(f"  {table}: fetched {len(rows)} rows so far…")
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


out = Path("backups")
out.mkdir(exist_ok=True)

print("── Runs ─────────────────────────────────────")
runs = fetch_all("runs")
pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
print(f"  Saved {len(runs)} rows → backups/runs.csv")

print("── Measurements ─────────────────────────────")
meas = fetch_all("measurements")
pd.DataFrame(meas).to_csv(out / "measurements.csv", index=False)
print(f"  Saved {len(meas)} rows → backups/measurements.csv")

print("Done.")
