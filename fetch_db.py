"""
Drop-in replacement for fetch_sheets.fetch_all_tabs().

Reads completed experiment data from Supabase and returns a DataFrame with the
same column names and _meta_* structure that the rest of the app expects.
"""

import logging
import pandas as pd

from supabase_utils import (
    get_client,
    SQL_TO_DF_COL,
    RUNS_META_MAP,
    ALL_MEASUREMENT_SQL_COLS,
)

logger = logging.getLogger(__name__)


def fetch_all_tabs() -> pd.DataFrame:
    """
    Fetch all completed runs from Supabase and return a combined DataFrame.

    Column names match the output of fetch_sheets.fetch_all_tabs() so the rest
    of the application (analyze.py, app.py) works without modification.
    """
    client = get_client()

    def _fetch_all(table: str, cols: str, page_size: int = 1000) -> list[dict]:
        """Paginate through all rows in a table."""
        rows, offset = [], 0
        while True:
            batch = (
                client.table(table)
                .select(cols)
                .range(offset, offset + page_size - 1)
                .execute()
                .data
            )
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return rows

    # ── Fetch runs ────────────────────────────────────────────────────────────
    runs_data = _fetch_all("runs", "*")
    if not runs_data:
        raise RuntimeError("No runs found in the Supabase database.")
    runs_df = pd.DataFrame(runs_data)

    # ── Fetch measurements ────────────────────────────────────────────────────
    meas_cols = ", ".join(["id", "run_id", "row_order", "extra_data"] + ALL_MEASUREMENT_SQL_COLS)
    meas_data = _fetch_all("measurements", meas_cols)
    if not meas_data:
        raise RuntimeError("No measurements found in the Supabase database.")
    meas_df = pd.DataFrame(meas_data)

    # ── Join ──────────────────────────────────────────────────────────────────
    merged = meas_df.merge(runs_df, left_on="run_id", right_on="id", suffixes=("", "_run"))

    # ── Rename SQL measurement columns back to original DataFrame names ───────
    rename_meas = {sql: df_col for sql, df_col in SQL_TO_DF_COL.items() if sql in merged.columns}
    merged = merged.rename(columns=rename_meas)

    # ── Rename runs columns to _meta_* names ─────────────────────────────────
    merged = merged.rename(columns=RUNS_META_MAP)

    # ── Reconstruct _run_id as a human-readable tab_name + sequential index ──
    # Assign a stable per-tab run number (ordered by migrated_at then source_key).
    runs_df_sorted = runs_df.sort_values(["tab_name", "migrated_at", "source_key"])
    uuid_to_run_id: dict[str, str]   = {}
    uuid_to_run_idx: dict[str, int]  = {}
    for tab, grp in runs_df_sorted.groupby("tab_name", sort=False):
        for idx, (_, row) in enumerate(grp.iterrows()):
            uuid_to_run_id[row["id"]]  = f"{tab}_run{idx:02d}"
            uuid_to_run_idx[row["id"]] = idx

    merged["_run_id"]    = merged["run_id"].map(uuid_to_run_id)
    merged["_run_index"] = merged["run_id"].map(uuid_to_run_idx)

    # ── Sort by run then row order ────────────────────────────────────────────
    merged = merged.sort_values(["_run_id", "row_order"], ignore_index=True)

    # ── Preserve run UUID for in-app edits ────────────────────────────────────
    merged["_run_uuid"] = merged["run_id"]

    # ── Drop internal DB columns ──────────────────────────────────────────────
    drop = {"id", "id_run", "run_id", "source_key", "migrated_at", "row_order", "extra_data"}
    merged = merged.drop(columns=[c for c in drop if c in merged.columns])

    # ── Rebuild _datetime and _run_start_date (same logic as fetch_sheets) ───
    date_col = "Date " if "Date " in merged.columns else ("Date" if "Date" in merged.columns else None)
    time_col = "time"  if "time"  in merged.columns else None
    if date_col and time_col:
        combined_dt = (
            merged[date_col].astype(str).str.strip()
            + " "
            + merged[time_col].astype(str).str.strip()
        )
        merged["_datetime"] = pd.to_datetime(combined_dt, errors="coerce", dayfirst=True)
    elif date_col:
        merged["_datetime"] = pd.to_datetime(
            merged[date_col].astype(str).str.strip(), errors="coerce", dayfirst=True
        )
    if "_datetime" in merged.columns:
        merged["_run_start_date"] = merged.groupby("_run_id")["_datetime"].transform("min")

    logger.info(
        f"Loaded {len(merged)} rows across {merged['_run_id'].nunique()} runs "
        f"from {merged['_meta_tab_name'].nunique()} tabs."
    )
    return merged
