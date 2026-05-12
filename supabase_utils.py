"""
Shared Supabase connection and DataFrame <-> database column mapping utilities.
Used by fetch_db.py, migrate_historical.py, and migrate_run.py.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column mapping: original DataFrame column name → SQL column name ─────────

MEASUREMENT_COL_MAP: dict[str, str] = {
    "Time (hours)":                       "time_h",
    "Time (seconds)":                     "time_s",
    "Date ":                              "date_col",   # trailing space variant
    "Date":                               "date_col",
    "time":                               "time_of_day",
    "Efficiency (%)":                     "efficiency_pct",
    "Current (A)":                        "current_a",
    "Voltage (V)":                        "voltage_v",
    "Average V (V)":                      "avg_voltage_v",
    "Current density (mA/cm²)":      "current_density_ma_cm2",
    "HFR":                                "hfr",
    "H2O2 current (A)":                   "h2o2_current_a",
    "H2O2 current density (mA/cm²)": "h2o2_current_density_ma_cm2",
    "Strip 1":                            "strip_1",
    "Strip 2":                            "strip_2",
    "Peroxide in DI water":               "peroxide_in_di",
    "Gas (LPM)":                          "gas_lpm",
    "Water flow (mL/s)":                  "water_flow_ml_s",
    "Conductivity (µS/cm)":          "conductivity_us_cm",
    "Diff Pressure (mbar)":               "diff_pressure_mbar",
    "Anode flow (mL/s)":                  "anode_flow_ml_s",
    "STK temp An out":                    "stk_temp_an_out",
    "STK temp Ca out":                    "stk_temp_ca_out",
    "Throughput (g/h)":                   "throughput_g_h",
    "Avg. throughput (g/h)":              "avg_throughput_g_h",
}

# Reverse map for reading from the DB back into the DataFrame.
# Only the first mapping wins (so "Date " takes precedence over "Date").
SQL_TO_DF_COL: dict[str, str] = {}
for _df, _sql in MEASUREMENT_COL_MAP.items():
    if _sql not in SQL_TO_DF_COL:
        SQL_TO_DF_COL[_sql] = _df

# All distinct SQL measurement column names (insertion / SELECT order)
ALL_MEASUREMENT_SQL_COLS: list[str] = list(dict.fromkeys(MEASUREMENT_COL_MAP.values()))

# Mapping: runs table SQL column → _meta_ DataFrame column (and vice-versa)
RUNS_META_MAP: dict[str, str] = {
    "tab_name":        "_meta_tab_name",
    "station_id":      "_meta_station_id",
    "operator":        "_meta_operator",
    "stack_id":        "_meta_stack_id",
    "date_start":      "_meta_date_start",
    "project":         "_meta_project",
    "aim":             "_meta_aim",
    "cabinet":         "_meta_cabinet",
    "n_cells":         "_meta_n_cells",
    "cell_area_cm2":   "_meta_cell_area_cm2",
    "current_ma_cm2":  "_meta_current_mA_cm2",
    "gdl":             "_meta_gdl",
    "foam_grid":       "_meta_foam_grid",
    "operation_note":  "_meta_operation_note",
    "is_informal":     "_meta_informal",
    "notes":           "_meta_notes",
}


# ── Supabase client ───────────────────────────────────────────────────────────

def get_client():
    """Return a Supabase client, loading credentials from Streamlit secrets or env vars."""
    import os
    from supabase import create_client

    url = key = ""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "supabase" in st.secrets:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["service_key"]
    except Exception:
        pass

    if not url:
        url = os.environ.get("SUPABASE_URL", "")
    if not key:
        key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found.\n"
            "  Local: set SUPABASE_URL and SUPABASE_KEY environment variables.\n"
            "  Streamlit Cloud: add [supabase] url / service_key to secrets.toml."
        )
    return create_client(url, key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_py(val):
    """Convert numpy scalars / NaN to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if np.isnan(v) else v
    if isinstance(val, float) and not np.isfinite(val):
        return None
    return val


# SQL columns that hold text (all others are numeric)
_TEXT_SQL_COLS = {"date_col", "time_of_day"}


def _to_db_numeric(val):
    """Return a float for numeric DB columns, or None if the value can't be parsed.

    Some cells in the source sheet contain comma-separated strings like
    "1.96, 1.91, 1.95" (multiple readings merged into one cell). PostgreSQL
    rejects those for numeric columns, so we drop them rather than fail.
    """
    v = _to_py(val)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def make_source_key(
    tab_name: str,
    stack_id: str | None,
    date_start: str | None,
    is_informal: bool,
    run_id_hint: str | None = None,
) -> str:
    """Stable deduplication key for a run — used as the UNIQUE constraint in `runs`.

    For formal runs with a stack_id and date_start the natural key is used.
    For any run without a date_start there is no reliable natural key, so the
    DataFrame _run_id (run_id_hint) is used as a stable positional fallback so
    that multiple informal runs on the same tab don't collide.
    """
    prefix = "informal" if is_informal else "formal"
    if not is_informal and stack_id and date_start:
        return f"formal::{stack_id}::{date_start}"
    if date_start:
        return f"{prefix}::{tab_name}::{date_start}"
    # No date recorded — use positional fallback to avoid collisions
    hint = run_id_hint or "unknown"
    return f"{prefix}::{hint}"


def meta_to_runs_row(df_group: pd.DataFrame, tab_name: str, run_id_hint: str | None = None) -> dict:
    """Build a `runs` table row dict from a per-run DataFrame group."""

    def get(col: str) -> str | None:
        full = f"_meta_{col}"
        if full not in df_group.columns:
            return None
        val = df_group[full].iloc[0]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip()
        return s if s else None

    def get_num(col: str) -> float | None:
        v = get(col)
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    stack_id    = get("stack_id")
    date_start  = get("date_start")
    is_informal = bool(df_group["_meta_informal"].iloc[0]) if "_meta_informal" in df_group.columns else False

    n = get_num("n_cells")
    return {
        "source_key":      make_source_key(tab_name, stack_id, date_start, is_informal, run_id_hint),
        "tab_name":        tab_name,
        "station_id":      get("station_id") or tab_name,
        "operator":        get("operator"),
        "stack_id":        stack_id,
        "date_start":      date_start,
        "project":         get("project"),
        "aim":             get("aim"),
        "cabinet":         get("cabinet"),
        "n_cells":         int(n) if n is not None else None,
        "cell_area_cm2":   get_num("cell_area_cm2"),
        "current_ma_cm2":  get_num("current_mA_cm2"),
        "gdl":             get("gdl"),
        "foam_grid":       get("foam_grid"),
        "operation_note":  get("operation_note"),
        "is_informal":     is_informal,
    }


def df_group_to_measurement_rows(df_group: pd.DataFrame, run_uuid: str) -> list[dict]:
    """Convert measurement rows for one run into dicts ready for `measurements` insert."""
    mapped_df_cols = set(MEASUREMENT_COL_MAP.keys())
    meta_cols      = {c for c in df_group.columns if c.startswith("_")}

    rows = []
    for row_order, (_, row) in enumerate(df_group.iterrows()):
        record: dict = {"run_id": run_uuid, "row_order": row_order}

        for df_col, sql_col in MEASUREMENT_COL_MAP.items():
            if df_col in df_group.columns and sql_col not in record:
                raw = row.get(df_col)
                if sql_col in _TEXT_SQL_COLS:
                    record[sql_col] = _to_py(raw)
                else:
                    record[sql_col] = _to_db_numeric(raw)

        extra: dict = {}
        for col in df_group.columns:
            if col not in meta_cols and col not in mapped_df_cols:
                v = _to_py(row.get(col))
                if v is not None:
                    extra[col] = v
        if extra:
            record["extra_data"] = extra

        rows.append(record)
    return rows


def insert_run(client, df_group: pd.DataFrame, tab_name: str, *, dry_run: bool = False) -> str | None:
    """
    Upsert one run (metadata + measurements) into Supabase.

    Uses a full-replace strategy for measurements: existing rows are deleted and
    reinserted so reruns of the migration script are idempotent.

    Returns the run UUID on success, None in dry-run mode.
    """
    run_id_hint = df_group["_run_id"].iloc[0] if "_run_id" in df_group.columns else None
    runs_row    = meta_to_runs_row(df_group, tab_name, run_id_hint=run_id_hint)
    source_key  = runs_row["source_key"]

    if dry_run:
        logger.info(f"[dry-run] Would upsert run: {source_key}  ({len(df_group)} rows)")
        return None

    result = (
        client.table("runs")
        .upsert(runs_row, on_conflict="source_key")
        .execute()
    )
    run_uuid = result.data[0]["id"]

    # Full replace: delete existing measurements then reinsert
    client.table("measurements").delete().eq("run_id", run_uuid).execute()

    measurement_rows = df_group_to_measurement_rows(df_group, run_uuid)
    chunk_size = 500
    for i in range(0, len(measurement_rows), chunk_size):
        client.table("measurements").insert(measurement_rows[i : i + chunk_size]).execute()

    logger.info(f"Upserted run {source_key}: {len(measurement_rows)} measurements")
    return run_uuid
