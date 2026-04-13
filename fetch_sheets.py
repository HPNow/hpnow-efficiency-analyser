"""
Fetch and normalise all station tabs from the HPNow Google Sheet.

True tab structure — each tab contains N runs, each run has:
  1. A metadata block (starts with "Initials" row, ~7 rows)      ← formal run
     OR a single "New experiment" marker row                      ← informal run
  2. Some blank / summary rows
  3. A fresh column header row (starts with "Time (hours)")
  4. Data rows until the next metadata block or end of sheet

Informal runs (no Initials block) inherit physical parameters
(current density, GDL, cell area, project, …) from the last formal
metadata block and are flagged with _meta_informal = True.

Returns a single combined DataFrame with one row per measurement,
with per-run metadata columns prefixed with _meta_ and _run_.
"""

import re
import logging
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Regex to detect informal run-boundary rows ("New experiment", "New stack", …)
INFORMAL_BOUNDARY = re.compile(r"^new\s+(experiment|stack|run)\b", re.IGNORECASE)

# Physical parameters that informal runs can inherit from the last formal block
INHERITABLE_META = (
    "current_mA_cm2", "gdl", "cell_area_cm2", "n_cells",
    "foam_grid", "project", "cabinet", "operation_note",
)

# Row labels to skip when collecting data rows (summaries / sub-headers)
SKIP_LABELS = frozenset({
    "Time (hours)", "Last measured hour", "Last measured efficiency",
    "Latest comment", "Gap", "Date start", "Cell area",
    "Current ", "Current", "Project", "Cabinet",
})


def _get_credentials():
    # On Streamlit Community Cloud, credentials are stored in st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            return Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]), scopes=SCOPES
            )
    except Exception:
        pass
    # Local development: fall back to the service account JSON file
    return Credentials.from_service_account_file(
        config.CREDENTIALS_FILE, scopes=SCOPES
    )


def _extract_run_metadata(block_rows):
    """
    Extract metadata from a run's header block (the ~7 orange rows).
    block_rows: list of raw row lists starting from the 'Initials' row.
    """
    meta = {}
    for row in block_rows:
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        if label == "Initials":
            meta["operator"] = row[2].strip() if len(row) > 2 else None
            # Stack ID label is at col3, value at col5
            meta["stack_id"] = (
                row[5].strip() if len(row) > 5 and row[5].strip() else None
            )
        elif label == "Date start":
            meta["date_start"] = row[2].strip() if len(row) > 2 else None
            meta["aim"]        = row[5].strip() if len(row) > 5 else None
        elif label in ("Cell area",):
            meta["cell_area_cm2"] = row[2].strip() if len(row) > 2 else None
            meta["n_cells"]       = row[5].strip() if len(row) > 5 else None
        elif label.strip() in ("Current ", "Current"):
            meta["current_mA_cm2"] = row[2].strip() if len(row) > 2 else None
            meta["gdl"]            = row[5].strip() if len(row) > 5 else None
        elif label == "Project":
            meta["project"]   = row[2].strip() if len(row) > 2 else None
            meta["foam_grid"] = row[5].strip() if len(row) > 5 else None
        elif label == "Cabinet":
            meta["cabinet"]        = row[2].strip() if len(row) > 2 else None
            meta["operation_note"] = row[5].strip() if len(row) > 5 else None
    return meta


def _dedup_headers(headers):
    """Deduplicate column names."""
    seen = {}
    result = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            result.append(h)
    return result


def _collect_run_data(raw, data_start, headers, meta, tab_name, run_index):
    """
    Collect data rows from data_start until the next formal or informal boundary.
    Returns (df_or_None, next_i).
    """
    data_rows = []
    m = data_start
    while m < len(raw):
        r = raw[m]
        if not r:
            m += 1
            continue
        first = r[0].strip()
        if first == "Initials":
            break
        if INFORMAL_BOUNDARY.match(first):
            break
        if first in SKIP_LABELS:
            m += 1
            continue
        data_rows.append(r)
        m += 1

    if len(data_rows) < config.MIN_ROWS:
        logger.debug(
            f"  {tab_name} run{run_index:02d} ({'informal' if meta.get('informal') else 'formal'}): "
            f"only {len(data_rows)} data rows, skipping."
        )
        return None, m

    padded = [r + [""] * max(0, len(headers) - len(r)) for r in data_rows]
    df = pd.DataFrame(padded, columns=headers)
    df = df.replace("", np.nan).dropna(how="all")

    df["_run_index"]      = run_index
    df["_run_id"]         = f"{tab_name}_run{run_index:02d}"
    df["_meta_tab_name"]  = tab_name
    for key, val in meta.items():
        df[f"_meta_{key}"] = val if val else None

    logger.debug(
        f"  {tab_name} run{run_index:02d} ({'informal' if meta.get('informal') else 'formal'}): "
        f"{len(df)} rows, project={meta.get('project')}"
    )
    return df, m


def _parse_tab(worksheet):
    """
    Parse a single worksheet into a list of per-run DataFrames.

    Handles two kinds of run boundaries:
      - Formal:   "Initials" row starting a metadata block
      - Informal: "New experiment" / "New stack" single-row marker
                  (inherits physical params from last formal block)
    """
    raw = worksheet.get_all_values()
    tab_name = worksheet.title
    run_dfs = []
    run_index = 0

    last_formal_meta = {}   # physical params from most recent Initials block
    last_headers     = None  # column headers from most recent Time(hours) row

    i = 0
    while i < len(raw):
        row = raw[i]
        if not row:
            i += 1
            continue
        first = row[0].strip()

        # ── Formal boundary: Initials block ──────────────────────────────────
        if first == "Initials":
            # Collect up to 12 rows of metadata
            block = []
            j = i
            while j < len(raw) and j < i + 12:
                if raw[j] and any(c.strip() for c in raw[j]):
                    block.append(raw[j])
                j += 1

            meta = _extract_run_metadata(block)
            meta["tab_name"]  = tab_name
            meta["station_id"] = meta.get("stack_id") or tab_name
            meta["informal"]  = False

            # Save inheritable params for any following informal runs
            for key in INHERITABLE_META:
                if meta.get(key):
                    last_formal_meta[key] = meta[key]

            # Find "Time (hours)" header from i+1 (may be inside the 12-row window)
            k = i + 1
            while k < len(raw):
                r0 = raw[k][0].strip() if raw[k] else ""
                if r0 == "Time (hours)":
                    break
                if r0 == "Initials" or INFORMAL_BOUNDARY.match(r0):
                    k = -1  # signal: no header before next boundary
                    break
                k += 1

            if k < 0 or k >= len(raw):
                i = j  # skip this block
                continue

            last_headers = _dedup_headers(raw[k])
            df, m = _collect_run_data(raw, k + 1, last_headers, meta, tab_name, run_index)
            if df is not None:
                run_dfs.append(df)
                run_index += 1
            i = m

        # ── Informal boundary: "New experiment" / "New stack" marker ─────────
        elif INFORMAL_BOUNDARY.match(first):
            if last_headers is None:
                # No formal run parsed yet — can't determine columns, skip
                i += 1
                continue

            # Inherit physical parameters from the last formal block
            meta = {key: last_formal_meta.get(key) for key in INHERITABLE_META}
            meta["tab_name"]  = tab_name
            meta["station_id"] = tab_name   # no stack ID for informal runs
            meta["stack_id"]  = None
            meta["informal"]  = True

            # Check if the very next non-empty rows have a "Time (hours)" header
            # (e.g. r0056 row 69). If so, update last_headers.
            k = i + 1
            found_new_header = False
            while k < min(i + 4, len(raw)):
                r0 = raw[k][0].strip() if raw[k] else ""
                if r0 == "Time (hours)":
                    last_headers = _dedup_headers(raw[k])
                    found_new_header = True
                    k += 1  # data starts after header
                    break
                if r0 == "Initials" or INFORMAL_BOUNDARY.match(r0):
                    break  # another boundary immediately — no data
                k += 1

            if not found_new_header:
                k = i + 1  # data starts right after the marker row

            df, m = _collect_run_data(raw, k, last_headers, meta, tab_name, run_index)
            if df is not None:
                run_dfs.append(df)
                run_index += 1
            i = m

        else:
            i += 1

    if not run_dfs:
        return _parse_tab_legacy(worksheet, raw)

    formal_count   = sum(1 for df in run_dfs if not df["_meta_informal"].any())
    informal_count = sum(1 for df in run_dfs if df["_meta_informal"].any())
    if informal_count:
        logger.info(
            f"Tab '{tab_name}': {len(run_dfs)} runs found "
            f"({formal_count} formal + {informal_count} informal)."
        )
    else:
        logger.info(f"Tab '{tab_name}': {len(run_dfs)} runs found.")
    return run_dfs


def _parse_tab_legacy(worksheet, raw):
    """
    Fallback for tabs without repeated metadata blocks (single-run or old format).
    """
    HEADER_ROW = 11
    DATA_START  = 12
    tab_name = worksheet.title

    if len(raw) <= HEADER_ROW:
        return []

    meta = {}
    for row in raw[:10]:
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        if label == "Initials":
            meta["operator"] = row[2].strip() if len(row) > 2 else None
            meta["stack_id"] = (
                row[5].strip() if len(row) > 5 and row[5].strip() else None
            )
        elif label == "Date start":
            meta["date_start"] = row[2].strip() if len(row) > 2 else None
        elif label == "Project":
            meta["project"] = row[2].strip() if len(row) > 2 else None
        elif label == "Cabinet":
            meta["cabinet"] = row[2].strip() if len(row) > 2 else None

    meta["tab_name"]   = tab_name
    meta["station_id"] = meta.get("stack_id") or tab_name
    meta["informal"]   = False

    headers   = _dedup_headers(raw[HEADER_ROW])
    data_rows = raw[DATA_START:]

    if len(data_rows) < config.MIN_ROWS:
        logger.warning(f"Tab '{tab_name}' (legacy): only {len(data_rows)} data rows, skipping.")
        return []

    padded = [r + [""] * max(0, len(headers) - len(r)) for r in data_rows]
    df = pd.DataFrame(padded, columns=headers)
    df = df.replace("", np.nan).dropna(how="all")
    df["_run_index"]     = 0
    df["_run_id"]        = f"{tab_name}_run00"
    df["_meta_tab_name"] = tab_name
    for key, val in meta.items():
        df[f"_meta_{key}"] = val if val else None

    logger.info(f"Tab '{tab_name}' (legacy): 1 run, {len(df)} rows.")
    return [df]


def _fix_time_hours(df):
    """Derive Time (hours) from Time (seconds) where hours looks wrong.

    MAX_PLAUSIBLE_HOURS is the longest run ever recorded at HPNow.
    Any value above this is treated as suspicious (likely stored in seconds).
    """
    MAX_PLAUSIBLE_HOURS = 12_000

    if "Time (hours)" not in df.columns:
        return df
    hours = pd.to_numeric(df["Time (hours)"], errors="coerce")

    if "Time (seconds)" in df.columns:
        # Preferred: derive from the dedicated seconds column
        seconds = pd.to_numeric(df["Time (seconds)"], errors="coerce")
        derived = seconds / 3600.0
        bad = hours.isna() | (hours < 0) | (hours > MAX_PLAUSIBLE_HOURS)
        hours = hours.where(~bad, derived)
    else:
        # Fallback: if value exceeds plausible range but dividing by 3600
        # gives a plausible result, the column was almost certainly recorded
        # in seconds rather than hours.
        suspicious = hours > MAX_PLAUSIBLE_HOURS
        as_hours = hours / 3600.0
        should_convert = suspicious & (as_hours <= MAX_PLAUSIBLE_HOURS)
        hours = hours.where(~should_convert, as_hours)

    # Hard clamp: any value still above the plausible limit after all
    # corrections is physically impossible — drop it rather than show it.
    hours = hours.where(hours <= MAX_PLAUSIBLE_HOURS)

    df["Time (hours)"] = hours
    return df


def _parse_datetime(df):
    """Build a _datetime column from Date + time columns."""
    date_col = "Date " if "Date " in df.columns else ("Date" if "Date" in df.columns else None)
    time_col = "time" if "time" in df.columns else None
    if date_col and time_col:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        df["_datetime"] = pd.to_datetime(combined, errors="coerce", dayfirst=True)
    elif date_col:
        df["_datetime"] = pd.to_datetime(
            df[date_col].astype(str).str.strip(), errors="coerce", dayfirst=True
        )
    if "_datetime" in df.columns and "_run_id" in df.columns:
        df["_run_start_date"] = df.groupby("_run_id")["_datetime"].transform("min")
    return df


def _coerce_numerics(df):
    """Convert data columns to numeric where ≥50% of values parse successfully."""
    meta_cols = {c for c in df.columns if c.startswith("_")}
    for col in df.columns:
        if col in meta_cols:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= df[col].notna().sum() * 0.5:
            df[col] = converted
    return df


def _trim_low_start_efficiency(df, min_start_eff: float = 10.0):
    """Drop leading rows of each run where Efficiency (%) < min_start_eff.

    Readings below 10 % at the very start of a run are measurement errors
    (sensor warm-up, priming artefacts, etc.).  Once efficiency has risen
    above the threshold for the first time, subsequent low readings are
    kept — they represent genuine degradation, not start-up noise.
    """
    if "Efficiency (%)" not in df.columns or "_run_id" not in df.columns:
        return df

    result = []
    for _, grp in df.groupby("_run_id", sort=False):
        if "Time (hours)" in grp.columns:
            grp = grp.sort_values("Time (hours)")
        eff = pd.to_numeric(grp["Efficiency (%)"], errors="coerce")
        above = eff >= min_start_eff
        if above.any():
            # Drop every row that comes before the first good reading
            first_good_pos = int(above.values.argmax())
            grp = grp.iloc[first_good_pos:]
        result.append(grp)

    return pd.concat(result, ignore_index=True) if result else df


def _merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce columns that represent the same measurement but use different
    Unicode characters or abbreviations in their header text.

    For each alias group the first listed name is treated as the canonical
    column name.  If the canonical column already exists it is filled from the
    alias; if it doesn't yet exist the alias is simply renamed.  The original
    alias column is dropped after merging.
    """
    COLUMN_ALIASES: dict[str, list[str]] = {
        # µ (U+00B5) vs "micro" written out
        "Conductivity (µS/cm)": ["Conductivity (micro S/cm)"],
    }
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias not in df.columns:
                continue
            if canonical in df.columns:
                # Fill NaN in the canonical column from the alias
                df[canonical] = df[canonical].combine_first(df[alias])
            else:
                df = df.rename(columns={alias: canonical})
            if alias in df.columns:
                df = df.drop(columns=[alias])
    return df


def _clean_data(df):
    """Filter physically impossible values."""
    if "Efficiency (%)" in df.columns:
        eff = pd.to_numeric(df["Efficiency (%)"], errors="coerce")
        df["Efficiency (%)"] = eff.where((eff >= 0) & (eff <= 100))
    if "Time (hours)" in df.columns:
        t = pd.to_numeric(df["Time (hours)"], errors="coerce")
        df["Time (hours)"] = t.where(t >= 0)
    return df


def fetch_all_tabs():
    """
    Main entry point. Returns a combined DataFrame of all runs across all tabs.
    """
    creds  = _get_credentials()
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(config.SHEET_ID)

    all_dfs    = []
    skipped    = []
    total_runs = 0

    for ws in sheet.worksheets():
        if ws.title in config.SKIP_TABS:
            continue
        logger.info(f"Fetching tab '{ws.title}' ...")
        run_dfs = _parse_tab(ws)
        if not run_dfs:
            skipped.append(ws.title)
            continue
        all_dfs.extend(run_dfs)
        total_runs += len(run_dfs)

    if not all_dfs:
        raise RuntimeError("No valid runs found in the sheet.")

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    combined = _coerce_numerics(combined)
    combined = _fix_time_hours(combined)
    combined = _trim_low_start_efficiency(combined)
    combined = _parse_datetime(combined)
    combined = _merge_duplicate_columns(combined)
    combined = _clean_data(combined)

    formal_runs   = combined["_run_id"].nunique() - combined[combined["_meta_informal"] == True]["_run_id"].nunique()
    informal_runs = combined[combined["_meta_informal"] == True]["_run_id"].nunique()

    logger.info(
        f"Loaded {len(combined)} rows across {total_runs} runs "
        f"({formal_runs} formal + {informal_runs} informal) "
        f"from {len(sheet.worksheets()) - len(skipped)} tabs. "
        f"Skipped: {skipped or 'none'}."
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import warnings; warnings.filterwarnings("ignore")
    df = fetch_all_tabs()

    formal   = df[df["_meta_informal"] != True]
    informal = df[df["_meta_informal"] == True]

    print(f"\nTotal: {len(df)} rows")
    print(f"Runs:  {df['_run_id'].nunique()} total  "
          f"({formal['_run_id'].nunique()} formal + {informal['_run_id'].nunique()} informal)")
    print(f"Test stations (tabs): {df['_meta_tab_name'].nunique()}")

    print("\n=== Runs per Test Station ===")
    summary = (df.groupby("_meta_tab_name")["_run_id"]
                 .nunique().sort_values(ascending=False))
    for tab, n in summary.items():
        inf_n = informal[informal["_meta_tab_name"] == tab]["_run_id"].nunique()
        flag  = f"  ({inf_n} informal)" if inf_n else ""
        print(f"  {tab:<25} {n:>3} runs{flag}")
