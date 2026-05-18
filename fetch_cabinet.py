"""
Cabinet sensor data ingestion.

Reads XLSM exports from the cabinet sensor database and computes per-run
aggregated statistics (mean, std, p5, p95, linear slope) for a set of
physics-relevant channels.  The resulting stats dict is stored in the
Supabase `cabinet_stats` table (one row per run).

Typical usage:
    from fetch_cabinet import find_cabinet_files, read_cabinet_xlsm, aggregate_run_stats
    files  = find_cabinet_files("cabinet_exports/", "r0054")
    serial, start_dt, df = read_cabinet_xlsm(files[0])
    stats  = aggregate_run_stats(df)
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Channels to aggregate — subset chosen for electrochemical relevance.
# Any channel not present in a given file is silently skipped.
KEY_CHANNELS = [
    # Electrical
    "Cell Current", "Cell Setpoint", "PS Voltage", "PS Temp",
    "Cells VTotal", "Cells DeltaV", "Cell 1", "Cell 2", "Cell 3",
    # Thermal
    "Water Temp", "Anode Water Temp", "Cathode Water Temp",
    "Ambient Temp", "O2 Temp", "Conc Temp",
    # Flow
    "Water Flow", "Anode Flow", "O2 Flow",
    # Pressure
    "Cell Pressure", "H2O2 Pressure", "Anode Pressure", "O2 Pressure",
    "Gas-Water Press.", "Pre-Filter Press.", "Δ prefilters",
    # Chemistry
    "H2O2 Level", "O2 Conc.", "Concentration",
    "Pre-RO Cond.", "Barrel Cond.", "DI Resin Cond.",
    # Environment / other
    "Ambient Humidity", "Bubble Age", "recovery",
]


def _safe_key(name: str) -> str:
    """Convert a channel name to a safe snake_case stat key prefix."""
    key = name.lower().replace("δ", "d").replace("Δ", "d")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def read_cabinet_xlsm(path: str | Path) -> tuple[str, pd.Timestamp | None, pd.DataFrame]:
    """
    Parse a cabinet XLSM export file.

    Returns
    -------
    serial    : str            – run serial from the Settings sheet (e.g. 'r0054')
    start_dt  : Timestamp|None – configured run start from Settings
    df        : DataFrame      – CabinetData sorted ascending by Time
    """
    path = Path(path)
    xl = pd.ExcelFile(path, engine="openpyxl")

    # ── Settings ──────────────────────────────────────────────────────────────
    settings_df = xl.parse("Settings")
    settings: dict = {}
    if "Key" in settings_df.columns and "Value" in settings_df.columns:
        for _, row in settings_df.iterrows():
            if pd.notna(row["Key"]):
                settings[str(row["Key"]).strip()] = row["Value"]

    serial   = str(settings.get("Serial", "")).strip()
    start_raw = settings.get("StartTime")
    start_dt  = pd.to_datetime(start_raw, dayfirst=True, errors="coerce") if start_raw else None

    # ── CabinetData ────────────────────────────────────────────────────────────
    df = xl.parse("CabinetData")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    logger.debug(f"Read {path.name}: serial={serial!r}, {len(df)} rows, "
                 f"{df['Time'].min()} → {df['Time'].max()}")
    return serial, start_dt, df


def aggregate_run_stats(
    df: pd.DataFrame,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
) -> dict:
    """
    Compute per-channel summary statistics over the run window.

    Statistics computed for each channel:
      _mean, _std, _p5, _p95, _slope   (slope = units/hour, linear regression)

    All keys are prefixed with ``cab_`` so they're distinguishable in the
    feature matrix.  Non-finite values are replaced with None for JSON safety.

    Parameters
    ----------
    df        : CabinetData DataFrame (sorted by Time)
    start_dt  : clip data before this timestamp (inclusive)
    end_dt    : clip data after this timestamp (inclusive)
    """
    if start_dt is not None:
        df = df[df["Time"] >= start_dt]
    if end_dt is not None:
        df = df[df["Time"] <= end_dt]

    if df.empty:
        logger.warning("aggregate_run_stats: empty DataFrame after time filter")
        return {}

    t0    = df["Time"].iloc[0]
    hours = (df["Time"] - t0).dt.total_seconds().values / 3600.0

    result: dict = {}

    for ch in KEY_CHANNELS:
        if ch not in df.columns:
            continue

        series = pd.to_numeric(df[ch], errors="coerce")
        mask   = series.notna().values
        vals   = series.values[mask]
        h_vals = hours[mask]

        if len(vals) < 3:
            continue

        key = f"cab_{_safe_key(ch)}"

        result[f"{key}_mean"] = _clean(float(np.mean(vals)))
        result[f"{key}_std"]  = _clean(float(np.std(vals)))
        result[f"{key}_p5"]   = _clean(float(np.percentile(vals, 5)))
        result[f"{key}_p95"]  = _clean(float(np.percentile(vals, 95)))

        if len(h_vals) >= 4 and h_vals.std() > 0:
            slope, *_ = scipy_stats.linregress(h_vals, vals)
            result[f"{key}_slope"] = _clean(float(slope))

    logger.debug(f"Aggregated {len(result)} stats from {len(df)} rows "
                 f"({len(KEY_CHANNELS)} channels attempted)")
    return result


def find_cabinet_files(cabinet_dir: str | Path, serial: str) -> list[Path]:
    """
    Scan *cabinet_dir* for XLSM/XLSX files whose Settings.Serial matches *serial*.

    If multiple files match, they are returned in ascending order of data
    density (most points last) so callers can simply take ``files[-1]``.
    """
    cabinet_dir = Path(cabinet_dir)
    if not cabinet_dir.is_dir():
        logger.warning(f"Cabinet directory not found: {cabinet_dir}")
        return []

    candidates = sorted(cabinet_dir.glob("*.xlsm")) + sorted(cabinet_dir.glob("*.xlsx"))
    matches: list[tuple[int, Path]] = []

    for path in candidates:
        try:
            xl = pd.ExcelFile(path, engine="openpyxl")
            if "Settings" not in xl.sheet_names:
                continue
            s_df = xl.parse("Settings")
            if "Key" not in s_df.columns or "Value" not in s_df.columns:
                continue
            for _, row in s_df.iterrows():
                if pd.notna(row["Key"]) and str(row["Key"]).strip() == "Serial":
                    if str(row.get("Value", "")).strip() == serial:
                        # Count data rows as a proxy for completeness
                        n = len(xl.parse("CabinetData")) if "CabinetData" in xl.sheet_names else 0
                        matches.append((n, path))
                    break
        except Exception as exc:
            logger.warning(f"Could not read {path.name}: {exc}")

    matches.sort(key=lambda t: t[0])
    return [p for _, p in matches]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(v: float | None) -> float | None:
    """Replace non-finite floats with None for JSON safety."""
    if v is None:
        return None
    return None if not np.isfinite(v) else v
