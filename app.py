"""
HPNow — Faradaic Efficiency Degradation Analyser
Interactive Streamlit app with Claude-powered chat assistant.

Primary analytical focus:
  What operating conditions or design choices are associated
  with faster Faradaic efficiency degradation over time?
"""

import io
import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Generator

from google import genai
from google.genai import types as genai_types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#dde3ec",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        "#eaeef2",
    "grid.linewidth":    0.7,
    "font.size":         10.5,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

# ── Suppress noisy logs from the data layer ───────────────────────────────────
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from fetch_sheets import fetch_all_tabs

# ── Constants ─────────────────────────────────────────────────────────────────
# Default fallback chain — overridden by user selection in sidebar
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

CONFIG_FILE = Path(__file__).parent / ".app_config.json"

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}

def save_config(data: dict):
    try:
        existing = load_config()
        existing.update(data)
        CONFIG_FILE.write_text(json.dumps(existing))
    except Exception:
        pass


COLORS = {
    "degrading":    "#e74c3c",
    "stable":       "#2ecc71",
    "improving":    "#27ae60",
    "short":        "#bdc3c7",
    "inconclusive": "#95a5a6",
    "no_data":      "#d5d8dc",
    "neutral":      "#3498db",
}

# Minimum number of data points to attempt regression
MIN_POINTS_FOR_REGRESSION = 4

# Slope thresholds for classification (%/h)
DEGRADING_SLOPE  = -0.03   # worse than –3%/100h → degrading
IMPROVING_SLOPE  =  0.01   # better  than +1%/100h → improving

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HPNow | Efficiency Analyser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Loading data from Google Sheets…")
def load_data() -> pd.DataFrame:
    return fetch_all_tabs()


@st.cache_data(ttl=3600, show_spinner="Computing run statistics…")
def compute_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-run degradation metrics.

    Primary metric: degradation rate = % efficiency lost per 100 hours,
    derived from a linear regression of Efficiency(%) vs Time(hours).
    A positive value means the run is degrading.
    """
    records = []
    for run_id, grp_raw in df.groupby("_run_id"):
        meta = grp_raw.iloc[0]

        def _safe(col, default="—"):
            v = meta.get(col, None)
            return str(v).strip() if v and str(v).strip() else default

        # Shared metadata fields
        _date_start_raw = _safe("_meta_date_start", default="")
        try:
            _year = int(pd.to_datetime(_date_start_raw, dayfirst=True).year)
        except Exception:
            _year = None

        base = {
            "run_id":          run_id,
            "station":         _safe("_meta_tab_name"),
            "stack_id":        _safe("_meta_station_id"),
            "project":         _safe("_meta_project"),
            "operator":        _safe("_meta_operator"),
            "gdl":             _safe("_meta_gdl"),
            "current_mA_cm2":  _safe("_meta_current_mA_cm2"),
            "date_start":      _safe("_meta_date_start"),
            "year":            _year,
            "aim":             _safe("_meta_aim"),
            "informal":        bool(meta.get("_meta_informal", False)),
        }

        # Duration from all time values (even rows with missing efficiency)
        t_all = pd.to_numeric(
            grp_raw["Time (hours)"] if "Time (hours)" in grp_raw.columns
            else pd.Series(dtype=float),
            errors="coerce"
        ).dropna()
        duration_all = float(t_all.max() - t_all.min()) if len(t_all) >= 2 else 0.0

        # Rows usable for regression (need both Time and Efficiency)
        grp = (
            grp_raw.dropna(subset=["Time (hours)", "Efficiency (%)"])
                   .sort_values("Time (hours)")
        )
        n_valid = len(grp)

        if n_valid < MIN_POINTS_FOR_REGRESSION:
            # Still record the run — just flag it as no_data
            records.append({
                **base,
                "n_points":        n_valid,
                "duration_h":      round(duration_all, 1),
                "eff_start_%":     np.nan,
                "eff_end_%":       np.nan,
                "eff_drop_%":      np.nan,
                "slope_%_per_h":   np.nan,
                "deg_rate_%/100h": np.nan,
                "r2":              np.nan,
                "p_value":         np.nan,
                "label":           "no_data",
            })
            continue

        t   = grp["Time (hours)"].values.astype(float)
        eff = grp["Efficiency (%)"].values.astype(float)

        duration  = float(t.max() - t.min())
        eff_start = float(np.mean(eff[:min(3, len(eff))]))
        eff_end   = float(np.mean(eff[-min(3, len(eff)):]))
        eff_drop  = eff_start - eff_end

        if duration > 0:
            slope, _, r_value, p_value, _ = stats.linregress(t, eff)
        else:
            slope = r_value = p_value = np.nan

        deg_rate = float(-slope * 100) if not np.isnan(slope) else np.nan

        if duration < 20:
            label = "short"
        elif np.isnan(slope):
            label = "inconclusive"
        elif slope < DEGRADING_SLOPE:
            label = "degrading"
        elif slope > IMPROVING_SLOPE:
            label = "improving"
        else:
            label = "stable"

        records.append({
            **base,
            "n_points":        n_valid,
            "duration_h":      round(duration, 1),
            "eff_start_%":     round(eff_start, 1),
            "eff_end_%":       round(eff_end, 1),
            "eff_drop_%":      round(eff_drop, 1),
            "slope_%_per_h":   round(slope, 5) if not np.isnan(slope) else np.nan,
            "deg_rate_%/100h": round(deg_rate, 2) if not np.isnan(deg_rate) else np.nan,
            "r2":              round(r_value**2, 3) if not np.isnan(r_value) else np.nan,
            "p_value":         round(p_value, 4) if not np.isnan(p_value) else np.nan,
            "label":           label,
        })

    return pd.DataFrame(records)


def apply_filters(
    df: pd.DataFrame,
    run_stats: pd.DataFrame,
    stations: list,
    projects: list,
    years: list,
    min_hours: float,
    exclude_informal: bool,
    exclude_short: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return filtered (raw_df, run_stats_df) based on sidebar selections."""
    mask = pd.Series(True, index=run_stats.index)
    if stations:
        mask &= run_stats["station"].isin(stations)
    if projects:
        mask &= run_stats["project"].isin(projects)
    if years:
        mask &= run_stats["year"].isin(years)
    # NaN duration (no_data runs) → treat as 0h so they obey the min_hours filter
    mask &= run_stats["duration_h"].fillna(0) >= min_hours
    if exclude_informal:
        mask &= ~run_stats["informal"]
    if exclude_short:
        mask &= run_stats["label"] != "short"

    fs = run_stats[mask].copy()
    fd = df[df["_run_id"].isin(fs["run_id"])].copy()
    return fd, fs


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_chat_context(
    filtered_df: pd.DataFrame,
    filtered_stats: pd.DataFrame,
    filter_summary: str,
) -> str:
    """Serialise the current filtered dataset into a text block for Claude."""
    rs = filtered_stats
    label_counts = rs["label"].value_counts().to_dict()
    deg_vals = rs["deg_rate_%/100h"].dropna()

    deg_summary = (
        f"mean = {deg_vals.mean():.2f}  |  "
        f"median = {deg_vals.median():.2f}  |  "
        f"std = {deg_vals.std():.2f}  %/100h"
    ) if len(deg_vals) else "insufficient data"

    # Top 5 worst-degrading runs
    top_deg_cols = [
        "run_id", "station", "stack_id", "project",
        "gdl", "current_mA_cm2", "duration_h",
        "eff_start_%", "eff_end_%", "deg_rate_%/100h",
    ]
    top_deg = (
        rs[rs["label"] == "degrading"]
        .nlargest(5, "deg_rate_%/100h")[
            [c for c in top_deg_cols if c in rs.columns]
        ].to_string(index=False)
    )

    # Station summary
    station_agg = rs.groupby("station").agg(
        n_runs       = ("run_id", "count"),
        n_degrading  = ("label", lambda x: (x == "degrading").sum()),
        n_stable     = ("label", lambda x: (x == "stable").sum()),
        median_deg   = ("deg_rate_%/100h", "median"),
        mean_dur_h   = ("duration_h", "mean"),
    ).round(2).sort_values("median_deg", ascending=False)

    # GDL breakdown
    gdl_agg = (
        rs.groupby("gdl")["deg_rate_%/100h"]
        .describe()[["count", "mean", "50%"]].round(2)
        if "gdl" in rs.columns else "—"
    )

    # Project breakdown
    proj_agg = (
        rs.groupby("project")["deg_rate_%/100h"]
        .describe()[["count", "mean", "50%"]].round(2)
        if "project" in rs.columns else "—"
    )

    # Point-level Pearson correlations with Efficiency(%)
    corr_lines = []
    if "Efficiency (%)" in filtered_df.columns:
        meta_cols = {c for c in filtered_df.columns if c.startswith("_")}
        eff = pd.to_numeric(filtered_df["Efficiency (%)"], errors="coerce")
        for col in filtered_df.columns:
            if col in meta_cols or col == "Efficiency (%)":
                continue
            if not pd.api.types.is_numeric_dtype(filtered_df[col]):
                continue
            x = pd.to_numeric(filtered_df[col], errors="coerce")
            valid = eff.notna() & x.notna()
            if valid.sum() >= 100:
                r, _ = stats.pearsonr(x[valid], eff[valid])
                corr_lines.append((col, r, int(valid.sum())))
        corr_lines.sort(key=lambda t: abs(t[1]), reverse=True)
        corr_text = "\n".join(
            f"  {col:<35} r = {r:+.3f}  (n = {n})"
            for col, r, n in corr_lines[:12]
        )
    else:
        corr_text = "  (no efficiency column found)"

    return f"""
## DATASET CONTEXT (currently filtered)

Active filters: {filter_summary}
Runs in view : {len(rs)} total
  → degrading    : {label_counts.get('degrading', 0)}
  → stable       : {label_counts.get('stable', 0)}
  → improving    : {label_counts.get('improving', 0)}
  → short (<20h) : {label_counts.get('short', 0)}
  → inconclusive : {label_counts.get('inconclusive', 0)}
Test stations : {rs['station'].nunique()}
Unique stacks : {rs['stack_id'].nunique()}
Informal runs : {rs['informal'].sum()} (no stack ID, inherited metadata)

### Degradation Rate Distribution (% efficiency lost per 100 h)
Positive = degrading, Negative = improving
{deg_summary}

### Top 5 Fastest-Degrading Runs
{top_deg or '(none found)'}

### Per-Station Summary
{station_agg.to_string()}

### GDL Type vs Degradation Rate
{gdl_agg if isinstance(gdl_agg, str) else gdl_agg.to_string()}

### Project vs Degradation Rate
{proj_agg if isinstance(proj_agg, str) else proj_agg.to_string()}

### Point-Level Correlations with Faradaic Efficiency (%)
(High |r| near ±1 = strong relationship)
{corr_text}

### Data Dictionary
- deg_rate_%/100h : primary metric — % Faradaic efficiency lost per 100 hours
- eff_start_%     : mean efficiency over first 3 measurements
- eff_end_%       : mean efficiency over last 3 measurements
- eff_drop_%      : eff_start - eff_end (positive = degraded over the run)
- label           : degrading | stable | improving | short | inconclusive
- gdl             : gas diffusion layer type installed in that run
- current_mA_cm2  : applied current density (mA/cm²)
- informal        : True if run had no formal "Initials" metadata block
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_trajectories(
    filtered_df: pd.DataFrame,
    filtered_stats: pd.DataFrame,
    selected_run_ids: list,
    color_by: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.get_cmap("RdYlGn_r")

    deg_series = filtered_stats.set_index("run_id")["deg_rate_%/100h"]
    label_series = filtered_stats.set_index("run_id")["label"]
    vmin = filtered_stats["deg_rate_%/100h"].quantile(0.05)
    vmax = filtered_stats["deg_rate_%/100h"].quantile(0.95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    plotted = 0
    for run_id in selected_run_ids:
        run_df = (
            filtered_df[filtered_df["_run_id"] == run_id]
            .dropna(subset=["Time (hours)", "Efficiency (%)"])
            .sort_values("Time (hours)")
        )
        if len(run_df) < 2:
            continue

        if color_by == "Degradation rate":
            dr = deg_series.get(run_id, np.nan)
            color = cmap(norm(dr)) if not np.isnan(dr) else "#bdc3c7"
        else:
            label = label_series.get(run_id, "inconclusive")
            color = COLORS.get(label, "#bdc3c7")

        ax.plot(
            run_df["Time (hours)"], run_df["Efficiency (%)"],
            linewidth=1.3, alpha=0.75, color=color,
        )
        plotted += 1

    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Faradaic Efficiency (%)", fontsize=11)
    ax.set_title(f"Efficiency Trajectories ({plotted} runs)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)

    if color_by == "Degradation rate":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Deg. rate (%/100h)", shrink=0.85)

    fig.tight_layout()
    return fig


def fig_corr_bars(filtered_df: pd.DataFrame) -> plt.Figure:
    meta_cols = {c for c in filtered_df.columns if c.startswith("_")}
    eff = pd.to_numeric(filtered_df.get("Efficiency (%)", pd.Series(dtype=float)), errors="coerce")

    corrs = {}
    for col in filtered_df.columns:
        if col in meta_cols or col == "Efficiency (%)":
            continue
        if not pd.api.types.is_numeric_dtype(filtered_df[col]):
            continue
        x = pd.to_numeric(filtered_df[col], errors="coerce")
        valid = eff.notna() & x.notna()
        if valid.sum() >= 100:
            r, _ = stats.pearsonr(x[valid], eff[valid])
            corrs[col] = r

    if not corrs:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    cdf = (
        pd.Series(corrs)
        .to_frame("r")
        .assign(abs_r=lambda d: d["r"].abs())
        .sort_values("abs_r", ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(cdf) * 0.45)))
    colors = [COLORS["degrading"] if r < 0 else COLORS["stable"] for r in cdf["r"]]
    ax.barh(cdf.index[::-1], cdf["r"][::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with Efficiency (%)", fontsize=11)
    ax.set_title("Feature correlations with Faradaic Efficiency (point-level)", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_feature_vs_degradation(filtered_stats: pd.DataFrame, feature: str) -> plt.Figure:
    data = filtered_stats.dropna(subset=["deg_rate_%/100h"])
    data = data[data[feature].notna() & (data[feature] != "—")]

    counts = data[feature].value_counts()
    valid  = counts[counts >= 2].index
    data   = data[data[feature].isin(valid)]

    if data.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "Need ≥2 runs per category", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return fig

    order = (
        data.groupby(feature)["deg_rate_%/100h"]
        .median().sort_values().index.tolist()
    )

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 1.0 + 1.5), 5))
    sns.boxplot(
        data=data, x=feature, y="deg_rate_%/100h",
        order=order, ax=ax, palette="RdYlGn_r", width=0.55,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel("Degradation rate (%/100h)", fontsize=11)
    ax.set_title(f"Degradation rate by {feature}", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    # Annotate n
    for i, cat in enumerate(order):
        n = (data[feature] == cat).sum()
        ax.text(i, ax.get_ylim()[0] + 0.3, f"n={n}", ha="center", fontsize=8, color="#555")

    fig.tight_layout()
    return fig


def fig_station_boxplot(filtered_stats: pd.DataFrame) -> plt.Figure:
    data = filtered_stats.dropna(subset=["deg_rate_%/100h"])
    if data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    order = (
        data.groupby("station")["deg_rate_%/100h"]
        .median().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, max(4, len(order) * 0.45 + 1)))
    sns.boxplot(
        data=data, y="station", x="deg_rate_%/100h",
        order=order, ax=ax, palette="RdYlGn_r", width=0.55, orient="h",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Degradation rate (%/100h)", fontsize=11)
    ax.set_ylabel("Test station", fontsize=11)
    ax.set_title("Degradation rate by test station", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_deg_histogram(filtered_stats: pd.DataFrame) -> plt.Figure:
    vals = filtered_stats["deg_rate_%/100h"].dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(vals, bins=25, color=COLORS["neutral"], edgecolor="white", alpha=0.85)
    if len(vals):
        ax.axvline(vals.median(), color=COLORS["stable"], linewidth=2,
                   label=f"Median: {vals.median():.2f}")
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    ax.set_xlabel("Degradation rate (%/100h)", fontsize=10)
    ax.set_ylabel("# runs", fontsize=10)
    ax.set_title("Distribution of\ndegradation rates", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib figure to PNG bytes for st.download_button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def stats_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to Excel bytes for st.download_button."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Run statistics")
    buf.seek(0)
    return buf.getvalue()


def chat_to_markdown(history: list) -> str:
    """Format chat history as a readable Markdown document."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# HPNow AI Chat Export\n_Exported: {ts}_\n"]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**AI**"
        lines.append(f"### {role}\n{msg['content']}\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE STYLING
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_BG = {
    "degrading":    "#fde8e8",
    "stable":       "#e8f8ef",
    "improving":    "#d4f0e0",
    "short":        "#f2f2f2",
    "inconclusive": "#f2f2f2",
    "no_data":      "#f2f2f2",
}
_LABEL_FG = {
    "degrading":    "#c0392b",
    "stable":       "#1e8449",
    "improving":    "#145a32",
    "short":        "#7f8c8d",
    "inconclusive": "#7f8c8d",
    "no_data":      "#7f8c8d",
}

def _style_run_table(df: pd.DataFrame):
    def _label(val):
        bg = _LABEL_BG.get(str(val), "#f2f2f2")
        fg = _LABEL_FG.get(str(val), "#333")
        return f"background-color: {bg}; color: {fg}; font-weight: 600;"

    def _deg_rate(val):
        try:
            v = float(val)
            if v > 0.1:
                return "color: #c0392b; font-weight: 600;"
            elif v < -0.05:
                return "color: #1e8449; font-weight: 600;"
        except Exception:
            pass
        return ""

    styler = df.style
    if "label" in df.columns:
        styler = styler.map(_label, subset=["label"])
    if "deg_rate_%/100h" in df.columns:
        styler = styler.map(_deg_rate, subset=["deg_rate_%/100h"])
    return styler


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f4f6f9;
        border: 1px solid #dde3ec;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #5a6a7a; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #1a2e45; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 500;
        color: #5a6a7a;
    }
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        border-bottom: 2px solid #1a6ebd !important;
        color: #1a6ebd !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f8f9fb; }

    /* Buttons */
    .stButton > button      { border-radius: 6px; font-weight: 500; }
    .stDownloadButton > button { border-radius: 6px; font-weight: 500; }

    /* Headings */
    h1 { color: #1a2e45; letter-spacing: -0.5px; }
    h2, h3 { color: #1a2e45; }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚡ HPNow — Faradaic Efficiency Degradation Analyser")
    st.caption("**Primary question:** what conditions cause Faradaic efficiency to decline over time?")

    # ── Data (cached) — loaded before sidebar so year options are available ───
    df        = load_data()
    run_stats = compute_run_stats(df)

    # ── Session state (load persisted config on first run) ────────────────────
    if "initialised" not in st.session_state:
        cfg = load_config()
        st.session_state.chat_history      = []
        st.session_state.models            = cfg.get("models") or DEFAULT_MODELS
        st.session_state.discovered_models = cfg.get("discovered_models") or []
        # Priority: Streamlit secrets → persisted config → env var
        _secrets_key = ""
        try:
            _secrets_key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            pass
        st.session_state.api_key = (
            _secrets_key
            or cfg.get("api_key")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        st.session_state.last_loaded_at = datetime.now()
        st.session_state.initialised    = True

    # Detect whether the API key is pre-configured via Streamlit secrets
    _key_from_secrets = False
    try:
        _key_from_secrets = bool(st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:
        pass

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # ── AI settings (only show key input when not pre-configured) ─────────
        if not _key_from_secrets:
            st.header("⚙️ Settings")
            api_key = st.text_input(
                "Google AI Studio API key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste key, then press Enter",
                help="Free at aistudio.google.com → Get API key.",
            )
            if st.button("✅ Apply key", use_container_width=True):
                st.session_state.api_key = api_key
                save_config({"api_key": api_key})
                st.rerun()
            elif api_key != st.session_state.api_key:
                st.session_state.api_key = api_key

            if st.session_state.api_key:
                st.success("API key saved", icon="🔑")

                # Model discovery
                if st.button("🔍 Discover available models", use_container_width=True):
                    with st.spinner("Fetching models from Google AI…"):
                        try:
                            client = genai.Client(api_key=st.session_state.api_key)
                            found = []
                            for m in client.models.list():
                                name    = m.name.replace("models/", "")
                                display = getattr(m, "display_name", name)
                                if ("flash" in name.lower() or "lite" in name.lower()) and \
                                   "embed" not in name.lower() and "vision" not in name.lower():
                                    found.append((name, display))
                            st.session_state.discovered_models = found
                            save_config({"discovered_models": found})
                        except Exception as e:
                            st.error(f"Could not list models: {e}")

                if st.session_state.discovered_models:
                    label_map = {n: d for n, d in st.session_state.discovered_models}
                    st.caption("Tick models to use, in priority order:")
                    new_selection = []
                    for api_name, display_name in st.session_state.discovered_models:
                        if st.checkbox(display_name, value=api_name in st.session_state.models,
                                       key=f"model_cb_{api_name}", help=f"API name: {api_name}"):
                            new_selection.append(api_name)
                    if new_selection != st.session_state.models:
                        st.session_state.models = new_selection
                        save_config({"models": new_selection})

                if st.session_state.models:
                    label_map = {n: d for n, d in st.session_state.discovered_models} \
                                if st.session_state.discovered_models else {}
                    st.caption("**Active:** " + " → ".join(
                        label_map.get(m, m) for m in st.session_state.models
                    ))
                else:
                    st.caption("No models selected — using defaults")

            st.divider()

        if st.button("🔄 Refresh data from Sheets", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_loaded_at = datetime.now()
            st.rerun()
        _elapsed = int((datetime.now() - st.session_state.last_loaded_at).total_seconds() / 60)
        _freshness = "just now" if _elapsed < 1 else f"{_elapsed} min ago"
        st.caption(f"Data loaded {_freshness}")

        st.subheader("Filters")

        all_stations = sorted(df["_meta_tab_name"].dropna().unique())
        all_projects = sorted({
            str(p).strip()
            for p in df["_meta_project"].dropna()
            if str(p).strip() and str(p).strip() not in ("—", "nan")
        })
        all_years = sorted(
            [int(y) for y in run_stats["year"].dropna().unique() if y],
            reverse=True,
        )

        stations = st.multiselect(
            "Test stations", options=all_stations,
            placeholder="All stations", key="filter_stations",
        )
        projects = st.multiselect(
            "Projects", options=all_projects,
            placeholder="All projects", key="filter_projects",
        )
        years = st.multiselect(
            "Year", options=all_years,
            placeholder="All years", key="filter_years",
        )
        min_hours = st.slider(
            "Min. run duration (hours)", min_value=0, max_value=500, value=20, step=5,
        )
        exclude_informal = st.toggle("Exclude informal runs (no stack ID)", value=False)
        exclude_short    = st.toggle("Exclude short runs (<20 h)", value=True)

        if st.button("↺ Reset filters", use_container_width=True, type="secondary"):
            for _k in ("filter_stations", "filter_projects", "filter_years"):
                st.session_state[_k] = []
            st.rerun()

        st.divider()
        st.caption(
            f"Sheet: {df['_run_id'].nunique()} total runs  \n"
            f"{df['_meta_tab_name'].nunique()} test stations"
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered_df, filtered_stats = apply_filters(
        df, run_stats, stations, projects, years, min_hours, exclude_informal, exclude_short,
    )

    if filtered_stats.empty:
        st.warning("No runs match the current filters. Try broadening your selection.")
        return

    filter_summary = (
        f"stations={'all' if not stations else ', '.join(stations)}, "
        f"projects={'all' if not projects else ', '.join(projects)}, "
        f"years={'all' if not years else ', '.join(str(y) for y in years)}, "
        f"min_duration={min_hours}h, "
        f"exclude_informal={exclude_informal}"
    )

    # ── Top-line metrics ──────────────────────────────────────────────────────
    lc  = filtered_stats["label"].value_counts()
    dv  = filtered_stats["deg_rate_%/100h"].dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Runs in view",
        f"{len(filtered_stats)} / {len(run_stats)}",
        help="Runs passing current filters / total runs detected across all sheets",
    )
    c2.metric("Degrading",
              lc.get("degrading", 0),
              delta=f"{lc.get('degrading',0)/len(filtered_stats)*100:.0f}%",
              delta_color="inverse")
    c3.metric("Stable", lc.get("stable", 0),
              delta=f"{lc.get('stable',0)/len(filtered_stats)*100:.0f}%")
    c4.metric("Median deg. rate",
              f"{dv.median():.2f} %/100h" if len(dv) else "—",
              help="% Faradaic efficiency lost per 100 hours (linear regression slope × −100)")
    c5.metric("Test stations", filtered_stats["station"].nunique())

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    _n_runs     = len(filtered_stats)
    _n_stations = filtered_stats["station"].nunique()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"📉 Degradation Overview  ({_n_runs} runs)",
        "📈 Efficiency Trajectories",
        "🔍 Correlations",
        f"🏭 Station Comparison  ({_n_stations} stations)",
        "💬 Ask AI",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — DEGRADATION OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Per-Run Degradation Rates")
        st.caption(
            "**Degradation rate** = % Faradaic efficiency lost per 100 hours, "
            "computed via linear regression on the Efficiency(%) vs Time(hours) trajectory.  \n"
            "Positive = degrading  |  Negative = actually improving  |  Near 0 = stable"
        )

        col_table, col_hist = st.columns([3, 1], gap="large")

        with col_table:
            display_cols = [
                "run_id", "station", "stack_id", "project",
                "gdl", "current_mA_cm2", "date_start",
                "duration_h", "n_points",
                "eff_start_%", "eff_end_%", "eff_drop_%",
                "deg_rate_%/100h", "r2", "label",
            ]
            table_df = (
                filtered_stats[[c for c in display_cols if c in filtered_stats.columns]]
                .sort_values("deg_rate_%/100h", ascending=False)
            )
            st.dataframe(
                _style_run_table(table_df),
                hide_index=True,
                use_container_width=True,
                height=520,
                column_config={
                    "deg_rate_%/100h": st.column_config.NumberColumn(
                        "Deg. rate (%/100h)", format="%.2f",
                        help="% efficiency lost per 100 hours (positive = degrading)",
                    ),
                    "eff_start_%":  st.column_config.NumberColumn("Start eff. (%)", format="%.1f"),
                    "eff_end_%":    st.column_config.NumberColumn("End eff. (%)", format="%.1f"),
                    "eff_drop_%":   st.column_config.NumberColumn("Eff. drop (%)", format="%.1f"),
                    "duration_h":   st.column_config.NumberColumn("Duration (h)", format="%.1f"),
                    "r2":           st.column_config.NumberColumn("R²", format="%.3f"),
                },
            )
            st.download_button(
                label="⬇️ Download as Excel",
                data=stats_to_excel_bytes(table_df),
                file_name=f"hpnow_run_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_hist:
            fig = fig_deg_histogram(filtered_stats)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Label counts**")
            st.dataframe(
                filtered_stats["label"].value_counts().rename("runs").to_frame(),
                use_container_width=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — TRAJECTORIES
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Efficiency Trajectories Over Time")

        ctrl_col, plot_col = st.columns([1, 4], gap="large")

        with ctrl_col:
            color_by = st.radio(
                "Colour lines by",
                ["Degradation rate", "Classification label"],
            )
            max_runs = st.slider("Max runs to show", 5, 150, 40)
            sort_worst_first = st.toggle("Worst-degrading first", value=True)

            traj_stations = st.multiselect(
                "Station filter (trajectories)",
                options=sorted(filtered_stats["station"].unique()),
                default=[],
                placeholder="All stations",
            )
            traj_labels = st.multiselect(
                "Show labels",
                options=["degrading", "stable", "improving", "inconclusive", "short"],
                default=["degrading", "stable", "improving"],
            )

        with plot_col:
            traj = filtered_stats.copy()
            if traj_stations:
                traj = traj[traj["station"].isin(traj_stations)]
            if traj_labels:
                traj = traj[traj["label"].isin(traj_labels)]
            if sort_worst_first:
                traj = traj.nlargest(max_runs, "deg_rate_%/100h")
            else:
                traj = traj.head(max_runs)

            if traj.empty:
                st.info("No runs match the trajectory filters.")
            else:
                fig = fig_trajectories(filtered_df, filtered_stats, traj["run_id"].tolist(), color_by)
                st.pyplot(fig)
                st.download_button(
                    label="⬇️ Download chart (PNG)",
                    data=fig_to_png_bytes(fig),
                    file_name=f"hpnow_trajectories_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=True,
                )
                plt.close(fig)

                if color_by == "Classification label":
                    cols = st.columns(5)
                    for i, (lbl, clr) in enumerate(COLORS.items()):
                        cols[i].markdown(
                            f"<span style='color:{clr}'>■</span> {lbl}",
                            unsafe_allow_html=True,
                        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CORRELATIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("What Correlates with Faradaic Efficiency?")

        corr_l, corr_r = st.columns(2, gap="large")

        with corr_l:
            st.markdown(
                "**Numerical features vs Efficiency (%)** — point-level Pearson r  \n"
                "Red = negative correlation (higher value → lower efficiency)"
            )
            fig = fig_corr_bars(filtered_df)
            st.pyplot(fig)
            st.download_button(
                label="⬇️ Download chart (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"hpnow_correlations_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png",
                use_container_width=True,
                key="dl_corr_bars",
            )
            plt.close(fig)

        with corr_r:
            st.markdown("**Categorical feature vs Degradation rate** — box plots per category")
            cat_feature = st.selectbox(
                "Feature to compare",
                options=["gdl", "project", "operator", "current_mA_cm2", "station"],
                index=0,
            )
            fig = fig_feature_vs_degradation(filtered_stats, cat_feature)
            st.pyplot(fig)
            st.download_button(
                label="⬇️ Download chart (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"hpnow_{cat_feature}_vs_degrate_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png",
                use_container_width=True,
                key="dl_feature_box",
            )
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — STATION COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Station-by-Station Comparison")

        station_agg = (
            filtered_stats.groupby("station").agg(
                n_runs       = ("run_id", "count"),
                n_degrading  = ("label", lambda x: (x == "degrading").sum()),
                n_stable     = ("label", lambda x: (x == "stable").sum()),
                median_deg   = ("deg_rate_%/100h", "median"),
                mean_dur_h   = ("duration_h", "mean"),
                mean_eff_start = ("eff_start_%", "mean"),
                mean_eff_end   = ("eff_end_%", "mean"),
            )
            .round(2)
            .sort_values("median_deg", ascending=False)
        )

        st_l, st_r = st.columns([1, 2], gap="large")

        with st_l:
            st.markdown("**Summary table** (sorted by median degradation rate)")
            st.dataframe(station_agg, use_container_width=True, height=500)

        with st_r:
            fig = fig_station_boxplot(filtered_stats)
            st.pyplot(fig)
            st.download_button(
                label="⬇️ Download chart (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"hpnow_station_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime="image/png",
                use_container_width=True,
            )
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — CHAT
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("💬 Ask AI about Your Efficiency Data")
        st.caption(
            "The AI can query the live dataset on demand — it will fetch exactly "
            "the data it needs to answer each question."
        )

        if not st.session_state.api_key:
            st.warning(
                "Enter your **Google AI Studio API key** in the sidebar to activate the chat.  \n"
                "Get one free at [aistudio.google.com](https://aistudio.google.com) → *Get API key* "
                "(no credit card needed)."
            )
        else:
            data_context = build_chat_context(filtered_df, filtered_stats, filter_summary)

            system_prompt = (
                "You are an expert electrochemist and data scientist helping the HPNow team "
                "understand what causes Faradaic efficiency to decline in their electrolyser "
                "test stations.\n\n"
                "The PRIMARY question is: what operating conditions or design choices "
                "correlate with faster Faradaic efficiency degradation? Starting efficiency "
                "matters less than the rate of decline.\n\n"
                "You have access to a query_data tool that lets you fetch any slice of the "
                "dataset you need. Use it whenever the summary context is insufficient — "
                "for example, to get chronological run sequences, filter by specific "
                "conditions, or compute temporal trends per station.\n\n"
                "Key context:\n"
                "- Faradaic efficiency (%): how efficiently the electrolyser converts "
                "electrical energy to product. Decline over time is the main concern.\n"
                "- Degradation rate (%/100h): efficiency lost per 100 operating hours. "
                "Positive = degrading, negative = improving, near 0 = stable.\n"
                "- GDL: gas diffusion layer — a key electrolyser component.\n"
                "- Test station (tab): the physical test rig (r0054, r0056, s0146, etc.).\n"
                "- Stack ID: identifier for a specific assembled electrolyser stack.\n"
                "- run_index: integer order of a run within its station (0 = first run ever).\n\n"
                "When the data supports a quantitative answer, be specific. "
                "When data is insufficient, say so and suggest what would resolve it.\n\n"
                f"{data_context}"
            )

            # ── Tool definitions ──────────────────────────────────────────────
            QUERY_TOOL = genai_types.Tool(function_declarations=[
                genai_types.FunctionDeclaration(
                    name="query_data",
                    description=(
                        "Query the run-level dataset. Returns a text table of matching rows. "
                        "Use this to get data that is not in the summary context, such as "
                        "chronological run sequences, filtered subsets, or specific stations."
                    ),
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "mode": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description=(
                                    "What to return. One of: "
                                    "'station_timeline' (all runs for a station in order), "
                                    "'runs_table' (run-level table, filterable), "
                                    "'station_list' (which stations exist and run counts)."
                                ),
                            ),
                            "station": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Filter to this station name (e.g. 'r0056'). Optional.",
                            ),
                            "label": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Filter to runs with this label: degrading|stable|improving|short|inconclusive. Optional.",
                            ),
                            "min_runs": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="For station_list: only show stations with at least this many runs.",
                            ),
                        },
                        required=["mode"],
                    ),
                )
            ])

            def _execute_query(args: dict) -> str:
                """Execute a query_data tool call against filtered_stats."""
                mode    = args.get("mode", "runs_table")
                station = args.get("station")
                label   = args.get("label")
                min_runs = int(args.get("min_runs", 1))

                df = filtered_stats.copy()
                if station:
                    df = df[df["station"] == station]
                if label:
                    df = df[df["label"] == label]

                if mode == "station_list":
                    agg = (
                        filtered_stats.groupby("station")
                        .agg(n_runs=("run_id","count"),
                             n_degrading=("label", lambda x:(x=="degrading").sum()),
                             median_deg=("deg_rate_%/100h","median"))
                        .query(f"n_runs >= {min_runs}")
                        .sort_values("n_runs", ascending=False)
                        .round(2)
                    )
                    return agg.to_string()

                elif mode == "station_timeline":
                    cols = ["run_id","date_start","eff_start_%","eff_end_%",
                            "deg_rate_%/100h","duration_h","label","project","gdl","stack_id"]
                    out = df[[c for c in cols if c in df.columns]].sort_values("run_id")
                    if out.empty:
                        return f"No runs found for station='{station}'."
                    return out.to_string(index=False)

                else:  # runs_table
                    cols = ["run_id","station","date_start","eff_start_%","eff_end_%",
                            "deg_rate_%/100h","duration_h","label","project","gdl",
                            "current_mA_cm2","stack_id"]
                    out = df[[c for c in cols if c in df.columns]].sort_values(
                        ["station","run_id"])
                    if out.empty:
                        return "No runs match the specified filters."
                    return out.to_string(index=False)

            # ── Helper checks ─────────────────────────────────────────────────
            def _is_rate_limit(exc):
                err = str(exc)
                return any(k in err for k in ("429","RESOURCE_EXHAUSTED","quota","Quota"))

            def _is_bad_key(exc):
                err = str(exc)
                return any(k in err for k in ("API_KEY_INVALID","API key not valid","UNAUTHENTICATED"))

            def _is_model_missing(exc):
                err = str(exc)
                return any(k in err for k in ("404","NOT_FOUND","not found","is not supported"))

            # ── Agentic loop with tool use ────────────────────────────────────
            def agentic_stream(
                api_key: str,
                system_prompt: str,
                history: list,
                user_input: str,
                models: list,
            ) -> Generator[str, None, None]:
                """
                Multi-turn agentic loop: the model can call query_data() up to 4 times
                before producing its final answer. Falls back across models on rate-limit.
                """
                client = genai.Client(api_key=api_key)

                # Build initial contents
                def _build_contents(history, user_input):
                    contents = []
                    for m in history:
                        role = "model" if m["role"] == "assistant" else "user"
                        contents.append(genai_types.Content(
                            role=role,
                            parts=[genai_types.Part(text=m["content"])],
                        ))
                    contents.append(genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=user_input)],
                    ))
                    return contents

                last_exc = None
                for model_idx, model_name in enumerate(models):
                    is_last_model = model_idx == len(models) - 1
                    try:
                        contents = _build_contents(history, user_input)
                        config   = genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            tools=[QUERY_TOOL],
                            max_output_tokens=4096,
                        )

                        # Agentic loop: up to 4 tool-call rounds
                        for _round in range(5):
                            response = client.models.generate_content(
                                model=model_name,
                                contents=contents,
                                config=config,
                            )

                            candidate = response.candidates[0]
                            parts     = candidate.content.parts

                            # Collect any function calls
                            fn_calls = [p for p in parts
                                        if p.function_call and p.function_call.name]
                            text_parts = [p.text for p in parts if p.text]

                            if not fn_calls:
                                # Final answer — stream word by word for UX
                                final_text = "\n".join(text_parts)
                                for word in final_text.split(" "):
                                    yield word + " "
                                return

                            # Execute each tool call
                            contents.append(candidate.content)  # assistant turn
                            fn_result_parts = []
                            for fc in fn_calls:
                                fn_args = {k: v for k, v in fc.function_call.args.items()}
                                yield (
                                    f"*🔍 Querying: `{fc.function_call.name}`"
                                    f"({', '.join(f'{k}={v}' for k,v in fn_args.items())})*\n\n"
                                )
                                result_text = _execute_query(fn_args)
                                fn_result_parts.append(genai_types.Part(
                                    function_response=genai_types.FunctionResponse(
                                        name=fc.function_call.name,
                                        response={"result": result_text},
                                    )
                                ))

                            contents.append(genai_types.Content(
                                role="user", parts=fn_result_parts
                            ))

                        yield "\n*(Reached maximum query iterations)*\n"
                        return

                    except Exception as exc:
                        last_exc = exc
                        if _is_bad_key(exc):
                            raise
                        if (_is_rate_limit(exc) or _is_model_missing(exc)) and not is_last_model:
                            yield f"*[{model_name} limited — trying {models[model_idx+1]}…]*\n\n"
                            continue
                        raise

            active_models = st.session_state.models or DEFAULT_MODELS

            # ── Display history ───────────────────────────────────────────────
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # ── Chat input ────────────────────────────────────────────────────
            if user_input := st.chat_input("Ask about your efficiency data…"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    try:
                        history_so_far = st.session_state.chat_history[:-1]
                        active_models  = st.session_state.models or DEFAULT_MODELS

                        def run_stream() -> Generator[str, None, None]:
                            yield from agentic_stream(
                                st.session_state.api_key,
                                system_prompt,
                                history_so_far,
                                user_input,
                                active_models,
                            )

                        full_response = st.write_stream(run_stream())
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": full_response}
                        )

                    except Exception as exc:
                        if _is_bad_key(exc):
                            st.error("❌ Invalid API key — check your Google AI Studio key in the sidebar.")
                        elif _is_rate_limit(exc):
                            st.error(
                                "⏳ All models are rate-limited. "
                                "Wait a minute and try again, or check "
                                "[aistudio.google.com](https://aistudio.google.com) → usage."
                            )
                        else:
                            st.error(f"Error: {exc}")

            # Export / clear chat history
            if st.session_state.chat_history:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="💾 Export chat (Markdown)",
                        data=chat_to_markdown(st.session_state.chat_history),
                        file_name=f"hpnow_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with btn_col2:
                    if st.button("🗑️ Clear chat history", type="secondary", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()


if __name__ == "__main__":
    main()
