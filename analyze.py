"""
Correlation and trend analysis of HPNow Sheets data.

Produces:
  - per-run summaries (efficiency trajectory, classification)
  - cross-run correlation ranking vs faradaic efficiency
  - stable vs degrading run comparison
  - time-series plots per station
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

import config

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Columns to exclude from correlation analysis
NON_FEATURE_COLS = {
    "Strip 1", "Strip 2", "Corrected strip 1", "Corrected strip 2", "Average",
    "Correction for strip 1 \"b\"", "Correction for strip 1 \"m\"",
    "Correction for strip 2 \"b\"", "Correction for strip 2 \"m\"",
    "Peroxide in DI water",
    "H2O2 current (A)",          # derived from efficiency
    "Average efficiency (%)",    # derived from efficiency
    "H2O2 current density (mA/cm2)",  # derived
    "Avg. throughput (g/h)",     # derived
    "Date", "time", "Comments",
}


# ── Run classification ──────────────────────────────────────────────────────

def classify_runs(df):
    """
    For each individual run (identified by _run_id), compute:
      - run_max_hours       : total test duration
      - eff_first / eff_last: efficiency at start and end
      - eff_drop_pct        : total drop in efficiency (positive = declined)
      - run_class           : 'degrading', 'stable', 'short', or 'inconclusive'
    Returns a per-run summary DataFrame.
    """
    records = []
    group_col = "_run_id" if "_run_id" in df.columns else "_meta_station_id"

    for run_id, grp in df.groupby(group_col):
        grp = grp.sort_values("Time (hours)").dropna(subset=[config.TARGET_COL, "Time (hours)"])
        if len(grp) < config.MIN_ROWS:
            continue

        eff   = pd.to_numeric(grp[config.TARGET_COL], errors="coerce").dropna()
        hours = pd.to_numeric(grp["Time (hours)"], errors="coerce").dropna()

        if len(eff) < config.MIN_ROWS:
            continue

        eff_first = eff.iloc[:3].mean()
        eff_last  = eff.iloc[-3:].mean()
        eff_drop  = eff_first - eff_last
        max_hours = hours.max()

        if max_hours < 50:
            run_class = "short"
        elif eff_drop >= config.DEGRADATION_THRESHOLD_PCT:
            run_class = "degrading"
        elif max_hours >= config.STABLE_MIN_HOURS and abs(eff_drop) < config.DEGRADATION_THRESHOLD_PCT:
            run_class = "stable"
        else:
            run_class = "inconclusive"

        records.append({
            "run_id":        run_id,
            "station_id":    grp["_meta_station_id"].iloc[0],
            "run_index":     grp["_run_index"].iloc[0] if "_run_index" in grp.columns else 0,
            "run_start_date": grp["_run_start_date"].iloc[0] if "_run_start_date" in grp.columns else None,
            "project":       grp["_meta_project"].iloc[0],
            "cabinet":       grp["_meta_cabinet"].iloc[0],
            "n_cells":       grp["_meta_n_cells"].iloc[0],
            "gdl":           grp["_meta_gdl"].iloc[0],
            "operator":      grp["_meta_operator"].iloc[0],
            "run_max_hours": round(max_hours, 1),
            "n_samples":     len(grp),
            "eff_first":     round(eff_first, 1),
            "eff_last":      round(eff_last, 1),
            "eff_drop_pct":  round(eff_drop, 1),
            "run_class":     run_class,
        })

    summary = pd.DataFrame(records).sort_values("run_max_hours", ascending=False)
    return summary


# ── Feature matrix ──────────────────────────────────────────────────────────

def build_feature_matrix(df):
    """
    Return a cleaned numeric DataFrame of feature columns + target.
    Adds engineered features.
    """
    meta_cols  = [c for c in df.columns if c.startswith("_meta_")]
    skip_cols  = NON_FEATURE_COLS | set(meta_cols)
    feat_cols  = [c for c in df.columns if c not in skip_cols]

    numeric = df[feat_cols].copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    # Drop columns with <30% coverage
    thresh = int(0.3 * len(numeric))
    numeric = numeric.dropna(axis=1, thresh=thresh)

    # Add metadata back for grouping
    numeric["_meta_station_id"] = df["_meta_station_id"].values
    numeric["_meta_project"]    = df["_meta_project"].values

    return numeric


def _add_engineered_features(df, feat_matrix):
    """Add computed features that might capture physics."""
    # Cell voltage efficiency proxy: V per cell
    if "Average V (V)" in feat_matrix.columns and "_meta_n_cells" in df.columns:
        n = pd.to_numeric(df["_meta_n_cells"], errors="coerce")
        feat_matrix["v_per_cell"] = pd.to_numeric(
            feat_matrix.get("Average V (V)"), errors="coerce"
        ) / n

    # HFR ratio (membrane degradation indicator)
    if "HFR at 1 KHz (ohm)" in feat_matrix.columns and "HFR at 10  KHz (ohm)" in feat_matrix.columns:
        feat_matrix["hfr_ratio"] = (
            pd.to_numeric(feat_matrix["HFR at 1 KHz (ohm)"], errors="coerce") /
            pd.to_numeric(feat_matrix["HFR at 10  KHz (ohm)"], errors="coerce")
        )

    return feat_matrix


# ── Correlation analysis ────────────────────────────────────────────────────

def correlate_with_target(feat_matrix):
    """
    Compute Pearson and Spearman correlations of every feature vs TARGET_COL.
    Returns a ranked DataFrame.
    """
    if config.TARGET_COL not in feat_matrix.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found in data.")

    target = feat_matrix[config.TARGET_COL].astype(float)
    results = []

    for col in feat_matrix.columns:
        if col in (config.TARGET_COL, "_meta_station_id", "_meta_project"):
            continue
        series = feat_matrix[col].astype(float)
        valid  = target.notna() & series.notna()
        n      = valid.sum()
        if n < 10:
            continue

        r_p, p_p = stats.pearsonr(target[valid], series[valid])
        r_s, p_s = stats.spearmanr(target[valid], series[valid])

        results.append({
            "feature":       col,
            "n":             n,
            "pearson_r":     round(r_p, 3),
            "pearson_p":     round(p_p, 4),
            "spearman_r":    round(r_s, 3),
            "spearman_p":    round(p_s, 4),
            "abs_pearson":   abs(r_p),
        })

    corr_df = pd.DataFrame(results).sort_values("abs_pearson", ascending=False)
    return corr_df


def stable_vs_degrading_comparison(df, run_summary):
    """
    Compare mean operating conditions between stable and degrading runs.
    Returns a DataFrame with mean values and significance test per feature.
    """
    stable_ids    = set(run_summary[run_summary.run_class == "stable"].run_id)
    degrading_ids = set(run_summary[run_summary.run_class == "degrading"].run_id)

    group_col = "_run_id" if "_run_id" in df.columns else "_meta_station_id"
    stable_df    = df[df[group_col].isin(stable_ids)]
    degrading_df = df[df[group_col].isin(degrading_ids)]

    if stable_df.empty or degrading_df.empty:
        logger.warning("Not enough stable or degrading runs for comparison.")
        return pd.DataFrame()

    feat_matrix = build_feature_matrix(df)
    feat_cols = [c for c in feat_matrix.columns
                 if c not in ("_meta_station_id", "_meta_project")]

    records = []
    for col in feat_cols:
        s = pd.to_numeric(stable_df[col], errors="coerce").dropna()
        d = pd.to_numeric(degrading_df[col], errors="coerce").dropna()
        if len(s) < 5 or len(d) < 5:
            continue
        _, p = stats.mannwhitneyu(s, d, alternative="two-sided")
        records.append({
            "feature":        col,
            "mean_stable":    round(s.mean(), 3),
            "mean_degrading": round(d.mean(), 3),
            "diff":           round(s.mean() - d.mean(), 3),
            "p_value":        round(p, 4),
            "significant":    p < 0.05,
        })

    comp = pd.DataFrame(records).sort_values("p_value")
    return comp


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_efficiency_trajectories(df, run_summary):
    """One subplot per station showing efficiency over time, coloured by class."""
    colour_map = {"stable": "green", "degrading": "red",
                  "inconclusive": "steelblue", "short": "grey"}

    run_ids = run_summary.run_id.tolist()
    n = len(run_ids)
    if n == 0:
        return

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    group_col = "_run_id" if "_run_id" in df.columns else "_meta_station_id"

    for idx, rid in enumerate(run_ids):
        ax = axes[idx // ncols][idx % ncols]
        grp = df[df[group_col] == rid].sort_values("Time (hours)")
        cls = run_summary[run_summary.run_id == rid].run_class.values[0]
        sid = run_summary[run_summary.run_id == rid].station_id.values[0]
        colour = colour_map.get(cls, "grey")

        x = pd.to_numeric(grp["Time (hours)"], errors="coerce")
        y = pd.to_numeric(grp[config.TARGET_COL], errors="coerce")
        ax.plot(x, y, "o-", color=colour, markersize=4, linewidth=1.2)
        ax.set_title(rid, fontsize=7, fontweight="bold")
        ax.set_xlabel("Hours", fontsize=7)
        ax.set_ylabel("Efficiency (%)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Faradaic Efficiency Trajectories by Station", fontsize=11, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "efficiency_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_correlation_heatmap(corr_df, top_n=20):
    """Bar chart of top feature correlations with efficiency."""
    top = corr_df.head(top_n).copy()
    colours = ["#2ecc71" if r > 0 else "#e74c3c" for r in top.pearson_r]

    fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 1.5))
    bars = ax.barh(top.feature[::-1], top.pearson_r[::-1], color=colours[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with Efficiency (%)", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Correlations with Faradaic Efficiency", fontsize=11)
    ax.tick_params(labelsize=8)

    path = OUTPUT_DIR / "correlation_ranking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_scatter_top_features(df, corr_df, top_n=6):
    """Scatter plots of the top correlated features vs efficiency."""
    top_feats = corr_df.head(top_n).feature.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    stations = df["_meta_station_id"].unique()
    palette = {s: c for s, c in zip(stations, cm.tab20.colors)}
    group_col = "_run_id" if "_run_id" in df.columns else "_meta_station_id"

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        for sid, grp in df.groupby("_meta_station_id"):
            x = pd.to_numeric(grp[feat], errors="coerce")
            y = pd.to_numeric(grp[config.TARGET_COL], errors="coerce")
            valid = x.notna() & y.notna()
            ax.scatter(x[valid], y[valid], s=18, alpha=0.6,
                       color=palette.get(sid, "grey"), label=sid)

        r = corr_df[corr_df.feature == feat].pearson_r.values[0]
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Efficiency (%)", fontsize=8)
        ax.set_title(f"r = {r:+.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=palette.get(s, "grey"), markersize=6, label=s)
               for s in stations]
    fig.legend(handles=handles, loc="lower right", fontsize=7,
               ncol=max(1, len(stations) // 8), title="Station")
    fig.suptitle("Top Feature Scatter Plots vs Faradaic Efficiency", fontsize=11)
    plt.tight_layout()

    path = OUTPUT_DIR / "scatter_top_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ── Main analysis function ───────────────────────────────────────────────────

def run_analysis(df):
    """
    Run the full analysis pipeline.
    Returns a dict of result DataFrames and plot paths.
    """
    logger.info("Classifying runs ...")
    run_summary = classify_runs(df)

    logger.info("Building feature matrix ...")
    feat_matrix = build_feature_matrix(df)
    feat_matrix = _add_engineered_features(df, feat_matrix)

    logger.info("Computing correlations ...")
    corr_df = correlate_with_target(feat_matrix)

    logger.info("Comparing stable vs degrading ...")
    comparison = stable_vs_degrading_comparison(df, run_summary)

    logger.info("Generating plots ...")
    p1 = plot_efficiency_trajectories(df, run_summary)
    p2 = plot_correlation_heatmap(corr_df)
    p3 = plot_scatter_top_features(df, corr_df)

    return {
        "run_summary":   run_summary,
        "correlations":  corr_df,
        "comparison":    comparison,
        "plot_trajectories":    str(p1) if p1 else None,
        "plot_correlations":    str(p2) if p2 else None,
        "plot_scatters":        str(p3) if p3 else None,
    }
