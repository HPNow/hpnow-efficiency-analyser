"""
Generate a narrative Markdown analysis report from the analysis results.
"""

import datetime
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _run_distribution(run_summary):
    counts = run_summary.run_class.value_counts().to_dict()
    total  = len(run_summary)
    lines  = [f"- **{cls.capitalize()}**: {n} of {total} runs"
              for cls, n in sorted(counts.items())]
    return "\n".join(lines)


def _top_correlations_section(corr_df, n=10):
    top = corr_df.head(n)
    lines = []
    for _, row in top.iterrows():
        direction = "positively" if row.pearson_r > 0 else "negatively"
        sig = "✓" if row.pearson_p < 0.05 else "~"
        lines.append(
            f"| {sig} | **{row.feature}** | {row.pearson_r:+.3f} | "
            f"{row.spearman_r:+.3f} | {row.n} |"
        )
    header = (
        "| Sig | Feature | Pearson r | Spearman r | n |\n"
        "|-----|---------|-----------|------------|---|"
    )
    return header + "\n" + "\n".join(lines)


def _hypotheses_section(corr_df, comparison_df):
    """
    Generate ranked hypothesis bullets from correlation and comparison data.
    """
    hypotheses = []
    rank = 1

    # Strong correlators → direct hypotheses
    strong = corr_df[corr_df.abs_pearson >= 0.3].head(8)
    for _, row in strong.iterrows():
        direction = "higher" if row.pearson_r < 0 else "lower"
        hypotheses.append(
            f"**H{rank}** — `{row.feature}` correlates with efficiency "
            f"(r = {row.pearson_r:+.3f}, n={row.n}): "
            f"*{direction} {row.feature} is associated with lower efficiency.*"
        )
        rank += 1

    # Significant stable vs degrading differences
    if not comparison_df.empty:
        sig_diff = comparison_df[comparison_df.significant].head(5)
        for _, row in sig_diff.iterrows():
            direction = "higher" if row.mean_stable > row.mean_degrading else "lower"
            hypotheses.append(
                f"**H{rank}** — Stable runs show {direction} `{row.feature}` "
                f"(stable mean: {row.mean_stable:.2f}, degrading mean: {row.mean_degrading:.2f}, "
                f"p={row.p_value:.4f})."
            )
            rank += 1

    if not hypotheses:
        return "_Insufficient data to generate hypotheses. More runs needed._"

    return "\n\n".join(hypotheses)


def _stable_run_profile(run_summary, df):
    stable = run_summary[run_summary.run_class == "stable"]
    if stable.empty:
        return "_No stable runs identified with current thresholds._"

    lines = [
        f"- **{row.run_id}** (station {row.station_id}): "
        f"{row.run_max_hours:.0f}h, "
        f"efficiency {row.eff_first:.1f}% → {row.eff_last:.1f}% "
        f"(Δ {-row.eff_drop_pct:+.1f}%), "
        f"project: {row.project or 'N/A'}, cabinet: {row.cabinet or 'N/A'}"
        for _, row in stable.iterrows()
    ]
    return "\n".join(lines)


def _degrading_run_profile(run_summary):
    deg = run_summary[run_summary.run_class == "degrading"]
    if deg.empty:
        return "_No clearly degrading runs identified with current thresholds._"

    lines = [
        f"- **{row.run_id}** (station {row.station_id}): "
        f"{row.run_max_hours:.0f}h, "
        f"efficiency {row.eff_first:.1f}% → {row.eff_last:.1f}% "
        f"(Δ {-row.eff_drop_pct:+.1f}%), "
        f"project: {row.project or 'N/A'}"
        for _, row in deg.iterrows()
    ]
    return "\n".join(lines)


def _run_summary_table(run_summary):
    cols = ["run_id", "station_id", "run_start_date", "run_max_hours",
            "n_samples", "eff_first", "eff_last", "eff_drop_pct", "run_class", "project"]
    cols = [c for c in cols if c in run_summary.columns]
    sub  = run_summary[cols].copy()
    sub.columns = [c.replace("_", " ").title() for c in sub.columns]

    # Markdown table
    header = "| " + " | ".join(sub.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |"
              for row in sub.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def generate_report(results, df):
    """
    Write the Markdown report to output/report.md and return the path.
    """
    run_summary   = results["run_summary"]
    corr_df       = results["correlations"]
    comparison_df = results.get("comparison")
    if comparison_df is None:
        import pandas as _pd
        comparison_df = _pd.DataFrame()

    n_stations = df["_meta_station_id"].nunique()
    n_rows     = len(df)
    date_str   = datetime.date.today().isoformat()

    report = f"""# HPNow Faradaic Efficiency Analysis
_Generated: {date_str}_

---

## 1. Overview

This report analyses **{n_rows} daily measurement rows** from **{n_stations} test stations**
sourced from the HPNow Google Sheet. The target variable is **{config.TARGET_COL}**.

### Run Classification

A run is classified as:
- **Degrading**: efficiency dropped ≥ {config.DEGRADATION_THRESHOLD_PCT}% over the test
- **Stable**: ran ≥ {config.STABLE_MIN_HOURS}h with < {config.DEGRADATION_THRESHOLD_PCT}% efficiency drop
- **Short**: fewer than 50 hours total
- **Inconclusive**: other

{_run_distribution(run_summary)}

---

## 2. All Runs Summary

{_run_summary_table(run_summary)}

---

## 3. Stable Run Profiles

These are the long-running tests that maintained high faradaic efficiency — the
"gold standard" operating conditions we want to understand and replicate.

{_stable_run_profile(run_summary, df)}

---

## 4. Degrading Run Profiles

{_degrading_run_profile(run_summary)}

---

## 5. Feature Correlations with {config.TARGET_COL}

Top features ranked by absolute Pearson correlation. ✓ = statistically significant (p < 0.05).

{_top_correlations_section(corr_df)}

![Correlation ranking]({results.get('plot_correlations', '')})

---

## 6. Scatter Plots — Top Features vs Efficiency

![Scatter plots]({results.get('plot_scatters', '')})

---

## 7. Efficiency Trajectories by Station

![Efficiency trajectories]({results.get('plot_trajectories', '')})

---

## 8. Ranked Hypotheses

The following hypotheses are ranked by statistical strength. These are starting
points for further investigation — not conclusions.

{_hypotheses_section(corr_df, comparison_df)}

---

## 9. Stable vs Degrading Comparison

Features with statistically significant differences between stable and degrading runs
(Mann-Whitney U test, p < 0.05):

"""

    if not comparison_df.empty:
        sig = comparison_df[comparison_df.significant].copy()
        if not sig.empty:
            header = "| Feature | Mean (stable) | Mean (degrading) | Diff | p-value |\n"
            sep    = "|---------|---------------|-----------------|------|---------|"
            rows   = "\n".join(
                f"| {r.feature} | {r.mean_stable:.3f} | {r.mean_degrading:.3f} | "
                f"{r['diff']:+.3f} | {r.p_value:.4f} |"
                for _, r in sig.iterrows()
            )
            report += header + sep + "\n" + rows
        else:
            report += "_No statistically significant differences found (may need more runs)._"
    else:
        report += "_Comparison not available (need at least one stable and one degrading run)._"

    report += """

---

## 10. Limitations & Next Steps

- This analysis uses **daily-sampled data only**. Adding 30-second SQL sensor data
  will reveal transient events (pressure spikes, temperature excursions) that may
  precede efficiency decline.
- Schema drift across early vs late tabs means some columns (HFR, temperatures)
  are only present in a subset of runs — treat those correlations with caution.
- With more runs, a **logistic regression** or **random forest** on run_class would
  give more reliable feature importance rankings.
"""

    path = OUTPUT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    logger.info(f"Report written to {path}")
    return path
