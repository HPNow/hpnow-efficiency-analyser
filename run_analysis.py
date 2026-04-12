"""
Main entry point for the HPNow Sheets analysis.

Usage:
    python run_analysis.py

On first run you will be prompted to log in via your browser (Google OAuth2).
A token.json file is saved so subsequent runs are silent.

Outputs written to ./output/:
  - report.md
  - efficiency_trajectories.png
  - correlation_ranking.png
  - scatter_top_features.png
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analysis.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # ── Check credentials file ──────────────────────────────────────────
    if not Path("service_account.json").exists():
        print("\n" + "=" * 60)
        print("ERROR: service_account.json not found.")
        print("Save your GCP service account key as 'service_account.json' in this folder.")
        print("=" * 60 + "\n")
        sys.exit(1)

    # ── Fetch data ──────────────────────────────────────────────────────
    logger.info("Fetching Google Sheets data ...")
    from fetch_sheets import fetch_all_tabs
    df = fetch_all_tabs()

    logger.info(
        f"Loaded {len(df)} rows from {df['_meta_station_id'].nunique()} stations. "
        f"Columns: {df.shape[1]}"
    )

    # ── Run analysis ────────────────────────────────────────────────────
    logger.info("Running analysis ...")
    from analyze import run_analysis
    results = run_analysis(df)

    run_summary = results["run_summary"]
    corr_df     = results["correlations"]

    print("\n" + "=" * 60)
    print("RUN CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(run_summary[["station_id", "run_max_hours", "eff_first",
                        "eff_last", "eff_drop_pct", "run_class"]].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"TOP 10 CORRELATES WITH {__import__('config').TARGET_COL}")
    print("=" * 60)
    print(corr_df[["feature", "pearson_r", "pearson_p", "n"]].head(10).to_string(index=False))

    # ── Write report ────────────────────────────────────────────────────
    logger.info("Generating report ...")
    from report import generate_report
    report_path = generate_report(results, df)

    print(f"\nReport written to: {report_path.resolve()}")
    print("Plots written to:  output/")
    print("\nDone.")


if __name__ == "__main__":
    main()
