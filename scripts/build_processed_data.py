#!/usr/bin/env python3
"""Build reproducible processed datasets from the raw master CSV.

Outputs (to ``data/processed/``):
  * ``budgets_wide.{parquet,csv}``  — validated wide table + a few derived features
  * ``budgets_long.{parquet,csv}``  — tidy long format (country-year-category)
  * ``country_profiles.csv``        — mean spending profile per country
  * ``validation_report.json``      — the integrity-check report

Usage:
    python scripts/build_processed_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the local package importable without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalbudget import cleaning, data_loader, features  # noqa: E402


def main() -> int:
    out_dir = data_loader.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading master table ...")
    wide = data_loader.load_master()

    print("Validating ...")
    report = cleaning.validate(wide)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("Engineering features ...")
    wide_feat = features.build_ml_frame(wide)

    print("Reshaping to long format ...")
    long = cleaning.to_long(wide)

    print("Building country profiles ...")
    profiles = features.spending_profile_matrix(wide)  # country x category means

    # Write outputs (parquet for speed/typing, CSV for portability/GitHub diffs).
    wide_feat.to_parquet(out_dir / "budgets_wide.parquet", index=False)
    wide_feat.to_csv(out_dir / "budgets_wide.csv", index=False)
    long.to_parquet(out_dir / "budgets_long.parquet", index=False)
    long.to_csv(out_dir / "budgets_long.csv", index=False)
    profiles.to_csv(out_dir / "country_profiles.csv")
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nWrote processed data to {out_dir}")
    print("  budgets_wide.parquet / .csv")
    print("  budgets_long.parquet / .csv")
    print("  country_profiles.csv")
    print("  validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
