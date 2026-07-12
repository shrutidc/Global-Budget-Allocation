"""Loading helpers for the Global Budget dataset.

All paths are resolved relative to the project root so the functions work
whether they are called from a notebook, a script, or the package itself.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

# Project layout: <root>/src/globalbudget/data_loader.py  ->  root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MASTER_CSV = RAW_DIR / "Master_Global_Budgets_Historical.csv"
INDIVIDUAL_DIR = RAW_DIR / "individual_countries"

# The nine spending categories (column prefixes).
CATEGORIES = [
    "Defense",
    "Education",
    "Health",
    "Interest_Payments",
    "Infrastructure",
    "Agriculture",
    "State_Transfers",
    "Social_Welfare",
    "Administration_and_Others",
]

PCT_COLS = [f"{c}_Percentage" for c in CATEGORIES]
AMOUNT_COLS = [f"{c}_Amount_Billions_USD" for c in CATEGORIES]
TOTAL_COL = "Total_Budget_Billions_USD"


@lru_cache(maxsize=1)
def load_master(path: str | Path | None = None) -> pd.DataFrame:
    """Load the wide master table (one row per country-year).

    Cached so repeated calls in a notebook are cheap. Pass an explicit ``path``
    to bypass the default location (also bypasses the cache key collision by
    virtue of the argument).
    """
    csv_path = Path(path) if path is not None else MASTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Master CSV not found at {csv_path}. Expected raw data under {RAW_DIR}."
        )
    df = pd.read_csv(csv_path)
    df["Country"] = df["Country"].astype("string")
    df["Year"] = df["Year"].astype(int)
    return df


def list_countries() -> list[str]:
    """Return the sorted list of countries present in the master table."""
    return sorted(load_master()["Country"].unique().tolist())


def load_country(country: str) -> pd.DataFrame:
    """Return the master rows for a single country, sorted by year.

    ``country`` is matched case-insensitively against the master table.
    """
    master = load_master()
    mask = master["Country"].str.lower() == country.lower()
    sub = master.loc[mask].sort_values("Year").reset_index(drop=True)
    if sub.empty:
        raise KeyError(
            f"Country '{country}' not found. Available: {', '.join(list_countries())}"
        )
    return sub


def load_processed(name: str = "budgets_long") -> pd.DataFrame:
    """Load a processed artifact produced by ``scripts/build_processed_data.py``.

    Prefers parquet (fast, typed) and falls back to CSV.
    """
    parquet = PROCESSED_DIR / f"{name}.parquet"
    csv = PROCESSED_DIR / f"{name}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No processed file '{name}' found under {PROCESSED_DIR}. "
        "Run `python scripts/build_processed_data.py` first."
    )
