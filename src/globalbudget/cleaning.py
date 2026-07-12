"""Validation and reshaping for the Global Budget dataset.

The raw data is already tidy-ish and clean, but these helpers make the
guarantees explicit (so downstream analysis can trust them) and provide the
long/tidy reshape that most analysis code prefers.
"""

from __future__ import annotations

import pandas as pd

from .data_loader import AMOUNT_COLS, CATEGORIES, PCT_COLS, TOTAL_COL


def validate(df: pd.DataFrame, pct_tolerance: float = 0.5) -> dict:
    """Run integrity checks and return a report dict (raises on hard failures).

    Checks performed:
      * no missing values in the modelled columns
      * no duplicate country-year rows
      * category percentages sum to ~100 (within ``pct_tolerance``)
      * amounts are non-negative
    """
    report: dict = {}

    modelled = ["Country", "Year", TOTAL_COL] + PCT_COLS + AMOUNT_COLS
    n_missing = int(df[modelled].isna().sum().sum())
    report["missing_values"] = n_missing

    dupes = int(df.duplicated(["Country", "Year"]).sum())
    report["duplicate_country_years"] = dupes

    pct_sum = df[PCT_COLS].sum(axis=1)
    off = df.loc[(pct_sum - 100).abs() > pct_tolerance]
    report["rows_pct_not_100"] = int(len(off))
    report["pct_sum_min"] = float(pct_sum.min())
    report["pct_sum_max"] = float(pct_sum.max())

    neg_amounts = int((df[AMOUNT_COLS] < 0).sum().sum())
    report["negative_amounts"] = neg_amounts

    report["n_rows"] = int(len(df))
    report["n_countries"] = int(df["Country"].nunique())
    report["year_min"] = int(df["Year"].min())
    report["year_max"] = int(df["Year"].max())

    if n_missing:
        raise ValueError(f"Data integrity failure: {n_missing} missing values.")
    if dupes:
        raise ValueError(f"Data integrity failure: {dupes} duplicate country-year rows.")

    return report


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide master table into a tidy long format.

    Output columns: ``Country, Year, Category, Percentage, Amount_Billions_USD,
    Total_Budget_Billions_USD`` — one row per country-year-category.
    """
    pct = df.melt(
        id_vars=["Country", "Year", TOTAL_COL],
        value_vars=PCT_COLS,
        var_name="Category",
        value_name="Percentage",
    )
    pct["Category"] = pct["Category"].str.replace("_Percentage", "", regex=False)

    amt = df.melt(
        id_vars=["Country", "Year"],
        value_vars=AMOUNT_COLS,
        var_name="Category",
        value_name="Amount_Billions_USD",
    )
    amt["Category"] = amt["Category"].str.replace(
        "_Amount_Billions_USD", "", regex=False
    )

    long = pct.merge(amt, on=["Country", "Year", "Category"], how="inner")
    # Stable, readable ordering.
    long["Category"] = pd.Categorical(long["Category"], categories=CATEGORIES, ordered=True)
    long = long.sort_values(["Country", "Year", "Category"]).reset_index(drop=True)
    return long[
        [
            "Country",
            "Year",
            "Category",
            "Percentage",
            "Amount_Billions_USD",
            TOTAL_COL,
        ]
    ]
