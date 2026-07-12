"""Derived features for analysis, ML and forecasting.

These operate on the wide master table (one row per country-year) and add
columns that are convenient for modelling and comparison. Nothing here mutates
its input in place — a copy is always returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import AMOUNT_COLS, CATEGORIES, PCT_COLS, TOTAL_COL


def add_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year growth of the total budget, per country (%).

    Growth is computed within each country in year order, so gaps or differing
    start years are handled correctly.
    """
    out = df.sort_values(["Country", "Year"]).copy()
    out["Total_Budget_Growth_Pct"] = (
        out.groupby("Country", observed=True)[TOTAL_COL].pct_change() * 100
    )
    return out


def add_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add a rolling mean of the total budget per country (trailing window)."""
    out = df.sort_values(["Country", "Year"]).copy()
    col = f"Total_Budget_Roll{window}"
    out[col] = (
        out.groupby("Country", observed=True)[TOTAL_COL]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return out


def guns_vs_butter(df: pd.DataFrame) -> pd.DataFrame:
    """Add a defense-vs-social ratio and a combined social-spending share.

    * ``Social_Spend_Pct`` = Education + Health + Social_Welfare + State_Transfers
    * ``Guns_Butter_Ratio`` = Defense_Percentage / Social_Spend_Pct
    """
    out = df.copy()
    social_cols = [
        "Education_Percentage",
        "Health_Percentage",
        "Social_Welfare_Percentage",
        "State_Transfers_Percentage",
    ]
    out["Social_Spend_Pct"] = out[social_cols].sum(axis=1)
    out["Guns_Butter_Ratio"] = out["Defense_Percentage"] / out["Social_Spend_Pct"].replace(
        0, np.nan
    )
    return out


def spending_profile_matrix(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Return a country x category matrix of spending percentages.

    If ``year`` is given, use that single year; otherwise use each country's
    mean profile across all available years. Handy for clustering / similarity.
    """
    if year is not None:
        sub = df.loc[df["Year"] == year]
        if sub.empty:
            raise ValueError(f"No rows for year {year}.")
        mat = sub.set_index("Country")[PCT_COLS]
    else:
        mat = df.groupby("Country", observed=True)[PCT_COLS].mean()
    mat.columns = [c.replace("_Percentage", "") for c in mat.columns]
    return mat


def era(year: int) -> str:
    """Bucket a year into a coarse geopolitical/economic era (for grouping)."""
    if year < 1946:
        return "WWII & earlier"
    if year <= 1991:
        return "Cold War (1946-1991)"
    if year <= 2007:
        return "Post-Cold-War (1992-2007)"
    if year <= 2019:
        return "Post-GFC (2008-2019)"
    return "COVID era (2020+)"


def add_era(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical ``Era`` column derived from ``Year``."""
    out = df.copy()
    out["Era"] = out["Year"].map(era)
    return out


def build_ml_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a modelling frame with growth, rolling stats, ratios and era.

    Convenience wrapper chaining the feature helpers, used by the ML and
    forecasting notebooks.
    """
    out = add_growth_rates(df)
    out = add_rolling(out, window=5)
    out = guns_vs_butter(out)
    out = add_era(out)
    return out
