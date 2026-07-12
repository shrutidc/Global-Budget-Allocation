"""Tests for globalbudget.cleaning."""

import numpy as np
import pytest

from globalbudget import cleaning
from globalbudget.data_loader import CATEGORIES, PCT_COLS


def test_validate_report_is_clean(master):
    report = cleaning.validate(master)
    assert report["missing_values"] == 0
    assert report["duplicate_country_years"] == 0
    assert report["rows_pct_not_100"] == 0
    assert report["negative_amounts"] == 0
    assert report["n_countries"] == 45


def test_validate_raises_on_missing(master):
    broken = master.copy()
    broken.loc[0, "Total_Budget_Billions_USD"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cleaning.validate(broken)


def test_validate_raises_on_duplicates(master):
    broken = master.copy()
    broken = broken.iloc[[0, 0]].copy()  # duplicate the same country-year
    with pytest.raises(ValueError, match="duplicate"):
        cleaning.validate(broken)


def test_to_long_shape_and_columns(master):
    long = cleaning.to_long(master)
    # One row per country-year-category.
    assert len(long) == len(master) * len(CATEGORIES)
    assert list(long.columns) == [
        "Country",
        "Year",
        "Category",
        "Percentage",
        "Amount_Billions_USD",
        "Total_Budget_Billions_USD",
    ]


def test_to_long_preserves_percentage_sums(master):
    long = cleaning.to_long(master)
    sums = long.groupby(["Country", "Year"], observed=True)["Percentage"].sum()
    assert np.allclose(sums.to_numpy(), 100.0, atol=0.5)


def test_to_long_category_values(master):
    long = cleaning.to_long(master)
    assert set(long["Category"].unique()) == set(CATEGORIES)
