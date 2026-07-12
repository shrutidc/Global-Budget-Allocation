"""Tests for globalbudget.data_loader."""

import pandas as pd
import pytest

from globalbudget import data_loader


def test_master_shape_and_columns(master):
    # 45 countries, up to 91 years each -> known row count, and the full schema.
    assert master.shape == (3654, 21)
    assert master["Country"].nunique() == 45
    assert {"Country", "Year", "Total_Budget_Billions_USD"} <= set(master.columns)
    for col in data_loader.PCT_COLS + data_loader.AMOUNT_COLS:
        assert col in master.columns


def test_year_bounds(master):
    assert master["Year"].min() == 1936
    assert master["Year"].max() == 2026


def test_list_countries_sorted_and_unique():
    countries = data_loader.list_countries()
    assert len(countries) == 45
    assert countries == sorted(countries)
    assert len(set(countries)) == len(countries)


def test_load_country_case_insensitive():
    lower = data_loader.load_country("usa")
    upper = data_loader.load_country("USA")
    pd.testing.assert_frame_equal(lower, upper)
    # Rows are sorted by year and belong to exactly one country.
    assert lower["Year"].is_monotonic_increasing
    assert set(lower["Country"]) == {"USA"}


def test_load_country_unknown_raises():
    with pytest.raises(KeyError):
        data_loader.load_country("Atlantis")


def test_load_processed_roundtrip():
    # The pipeline artifact should load and share the master's country set.
    long = data_loader.load_processed("budgets_long")
    assert {"Country", "Year", "Category"} <= set(long.columns)
    assert long["Country"].nunique() == 45
