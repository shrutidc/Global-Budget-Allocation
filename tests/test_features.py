"""Tests for globalbudget.features."""

import numpy as np

from globalbudget import features
from globalbudget.data_loader import CATEGORIES


def test_add_growth_rates_first_year_is_nan(master):
    out = features.add_growth_rates(master)
    assert "Total_Budget_Growth_Pct" in out.columns
    # The earliest year of each country has no prior year -> NaN growth.
    first_rows = out.sort_values("Year").groupby("Country").head(1)
    assert first_rows["Total_Budget_Growth_Pct"].isna().all()


def test_add_rolling_matches_expanding_at_start(master):
    out = features.add_rolling(master, window=5)
    usa = out[out["Country"] == "USA"].sort_values("Year")
    # With min_periods=1, the first rolling value equals the first raw value.
    assert np.isclose(
        usa["Total_Budget_Roll5"].iloc[0], usa["Total_Budget_Billions_USD"].iloc[0]
    )


def test_guns_vs_butter_columns(master):
    out = features.guns_vs_butter(master)
    assert "Social_Spend_Pct" in out.columns
    assert "Guns_Butter_Ratio" in out.columns
    # Social share is a sum of four category shares -> bounded within [0, 100].
    assert (out["Social_Spend_Pct"] >= 0).all()
    assert (out["Social_Spend_Pct"] <= 100).all()


def test_era_buckets():
    assert features.era(1960) == "Cold War (1946-1991)"
    assert features.era(2000) == "Post-Cold-War (1992-2007)"
    assert features.era(2010) == "Post-GFC (2008-2019)"
    assert features.era(2021) == "COVID era (2020+)"


def test_spending_profile_matrix_shape(master):
    profiles = features.spending_profile_matrix(master)
    assert profiles.shape == (45, len(CATEGORIES))
    # Each country's mean profile should roughly sum to 100%.
    assert np.allclose(profiles.sum(axis=1).to_numpy(), 100.0, atol=1.0)


def test_build_ml_frame_adds_all_features(master):
    out = features.build_ml_frame(master)
    for col in [
        "Total_Budget_Growth_Pct",
        "Total_Budget_Roll5",
        "Social_Spend_Pct",
        "Guns_Butter_Ratio",
        "Era",
    ]:
        assert col in out.columns
    # No rows dropped.
    assert len(out) == len(master)
