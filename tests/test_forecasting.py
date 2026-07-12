"""Tests for globalbudget.forecasting."""

import numpy as np
import pandas as pd
import pytest

from globalbudget import forecasting


@pytest.fixture
def linear_series():
    """A clean upward-trending series where drift should be near-perfect."""
    years = np.arange(1990, 2026)
    return pd.Series(2.0 * (years - 1990) + 10.0, index=years, name="synthetic")


def test_country_series_is_year_indexed(master):
    s = forecasting.country_series(master, "USA", "Total_Budget_Billions_USD")
    assert s.index.min() == 1936
    assert s.index.max() == 2026
    assert s.index.is_monotonic_increasing


def test_naive_repeats_last_value(linear_series):
    pred = forecasting.forecast_naive(linear_series, horizon=3)
    assert np.allclose(pred, linear_series.iloc[-1])


def test_drift_recovers_linear_trend(linear_series):
    # A perfectly linear series -> drift forecast should be (almost) exact.
    res = forecasting.backtest(linear_series, model="drift", test_size=5)
    assert res.mape < 1e-6
    assert res.rmse < 1e-6


def test_backtest_alignment(linear_series):
    res = forecasting.backtest(linear_series, model="ets", test_size=5)
    assert len(res.actual) == len(res.predicted) == 5
    assert len(res.test_index) == 5
    # Test window is the final 5 years.
    assert res.test_index[0] == 2021
    assert res.test_index[-1] == 2025


def test_backtest_rejects_short_series():
    short = pd.Series([1, 2, 3], index=[2000, 2001, 2002])
    with pytest.raises(ValueError):
        forecasting.backtest(short, model="ets", test_size=10)


def test_backtest_unknown_model(linear_series):
    with pytest.raises(ValueError, match="Unknown model"):
        forecasting.backtest(linear_series, model="magic", test_size=5)


def test_compare_models_returns_all(linear_series):
    table = forecasting.compare_models(linear_series, test_size=5)
    assert set(table["model"]) == set(forecasting.MODELS)
    # Sorted ascending by MAPE, so the best model is first.
    assert table["MAPE_%"].is_monotonic_increasing


def test_forecast_future_years(linear_series):
    fc = forecasting.forecast_future(linear_series, horizon=4, model="drift")
    assert len(fc) == 4
    assert list(fc.index) == [2026, 2027, 2028, 2029]
    # Continues the upward trend.
    assert fc.iloc[-1] > linear_series.iloc[-1]
