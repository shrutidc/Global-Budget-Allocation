"""Time-series forecasting helpers: baselines, ETS/ARIMA, and backtesting.

Everything here works on a single univariate series indexed by integer year.
The forecasting notebook loops these over countries/categories.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def country_series(df: pd.DataFrame, country: str, column: str) -> pd.Series:
    """Extract a year-indexed series for one country and one numeric column."""
    sub = df[df["Country"] == country].sort_values("Year")
    s = pd.Series(sub[column].to_numpy(), index=sub["Year"].to_numpy(), name=column)
    return s


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def forecast_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Random-walk baseline: repeat the last observed value."""
    return np.repeat(train.iloc[-1], horizon)


def forecast_drift(train: pd.Series, horizon: int) -> np.ndarray:
    """Drift baseline: extrapolate the average per-step change."""
    if len(train) < 2:
        return forecast_naive(train, horizon)
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    return train.iloc[-1] + slope * np.arange(1, horizon + 1)


def forecast_ets(train: pd.Series, horizon: int) -> np.ndarray:
    """Exponential smoothing (additive trend). Falls back to drift on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train.to_numpy(dtype=float), trend="add", seasonal=None
            ).fit()
        return np.asarray(model.forecast(horizon), dtype=float)
    except Exception:
        return forecast_drift(train, horizon)


def forecast_arima(train: pd.Series, horizon: int, order=(1, 1, 1)) -> np.ndarray:
    """ARIMA forecast (default order (1,1,1)). Falls back to drift on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train.to_numpy(dtype=float), order=order).fit()
        return np.asarray(model.forecast(horizon), dtype=float)
    except Exception:
        return forecast_drift(train, horizon)


MODELS = {
    "naive": forecast_naive,
    "drift": forecast_drift,
    "ets": forecast_ets,
    "arima": forecast_arima,
}


@dataclass
class BacktestResult:
    model: str
    mape: float
    rmse: float
    actual: np.ndarray
    predicted: np.ndarray
    test_index: np.ndarray


def backtest(series: pd.Series, model: str = "ets", test_size: int = 10) -> BacktestResult:
    """Hold out the last ``test_size`` points and score a one-shot forecast.

    Returns a :class:`BacktestResult` with MAPE/RMSE and the aligned arrays so
    the caller can plot the fit.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODELS)}.")
    if len(series) <= test_size + 2:
        raise ValueError("Series too short for the requested test_size.")

    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    pred = MODELS[model](train, test_size)
    return BacktestResult(
        model=model,
        mape=_mape(test.to_numpy(), pred),
        rmse=_rmse(test.to_numpy(), pred),
        actual=test.to_numpy(),
        predicted=np.asarray(pred, dtype=float),
        test_index=test.index.to_numpy(),
    )


def compare_models(series: pd.Series, test_size: int = 10) -> pd.DataFrame:
    """Backtest every model on a series and return a sorted score table."""
    rows = []
    for name in MODELS:
        try:
            r = backtest(series, model=name, test_size=test_size)
            rows.append({"model": name, "MAPE_%": r.mape, "RMSE": r.rmse})
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({"model": name, "MAPE_%": np.nan, "RMSE": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True)


def forecast_future(series: pd.Series, horizon: int, model: str = "ets") -> pd.Series:
    """Fit on the full series and forecast ``horizon`` future years."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODELS)}.")
    preds = MODELS[model](series, horizon)
    future_years = np.arange(series.index[-1] + 1, series.index[-1] + 1 + horizon)
    return pd.Series(np.asarray(preds, dtype=float), index=future_years, name=f"{series.name}_forecast")
