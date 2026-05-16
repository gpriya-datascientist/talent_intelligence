"""
forecaster.py
-------------
Produce Predicted vs Actual forecasts for each KPI dimension.

Models available
----------------
- LinearTrend   : OLS linear regression on time index (fast baseline)
- ExponentialSmoothing : Holt-Winters (captures trend + seasonality)
- SARIMA        : Seasonal ARIMA via statsmodels (best accuracy)

Auto-selection: tries SARIMA → ExponentialSmoothing → LinearTrend
based on data length and stationarity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # suppress statsmodels convergence warnings


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    kpi_name:    str
    dimension:   str
    model_name:  str
    history:     pd.DataFrame   # date, value (actual)
    forecast:    pd.DataFrame   # date, predicted, lower_ci, upper_ci
    metrics:     dict           # mape, rmse, mae on held-out test set
    horizon:     int            # number of periods forecast ahead


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_kpi(
    df: pd.DataFrame,
    kpi_name: str,
    dimension: str = "Overall",
    horizon: int = 12,
    test_periods: int = 8,
    model_preference: Optional[str] = None,
) -> ForecastResult:
    """
    Fit a model and produce Predicted vs Actual for one KPI/dimension.

    Parameters
    ----------
    df               : Preprocessed KPI DataFrame
    kpi_name         : Name of the KPI to forecast
    dimension        : Dimension slice
    horizon          : Periods to forecast into the future
    test_periods     : Held-out periods for backtesting metrics
    model_preference : Force a specific model ('linear', 'ets', 'sarima')

    Returns
    -------
    ForecastResult
    """
    series = _extract_series(df, kpi_name, dimension)
    if len(series) < 8:
        raise ValueError(
            f"Insufficient data for {kpi_name}/{dimension}: {len(series)} rows (need ≥8)"
        )

    model_name, fitted_model, fit_fn = _select_model(series, model_preference)
    logger.info(f"Forecasting {kpi_name}/{dimension} with {model_name} ({len(series)} obs)")

    train = series.iloc[:-test_periods]
    test  = series.iloc[-test_periods:]

    # Backtest on held-out period
    backtest_pred = fit_fn(train, len(test))
    metrics = _compute_metrics(test["value"].values, backtest_pred)

    # Refit on full series and project forward
    future_dates = _future_dates(series, horizon)
    future_pred  = fit_fn(series, horizon)

    forecast_df = pd.DataFrame({
        "date":       future_dates,
        "predicted":  future_pred,
        "lower_ci":   future_pred * 0.90,  # naive ±10% CI
        "upper_ci":   future_pred * 1.10,
    })

    # Predicted vs actual for historical window (in-sample)
    in_sample = fit_fn(series, 0, return_in_sample=True)
    history_df = series.copy()
    history_df["predicted"] = in_sample

    logger.success(
        f"Forecast complete | model={model_name} | MAPE={metrics['mape']:.2f}% | horizon={horizon}"
    )
    return ForecastResult(
        kpi_name=kpi_name,
        dimension=dimension,
        model_name=model_name,
        history=history_df,
        forecast=forecast_df,
        metrics=metrics,
        horizon=horizon,
    )


def forecast_all(
    df: pd.DataFrame,
    horizon: int = 12,
    test_periods: int = 8,
) -> list[ForecastResult]:
    """
    Forecast every (kpi_name, dimension) combination in the DataFrame.

    Returns
    -------
    List of ForecastResult, sorted by MAPE ascending.
    """
    pairs = df[["kpi_name", "dimension"]].drop_duplicates().values.tolist()
    results = []
    for kpi, dim in pairs:
        try:
            r = forecast_kpi(df, kpi, dim, horizon=horizon, test_periods=test_periods)
            results.append(r)
        except Exception as exc:
            logger.warning(f"Skipping {kpi}/{dim}: {exc}")
    results.sort(key=lambda r: r.metrics.get("mape", 999))
    return results


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _select_model(series: pd.DataFrame, preference: Optional[str]):
    """Return (model_name, model_obj, fit_fn)."""
    n = len(series)

    if preference == "linear":
        return "LinearTrend", None, _linear_fit
    if preference == "ets":
        return "ExponentialSmoothing", None, _ets_fit
    if preference == "sarima":
        return "SARIMA", None, _sarima_fit

    # Auto: need ≥24 obs for SARIMA
    if n >= 24:
        try:
            _sarima_fit(series.head(20), 4)  # quick probe
            return "SARIMA", None, _sarima_fit
        except Exception:
            pass

    if n >= 12:
        return "ExponentialSmoothing", None, _ets_fit

    return "LinearTrend", None, _linear_fit


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------

def _linear_fit(
    train: pd.DataFrame,
    horizon: int,
    return_in_sample: bool = False,
) -> np.ndarray:
    y = train["value"].values
    X = np.arange(len(y)).reshape(-1, 1)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_sc, y)

    if return_in_sample:
        return model.predict(X_sc)

    if horizon == 0:
        return model.predict(X_sc)

    X_fut = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
    return model.predict(scaler.transform(X_fut))


def _ets_fit(
    train: pd.DataFrame,
    horizon: int,
    return_in_sample: bool = False,
) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = train["value"].values
    period = _detect_seasonality(train)

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add" if len(y) >= 2 * period else None,
        seasonal_periods=period if len(y) >= 2 * period else None,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, disp=False)

    if return_in_sample:
        return fitted.fittedvalues

    if horizon == 0:
        return fitted.fittedvalues

    return fitted.forecast(horizon)


def _sarima_fit(
    train: pd.DataFrame,
    horizon: int,
    return_in_sample: bool = False,
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train["value"].values
    period = _detect_seasonality(train)

    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=100)

    if return_in_sample:
        return fitted.fittedvalues

    if horizon == 0:
        return fitted.fittedvalues

    forecast = fitted.forecast(horizon)
    return forecast


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_series(df: pd.DataFrame, kpi: str, dim: str) -> pd.DataFrame:
    mask = (df["kpi_name"] == kpi) & (df["dimension"] == dim)
    s = df[mask][["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
    return s


def _detect_seasonality(series: pd.DataFrame) -> int:
    """Infer seasonal period from date frequency."""
    if len(series) < 2:
        return 4
    delta = (series["date"].iloc[1] - series["date"].iloc[0]).days
    if delta <= 1:
        return 7     # daily → weekly season
    if delta <= 8:
        return 52    # weekly → annual
    if delta <= 35:
        return 12    # monthly → annual
    return 4         # quarterly default


def _future_dates(series: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """Generate future date index matching the series frequency."""
    freq = pd.infer_freq(series["date"]) or "W"
    last_date = series["date"].iloc[-1]
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p  = actual[mask], predicted[mask]

    if len(a) == 0:
        return {"mape": 999.0, "rmse": 999.0, "mae": 999.0}

    mape = mean_absolute_percentage_error(a, p) * 100
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = np.mean(np.abs(a - p))
    return {"mape": round(mape, 2), "rmse": round(rmse, 2), "mae": round(mae, 2)}
