from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    asset: str,
    interval: str,
    artifacts_dir: Path,
    reports_dir: Path,
    transaction_cost_bps: float = 10.0,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    if len(frame) < 120:
        raise ValueError(
            f"Only {len(frame)} usable rows are available; at least 120 are required "
            "for walk-forward evaluation."
        )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    target_columns = [f"target_{horizon}" for horizon in range(1, 6)]
    training_frame = frame.dropna(subset=target_columns)
    if len(training_frame) < 120:
        raise ValueError(
            f"Only {len(training_frame)} rows have complete forecast targets; "
            "at least 120 are required for walk-forward evaluation."
        )
    x = training_frame[feature_columns].astype(float)
    split_count = min(5, max(2, len(training_frame) // 40))
    test_size = max(1, len(training_frame) // (split_count + 2))
    splitter = TimeSeriesSplit(n_splits=split_count, test_size=test_size, gap=5)

    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    live_forecasts: dict[str, float] = {}
    for horizon in range(1, 6):
        target_column = f"target_{horizon}"
        y = training_frame[target_column].astype(float) / training_frame["close"] - 1
        for fold, (train_indices, test_indices) in enumerate(splitter.split(x), start=1):
            model = _new_model()
            model.fit(x.iloc[train_indices], y.iloc[train_indices])
            predicted_returns = model.predict(x.iloc[test_indices])
            train_returns = y.iloc[train_indices]
            positive_count = int((train_returns > 0).sum())
            negative_count = int((train_returns < 0).sum())
            majority_direction = 1 if positive_count >= negative_count else -1
            for index, predicted_return in zip(test_indices, predicted_returns):
                base = float(training_frame["close"].iloc[index])
                actual = float(y.iloc[index])
                predicted_return = max(float(predicted_return), -0.999999)
                prediction_rows.append(
                    {
                        "timestamp": training_frame.index[index].isoformat(),
                        "fold": fold,
                        "horizon": horizon,
                        "base_close": base,
                        "actual_close": base * (1 + actual),
                        "predicted_close": base * (1 + predicted_return),
                        "actual_return_pct": actual * 100,
                        "predicted_return_pct": predicted_return * 100,
                        "majority_direction": majority_direction,
                    }
                )
        horizon_rows = [row for row in prediction_rows if row["horizon"] == horizon]
        metrics[f"n+{horizon}"] = _metrics(
            horizon_rows, interval, horizon, transaction_cost_bps
        )

        final_model = _new_model()
        final_model.fit(x, y)
        latest_x = frame[feature_columns].iloc[[-1]].astype(float)
        predicted_return = max(float(final_model.predict(latest_x)[0]), -0.999999)
        live_forecasts[f"n+{horizon}"] = float(frame["close"].iloc[-1]) * (
            1 + predicted_return
        )
        model_path = artifacts_dir / f"{_safe_asset(asset)}_{interval}_h{horizon}.joblib"
        joblib.dump(
            {
                "model": final_model,
                "feature_columns": feature_columns,
                "asset": asset,
                "interval": interval,
                "horizon": horizon,
            },
            model_path,
        )

    predictions_frame = pd.DataFrame(prediction_rows).sort_values(["timestamp", "horizon"])
    predictions_path = reports_dir / f"{_safe_asset(asset)}_{interval}_oos_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    current_price = float(frame["close"].iloc[-1])
    forecasts = {"n+0": current_price, **live_forecasts}
    forecast_path = reports_dir / f"{_safe_asset(asset)}_{interval}_forecast.json"
    forecast_path.write_text(
        json.dumps(
            {
                "asset": asset,
                "interval": interval,
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "n+0_semantics": "latest observed completed bar close (baseline, not a model prediction)",
                "forecasts": forecasts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, predictions_frame, forecasts


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=4,
        max_features=0.9,
        n_jobs=-1,
        random_state=42,
    )


def _metrics(
    rows: list[dict[str, Any]],
    interval: str,
    horizon: int = 1,
    transaction_cost_bps: float = 10.0,
) -> dict[str, float]:
    actual = np.asarray([row["actual_close"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_close"] for row in rows], dtype=float)
    actual_returns = np.asarray([row["actual_return_pct"] / 100 for row in rows], dtype=float)
    predicted_returns = np.asarray([row["predicted_return_pct"] / 100 for row in rows], dtype=float)
    persistence_predictions = np.asarray([row["base_close"] for row in rows], dtype=float)
    majority_direction = np.asarray(
        [row.get("majority_direction", 0) for row in rows], dtype=float
    )
    non_overlapping_rows = rows[::horizon]
    strategy_returns = (
        np.sign([row["predicted_return_pct"] for row in non_overlapping_rows])
        * np.asarray([row["actual_return_pct"] for row in non_overlapping_rows], dtype=float)
        / 100
        - transaction_cost_bps / 10_000
    )
    equity = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdowns = equity / peak - 1
    downside_deviation = float(np.sqrt(np.mean(np.minimum(strategy_returns, 0) ** 2)))
    periods_per_year = (24 * 365 if interval == "hourly" else 252) / horizon
    strategy_std = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    persistence_errors = actual - persistence_predictions
    actual_directions = np.sign(actual_returns)
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "persistence_mae": float(np.mean(np.abs(persistence_errors))),
        "persistence_rmse": float(np.sqrt(np.mean(persistence_errors**2))),
        "mae_improvement_vs_persistence_pct": (
            float((1 - np.mean(np.abs(actual - predicted)) / np.mean(np.abs(persistence_errors))) * 100)
            if np.mean(np.abs(persistence_errors)) > 0
            else 0.0
        ),
        "directional_accuracy": float(np.mean(actual_directions == np.sign(predicted_returns))),
        "majority_direction_accuracy": float(
            np.mean(actual_directions == majority_direction)
        ),
        "sharpe_ratio": (
            float(np.mean(strategy_returns) / strategy_std * np.sqrt(periods_per_year))
            if strategy_std > 0
            else 0.0
        ),
        "sortino_ratio": (
            float(np.mean(strategy_returns) / downside_deviation * np.sqrt(periods_per_year))
            if downside_deviation > 0
            else 0.0
        ),
        "max_drawdown": float(np.min(drawdowns)) if len(drawdowns) else 0.0,
        "oos_samples": int(len(rows)),
        "non_overlapping_trades": int(len(strategy_returns)),
        "transaction_cost_bps_round_trip": float(transaction_cost_bps),
    }


def metrics_from_predictions(
    predictions: pd.DataFrame, interval: str
) -> dict[str, dict[str, float]]:
    required = {
        "horizon",
        "base_close",
        "actual_close",
        "predicted_close",
        "actual_return_pct",
        "predicted_return_pct",
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return {}
    return {
        f"n+{int(horizon)}": _metrics(
            group.to_dict(orient="records"), interval, int(horizon)
        )
        for horizon, group in predictions.groupby("horizon", sort=True)
    }


def _safe_asset(asset: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in asset)
