from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cwt_agent.sentiment_model import load_sentiment_classifier


def build_feature_frame(
    prices: pd.DataFrame,
    tweets: pd.DataFrame,
    interval: str,
    sentiment_model_path: Path | None = None,
    precomputed_sentiment: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if prices.empty:
        raise ValueError("No price history is stored. Run the ingest command first.")
    if interval not in {"hourly", "daily"}:
        raise ValueError("interval must be either 'hourly' or 'daily'.")
    frequency = "h" if interval == "hourly" else "D"
    bars = prices.copy()
    bars.index = pd.to_datetime(bars.index, utc=True).floor(frequency)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["close"])

    sentiment = _aggregate_sentiment(tweets, frequency, sentiment_model_path)
    if precomputed_sentiment is not None:
        sentiment = _combine_sentiment(
            sentiment, precomputed_sentiment, frequency
        )
    frame = bars.join(sentiment, how="left")
    frame["sentiment_available"] = frame["post_volume"].notna().astype(float)
    frame[["sentiment_mean", "positive_share", "post_volume"]] = (
        frame[["sentiment_mean", "positive_share", "post_volume"]]
        .fillna(0.0)
    )

    frame["return_1"] = frame["close"].pct_change(1)
    frame["return_2"] = frame["close"].pct_change(2)
    frame["return_6"] = frame["close"].pct_change(6)
    frame["return_24"] = frame["close"].pct_change(24)
    frame["volatility_6"] = frame["return_1"].rolling(6).std()
    frame["volatility_24"] = frame["return_1"].rolling(24).std()
    frame["close_vs_sma_24"] = frame["close"] / frame["close"].rolling(24).mean() - 1
    frame["sentiment_rolling_6"] = (
        frame["sentiment_mean"]
        .where(frame["sentiment_available"] > 0)
        .rolling(6, min_periods=1)
        .mean()
        .fillna(0.0)
    )
    frame["volume_rolling_6"] = frame["post_volume"].rolling(6, min_periods=1).mean()

    feature_columns = [
        "sentiment_mean",
        "positive_share",
        "sentiment_available",
        "post_volume",
        "return_1",
        "return_2",
        "return_6",
        "return_24",
        "volatility_6",
        "volatility_24",
        "close_vs_sma_24",
        "sentiment_rolling_6",
        "volume_rolling_6",
    ]
    for horizon in range(1, 6):
        frame[f"target_{horizon}"] = frame["close"].shift(-horizon)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=feature_columns)
    return frame, feature_columns


def sentiment_coverage(
    prices: pd.DataFrame,
    tweets: pd.DataFrame,
    interval: str,
) -> dict[str, int | float | str | None]:
    if prices.empty:
        raise ValueError("No price history is available for coverage diagnostics.")
    frequency = "h" if interval == "hourly" else "D"
    price_index = pd.to_datetime(prices.index, utc=True).floor(frequency)
    total_bars = int(price_index.nunique())
    if tweets.empty:
        return {
            "post_count": 0,
            "active_periods": 0,
            "bar_count": total_bars,
            "bar_coverage_pct": 0.0,
            "active_days": 0,
            "first_post_utc": None,
            "last_post_utc": None,
        }
    timestamps = pd.to_datetime(tweets["timestamp"], utc=True, errors="coerce").dropna()
    start = pd.Timestamp(price_index.min())
    end = pd.Timestamp(price_index.max()) + (
        pd.Timedelta(hours=1) if interval == "hourly" else pd.Timedelta(days=1)
    )
    timestamps = timestamps[(timestamps >= start) & (timestamps < end)]
    active_periods = int(timestamps.dt.floor(frequency).nunique())
    return {
        "post_count": int(len(timestamps)),
        "active_periods": active_periods,
        "bar_count": total_bars,
        "bar_coverage_pct": active_periods / total_bars * 100 if total_bars else 0.0,
        "active_days": int(timestamps.dt.floor("D").nunique()),
        "first_post_utc": timestamps.min().isoformat() if not timestamps.empty else None,
        "last_post_utc": timestamps.max().isoformat() if not timestamps.empty else None,
    }


def _aggregate_sentiment(
    tweets: pd.DataFrame,
    frequency: str,
    sentiment_model_path: Path | None = None,
) -> pd.DataFrame:
    columns = ["sentiment_mean", "positive_share", "post_volume"]
    if tweets.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"))
    if not {"timestamp", "text"}.issubset(tweets.columns):
        raise ValueError("Tweet data must include timestamp and text columns.")
    posts = tweets.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    posts = posts.dropna(subset=["timestamp"])
    if posts.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"))
    posts["period"] = posts["timestamp"].dt.floor(frequency)
    texts = posts["text"].fillna("").astype(str).tolist()
    if sentiment_model_path is None:
        analyzer = SentimentIntensityAnalyzer()
        posts["sentiment"] = [
            analyzer.polarity_scores(text)["compound"] for text in texts
        ]
    else:
        if not sentiment_model_path.is_file():
            raise FileNotFoundError(
                f"Sentiment model does not exist: {sentiment_model_path}"
            )
        posts["sentiment"] = load_sentiment_classifier(sentiment_model_path).score(texts)
    posts["is_positive"] = (posts["sentiment"] > 0).astype(float)
    grouped = posts.groupby("period", sort=True)
    result = grouped.agg(
        sentiment_mean=("sentiment", "mean"),
        positive_share=("is_positive", "mean"),
        post_volume=("sentiment", "size"),
    )
    return result[columns]


def _combine_sentiment(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    columns = ["sentiment_mean", "positive_share", "post_volume"]
    weighted_frames = []
    for source in (current, historical):
        if source.empty:
            continue
        if not set(columns).issubset(source.columns):
            raise ValueError(
                "Precomputed sentiment must include sentiment_mean, positive_share, "
                "and post_volume."
            )
        frame = source[columns].copy()
        frame.index = pd.to_datetime(frame.index, utc=True).floor(frequency)
        frame["post_volume"] = pd.to_numeric(frame["post_volume"], errors="coerce")
        frame["sentiment_mean"] = pd.to_numeric(
            frame["sentiment_mean"], errors="coerce"
        )
        frame["positive_share"] = pd.to_numeric(
            frame["positive_share"], errors="coerce"
        )
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Precomputed sentiment values must all be finite.")
        if (
            (frame["post_volume"] < 0).any()
            or not frame["sentiment_mean"].between(-1, 1).all()
            or not frame["positive_share"].between(0, 1).all()
        ):
            raise ValueError("Precomputed sentiment values are outside valid ranges.")
        frame["sentiment_sum"] = frame["sentiment_mean"] * frame["post_volume"]
        frame["positive_sum"] = frame["positive_share"] * frame["post_volume"]
        weighted_frames.append(frame[["sentiment_sum", "positive_sum", "post_volume"]])
    if not weighted_frames:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"))
    combined = pd.concat(weighted_frames).groupby(level=0, sort=True).sum()
    combined = combined[combined["post_volume"] > 0]
    return pd.DataFrame(
        {
            "sentiment_mean": combined["sentiment_sum"] / combined["post_volume"],
            "positive_share": combined["positive_sum"] / combined["post_volume"],
            "post_volume": combined["post_volume"],
        },
        index=combined.index,
    )
