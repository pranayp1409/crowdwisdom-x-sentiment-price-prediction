from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def write_evaluation_report(
    asset: str,
    interval: str,
    metrics: dict[str, Any],
    forecasts: dict[str, float],
    path: Path,
    data_quality: dict[str, int | float | str | None] | None = None,
) -> None:
    lines = [
        f"# {asset} sentiment + price evaluation",
        "",
        f"- Bar interval: {interval}",
        "- Validation: expanding-window walk-forward splits with a five-bar purge gap.",
        "- Inputs: X sentiment and price-derived features only; news is excluded from model inputs.",
        "- n+0 is the latest observed completed bar close; it is a baseline, not a model prediction.",
        "- Trading metrics are historical estimates; the stated round-trip cost is assumed, and actual fees/slippage may differ.",
        "- Directional strategy metrics use non-overlapping horizon-length trades and the configured round-trip cost.",
        "",
    ]
    if data_quality is not None:
        lines.extend(
            [
                "## X data coverage",
                "",
                f"- Posts in price window: {data_quality['post_count']}",
                f"- Active periods: {data_quality['active_periods']} / {data_quality['bar_count']} "
                f"({data_quality['bar_coverage_pct']:.2f}%)",
                f"- Active calendar days: {data_quality['active_days']}",
                f"- First post: {data_quality['first_post_utc'] or 'none'}",
                f"- Last post: {data_quality['last_post_utc'] or 'none'}",
                "",
            ]
        )
        if data_quality["post_count"] < 100 or data_quality["bar_coverage_pct"] < 10:
            lines.extend(
                [
                    "> **Warning:** X coverage is sparse. Treat sentiment-dependent results "
                    "as preliminary; improve date coverage before drawing conclusions.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Latest price forecast",
            "",
            "| Horizon | Price |",
            "|---|---:|",
        ]
    )
    for horizon, price in forecasts.items():
        lines.append(f"| {horizon} | {price:.6g} |")
    lines.extend(
        [
            "",
            "## Out-of-sample metrics by horizon",
            "",
            "| Horizon | MAE | Persistence MAE | MAE change vs baseline | Direction accuracy | Majority baseline | Sharpe | Sortino | Max drawdown | OOS rows | Trades |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, values in metrics.items():
        lines.append(
            f"| {horizon} | {values['mae']:.6g} | {values['persistence_mae']:.6g} | "
            f"{values['mae_improvement_vs_persistence_pct']:+.2f}% | "
            f"{values['directional_accuracy']:.3f} | {values['majority_direction_accuracy']:.3f} | "
            f"{values['sharpe_ratio']:.3f} | {values['sortino_ratio']:.3f} | "
            f"{values['max_drawdown']:.3%} | {values['oos_samples']} | "
            f"{values['non_overlapping_trades']} |"
        )
    lines.extend(
        [
            "",
            f"Round-trip transaction cost assumed: {metrics[next(iter(metrics))]['transaction_cost_bps_round_trip']:.1f} bps."
            if metrics
            else "No metric rows available.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sentiment_summary(
    asset: str,
    prices: pd.DataFrame,
    tweets: pd.DataFrame,
    news: pd.DataFrame,
    path: Path,
) -> None:
    if prices.empty:
        raise ValueError("No prices are available for the sentiment summary.")
    latest = prices.sort_index().iloc[-1]
    latest_time = pd.Timestamp(prices.index[-1])
    posts = tweets.copy()
    score = 0.0
    if not posts.empty:
        posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
        recent = posts[posts["timestamp"] >= latest_time - pd.Timedelta(hours=24)]
        if not recent.empty:
            analyzer = SentimentIntensityAnalyzer()
            scores = recent["text"].fillna("").map(
                lambda text: analyzer.polarity_scores(str(text))["compound"]
            )
            score = float(scores.mean())
    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    change = None
    if len(prices) > 1:
        previous = float(prices["close"].iloc[-2])
        change = float(latest["close"]) / previous - 1 if previous else None
    lines = [
        f"# {asset}: current sentiment outlook",
        "",
        f"**As of:** {latest_time.isoformat()}  ",
        f"**Latest close:** {float(latest['close']):.6g}  ",
        f"**X sentiment (last 24h):** {label} ({score:+.3f}, {len(recent) if not posts.empty else 0} posts)  ",
        f"**Latest-bar price change:** {change:+.2%}" if change is not None else "**Latest-bar price change:** unavailable",
        "",
        "## Readout",
        "",
        _outlook_text(score, change),
        "",
        "This is a descriptive research summary, not investment advice. News context is shown below "
        "but is not fed into the forecasting model.",
        "",
        "## Recent news",
        "",
    ]
    if news.empty:
        lines.append("No news has been ingested. Configure Tavily or Exa and run `cwt-agent ingest`.")
    else:
        for _, item in news.head(8).iterrows():
            lines.append(f"- [{item['title']}]({item['url']}) — {item['source']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_charts(
    asset: str,
    interval: str,
    prices: pd.DataFrame,
    tweets: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    if prices.empty:
        raise ValueError("No price data available to chart.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency = "h" if interval == "hourly" else "D"
    chart_prices = prices.copy()
    chart_prices.index = pd.to_datetime(chart_prices.index, utc=True).floor(frequency)
    chart_prices = chart_prices[~chart_prices.index.duplicated(keep="last")]
    sentiment = _period_sentiment(tweets, chart_prices.index, interval)
    plot_sentiment_price(asset, chart_prices, sentiment, output_dir / "sentiment_price.png")
    chart_paths = [output_dir / "sentiment_price.png"]
    if not predictions.empty:
        pivot = predictions.pivot_table(
            index="horizon", columns="timestamp", values="predicted_return_pct", aggfunc="last"
        )
        if pivot.shape[1] > 72:
            pivot = pivot.iloc[:, -72:]
        fig, axis = plt.subplots(figsize=(13, 4))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdYlGn")
        axis.set_yticks(range(len(pivot.index)), [f"n+{value}" for value in pivot.index])
        tick_positions = np.linspace(0, max(0, len(pivot.columns) - 1), min(6, len(pivot.columns)), dtype=int)
        axis.set_xticks(tick_positions, [pivot.columns[index][11:16] for index in tick_positions])
        axis.set_title(f"{asset} walk-forward predicted returns (%)")
        axis.set_xlabel("Out-of-sample bar time (UTC)")
        fig.colorbar(image, ax=axis, label="Predicted return (%)")
        fig.tight_layout()
        destination = output_dir / "forecast_matrix.png"
        fig.savefig(destination, dpi=140)
        plt.close(fig)
        chart_paths.append(destination)
    return chart_paths


def plot_sentiment_price(
    asset: str,
    prices: pd.DataFrame,
    sentiment: pd.Series,
    destination: Path,
) -> None:
    frame = prices[["close"]].copy()
    frame["sentiment"] = sentiment.reindex(frame.index).fillna(0.0)
    frame = frame.tail(24 * 30)
    fig, price_axis = plt.subplots(figsize=(13, 5))
    sentiment_axis = price_axis.twinx()
    price_axis.plot(frame.index, frame["close"], color="#2455a4", linewidth=1.2, label="Close")
    sentiment_axis.plot(
        frame.index,
        frame["sentiment"],
        color="#df7c22",
        linewidth=0.9,
        alpha=0.85,
        label="Mean X sentiment",
    )
    price_axis.set_title(f"{asset}: price and X sentiment")
    price_axis.set_ylabel("Close price")
    sentiment_axis.set_ylabel("Mean sentiment (-1 to +1)")
    price_axis.set_xlabel("Time (UTC)")
    handles_a, labels_a = price_axis.get_legend_handles_labels()
    handles_b, labels_b = sentiment_axis.get_legend_handles_labels()
    price_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def _period_sentiment(tweets: pd.DataFrame, price_index: pd.Index, interval: str) -> pd.Series:
    if tweets.empty:
        return pd.Series(dtype=float, index=price_index)
    analyzer = SentimentIntensityAnalyzer()
    posts = tweets.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    posts = posts.dropna(subset=["timestamp"])
    frequency = "h" if interval == "hourly" else "D"
    posts["period"] = posts["timestamp"].dt.floor(frequency)
    posts["sentiment"] = posts["text"].fillna("").map(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    return posts.groupby("period")["sentiment"].mean()


def _outlook_text(score: float, change: float | None) -> str:
    if score > 0.05:
        direction = "X posts are net positive"
    elif score < -0.05:
        direction = "X posts are net negative"
    else:
        direction = "X posts are broadly neutral"
    price_text = "price-change context is unavailable"
    if change is not None:
        price_text = f"the latest bar moved {change:+.2%}"
    return (
        f"{direction} over the last 24 hours (mean VADER compound score {score:+.3f}); "
        f"{price_text}. This small-window reading can be noisy and should be checked against "
        "the walk-forward evaluation and source coverage."
    )
