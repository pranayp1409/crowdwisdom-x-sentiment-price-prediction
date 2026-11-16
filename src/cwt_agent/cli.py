from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cwt_agent.archive_data import aggregate_bitcoin_tweet_archive
from cwt_agent.config import Settings
from cwt_agent.database import Database
from cwt_agent.features import build_feature_frame, sentiment_coverage
from cwt_agent.model import train_and_evaluate
from cwt_agent.providers import (
    fetch_news,
    fetch_prices,
    fetch_tweets,
    limit_tweets_to_lookback,
    parse_tweet_export,
)
from cwt_agent.reports import create_charts, write_evaluation_report, write_sentiment_summary
from cwt_agent.sentiment_model import train_sentiment_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cwt-agent",
        description="Research pipeline for X sentiment and price forecasting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Fetch price, X, and news data.")
    ingest_parser.add_argument("--skip-x", action="store_true", help="Do not call the Apify actor.")
    ingest_parser.add_argument("--skip-news", action="store_true", help="Do not call news providers.")
    ingest_parser.add_argument(
        "--tweets-file",
        type=Path,
        help="Import historical actor-style CSV or JSON posts in addition to live X posts.",
    )
    subparsers.add_parser("train", help="Train models and run walk-forward evaluation.")
    sentiment_parser = subparsers.add_parser(
        "train-sentiment",
        help="Train a local sentiment classifier from a Sentiment140 CSV.",
    )
    sentiment_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the headerless Sentiment140 CSV.",
    )
    archive_parser = subparsers.add_parser(
        "train-btc-archive",
        help="Train daily BTC forecasts using a streamed historical Bitcoin tweet archive.",
    )
    archive_parser.add_argument(
        "--tweets-file",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV tweet archives with date/timestamp and text columns.",
    )
    subparsers.add_parser("report", help="Write the current sentiment summary and charts.")
    run_parser = subparsers.add_parser("run", help="Ingest, train, and generate reports.")
    run_parser.add_argument("--tweets-file", type=Path)
    run_parser.add_argument("--skip-x", action="store_true")
    run_parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    database = Database(settings.database_path)

    if args.command == "ingest":
        ingest(settings, database, args.skip_x, args.skip_news, args.tweets_file)
    elif args.command == "train":
        train(settings, database)
    elif args.command == "train-sentiment":
        metrics = train_sentiment_classifier(
            args.dataset,
            settings.sentiment_model_path,
            settings.reports_dir / "sentiment_model_evaluation.json",
        )
        print(f"Sentiment classifier saved to {settings.sentiment_model_path}.")
        print(json.dumps(metrics, indent=2))
    elif args.command == "train-btc-archive":
        train_btc_archive(settings, database, args.tweets_file)
    elif args.command == "report":
        report(settings, database, pd.DataFrame())
    elif args.command == "run":
        ingest(settings, database, args.skip_x, args.skip_news, args.tweets_file)
        _, predictions, _ = train(settings, database)
        report(settings, database, predictions)


def ingest(
    settings: Settings,
    database: Database,
    skip_x: bool,
    skip_news: bool,
    tweets_file: Path | None,
) -> None:
    price_rows = fetch_prices(settings)
    database.save_prices(settings.asset, price_rows)
    print(f"Stored {len(price_rows)} {settings.price_interval} price bars for {settings.asset}.")
    if not skip_x:
        tweet_rows = fetch_tweets(settings)
        database.save_tweets(settings.asset, tweet_rows)
        print(f"Stored {len(tweet_rows)} X posts from Apify.")
    if tweets_file:
        tweet_rows = limit_tweets_to_lookback(
            _load_tweet_file(tweets_file), settings.lookback_days
        )
        if not tweet_rows:
            raise ValueError(
                f"No imported posts are within the configured {settings.lookback_days}-day window."
            )
        database.save_tweets(settings.asset, tweet_rows)
        print(f"Imported {len(tweet_rows)} historical posts from {tweets_file}.")
    if not skip_news:
        news_rows = fetch_news(settings)
        database.save_news(settings.asset, news_rows)
        print(f"Stored {len(news_rows)} news items.")
    prices = database.load_prices(settings.asset)
    tweets = database.load_tweets(settings.asset)
    coverage = sentiment_coverage(prices, tweets, settings.interval)
    print(
        "X coverage: "
        f"{coverage['post_count']} posts across {coverage['active_periods']} of "
        f"{coverage['bar_count']} price bars ({coverage['bar_coverage_pct']:.2f}%), "
        f"on {coverage['active_days']} calendar days."
    )
    if coverage["post_count"] < 100 or coverage["bar_coverage_pct"] < 10:
        print(
            "Warning: X coverage is sparse. Confirm your Apify plan/result limits, "
            "Actor input schema, search query, and date-range support."
        )


def train(
    settings: Settings, database: Database
) -> tuple[dict, pd.DataFrame, dict[str, float]]:
    prices = database.load_prices(settings.asset)
    tweets = database.load_tweets(settings.asset)
    sentiment_model_path = (
        settings.sentiment_model_path
        if settings.sentiment_model_path.is_file()
        else None
    )
    frame, feature_columns = build_feature_frame(
        prices, tweets, settings.interval, sentiment_model_path
    )
    metrics, predictions, forecasts = train_and_evaluate(
        frame,
        feature_columns,
        settings.asset,
        settings.interval,
        settings.artifacts_dir,
        settings.reports_dir,
    )
    evaluation_path = settings.reports_dir / (
        f"{_asset_name(settings.asset)}_{settings.interval}_evaluation.md"
    )
    coverage = sentiment_coverage(prices, tweets, settings.interval)
    write_evaluation_report(
        settings.asset,
        settings.interval,
        metrics,
        forecasts,
        evaluation_path,
        coverage,
        "Sentiment140 classifier" if sentiment_model_path else "VADER",
    )
    print(f"Walk-forward evaluation and forecasts written to {settings.reports_dir}.")
    print(json.dumps(metrics, indent=2))
    return metrics, predictions, forecasts


def train_btc_archive(
    settings: Settings,
    database: Database,
    tweets_files: list[Path],
) -> tuple[dict, pd.DataFrame, dict[str, float]]:
    if settings.asset != "BTC-USD":
        raise ValueError("The Bitcoin tweet archive can only be used with ASSET=BTC-USD.")
    missing_files = [path for path in tweets_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Tweet archive file(s) do not exist: "
            + ", ".join(str(path) for path in missing_files)
        )
    if not settings.sentiment_model_path.is_file():
        raise FileNotFoundError(
            f"Sentiment model not found at {settings.sentiment_model_path}. "
            "Run `cwt-agent train-sentiment` first."
        )

    archive_artifacts_dir = settings.artifacts_dir / "btc_archive_daily"
    archive_reports_dir = settings.reports_dir / "btc_archive_daily"
    daily_settings = replace(settings, interval="daily", lookback_days=4_500)
    price_rows = fetch_prices(daily_settings)
    prices = pd.DataFrame(price_rows)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.set_index("timestamp")
    historical_sentiment, archive_stats = aggregate_bitcoin_tweet_archive(
        tweets_files,
        settings.sentiment_model_path,
        prices.index.min(),
        prices.index.max(),
        progress=lambda rows: print(f"Scanned {rows:,} rows from the Bitcoin archive."),
    )
    tweets = database.load_tweets(settings.asset)
    frame, feature_columns = build_feature_frame(
        prices,
        tweets,
        "daily",
        sentiment_model_path=settings.sentiment_model_path,
        precomputed_sentiment=historical_sentiment,
    )
    metrics, predictions, forecasts = train_and_evaluate(
        frame,
        feature_columns,
        settings.asset,
        "daily",
        archive_artifacts_dir,
        archive_reports_dir,
    )
    historical_sentiment_path = archive_artifacts_dir / "daily_archive_sentiment.csv"
    historical_sentiment.to_csv(historical_sentiment_path)
    current_coverage = sentiment_coverage(prices, tweets, "daily")
    archive_days = int(archive_stats["active_days"])
    combined_days = archive_days + int(current_coverage["active_days"])
    active_periods = int(archive_stats["active_days"]) + int(
        current_coverage["active_periods"]
    )
    first_posts = [
        value
        for value in (
            archive_stats["first_post_utc"],
            current_coverage["first_post_utc"],
        )
        if value
    ]
    last_posts = [
        value
        for value in (
            archive_stats["last_post_utc"],
            current_coverage["last_post_utc"],
        )
        if value
    ]
    coverage = {
        "post_count": int(archive_stats["scored_rows"])
        + int(current_coverage["post_count"]),
        "active_periods": active_periods,
        "bar_count": int(current_coverage["bar_count"]),
        "bar_coverage_pct": (
            active_periods / int(current_coverage["bar_count"]) * 100
            if current_coverage["bar_count"]
            else 0.0
        ),
        "active_days": combined_days,
        "first_post_utc": min(first_posts) if first_posts else None,
        "last_post_utc": max(last_posts) if last_posts else None,
    }
    evaluation_path = archive_reports_dir / f"{_asset_name(settings.asset)}_daily_evaluation.md"
    write_evaluation_report(
        settings.asset,
        "daily",
        metrics,
        forecasts,
        evaluation_path,
        coverage,
        "Sentiment140 classifier applied to historical BTC tweets",
    )
    print(
        f"Scored {archive_stats['scored_rows']:,} Bitcoin-related posts across "
        f"{archive_stats['active_days']:,} historical days "
        f"({archive_stats['first_post_utc']} to {archive_stats['last_post_utc']})."
    )
    if archive_stats["overlapping_days_skipped"]:
        print(
            f"Skipped {archive_stats['overlapping_rows_skipped']:,} posts on "
            f"{archive_stats['overlapping_days_skipped']:,} days already represented "
            "in an earlier archive file."
        )
    print(f"Historical daily sentiment aggregates saved to {historical_sentiment_path}.")
    print(f"Walk-forward evaluation and forecasts written to {archive_reports_dir}.")
    print(json.dumps(metrics, indent=2))
    return metrics, predictions, forecasts


def report(
    settings: Settings,
    database: Database,
    predictions: pd.DataFrame,
) -> None:
    prices = database.load_prices(settings.asset)
    tweets = database.load_tweets(settings.asset)
    news = database.load_news(settings.asset)
    summary_path = settings.reports_dir / f"{_asset_name(settings.asset)}_sentiment_summary.md"
    write_sentiment_summary(settings.asset, prices, tweets, news, summary_path)
    chart_paths = create_charts(
        settings.asset, settings.interval, prices, tweets, predictions, settings.reports_dir
    )
    print(f"Sentiment summary written to {summary_path}.")
    print("Charts written: " + ", ".join(str(path) for path in chart_paths))


def _load_tweet_file(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Tweet file does not exist: {path}")
    try:
        return parse_tweet_export(path.name, path.read_bytes())
    except ValueError as error:
        raise ValueError(f"Could not import {path}: {error}") from error


def _asset_name(asset: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in asset)


if __name__ == "__main__":
    main()
