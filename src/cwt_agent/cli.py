from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cwt_agent.config import Settings
from cwt_agent.database import Database
from cwt_agent.features import build_feature_frame, sentiment_coverage
from cwt_agent.model import train_and_evaluate
from cwt_agent.providers import (
    fetch_news,
    fetch_prices,
    fetch_tweets,
    limit_tweets_to_lookback,
    normalize_tweets,
)
from cwt_agent.reports import create_charts, write_evaluation_report, write_sentiment_summary


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
    frame, feature_columns = build_feature_frame(prices, tweets, settings.interval)
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
    )
    print(f"Walk-forward evaluation and forecasts written to {settings.reports_dir}.")
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
    if path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    elif path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("data", [])
    else:
        raise ValueError("Tweet import supports CSV, JSON, or JSONL files.")
    rows = normalize_tweets(records)
    if not rows:
        raise ValueError("No valid posts found; ensure the file has text and timestamp columns.")
    return rows


def _asset_name(asset: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in asset)


if __name__ == "__main__":
    main()
