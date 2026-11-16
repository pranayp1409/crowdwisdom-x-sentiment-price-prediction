from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from cwt_agent.sentiment_model import load_sentiment_classifier

_BITCOIN_TEXT = re.compile(r"\b(?:bitcoin|btc)\b", re.IGNORECASE)
_ISO_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def aggregate_bitcoin_tweet_archive(
    tweets_path: Path | Sequence[Path],
    sentiment_model_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 32_768,
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int | str | None]]:
    tweets_paths = [tweets_path] if isinstance(tweets_path, Path) else list(tweets_path)
    if not tweets_paths:
        raise ValueError("At least one tweet archive path is required.")
    for path in tweets_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Tweet archive does not exist: {path}")
    if not sentiment_model_path.is_file():
        raise FileNotFoundError(
            f"Sentiment model does not exist: {sentiment_model_path}. "
            "Run `cwt-agent train-sentiment` first."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    start_day = _utc_day(start)
    end_day = _utc_day(end)
    start_date = start_day.strftime("%Y-%m-%d")
    end_date = end_day.strftime("%Y-%m-%d")
    if start_date > end_date:
        raise ValueError("Archive start date must not be after its end date.")

    classifier = load_sentiment_classifier(sentiment_model_path)
    aggregate: dict[pd.Timestamp, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    pending_timestamps: list[str] = []
    pending_texts: list[str] = []
    scanned_rows = 0
    bitcoin_rows_in_range = 0
    invalid_timestamps = 0
    missing_text_rows = 0
    overlapping_rows_skipped = 0
    scored_rows = 0
    active_days: set[str] = set()
    claimed_days: set[pd.Timestamp] = set()
    overlapping_days: set[str] = set()
    first_post: str | None = None
    last_post: str | None = None

    def score_batch() -> None:
        nonlocal invalid_timestamps, overlapping_rows_skipped, scored_rows
        nonlocal first_post, last_post
        if not pending_texts:
            return
        timestamps = pd.to_datetime(pending_timestamps, utc=True, errors="coerce")
        scores = classifier.score(pending_texts)
        valid = ~timestamps.isna()
        invalid_timestamps += int((~valid).sum())
        periods = timestamps.floor("D")
        in_window = (
            valid
            & (periods >= start_day)
            & (periods <= end_day)
        )
        is_duplicate_day = in_window & periods.isin(claimed_days)
        overlapping_rows_skipped += int(is_duplicate_day.sum())
        overlapping_days.update(
            periods[is_duplicate_day].strftime("%Y-%m-%d").unique().tolist()
        )
        within_range = in_window & ~periods.isin(claimed_days)
        if within_range.any():
            periods = periods[within_range]
            score_values = scores[within_range]
            batch = pd.DataFrame(
                {
                    "period": periods,
                    "score_sum": score_values,
                    "positive_count": (score_values > 0).astype(int),
                    "post_volume": 1,
                }
            )
            for period, values in batch.groupby("period", sort=False).sum().iterrows():
                totals = aggregate[pd.Timestamp(period)]
                totals[0] += float(values["score_sum"])
                totals[1] += float(values["positive_count"])
                totals[2] += float(values["post_volume"])
            scored_rows += int(within_range.sum())
            source_days.update(periods.unique().tolist())
            valid_timestamps = timestamps[within_range]
            batch_first = valid_timestamps.min().isoformat()
            batch_last = valid_timestamps.max().isoformat()
            first_post = min(first_post, batch_first) if first_post else batch_first
            last_post = max(last_post, batch_last) if last_post else batch_last
            active_days.update(periods.strftime("%Y-%m-%d").unique().tolist())
        pending_timestamps.clear()
        pending_texts.clear()

    for tweets_file in tweets_paths:
        source_days: set[pd.Timestamp] = set()
        with tweets_file.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as stream:
            sample = stream.read(65_536)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error as error:
                raise ValueError(
                    f"Could not determine the CSV delimiter for {tweets_file}."
                ) from error
            reader = csv.DictReader(stream, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError(f"Tweet archive has no header: {tweets_file}")
            timestamp_column = _find_column(
                reader.fieldnames,
                ("timestamp", "created_at", "createdat", "datetime", "date", "tweetat"),
            )
            text_column = _find_column(
                reader.fieldnames,
                ("text", "full_text", "fulltext", "tweet_text", "tweettext", "content", "message", "tweet"),
            )
            if timestamp_column is None or text_column is None:
                raise ValueError(
                    f"Tweet archive {tweets_file} must include a timestamp/date "
                    "column and a text column."
                )
            for row in reader:
                scanned_rows += 1
                timestamp = (row.get(timestamp_column) or "").strip()
                text = row.get(text_column) or ""
                if not text.strip():
                    missing_text_rows += 1
                if not _ISO_DATE_PREFIX.match(timestamp):
                    invalid_timestamps += 1
                    continue
                if not text.strip():
                    continue
                date_prefix = timestamp[:10]
                if date_prefix < start_date or date_prefix > end_date:
                    continue
                if not _BITCOIN_TEXT.search(text):
                    continue
                bitcoin_rows_in_range += 1
                pending_timestamps.append(timestamp)
                pending_texts.append(text)
                if len(pending_texts) >= batch_size:
                    score_batch()
                if progress is not None and scanned_rows % 2_000_000 == 0:
                    progress(scanned_rows)
        score_batch()
        claimed_days.update(source_days)

    if not scored_rows:
        raise ValueError(
            "No Bitcoin-related posts with usable timestamps were found inside "
            "the available price history."
        )
    rows = []
    index = []
    for period, (score_sum, positive_count, volume) in sorted(aggregate.items()):
        if volume <= 0:
            continue
        index.append(period)
        rows.append(
            {
                "sentiment_mean": score_sum / volume,
                "positive_share": positive_count / volume,
                "post_volume": volume,
            }
        )
    sentiment = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(index, tz="UTC", name="timestamp"),
        columns=["sentiment_mean", "positive_share", "post_volume"],
    )
    stats: dict[str, int | str | None] = {
        "scanned_rows": scanned_rows,
        "bitcoin_rows_in_range": bitcoin_rows_in_range,
        "scored_rows": scored_rows,
        "invalid_timestamps": invalid_timestamps,
        "missing_text_rows": missing_text_rows,
        "overlapping_rows_skipped": overlapping_rows_skipped,
        "overlapping_days_skipped": len(overlapping_days),
        "active_days": len(active_days),
        "first_post_utc": first_post,
        "last_post_utc": last_post,
    }
    return sentiment, stats


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", fieldname.lower()): fieldname
        for fieldname in fieldnames
    }
    for candidate in candidates:
        match = normalized.get(re.sub(r"[^a-z0-9]", "", candidate.lower()))
        if match is not None:
            return match
    return None


def _utc_day(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()
