from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from cwt_agent.config import Settings

logger = logging.getLogger(__name__)
APIFY_API_BASE = "https://api.apify.com/v2"
APIFY_DATASET_PAGE_SIZE = 1_000
APIFY_POLL_INTERVAL_SECONDS = 5


def _utc_iso(value: Any) -> str:
    if isinstance(value, (int, float)):
        epoch_unit = "ms" if abs(value) >= 1e11 else "s"
        timestamp = pd.to_datetime(value, unit=epoch_unit, utc=True, errors="coerce")
    else:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Unusable timestamp value: {value!r}")
    return timestamp.isoformat()


def fetch_prices(settings: Settings) -> list[dict[str, Any]]:
    import yfinance as yf

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=settings.lookback_days)
    frame = yf.download(
        settings.asset,
        start=start,
        end=end,
        interval=settings.price_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError(
            f"No price bars returned for {settings.asset} at {settings.price_interval}."
        )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = {"Close"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Price provider returned unexpected columns: {list(frame.columns)}")

    source_index = pd.DatetimeIndex(frame.index)
    utc_index = (
        source_index.tz_localize("UTC")
        if source_index.tz is None
        else source_index.tz_convert("UTC")
    )
    now = pd.Timestamp.now(tz="UTC")
    if settings.interval == "hourly":
        complete = utc_index + pd.Timedelta(hours=1) <= now
    else:
        local_now = now.tz_convert(source_index.tz) if source_index.tz else now
        complete = [timestamp.date() < local_now.date() for timestamp in source_index]
    frame = frame.loc[complete].copy()
    if frame.empty:
        raise RuntimeError(f"No completed price bars returned for {settings.asset}.")
    frame.index = utc_index[complete]
    rows: list[dict[str, Any]] = []
    for timestamp, values in frame.iterrows():
        row: dict[str, Any] = {"timestamp": timestamp.isoformat()}
        for source, target in (
            ("Open", "open"),
            ("High", "high"),
            ("Low", "low"),
            ("Close", "close"),
            ("Volume", "volume"),
        ):
            value = values.get(source)
            if value is None or pd.isna(value):
                row[target] = None
            else:
                row[target] = float(value)
        if row["close"] is not None and math.isfinite(row["close"]):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Price provider returned no valid closing prices for {settings.asset}.")
    return rows


def _first(record: dict[str, Any], names: tuple[str, ...], default: Any = 0) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    normalized_record = {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): value
        for key, value in record.items()
    }
    for name in names:
        value = normalized_record.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None:
            return value
    return default


def normalize_tweets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        text = _first(record, ("text", "fullText", "full_text", "content", "tweetText"), "")
        timestamp = _first(
            record,
            ("createdAt", "created_at", "timestamp", "datetime", "date", "postedAt"),
            None,
        )
        if text is None or (not isinstance(text, (dict, list)) and pd.isna(text)):
            continue
        if isinstance(text, dict):
            text = text.get("text", "")
        if (
            not str(text).strip()
            or timestamp is None
            or (not isinstance(timestamp, (dict, list)) and pd.isna(timestamp))
        ):
            continue
        try:
            timestamp_iso = _utc_iso(timestamp)
        except ValueError:
            continue
        user = record.get("author") or record.get("user") or {}
        if not isinstance(user, dict):
            user = {}
        likes = _first(record, ("likeCount", "like_count", "likes"), 0)
        reposts = _first(record, ("retweetCount", "retweet_count", "repostCount", "reposts"), 0)
        views = _first(record, ("viewCount", "view_count", "views"), 0)
        followers = _first(
            record,
            ("authorFollowers", "followersCount", "followers_count", "followerCount"),
            _first(user, ("followersCount", "followers_count", "followers"), 0),
        )
        post_id = _first(record, ("id", "id_str", "tweetId", "url"), None)
        if post_id is None:
            digest = hashlib.sha256(f"{timestamp_iso}\0{text}".encode("utf-8")).hexdigest()
            post_id = digest
        rows.append(
            {
                "id": str(post_id),
                "timestamp": timestamp_iso,
                "text": str(text),
                "likes": _number(likes),
                "reposts": _number(reposts),
                "views": _number(views),
                "followers": _number(followers),
            }
        )
    return rows


def parse_tweet_export(file_name: str, content: bytes) -> list[dict[str, Any]]:
    suffix = file_name.lower().rsplit(".", maxsplit=1)[-1] if "." in file_name else ""
    if suffix == "csv":
        try:
            records = pd.read_csv(io.BytesIO(content)).to_dict(orient="records")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError(f"Could not read the CSV export: {error}") from error
    elif suffix in {"json", "jsonl"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("The export must be UTF-8 encoded.") from error
        if suffix == "jsonl":
            records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error.msg}."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object on line {line_number} of the JSONL export."
                    )
                records.append(record)
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON export: {error.msg}.") from error
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = next(
                    (
                        payload[key]
                        for key in ("data", "items", "tweets")
                        if isinstance(payload.get(key), list)
                    ),
                    None,
                )
                if records is None:
                    records = [payload]
            else:
                raise ValueError("JSON export must contain an array or an object of posts.")
    else:
        raise ValueError("Upload a CSV, JSON, or JSONL post export.")

    if not records:
        raise ValueError("The uploaded export contains no post records.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Each post in the uploaded export must be a JSON object or CSV row.")
    rows = normalize_tweets(records)
    if not rows:
        fields = sorted({field for record in records[:5] for field in record})
        raise ValueError(
            "No usable posts found. Each row needs recognized text and timestamp fields "
            "(for example `text` and `createdAt`). "
            f"Fields found: {fields or 'none'}."
        )
    return list({row["id"]: row for row in rows}.values())


def limit_tweets_to_lookback(
    rows: list[dict[str, Any]], lookback_days: int
) -> list[dict[str, Any]]:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=lookback_days)
    return [
        row
        for row in rows
        if cutoff <= pd.Timestamp(row["timestamp"]) <= now
    ]


def _number(value: Any) -> float:
    try:
        parsed = float(value or 0)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def fetch_tweets(settings: Settings) -> list[dict[str, Any]]:
    if not settings.apify_api_token or not settings.apify_actor_id:
        raise RuntimeError("Set APIFY_API_TOKEN and APIFY_ACTOR_ID in your local .env file.")
    if settings.apify_input_json:
        try:
            actor_input = json.loads(settings.apify_input_json)
        except json.JSONDecodeError as error:
            raise ValueError("APIFY_INPUT_JSON must contain valid JSON.") from error
    else:
        actor_input = {"searchTerms": [settings.asset]}
    if not isinstance(actor_input, dict):
        raise ValueError("APIFY_INPUT_JSON must contain a JSON object.")
    actor_id = quote(settings.apify_actor_id, safe="~")
    serialized_input = json.dumps(actor_input)
    has_since = "{{since}}" in serialized_input
    has_until = "{{until}}" in serialized_input
    if has_since != has_until:
        raise ValueError("APIFY_INPUT_JSON must include both {{since}} and {{until}}, or neither.")
    has_since_time = "{{since_time}}" in serialized_input
    has_until_time = "{{until_time}}" in serialized_input
    if has_since_time != has_until_time:
        raise ValueError(
            "APIFY_INPUT_JSON must include both {{since_time}} and {{until_time}}, or neither."
        )
    has_start = "start" in actor_input
    has_end = "end" in actor_input
    if has_start != has_end:
        raise ValueError("APIFY_INPUT_JSON must include both 'start' and 'end' date fields.")
    if has_since_time and (has_since or has_start):
        raise ValueError("APIFY_INPUT_JSON must use only one date-filter placeholder format.")

    windows: list[tuple[str, str]] = []
    if has_since_time:
        windows = _daily_date_windows(settings.lookback_days)
        actor_inputs = [_expand_timestamp_queries(actor_input, windows)]
    elif has_start:
        windows = _monthly_date_windows(settings.lookback_days)
        actor_inputs = [
            {**actor_input, "start": since, "end": until}
            for since, until in windows
        ]
    elif has_since:
        windows = _monthly_date_windows(settings.lookback_days)
        actor_inputs = [_expand_date_queries(actor_input, windows)]
    else:
        actor_inputs = [actor_input]

    records: list[dict[str, Any]] = []
    for window_input in actor_inputs:
        try:
            records.extend(
                _run_apify_actor(
                    actor_id,
                    settings.apify_api_token,
                    window_input,
                    settings.apify_run_timeout_seconds,
                    settings.apify_max_dataset_items,
                )
            )
        except (requests.RequestException, RuntimeError, TimeoutError, ValueError) as error:
            if has_start:
                raise RuntimeError(
                    "Apify collection failed for date window "
                    f"{window_input['start']} through {window_input['end']}."
                ) from error
            raise
    normalized_rows = normalize_tweets(records)
    if not normalized_rows:
        if not records:
            result_detail = "the Actor returned zero dataset records"
        else:
            field_names = sorted(
                {name for record in records[:5] for name in record}
            )
            result_detail = (
                f"the Actor returned {len(records)} dataset records, but none had "
                "recognized non-empty text and timestamp fields. "
                f"Fields in the first records: {field_names or 'none'}."
            )
        raise RuntimeError(
            f"Apify collection completed for {len(actor_inputs)} input(s): "
            f"{result_detail} Check the Actor run datasets and input schema."
        )

    rows = limit_tweets_to_lookback(normalized_rows, settings.lookback_days)
    rows_by_id = {row["id"]: row for row in rows}
    rows = list(rows_by_id.values())
    if windows:
        _warn_about_empty_search_windows(rows, windows)
    if not rows:
        timestamps = pd.to_datetime(
            [row["timestamp"] for row in normalized_rows], utc=True, errors="coerce"
        ).dropna()
        returned_range = (
            f"{timestamps.min().isoformat()} through {timestamps.max().isoformat()} UTC"
            if not timestamps.empty
            else "an unrecognized timestamp range"
        )
        raise RuntimeError(
            f"Apify returned {len(normalized_rows)} valid posts, but none fell within "
            f"the last {settings.lookback_days} days ({returned_range}). Check the "
            "Actor's start/end date filters, timezone, and the requested lookback."
        )
    return rows


def _expand_date_queries(
    actor_input: Any, windows: list[tuple[str, str]]
) -> Any:
    return _expand_query_placeholders(actor_input, windows, "{{since}}", "{{until}}")


def _expand_timestamp_queries(
    actor_input: Any, windows: list[tuple[str, str]]
) -> Any:
    timestamp_windows = [
        (
            str(int(pd.Timestamp(since, tz="UTC").timestamp())),
            str(int(pd.Timestamp(until, tz="UTC").timestamp())),
        )
        for since, until in windows
    ]
    return _expand_query_placeholders(
        actor_input, timestamp_windows, "{{since_time}}", "{{until_time}}"
    )


def _expand_query_placeholders(
    actor_input: Any,
    windows: list[tuple[str, str]],
    since_placeholder: str,
    until_placeholder: str,
) -> Any:
    if isinstance(actor_input, dict):
        return {
            key: _expand_query_placeholders(
                value, windows, since_placeholder, until_placeholder
            )
            for key, value in actor_input.items()
        }
    if isinstance(actor_input, list):
        expanded: list[Any] = []
        for value in actor_input:
            if isinstance(value, str) and (
                since_placeholder in value or until_placeholder in value
            ):
                if since_placeholder not in value or until_placeholder not in value:
                    raise ValueError(
                        "Each APIFY_INPUT_JSON date query must include both "
                        f"{since_placeholder} and {until_placeholder}."
                    )
                expanded.extend(
                    value.replace(since_placeholder, since).replace(
                        until_placeholder, until
                    )
                    for since, until in windows
                )
            else:
                expanded.append(
                    _expand_query_placeholders(
                        value, windows, since_placeholder, until_placeholder
                    )
                )
        return expanded
    if isinstance(actor_input, str) and (
        since_placeholder in actor_input or until_placeholder in actor_input
    ):
        raise ValueError(
            "Put date placeholders inside an APIFY_INPUT_JSON array such as searchTerms."
        )
    return actor_input


def _run_apify_actor(
    actor_id: str,
    api_token: str,
    actor_input: dict[str, Any],
    timeout_seconds: int,
    max_dataset_items: int,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(
        f"{APIFY_API_BASE}/acts/{actor_id}/runs",
        headers=headers,
        json=actor_input,
        timeout=60,
    )
    response.raise_for_status()
    run_payload = response.json()
    run = run_payload.get("data", run_payload)
    if not isinstance(run, dict) or not run.get("id"):
        raise RuntimeError("Apify did not return an Actor run ID.")
    run_id = str(run["id"])
    deadline = time.monotonic() + timeout_seconds
    status = str(run.get("status", "READY")).upper()
    while status not in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Apify Actor run {run_id} did not finish within {timeout_seconds} seconds. "
                "Check its status and logs in Apify Console."
            )
        time.sleep(min(APIFY_POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))
        status_response = requests.get(
            f"{APIFY_API_BASE}/actor-runs/{quote(run_id, safe='')}",
            headers=headers,
            timeout=30,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        run = status_payload.get("data", status_payload)
        if not isinstance(run, dict):
            raise RuntimeError(f"Apify returned an invalid status for Actor run {run_id}.")
        status = str(run.get("status", "")).upper()
    if status != "SUCCEEDED":
        message = run.get("statusMessage")
        detail = f" ({message})" if message else ""
        raise RuntimeError(
            f"Apify Actor run {run_id} ended with status {status}{detail}. "
            "Check the Actor input and run logs in Apify Console."
        )
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError(
            f"Apify Actor run {run_id} succeeded but returned no default dataset ID."
        )
    return _get_apify_dataset_items(
        str(dataset_id), headers, max_dataset_items
    )


def _get_apify_dataset_items(
    dataset_id: str,
    headers: dict[str, str],
    max_items: int,
) -> list[dict[str, Any]]:
    dataset_url = f"{APIFY_API_BASE}/datasets/{quote(dataset_id, safe='')}/items"
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < max_items:
        page_size = min(APIFY_DATASET_PAGE_SIZE, max_items - offset)
        response = requests.get(
            dataset_url,
            headers=headers,
            params={
                "format": "json",
                "clean": "true",
                "offset": offset,
                "limit": page_size,
            },
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError("Apify returned an unexpected dataset page; expected a list.")
        records.extend(record for record in page if isinstance(record, dict))
        offset += len(page)
        if len(page) < page_size:
            return records
    logger.warning(
        "Apify dataset reached APIFY_MAX_DATASET_ITEMS=%s; results may be truncated.",
        max_items,
    )
    return records


def _warn_about_empty_search_windows(
    rows: list[dict[str, Any]], windows: list[tuple[str, str]]
) -> None:
    post_times = pd.to_datetime(
        [row["timestamp"] for row in rows], utc=True, errors="coerce"
    )
    empty_windows = [
        (since, until)
        for since, until in windows
        if not ((post_times >= pd.Timestamp(since, tz="UTC")) & (post_times < pd.Timestamp(until, tz="UTC"))).any()
    ]
    if empty_windows:
        logger.warning(
            "No returned posts fell within %s of %s requested date windows. "
            "The Actor may not support the query syntax, a date range may have no matches, "
            "or the plan/run result cap may have limited results.",
            len(empty_windows),
            len(windows),
        )


def _monthly_date_windows(lookback_days: int) -> list[tuple[str, str]]:
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor < end:
        next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
        window_end = min(next_month, end)
        windows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end
    return windows


def _daily_date_windows(lookback_days: int) -> list[tuple[str, str]]:
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + pd.Timedelta(days=1), end)
        windows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end
    return windows


def fetch_news(settings: Settings) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    query = f"{settings.asset} market news"
    if settings.tavily_api_key:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "topic": "news",
                "days": min(settings.lookback_days, 30),
                "max_results": 10,
                "include_answer": False,
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            results.append(
                {
                    "url": str(item.get("url") or ""),
                    "timestamp": _optional_iso(item.get("published_date")),
                    "title": str(item.get("title") or ""),
                    "content": str(item.get("content") or ""),
                    "source": "tavily",
                }
            )
    if settings.exa_api_key:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "category": "news",
                "numResults": 10,
                "contents": {"text": {"maxCharacters": 1500}},
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            results.append(
                {
                    "url": str(item.get("url") or ""),
                    "timestamp": _optional_iso(item.get("publishedDate")),
                    "title": str(item.get("title") or ""),
                    "content": str(item.get("text") or ""),
                    "source": "exa",
                }
            )
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in results:
        if row["url"] and row["title"]:
            deduplicated[row["url"]] = row
    if not deduplicated:
        raise RuntimeError(
            "No news results available. Configure TAVILY_API_KEY or EXA_API_KEY, "
            "or explicitly skip news ingestion."
        )
    return list(deduplicated.values())


def _optional_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return _utc_iso(value)
    except ValueError:
        return None
