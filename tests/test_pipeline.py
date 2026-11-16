from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cwt_agent.config import Settings
from cwt_agent.archive_data import aggregate_bitcoin_tweet_archive
from cwt_agent.database import Database
from cwt_agent.features import build_feature_frame
from cwt_agent.features import sentiment_coverage
from cwt_agent.model import _metrics, metrics_from_predictions, train_and_evaluate
from cwt_agent.providers import (
    _monthly_date_windows,
    fetch_prices,
    fetch_tweets,
    limit_tweets_to_lookback,
    normalize_tweets,
    parse_tweet_export,
)
from cwt_agent.reports import create_charts, write_evaluation_report, write_sentiment_summary
from cwt_agent.sentiment_model import (
    load_sentiment_classifier,
    train_sentiment_classifier,
)


def test_normalize_tweets_maps_common_actor_fields() -> None:
    rows = normalize_tweets(
        [
            {
                "id": 123,
                "createdAt": "2026-01-01T10:15:00Z",
                "fullText": "BTC looks strong",
                "likeCount": 4,
                "retweetCount": 2,
                "author": {"followersCount": 100},
            }
        ]
    )
    assert rows == [
        {
            "id": "123",
            "timestamp": "2026-01-01T10:15:00+00:00",
            "text": "BTC looks strong",
            "likes": 4.0,
            "reposts": 2.0,
            "views": 0.0,
            "followers": 100.0,
        }
    ]


def test_parse_tweet_export_supports_csv_json_and_jsonl() -> None:
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    csv_rows = parse_tweet_export(
        "posts.csv",
        f"id,text,createdAt\n1,Bitcoin is bullish,{timestamp}\n".encode(),
    )
    json_rows = parse_tweet_export(
        "posts.json",
        (
            '[{"id":"2","fullText":"BTC update","createdAt":"'
            + timestamp
            + '"}]'
        ).encode(),
    )
    jsonl_rows = parse_tweet_export(
        "posts.jsonl",
        (
            '{"id":"3","text":"BTC rises","createdAt":"'
            + timestamp
            + '"}\n'
        ).encode(),
    )

    assert [row["id"] for row in csv_rows] == ["1"]
    assert [row["id"] for row in json_rows] == ["2"]
    assert [row["id"] for row in jsonl_rows] == ["3"]


def test_parse_tweet_export_reports_invalid_shape_and_jsonl_line() -> None:
    with pytest.raises(ValueError, match="text and timestamp"):
        parse_tweet_export("posts.json", b'[{"demo":true}]')
    with pytest.raises(ValueError, match="line 2"):
        parse_tweet_export("posts.jsonl", b'{"text":"ok"}\n{bad json}\n')
    with pytest.raises(ValueError, match="CSV, JSON, or JSONL"):
        parse_tweet_export("posts.txt", b"posts")


def test_limit_tweets_to_lookback_excludes_old_and_future_posts() -> None:
    now = pd.Timestamp.now(tz="UTC")
    rows = [
        {"timestamp": (now - pd.Timedelta(days=181)).isoformat()},
        {"timestamp": (now - pd.Timedelta(days=20)).isoformat()},
        {"timestamp": (now + pd.Timedelta(days=1)).isoformat()},
    ]
    assert limit_tweets_to_lookback(rows, 180) == [rows[1]]


def test_price_ingestion_drops_incomplete_hour(monkeypatch, tmp_path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    index = pd.DatetimeIndex(
        [now.floor("h") - pd.Timedelta(hours=2), now.floor("h") - pd.Timedelta(hours=1), now.floor("h")]
    )
    source = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [1.0, 2.0, 3.0],
        },
        index=index,
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: source)
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=tmp_path / "db.sqlite3",
        artifacts_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        apify_api_token=None,
        apify_actor_id=None,
        apify_input_json=None,
        tavily_api_key=None,
        exa_api_key=None,
    )
    rows = fetch_prices(settings)
    assert len(rows) == 2
    assert rows[-1]["close"] == 2.0


def test_apify_token_is_sent_in_authorization_header(monkeypatch) -> None:
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    captured = {"post": {}, "get": []}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_post(url, **kwargs):
        captured["post"]["url"] = url
        captured["post"].update(kwargs)
        return Response(
            {
                "data": {
                    "id": "run-1",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "dataset-1",
                }
            }
        )

    def fake_get(url, **kwargs):
        captured["get"].append((url, kwargs))
        return Response([{"id": "1", "createdAt": timestamp, "text": "BTC update"}])

    monkeypatch.setattr("cwt_agent.providers.requests.post", fake_post)
    monkeypatch.setattr("cwt_agent.providers.requests.get", fake_get)
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="user~actor",
        apify_input_json="{}",
        tavily_api_key=None,
        exa_api_key=None,
    )
    assert fetch_tweets(settings)[0]["text"] == "BTC update"
    assert captured["post"]["headers"]["Authorization"] == "Bearer unit-test-token"
    assert captured["get"][0][1]["headers"]["Authorization"] == "Bearer unit-test-token"
    assert "token=" not in captured["post"]["url"]
    assert captured["get"][0][1]["params"]["offset"] == 0


def test_apify_date_placeholders_batch_months_in_one_run(monkeypatch) -> None:
    from cwt_agent.providers import _monthly_date_windows

    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response(
            {
                "data": {
                    "id": "run-1",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "dataset-1",
                }
            }
        )

    def fake_get(url, **kwargs):
        return Response(
            [
                {
                    "id": "post-1",
                    "createdAt": pd.Timestamp.now(tz="UTC").isoformat(),
                    "text": "BTC date-window post",
                }
            ]
        )

    monkeypatch.setattr("cwt_agent.providers.requests.post", fake_post)
    monkeypatch.setattr("cwt_agent.providers.requests.get", fake_get)
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="user~actor",
        apify_input_json=(
            '{"searchTerms":["BTC since:{{since}} until:{{until}}"],"maxItems":1000}'
        ),
        tavily_api_key=None,
        exa_api_key=None,
    )
    rows = fetch_tweets(settings)
    assert len(calls) == 1
    assert len(calls[0]["searchTerms"]) == len(_monthly_date_windows(180))
    assert len(rows) == 1
    assert all("{{since}}" not in term for term in calls[0]["searchTerms"])
    assert all("{{until}}" not in term for term in calls[0]["searchTerms"])


def test_apify_timestamp_placeholders_batch_daily_queries_in_one_run(monkeypatch) -> None:
    import cwt_agent.providers as providers

    today = pd.Timestamp.now(tz="UTC").normalize()
    windows = [
        (
            (today - pd.Timedelta(days=1)).date().isoformat(),
            today.date().isoformat(),
        ),
        (
            today.date().isoformat(),
            (today + pd.Timedelta(days=1)).date().isoformat(),
        ),
    ]
    monkeypatch.setattr(providers, "_daily_date_windows", lambda _: windows)
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response(
            {
                "data": {
                    "id": "run-1",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "dataset-1",
                }
            }
        )

    records = [
        {
            "id": f"post-{index}",
            "createdAt": (
                pd.Timestamp(since, tz="UTC") + pd.Timedelta(hours=6)
            ).isoformat(),
            "text": "BTC daily-window post",
        }
        for index, (since, _) in enumerate(windows)
    ]

    def fake_get(url, **kwargs):
        return Response(records)

    monkeypatch.setattr("cwt_agent.providers.requests.post", fake_post)
    monkeypatch.setattr("cwt_agent.providers.requests.get", fake_get)
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest",
        apify_input_json=(
            '{"searchTerms":["BTC since_time:{{since_time}} '
            'until_time:{{until_time}}"],"queryType":"Latest","maxItems":20}'
        ),
        tavily_api_key=None,
        exa_api_key=None,
    )

    rows = fetch_tweets(settings)

    expected_terms = [
        "BTC since_time:"
        + str(int(pd.Timestamp(since, tz="UTC").timestamp()))
        + " until_time:"
        + str(int(pd.Timestamp(until, tz="UTC").timestamp()))
        for since, until in windows
    ]
    assert len(calls) == 1
    assert calls[0]["searchTerms"] == expected_terms
    assert calls[0]["queryType"] == "Latest"
    assert calls[0]["maxItems"] == 20
    assert len(rows) == 2


def test_apify_start_end_fields_run_each_month_and_deduplicate(monkeypatch) -> None:
    import cwt_agent.providers as providers

    windows = [("2026-05-13", "2026-06-01"), ("2026-06-01", "2026-11-14")]
    monkeypatch.setattr(providers, "_monthly_date_windows", lambda _: windows)
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response(
            {
                "data": {
                    "id": f"run-{len(calls)}",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": f"dataset-{len(calls)}",
                }
            }
        )

    timestamp = pd.Timestamp.now(tz="UTC").isoformat()

    def fake_get(url, **kwargs):
        return Response(
            [{"id": "same-post", "createdAt": timestamp, "text": "BTC update"}]
        )

    monkeypatch.setattr("cwt_agent.providers.requests.post", fake_post)
    monkeypatch.setattr("cwt_agent.providers.requests.get", fake_get)
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="apidojo/twitter-scraper-lite",
        apify_input_json=(
            '{"searchTerms":["BTC"],"sort":"Latest","maxItems":20,'
            '"start":"2026-06-01","end":"2026-06-02"}'
        ),
        tavily_api_key=None,
        exa_api_key=None,
    )

    rows = fetch_tweets(settings)

    assert len(calls) == len(windows)
    assert [(call["start"], call["end"]) for call in calls] == windows
    assert all(call["maxItems"] == 20 for call in calls)
    assert len(rows) == 1


def test_apify_rejects_unpaired_start_end_fields(monkeypatch) -> None:
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="apidojo/twitter-scraper-lite",
        apify_input_json='{"searchTerms":["BTC"],"start":"2026-06-01"}',
        tavily_api_key=None,
        exa_api_key=None,
    )
    monkeypatch.setattr(
        "cwt_agent.providers.requests.post",
        lambda *args, **kwargs: pytest.fail("Actor must not run for invalid date input"),
    )

    with pytest.raises(ValueError, match="both 'start' and 'end'"):
        fetch_tweets(settings)


def test_apify_empty_dataset_error_explains_no_records(monkeypatch) -> None:
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="apidojo/twitter-scraper-lite",
        apify_input_json='{"searchTerms":["BTC"]}',
        tavily_api_key=None,
        exa_api_key=None,
    )
    monkeypatch.setattr("cwt_agent.providers._run_apify_actor", lambda *args: [])

    with pytest.raises(RuntimeError, match="returned zero dataset records"):
        fetch_tweets(settings)


def test_apify_unknown_output_schema_is_reported(monkeypatch) -> None:
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="apidojo/twitter-scraper-lite",
        apify_input_json='{"searchTerms":["BTC"]}',
        tavily_api_key=None,
        exa_api_key=None,
    )
    monkeypatch.setattr(
        "cwt_agent.providers._run_apify_actor",
        lambda *args: [{"tweet_text": "BTC update", "created": "2026-06-01"}],
    )

    with pytest.raises(RuntimeError, match="recognized non-empty text and timestamp") as error:
        fetch_tweets(settings)
    assert "created" in str(error.value)
    assert "tweet_text" in str(error.value)


def test_apify_dataset_is_paginated_until_short_page(monkeypatch) -> None:
    import cwt_agent.providers as providers

    monkeypatch.setattr(providers, "APIFY_DATASET_PAGE_SIZE", 2)
    offsets = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_post(url, **kwargs):
        return Response(
            {
                "data": {
                    "id": "run-1",
                    "status": "SUCCEEDED",
                    "defaultDatasetId": "dataset-1",
                }
            }
        )

    pages = {
        0: [{"id": "1"}, {"id": "2"}],
        2: [{"id": "3"}],
    }

    def fake_get(url, **kwargs):
        offset = kwargs["params"]["offset"]
        offsets.append(offset)
        return Response(pages[offset])

    monkeypatch.setattr("cwt_agent.providers.requests.post", fake_post)
    monkeypatch.setattr("cwt_agent.providers.requests.get", fake_get)
    rows = providers._run_apify_actor(
        "user~actor", "unit-test-token", {}, 30, 10
    )
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert offsets == [0, 2]


def test_apify_date_slicing_requires_both_placeholders() -> None:
    from cwt_agent.providers import fetch_tweets

    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=Path("unused.sqlite3"),
        artifacts_dir=Path("unused-models"),
        reports_dir=Path("unused-reports"),
        apify_api_token="unit-test-token",
        apify_actor_id="user~actor",
        apify_input_json='{"searchTerms":["BTC since:{{since}}"]}',
        tavily_api_key=None,
        exa_api_key=None,
    )
    with pytest.raises(ValueError, match="both"):
        fetch_tweets(settings)


def test_feature_frame_uses_only_current_and_past_features() -> None:
    index = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    prices = pd.DataFrame({"close": range(100, 180)}, index=index)
    tweets = pd.DataFrame(
        {
            "timestamp": [index[30].isoformat()],
            "text": ["BTC is great"],
            "likes": [2],
            "reposts": [1],
            "views": [10],
            "followers": [50],
        }
    )
    frame, feature_columns = build_feature_frame(prices, tweets, "hourly")
    assert "target_1" not in feature_columns
    assert "close" not in feature_columns
    assert frame.index.max() == index[-1]
    assert pd.isna(frame.loc[index[-1], "target_1"])
    assert frame.loc[index[30], "post_volume"] == 1
    assert frame.loc[index[30], "target_1"] == 131


def test_sentiment_features_are_equal_weight_and_mark_missing_bars() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    prices = pd.DataFrame({"close": range(100, 130)}, index=index)
    tweets = pd.DataFrame(
        [
            {
                "timestamp": index[25].isoformat(),
                "text": "Great positive rally",
                "likes": 1,
                "reposts": 0,
                "views": 10,
                "followers": 20,
            },
            {
                "timestamp": index[25].isoformat(),
                "text": "Terrible market collapse",
                "likes": 1_000_000,
                "reposts": 1_000_000,
                "views": 1_000_000,
                "followers": 1_000_000,
            },
        ]
    )
    frame, feature_columns = build_feature_frame(prices, tweets, "hourly")
    assert "weighted_sentiment" not in feature_columns
    assert frame.loc[index[25], "sentiment_available"] == 1
    assert frame.loc[index[24], "sentiment_available"] == 0
    analyzer = SentimentIntensityAnalyzer()
    expected_mean = (
        analyzer.polarity_scores("Great positive rally")["compound"]
        + analyzer.polarity_scores("Terrible market collapse")["compound"]
    ) / 2
    assert frame.loc[index[25], "sentiment_mean"] == expected_mean
    coverage = sentiment_coverage(prices, tweets, "hourly")
    assert coverage["post_count"] == 2
    assert coverage["active_periods"] == 1
    assert coverage["bar_count"] == 30


def test_sentiment_classifier_training_and_feature_scoring(tmp_path) -> None:
    import csv

    dataset_path = tmp_path / "small-sentiment140.csv"
    with dataset_path.open("w", encoding="latin-1", newline="") as stream:
        writer = csv.writer(stream)
        for label, phrase in ((0, "terrible disappointing"), (4, "excellent wonderful")):
            for index in range(100):
                writer.writerow(
                    [
                        label,
                        index,
                        "Mon Apr 06 22:19:45 PDT 2009",
                        "NO_QUERY",
                        "user",
                        f"{phrase} movie message number {index}",
                    ]
                )

    model_path = tmp_path / "artifacts" / "tweet_sentiment.joblib"
    evaluation_path = tmp_path / "reports" / "sentiment.json"
    metrics = train_sentiment_classifier(
        dataset_path, model_path, evaluation_path, batch_size=20
    )
    classifier = load_sentiment_classifier(model_path)
    scores = classifier.score(
        ["excellent wonderful movie", "terrible disappointing movie"]
    )

    assert model_path.is_file()
    assert evaluation_path.is_file()
    assert metrics["train_rows"] + metrics["validation_rows"] == 200
    assert metrics["validation_rows"] > 0
    assert scores[0] > scores[1]

    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    prices = pd.DataFrame({"close": range(100, 148)}, index=index)
    posts = pd.DataFrame(
        {
            "timestamp": [index[30], index[30]],
            "text": ["excellent wonderful movie", "terrible disappointing movie"],
        }
    )
    scored_frame, _ = build_feature_frame(
        prices, posts, "hourly", sentiment_model_path=model_path
    )
    expected_mean = float(sum(scores) / len(scores))
    assert scored_frame.loc[index[30], "sentiment_mean"] == pytest.approx(expected_mean)

    archive_path = tmp_path / "tweets.csv"
    with archive_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(
            ["id", "user", "fullname", "url", "timestamp", "replies", "likes", "retweets", "text"]
        )
        writer.writerow(
            ["1", "u1", "name", "url", "2026-01-31 08:00:00+00", 0, 0, 0, "Bitcoin is excellent"]
        )
        writer.writerow(
            ["2", "u2", "name", "url", "2026-01-31 09:00:00+00", 0, 0, 0, "BTC is terrible"]
        )
        writer.writerow(
            ["3", "u3", "name", "url", "2026-01-31 10:00:00+00", 0, 0, 0, "Unrelated post"]
        )
    latest_archive_path = tmp_path / "latest-tweets.csv"
    with latest_archive_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "text", "is_retweet"])
        writer.writerow(["2026-01-31 11:00:00", "Bitcoin duplicate day", "False"])
        writer.writerow(["2026-02-01 11:00:00", "Bitcoin adds a new day", "False"])
    archive_sentiment, archive_stats = aggregate_bitcoin_tweet_archive(
        [archive_path, latest_archive_path],
        model_path,
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC"),
    )
    assert archive_stats["scored_rows"] == 3
    assert archive_stats["overlapping_days_skipped"] == 1
    assert archive_stats["overlapping_rows_skipped"] == 1
    assert archive_sentiment.loc[pd.Timestamp("2026-01-31", tz="UTC"), "post_volume"] == 2
    assert archive_sentiment.loc[pd.Timestamp("2026-02-01", tz="UTC"), "post_volume"] == 1

    daily_index = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    daily_prices = pd.DataFrame({"close": range(100, 140)}, index=daily_index)
    daily_posts = pd.DataFrame(
        {
            "timestamp": [daily_index[30], daily_index[30]],
            "text": ["excellent wonderful movie", "terrible disappointing movie"],
        }
    )
    combined_frame, _ = build_feature_frame(
        daily_prices,
        daily_posts,
        "daily",
        sentiment_model_path=model_path,
        precomputed_sentiment=archive_sentiment,
    )
    assert combined_frame.loc[daily_index[30], "post_volume"] == 4


def test_dashboard_imports_uploaded_tweet_file(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=tmp_path / "dashboard.sqlite3",
        artifacts_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        apify_api_token=None,
        apify_actor_id=None,
        apify_input_json=None,
        tavily_api_key=None,
        exa_api_key=None,
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    app_path = Path(__file__).parents[1] / "src" / "cwt_agent" / "dashboard.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception

    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    csv_content = f"id,text,createdAt\npost-1,BTC update,{timestamp}\n".encode()
    app.file_uploader(key="tweet_export_upload").upload("posts.csv", csv_content)
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert app.success[0].value == "Imported 1 posts for BTC-USD."
    imported = Database(settings.database_path).load_tweets("BTC-USD")
    assert len(imported) == 1
    assert imported.iloc[0]["text"] == "BTC update"


def test_database_clear_all_data_removes_every_table(tmp_path) -> None:
    database = Database(tmp_path / "agent.sqlite3")
    database.save_prices(
        "BTC-USD",
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
            }
        ],
    )
    database.save_tweets(
        "BTC-USD",
        [
            {
                "id": "post-1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "text": "BTC test post",
                "likes": 0.0,
                "reposts": 0.0,
                "views": 0.0,
                "followers": 0.0,
            }
        ],
    )
    database.save_news(
        "BTC-USD",
        [
            {
                "url": "https://example.test/news",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "title": "Test news",
                "content": "Test content",
                "source": "test",
            }
        ],
    )

    assert database.clear_all_data() == {"prices": 1, "tweets": 1, "news": 1}
    assert database.load_prices("BTC-USD").empty
    assert database.load_tweets("BTC-USD").empty
    assert database.load_news("BTC-USD").empty


def test_dashboard_data_reset_requires_admin_and_clears_generated_data(
    monkeypatch, tmp_path
) -> None:
    import zipfile
    from io import BytesIO

    from streamlit.testing.v1 import AppTest

    from cwt_agent.dashboard import _build_data_backup

    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=tmp_path / "data" / "agent.sqlite3",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        apify_api_token=None,
        apify_actor_id=None,
        apify_input_json=None,
        tavily_api_key=None,
        exa_api_key=None,
    )
    database = Database(settings.database_path)
    database.save_tweets(
        "BTC-USD",
        [
            {
                "id": "post-1",
                "timestamp": "2026-11-14T12:00:00+00:00",
                "text": "BTC test post",
                "likes": 0.0,
                "reposts": 0.0,
                "views": 0.0,
                "followers": 0.0,
            }
        ],
    )
    database.save_news(
        "BTC-USD",
        [
            {
                "url": "https://example.test/news",
                "timestamp": "2026-11-14T12:00:00+00:00",
                "title": "Test news",
                "content": "Test content",
                "source": "test",
            }
        ],
    )
    settings.artifacts_dir.mkdir()
    settings.reports_dir.mkdir()
    (settings.artifacts_dir / "model.joblib").write_bytes(b"model")
    (settings.reports_dir / "report.md").write_text("report", encoding="utf-8")

    backup = _build_data_backup(
        settings.database_path, settings.artifacts_dir, settings.reports_dir
    )
    with zipfile.ZipFile(BytesIO(backup)) as archive:
        assert set(archive.namelist()) == {
            "database/agent.sqlite3",
            "artifacts/model.joblib",
            "reports/report.md",
        }

    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setenv("APP_ADMIN_PASSWORD", "test-admin-passphrase")
    app_path = Path(__file__).parents[1] / "src" / "cwt_agent" / "dashboard.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception

    app.session_state["app_data_backup_downloaded"] = True
    app.run()
    app.text_input(key="app_reset_password").set_value("wrong")
    app.text_input(key="app_reset_confirmation").set_value("DELETE ALL DATA")
    app.form_submit_button(key="app_reset_submit").click().run()
    assert not app.exception
    assert len(database.load_tweets("BTC-USD")) == 1

    app.text_input(key="app_reset_password").set_value("test-admin-passphrase")
    app.form_submit_button(key="app_reset_submit").click().run()

    assert not app.exception
    assert database.load_tweets("BTC-USD").empty
    assert database.load_news("BTC-USD").empty
    assert list(settings.artifacts_dir.iterdir()) == []
    assert list(settings.reports_dir.iterdir()) == []


def test_monthly_apify_windows_are_contiguous_and_expand_over_lookback() -> None:
    windows = _monthly_date_windows(180)
    assert len(windows) >= 6
    assert windows[0][0] < windows[-1][1]
    assert all(windows[index][1] == windows[index + 1][0] for index in range(len(windows) - 1))


def test_daily_apify_windows_cover_lookback_with_contiguous_days() -> None:
    from cwt_agent.providers import _daily_date_windows

    windows = _daily_date_windows(180)
    assert len(windows) == 180
    assert all(windows[index][1] == windows[index + 1][0] for index in range(len(windows) - 1))
    assert all(
        pd.Timestamp(until) - pd.Timestamp(since) == pd.Timedelta(days=1)
        for since, until in windows
    )


def test_metrics_compare_persistence_and_use_non_overlapping_costed_trades() -> None:
    rows = [
        {
            "base_close": 100.0,
            "actual_close": 101.0,
            "predicted_close": 100.5,
            "actual_return_pct": 1.0,
            "predicted_return_pct": 0.5,
            "majority_direction": 1,
        }
        for _ in range(6)
    ]
    metrics = _metrics(rows, "hourly", horizon=3, transaction_cost_bps=10)
    assert metrics["persistence_mae"] == 1.0
    assert metrics["mae"] == 0.5
    assert metrics["mae_improvement_vs_persistence_pct"] == 50.0
    assert metrics["non_overlapping_trades"] == 2
    assert metrics["transaction_cost_bps_round_trip"] == 10


def test_daily_trading_metrics_use_365_day_annualization() -> None:
    rows = [
        {
            "base_close": 100.0,
            "actual_close": actual_close,
            "predicted_close": predicted_close,
            "actual_return_pct": actual_return,
            "predicted_return_pct": predicted_return,
        }
        for actual_close, predicted_close, actual_return, predicted_return in [
            (101.0, 101.5, 1.0, 1.5),
            (99.0, 99.5, -1.0, -0.5),
            (102.0, 101.0, 2.0, 1.0),
        ]
    ]

    daily = _metrics(rows, "daily", transaction_cost_bps=0)
    hourly = _metrics(rows, "hourly", transaction_cost_bps=0)

    assert daily["sharpe_ratio"] == pytest.approx(
        hourly["sharpe_ratio"] * (1 / 24) ** 0.5
    )


def test_walk_forward_training_forecasts_from_latest_bar(tmp_path) -> None:
    index = pd.date_range("2026-01-01", periods=150, freq="h", tz="UTC")
    close = pd.Series(range(100, 250), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "close": close,
            "sentiment_mean": [0.2 if position % 3 else -0.1 for position in range(len(index))],
        },
        index=index,
    )
    for horizon in range(1, 6):
        frame[f"target_{horizon}"] = close.shift(-horizon)
    metrics, predictions, forecasts = train_and_evaluate(
        frame,
        ["close", "sentiment_mean"],
        "BTC-USD",
        "hourly",
        tmp_path / "models",
        tmp_path / "reports",
    )
    assert set(forecasts) == {f"n+{horizon}" for horizon in range(6)}
    assert forecasts["n+0"] == close.iloc[-1]
    assert predictions["horizon"].nunique() == 5
    assert all(values["oos_samples"] > 0 for values in metrics.values())
    forecast_path = tmp_path / "reports" / "BTC-USD_hourly_forecast.json"
    assert forecast_path.is_file()
    forecast_payload = json.loads(forecast_path.read_text(encoding="utf-8"))
    assert forecast_payload["forecast_sources"]["n+0"] == "observed_close"
    for horizon in range(1, 6):
        key = f"n+{horizon}"
        if metrics[key]["mae_improvement_vs_persistence_pct"] <= 0:
            assert forecast_payload["forecast_sources"][key] == "persistence_baseline"
            assert forecasts[key] == close.iloc[-1]
        else:
            assert forecast_payload["forecast_sources"][key] == "random_forest"
    assert set(metrics["n+5"]) >= {
        "persistence_mae",
        "majority_direction_accuracy",
        "non_overlapping_trades",
    }
    chart_paths = create_charts(
        "BTC-USD",
        "hourly",
        frame[["close"]],
        pd.DataFrame(),
        predictions,
        tmp_path / "reports",
    )
    assert len(chart_paths) == 2
    assert all(path.is_file() for path in chart_paths)
    report_path = tmp_path / "reports" / "evaluation.md"
    write_evaluation_report(
        "BTC-USD",
        "hourly",
        {
            "n+1": {
                **metrics["n+1"],
                "mae_improvement_vs_persistence_pct": -1.0,
            }
        },
        forecasts,
        report_path,
    )
    assert "did not beat the persistence baseline for n+1" in report_path.read_text(
        encoding="utf-8"
    )
    summary_path = tmp_path / "reports" / "summary.md"
    write_sentiment_summary(
        "BTC-USD", frame[["close"]], pd.DataFrame(), pd.DataFrame(), summary_path
    )
    assert "current sentiment outlook" in summary_path.read_text(encoding="utf-8")


def test_metrics_from_saved_predictions_and_local_news_summary() -> None:
    from cwt_agent.dashboard import _extractive_summary

    predictions = pd.DataFrame(
        [
            {
                "horizon": 1,
                "base_close": 100.0,
                "actual_close": 102.0,
                "predicted_close": 101.0,
                "actual_return_pct": 2.0,
                "predicted_return_pct": 1.0,
            },
            {
                "horizon": 1,
                "base_close": 102.0,
                "actual_close": 101.0,
                "predicted_close": 103.0,
                "actual_return_pct": -1.0 / 102.0 * 100,
                "predicted_return_pct": 1.0 / 102.0 * 100,
            },
        ]
    )
    metrics = metrics_from_predictions(predictions, "hourly")
    assert metrics["n+1"]["oos_samples"] == 2
    assert metrics["n+1"]["mae"] == 1.5

    summary = _extractive_summary(
        "Bitcoin rises after strong demand. Bitcoin demand lifts the market. "
        "Analysts discuss several unrelated details."
    )
    assert "Bitcoin rises after strong demand." in summary
    assert "Bitcoin demand lifts the market." in summary


def test_dashboard_loads_forecast_sources_and_accepts_legacy_forecast_files(
    tmp_path,
) -> None:
    from cwt_agent.dashboard import _load_forecast_sources

    reports_dir = tmp_path / "reports"
    settings = Settings(
        asset="BTC-USD",
        interval="hourly",
        lookback_days=180,
        database_path=tmp_path / "agent.sqlite3",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=reports_dir,
        apify_api_token=None,
        apify_actor_id=None,
        apify_input_json=None,
        tavily_api_key=None,
        exa_api_key=None,
    )
    assert _load_forecast_sources(settings) == {}

    reports_dir.mkdir()
    forecast_path = reports_dir / "BTC-USD_hourly_forecast.json"
    forecast_path.write_text(
        json.dumps(
            {
                "forecast_sources": {
                    "n+0": "observed_close",
                    "n+1": "persistence_baseline",
                }
            }
        ),
        encoding="utf-8",
    )
    assert _load_forecast_sources(settings) == {
        "n+0": "observed_close",
        "n+1": "persistence_baseline",
    }

    forecast_path.write_text(json.dumps({"forecasts": {"n+0": 100}}), encoding="utf-8")
    assert _load_forecast_sources(settings) == {}
