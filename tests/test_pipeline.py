from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cwt_agent.config import Settings
from cwt_agent.features import build_feature_frame
from cwt_agent.features import sentiment_coverage
from cwt_agent.model import _metrics, metrics_from_predictions, train_and_evaluate
from cwt_agent.providers import (
    _monthly_date_windows,
    fetch_prices,
    fetch_tweets,
    limit_tweets_to_lookback,
    normalize_tweets,
)
from cwt_agent.reports import create_charts, write_sentiment_summary


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


def test_monthly_apify_windows_are_contiguous_and_expand_over_lookback() -> None:
    windows = _monthly_date_windows(180)
    assert len(windows) >= 6
    assert windows[0][0] < windows[-1][1]
    assert all(windows[index][1] == windows[index + 1][0] for index in range(len(windows) - 1))


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
    assert (tmp_path / "reports" / "BTC-USD_hourly_forecast.json").is_file()
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
