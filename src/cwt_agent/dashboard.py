from __future__ import annotations

import hmac
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from streamlit.errors import StreamlitSecretNotFoundError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cwt_agent.config import Settings
from cwt_agent.database import Database
from cwt_agent.features import sentiment_coverage
from cwt_agent.model import metrics_from_predictions
from cwt_agent.providers import (
    fetch_news,
    fetch_prices,
    fetch_tweets,
    limit_tweets_to_lookback,
    parse_tweet_export,
)


STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "being", "but", "can", "could", "did", "does", "during", "each", "for", "from",
    "had", "has", "have", "her", "here", "him", "his", "how", "into", "its", "just",
    "more", "most", "not", "our", "out", "over", "said", "she", "should", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "under", "was", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
}


def main() -> None:
    st.set_page_config(
        page_title="CWT Sentiment Terminal",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    settings = Settings.from_env()
    database = Database(settings.database_path)
    available_assets = database.list_assets()
    if settings.asset not in available_assets:
        available_assets.insert(0, settings.asset)

    st.sidebar.markdown("## CWT · RESEARCH")
    asset = st.sidebar.selectbox(
        "Asset",
        available_assets,
        index=available_assets.index(settings.asset),
    )
    interval = st.sidebar.selectbox(
        "Bar interval",
        ["hourly", "daily"],
        index=0 if settings.interval == "hourly" else 1,
    )
    window_days = st.sidebar.select_slider(
        "Chart lookback",
        options=[7, 30, 90, 180],
        value=min([7, 30, 90, 180], key=lambda value: abs(value - settings.lookback_days)),
        format_func=lambda value: f"{value} days",
    )
    selected_settings = replace(settings, asset=asset, interval=interval)
    st.sidebar.divider()
    st.sidebar.caption(
        "Credentials are loaded from `.env` or deployment secrets and never shown here."
    )
    if st.sidebar.button("↻  Refresh data", width="stretch", type="primary"):
        refresh_errors: list[str] = []
        with st.spinner("Refreshing price, X, and news data…"):
            try:
                prices = fetch_prices(selected_settings)
                database.save_prices(asset, prices)
            except (requests.RequestException, RuntimeError, TimeoutError, ValueError) as error:
                refresh_errors.append(f"Price refresh failed: {error}")
            if selected_settings.apify_api_token and selected_settings.apify_actor_id:
                try:
                    database.save_tweets(asset, fetch_tweets(selected_settings))
                except (
                    requests.RequestException,
                    RuntimeError,
                    TimeoutError,
                    ValueError,
                ) as error:
                    refresh_errors.append(f"X refresh failed: {error}")
            else:
                st.sidebar.info("Apify is not configured; X posts were not refreshed.")
            if selected_settings.tavily_api_key or selected_settings.exa_api_key:
                try:
                    database.save_news(asset, fetch_news(selected_settings))
                except (requests.RequestException, RuntimeError, TimeoutError, ValueError) as error:
                    refresh_errors.append(f"News refresh failed: {error}")
            else:
                st.sidebar.info("No news provider is configured; saved news was left unchanged.")
        if refresh_errors:
            for error in refresh_errors:
                st.error(error)
        else:
            st.rerun()

    _render_data_reset(database, settings)

    prices = database.load_prices(asset)
    tweets = database.load_tweets(asset)
    news = database.load_news(asset, limit=30)
    predictions = _load_predictions(selected_settings)
    forecasts = _load_forecasts(selected_settings)
    forecast_sources = _load_forecast_sources(selected_settings)
    metrics = metrics_from_predictions(predictions, interval)
    st.markdown(
        '<div class="eyebrow">MARKET INTELLIGENCE · SENTIMENT & QUANT</div>',
        unsafe_allow_html=True,
    )
    st.title(f"{asset}  /  Sentiment overview")
    if not prices.empty:
        latest_time = pd.Timestamp(prices.index[-1])
        st.caption(
            f"Research dashboard · {interval} bars · latest stored price bar "
            f"{latest_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
    else:
        st.info("No price bars are stored for this asset yet. Use **Refresh data** to ingest prices.")

    if _render_tweet_import(database, asset, settings.lookback_days, tweets.empty):
        tweets = database.load_tweets(asset)

    if not prices.empty:
        recent_posts, recent_score = _recent_sentiment(tweets)
        _render_live_metrics(prices, recent_posts, recent_score)
        _render_data_quality(prices, tweets, interval)
        _render_price_sentiment(prices, tweets, interval, window_days)
        _render_forecasts(forecasts, prices, forecast_sources)
        _render_performance(metrics)
        _render_walk_forward(predictions, metrics)
        _render_news(news)
        _render_recent_posts(tweets)


def _render_tweet_import(
    database: Database, asset: str, lookback_days: int, expand_by_default: bool
) -> bool:
    imported = False
    with st.expander("Import historical X posts", expanded=expand_by_default):
        st.caption(
            "Upload an Apify CSV, JSON, or JSONL export with post text and timestamps. "
            f"Only posts from the last {lookback_days} days are imported."
        )
        with st.form("tweet_export_import"):
            uploaded_file = st.file_uploader(
                "Choose a post export",
                type=["csv", "json", "jsonl"],
                help="Expected fields include text and createdAt (or supported aliases).",
                key="tweet_export_upload",
            )
            submitted = st.form_submit_button("Import posts", width="stretch")
        if submitted:
            if uploaded_file is None:
                st.error("Choose a CSV, JSON, or JSONL file before importing.")
            else:
                try:
                    rows = parse_tweet_export(uploaded_file.name, uploaded_file.getvalue())
                    in_window = limit_tweets_to_lookback(rows, lookback_days)
                    if not in_window:
                        st.error(
                            f"Found {len(rows)} valid posts, but none are within the last "
                            f"{lookback_days} days. Check the export timestamps."
                        )
                    else:
                        database.save_tweets(asset, in_window)
                        ignored = len(rows) - len(in_window)
                        st.success(
                            f"Imported {len(in_window)} posts for {asset}."
                            + (
                                f" Ignored {ignored} posts outside the lookback window."
                                if ignored
                                else ""
                            )
                        )
                        imported = True
                except ValueError as error:
                    st.error(f"Could not import the post export: {error}")
    return imported


def _render_data_reset(database: Database, settings: Settings) -> None:
    with st.sidebar.expander("Reset application data"):
        st.warning(
            "This permanently clears prices, X posts, news, saved models, and reports "
            "for every asset. Download the backup before confirming. Credentials and "
            "source code are not included or removed.",
            title="Back up before clearing",
        )
        admin_password = _get_reset_admin_password()
        if not admin_password:
            st.info(
                "Reset is locked. Configure a private APP_ADMIN_PASSWORD in local "
                ".env or Streamlit Cloud Secrets."
            )
            return

        try:
            backup = _build_data_backup(
                database.path, settings.artifacts_dir, settings.reports_dir
            )
        except (OSError, sqlite3.Error, ValueError) as error:
            st.error(f"Could not prepare the backup: {error}")
            return

        downloaded = st.download_button(
            "Download data backup",
            data=backup,
            file_name=(
                f"cwt-{settings.asset.lower()}-backup-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
            ),
            mime="application/zip",
            key="app_data_backup_download",
            width="stretch",
        )
        if downloaded:
            st.session_state["app_data_backup_downloaded"] = True

        if not st.session_state.get("app_data_backup_downloaded", False):
            st.caption("Download the backup to unlock the reset confirmation.")
            return

        with st.form("app_data_reset_form"):
            password = st.text_input(
                "Admin passphrase",
                type="password",
                key="app_reset_password",
            )
            confirmation = st.text_input(
                'Type "DELETE ALL DATA" to confirm',
                key="app_reset_confirmation",
            )
            submitted = st.form_submit_button(
                "Clear all application data",
                key="app_reset_submit",
                width="stretch",
            )

        if submitted:
            if not hmac.compare_digest(password, admin_password):
                st.error("The admin passphrase is incorrect.")
            elif confirmation != "DELETE ALL DATA":
                st.error('Type "DELETE ALL DATA" exactly to confirm.')
            else:
                try:
                    _clear_generated_directory(settings.artifacts_dir, database.path)
                    _clear_generated_directory(settings.reports_dir, database.path)
                    database.clear_all_data()
                except (OSError, sqlite3.Error, ValueError) as error:
                    st.error(
                        f"Reset did not finish completely: {error}. "
                        "Restore from the downloaded backup if needed."
                    )
                else:
                    st.session_state["app_data_backup_downloaded"] = False
                    st.session_state["app_data_reset_complete"] = True
                    st.rerun()

        if st.session_state.get("app_data_reset_complete", False):
            st.success("Application data cleared for all assets.")
            st.session_state["app_data_reset_complete"] = False


def _get_reset_admin_password() -> str | None:
    password = os.getenv("APP_ADMIN_PASSWORD")
    if password:
        return password
    try:
        secret = st.secrets.get("APP_ADMIN_PASSWORD")
    except (KeyError, StreamlitSecretNotFoundError):
        return None
    return str(secret) if secret else None


def _build_data_backup(
    database_path: Path, artifacts_dir: Path, reports_dir: Path
) -> bytes:
    artifact_path = _resolve_generated_directory(artifacts_dir, database_path)
    report_path = _resolve_generated_directory(reports_dir, database_path)
    output = io.BytesIO()

    with tempfile.TemporaryDirectory(prefix="cwt-data-backup-") as temp_dir:
        snapshot_path = Path(temp_dir) / database_path.name
        with closing(sqlite3.connect(database_path)) as source:
            with closing(sqlite3.connect(snapshot_path)) as snapshot:
                source.backup(snapshot)

        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot_path, f"database/{database_path.name}")
            for directory, prefix in (
                (artifact_path, "artifacts"),
                (report_path, "reports"),
            ):
                if directory.exists():
                    if not directory.is_dir():
                        raise ValueError(f"Expected a directory at {directory}.")
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            archive.write(
                                path,
                                f"{prefix}/{path.relative_to(directory).as_posix()}",
                            )

    return output.getvalue()


def _resolve_generated_directory(directory: Path, database_path: Path) -> Path:
    if directory.is_symlink():
        raise ValueError(f"Refusing to reset a symlinked output directory: {directory}.")
    resolved = directory.resolve()
    project_root = Path.cwd().resolve()
    if resolved == project_root or resolved in project_root.parents:
        raise ValueError(f"Refusing to reset a protected directory: {resolved}.")
    database_resolved = database_path.resolve()
    if resolved == database_resolved or database_resolved in resolved.parents:
        raise ValueError(
            f"Output directory {resolved} must not contain the database file."
        )
    return resolved


def _clear_generated_directory(directory: Path, database_path: Path) -> None:
    resolved = _resolve_generated_directory(directory, database_path)
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise ValueError(f"Expected a directory at {resolved}.")
    for child in resolved.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _render_live_metrics(prices: pd.DataFrame, posts: pd.DataFrame, score: float) -> None:
    latest = float(prices["close"].iloc[-1])
    previous = float(prices["close"].iloc[-2]) if len(prices) > 1 else latest
    change = latest / previous - 1 if previous else 0.0
    sentiment_label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    columns = st.columns(2)
    columns[0].metric("Latest close", f"{latest:,.2f}", f"{change:+.2%} latest bar")
    columns[1].metric("24h X sentiment", sentiment_label, f"{score:+.3f} compound")
    columns = st.columns(2)
    columns[0].metric("Posts · last 24h", f"{len(posts):,}", "Apify / historical data")
    coverage = "No recent posts" if posts.empty else "Sentiment from VADER"
    columns[1].metric("Sentiment coverage", coverage, "UTC rolling window")


def _render_price_sentiment(
    prices: pd.DataFrame, tweets: pd.DataFrame, interval: str, lookback_days: int
) -> None:
    st.markdown('<div class="section-label">01 / MARKET & SOCIAL PULSE</div>', unsafe_allow_html=True)
    frequency = "h" if interval == "hourly" else "D"
    price_frame = prices[["close"]].copy()
    price_frame.index = pd.to_datetime(price_frame.index, utc=True).floor(frequency)
    price_frame = price_frame[~price_frame.index.duplicated(keep="last")]
    sentiment = _sentiment_series(tweets, frequency)
    chart = price_frame.join(sentiment.rename("sentiment"), how="left").fillna({"sentiment": 0.0})
    chart = chart[chart.index >= chart.index.max() - pd.Timedelta(days=lookback_days)]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["close"],
            name="Close",
            line={"color": "#67e8c1", "width": 2},
            hovertemplate="%{x|%b %d %H:%M UTC}<br>Close: %{y:,.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["sentiment"],
            name="Mean X sentiment",
            line={"color": "#f5a76c", "width": 1.4},
            fill="tozeroy",
            fillcolor="rgba(245, 167, 108, 0.08)",
            hovertemplate="%{x|%b %d %H:%M UTC}<br>Sentiment: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        **_chart_layout(height=390),
        hovermode="x unified",
    )
    figure.update_layout(legend={"orientation": "h", "y": 1.08, "x": 0})
    figure.update_yaxes(title_text="Close", secondary_y=False, showgrid=True)
    figure.update_yaxes(title_text="Sentiment (-1 to +1)", secondary_y=True, showgrid=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Sentiment is VADER compound score aggregated per bar. Missing-post periods are shown as zero."
    )


def _render_forecasts(
    forecasts: dict[str, float],
    prices: pd.DataFrame,
    forecast_sources: dict[str, str],
) -> None:
    st.markdown('<div class="section-label">02 / SIX-HORIZON FORECAST</div>', unsafe_allow_html=True)
    if not forecasts:
        st.info("Forecast file not found. Run `cwt-agent train` to create the six-horizon output.")
        return
    fallback_horizons = [
        horizon
        for horizon, source in forecast_sources.items()
        if source == "persistence_baseline"
    ]
    if fallback_horizons:
        st.info(
            "The Random Forest did not beat persistence MAE for "
            f"{', '.join(fallback_horizons)}. Those horizons use the latest close.",
            title="Persistence fallback",
        )
    price_now = float(prices["close"].iloc[-1])
    for row_start in (0, 3):
        columns = st.columns(3)
        for horizon, column in zip(range(row_start, row_start + 3), columns):
            key = f"n+{horizon}"
            value = forecasts.get(key)
            if value is None:
                column.metric(key, "—")
            else:
                delta = (
                    None
                    if horizon == 0 or price_now == 0
                    else f"{(value / price_now - 1):+.2%} vs n+0"
                )
                column.metric(key, f"{value:,.2f}", delta)
            source = forecast_sources.get(key)
            source_label = {
                "random_forest": "Random Forest",
                "persistence_baseline": "Persistence fallback",
            }.get(source)
            if source_label:
                column.caption(source_label)
    st.caption(
        "n+0 is the latest observed completed-bar close. n+1…n+5 are model forecasts, not financial advice."
    )


def _render_performance(metrics: dict[str, dict[str, float]]) -> None:
    st.markdown('<div class="section-label">03 / MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    if not metrics:
        st.info("No walk-forward predictions found. Run `cwt-agent train` to generate evaluation results.")
        return
    st.caption(
        "Metrics are for the raw Random Forest candidate; forecast horizons that fail "
        "to beat persistence MAE use a persistence fallback."
    )
    table = pd.DataFrame.from_dict(metrics, orient="index")
    table.index.name = "Horizon"
    table = table.rename(
        columns={
            "mae": "MAE",
            "rmse": "RMSE",
            "directional_accuracy": "Direction accuracy",
            "persistence_mae": "Persistence MAE",
            "mae_improvement_vs_persistence_pct": "MAE change vs baseline",
            "majority_direction_accuracy": "Majority direction",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "max_drawdown": "Max drawdown",
            "oos_samples": "OOS samples",
        }
    )
    styled = table.style.format(
        {
            "MAE": "{:,.4f}",
            "RMSE": "{:,.4f}",
            "Direction accuracy": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Max drawdown": "{:.1%}",
            "OOS samples": "{:,.0f}",
            "MAE change vs baseline": "{:+.1f}%",
        },
        na_rep="—",
    )
    st.dataframe(styled, width="stretch")
    st.caption(
        "Directional metrics use non-overlapping horizon-length trades with a 10 bps round-trip cost."
    )


def _render_data_quality(prices: pd.DataFrame, tweets: pd.DataFrame, interval: str) -> None:
    coverage = sentiment_coverage(prices, tweets, interval)
    st.markdown('<div class="section-label">DATA COVERAGE / FEATURE CHECK</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Posts in price window", f"{coverage['post_count']:,}")
    columns[1].metric(
        "Bars with posts",
        f"{coverage['active_periods']:,} / {coverage['bar_count']:,}",
    )
    columns[2].metric("Bar coverage", f"{coverage['bar_coverage_pct']:.2f}%")
    columns[3].metric("Active calendar days", f"{coverage['active_days']:,}")
    if coverage["post_count"] < 100 or coverage["bar_coverage_pct"] < 10:
        st.warning(
            "X coverage is sparse. These rows are suitable for pipeline checks, not "
            "reliable sentiment conclusions. Review the actor's date filters and dataset limits."
        )


def _render_walk_forward(
    predictions: pd.DataFrame, metrics: dict[str, dict[str, float]]
) -> None:
    st.markdown('<div class="section-label">04 / WALK-FORWARD VALIDATION</div>', unsafe_allow_html=True)
    if predictions.empty or not metrics:
        st.info("Walk-forward predictions are not available yet.")
        return
    available = sorted(metrics, key=lambda label: int(label.split("+")[1]))
    selected = st.selectbox("Validation horizon", available, index=0, key="validation_horizon")
    horizon = int(selected.split("+")[1])
    frame = predictions[predictions["horizon"] == horizon].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    if len(frame) > 500:
        frame = frame.tail(500)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=frame["timestamp"], y=frame["actual_close"], name="Actual", line={"color": "#67e8c1"})
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["predicted_close"],
            name="Walk-forward prediction",
            line={"color": "#a78bfa", "width": 1.5},
        )
    )
    figure.update_layout(**_chart_layout(height=340), hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    left, right = st.columns(2)
    left.metric("Validation rows", f"{len(frame):,}")
    residual = frame["predicted_close"] - frame["actual_close"]
    right.metric("Mean absolute error", f"{residual.abs().mean():,.4f}")
    st.caption("Expanding-window out-of-sample observations from saved fold predictions.")


def _render_news(news: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">05 / NEWS BRIEF</div>', unsafe_allow_html=True)
    st.caption(
        "Local extractive summaries select salient sentences from saved news; article text stays on this machine."
    )
    if news.empty:
        st.info("No news stored. Configure Tavily or Exa and click **Refresh data**.")
        return
    for _, item in news.head(8).iterrows():
        title = str(item.get("title") or "Untitled")
        body = str(item.get("content") or "")
        summary = _extractive_summary(f"{title}. {body}")
        source = str(item.get("source") or "news")
        timestamp = item.get("timestamp")
        date_text = ""
        if pd.notna(timestamp) and timestamp:
            parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
            if pd.notna(parsed):
                date_text = f" · {parsed.strftime('%b %d, %Y')}"
        url = str(item.get("url") or "")
        with st.container(border=True):
            st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
            st.caption(f"{source.upper()}{date_text}")
            st.write(summary)


def _render_recent_posts(tweets: pd.DataFrame) -> None:
    with st.expander("Recent X posts · sentiment context"):
        if tweets.empty:
            st.info("No X posts are stored for this asset.")
            return
        posts = tweets.copy()
        posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
        posts = posts.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False).head(10)
        analyzer = SentimentIntensityAnalyzer()
        for _, post in posts.iterrows():
            score = analyzer.polarity_scores(str(post.get("text") or ""))["compound"]
            stamp = post["timestamp"].strftime("%b %d · %H:%M UTC")
            st.markdown(f"`{stamp}` · sentiment **{score:+.2f}**")
            st.write(str(post.get("text") or ""))
            st.divider()


def _recent_sentiment(tweets: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if tweets.empty:
        return tweets, 0.0
    posts = tweets.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    recent = posts[posts["timestamp"] >= cutoff]
    if recent.empty:
        return recent, 0.0
    analyzer = SentimentIntensityAnalyzer()
    score = recent["text"].fillna("").map(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    ).mean()
    return recent, float(score)


def _sentiment_series(tweets: pd.DataFrame, frequency: str) -> pd.Series:
    if tweets.empty:
        return pd.Series(dtype=float, name="sentiment")
    analyzer = SentimentIntensityAnalyzer()
    posts = tweets.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    posts = posts.dropna(subset=["timestamp"])
    posts["period"] = posts["timestamp"].dt.floor(frequency)
    posts["sentiment"] = posts["text"].fillna("").map(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    return posts.groupby("period")["sentiment"].mean().rename("sentiment")


def _extractive_summary(text: str, sentence_limit: int = 2) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if len(sentences) <= sentence_limit:
        return " ".join(sentences) or "No article summary available."
    tokenized = [
        [word.lower() for word in re.findall(r"[A-Za-z0-9$'-]+", sentence) if len(word) > 2]
        for sentence in sentences
    ]
    frequencies: dict[str, int] = {}
    for words in tokenized:
        for word in words:
            if word not in STOP_WORDS:
                frequencies[word] = frequencies.get(word, 0) + 1
    scored = []
    for index, (sentence, words) in enumerate(zip(sentences, tokenized)):
        salient = [word for word in words if word not in STOP_WORDS]
        score = sum(frequencies[word] for word in salient) / max(len(salient), 1)
        scored.append((score, -index, sentence))
    selected = sorted(scored, reverse=True)[:sentence_limit]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: -item[1]))


def _load_predictions(settings: Settings) -> pd.DataFrame:
    path = settings.reports_dir / f"{_safe_asset(settings.asset)}_{settings.interval}_oos_predictions.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_forecasts(settings: Settings) -> dict[str, float]:
    path = settings.reports_dir / f"{_safe_asset(settings.asset)}_{settings.interval}_forecast.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    forecasts = payload.get("forecasts", {})
    return {str(key): float(value) for key, value in forecasts.items()}


def _load_forecast_sources(settings: Settings) -> dict[str, str]:
    path = settings.reports_dir / f"{_safe_asset(settings.asset)}_{settings.interval}_forecast.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("forecast_sources", {})
    if not isinstance(sources, dict):
        raise ValueError("Forecast sources in the forecast file must be a JSON object.")
    return {str(key): str(value) for key, value in sources.items()}


def _safe_asset(asset: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in asset)


def _chart_layout(height: int) -> dict:
    return {
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 28, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1"},
        "xaxis": {"gridcolor": "rgba(148,163,184,0.12)", "zerolinecolor": "rgba(148,163,184,0.12)"},
        "yaxis": {"gridcolor": "rgba(148,163,184,0.12)", "zerolinecolor": "rgba(148,163,184,0.12)"},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
    }


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #080d13; color: #e2e8f0; font-family: system-ui, sans-serif; }
        [data-testid="stSidebar"] { background: #0c131c; border-right: 1px solid #1e2936; }
        .block-container { max-width: 1440px; padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: -0.04em; }
        .eyebrow, .section-label { color: #70daba; font: 500 0.72rem 'DM Mono', monospace; letter-spacing: .12em; }
        .section-label { margin: 2.2rem 0 0.8rem; color: #8293a8; }
        [data-testid="stMetric"] { background: #101923; border: 1px solid #1d2b37; border-radius: 12px; padding: 16px 18px; }
        [data-testid="stMetricLabel"] { color: #93a4b7; }
        [data-testid="stMetricValue"] { color: #e6edf5; font-family: ui-monospace, monospace; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: #101923; border-color: #1d2b37; }
        a { color: #70daba !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
