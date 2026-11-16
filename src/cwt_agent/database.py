from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    asset TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL,
                    volume REAL,
                    PRIMARY KEY (asset, timestamp)
                );
                CREATE TABLE IF NOT EXISTS tweets (
                    id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    likes REAL NOT NULL DEFAULT 0,
                    reposts REAL NOT NULL DEFAULT 0,
                    views REAL NOT NULL DEFAULT 0,
                    followers REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_tweets_asset_time
                    ON tweets(asset, timestamp);
                CREATE TABLE IF NOT EXISTS news (
                    url TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timestamp TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_prices(self, asset: str, rows: list[dict[str, Any]]) -> None:
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO prices(asset, timestamp, open, high, low, close, volume)
                VALUES (:asset, :timestamp, :open, :high, :low, :close, :volume)
                ON CONFLICT(asset, timestamp) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume
                """,
                [dict(asset=asset, **row) for row in rows],
            )

    def save_tweets(self, asset: str, rows: list[dict[str, Any]]) -> None:
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO tweets(id, asset, timestamp, text, likes, reposts, views, followers)
                VALUES (:id, :asset, :timestamp, :text, :likes, :reposts, :views, :followers)
                ON CONFLICT(id) DO UPDATE SET
                    asset=excluded.asset, timestamp=excluded.timestamp,
                    text=excluded.text, likes=excluded.likes, reposts=excluded.reposts,
                    views=excluded.views, followers=excluded.followers
                """,
                [dict(asset=asset, **row) for row in rows],
            )

    def save_news(self, asset: str, rows: list[dict[str, Any]]) -> None:
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO news(url, asset, timestamp, title, content, source)
                VALUES (:url, :asset, :timestamp, :title, :content, :source)
                ON CONFLICT(url) DO UPDATE SET
                    asset=excluded.asset, timestamp=excluded.timestamp,
                    title=excluded.title, content=excluded.content, source=excluded.source
                """,
                [dict(asset=asset, **row) for row in rows],
            )

    def clear_all_data(self) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self.connection() as connection:
            for table in ("prices", "tweets", "news"):
                deleted[table] = connection.execute(
                    f"DELETE FROM {table}"
                ).rowcount
        return deleted

    def load_prices(self, asset: str) -> pd.DataFrame:
        with self.connection() as connection:
            frame = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM prices "
                "WHERE asset = ? ORDER BY timestamp",
                connection,
                params=(asset,),
            )
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        return frame

    def load_tweets(self, asset: str) -> pd.DataFrame:
        with self.connection() as connection:
            frame = pd.read_sql_query(
                "SELECT id, timestamp, text, likes, reposts, views, followers "
                "FROM tweets WHERE asset = ? ORDER BY timestamp",
                connection,
                params=(asset,),
            )
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def load_news(self, asset: str, limit: int = 20) -> pd.DataFrame:
        with self.connection() as connection:
            return pd.read_sql_query(
                "SELECT url, timestamp, title, content, source FROM news "
                "WHERE asset = ? ORDER BY timestamp DESC LIMIT ?",
                connection,
                params=(asset, limit),
            )

    def list_assets(self) -> list[str]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT asset FROM prices GROUP BY asset ORDER BY asset"
            ).fetchall()
        return [str(row["asset"]) for row in rows]
