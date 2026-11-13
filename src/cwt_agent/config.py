from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    asset: str
    interval: str
    lookback_days: int
    database_path: Path
    artifacts_dir: Path
    reports_dir: Path
    apify_api_token: str | None
    apify_actor_id: str | None
    apify_input_json: str | None
    tavily_api_key: str | None
    exa_api_key: str | None
    apify_max_dataset_items: int = 10_000
    apify_run_timeout_seconds: int = 1_800

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        interval = os.getenv("BAR_INTERVAL", "hourly").strip().lower()
        if interval not in {"hourly", "daily"}:
            raise ValueError("BAR_INTERVAL must be either 'hourly' or 'daily'.")
        lookback_days = int(os.getenv("LOOKBACK_DAYS", "180"))
        if lookback_days < 1:
            raise ValueError("LOOKBACK_DAYS must be a positive integer.")
        apify_max_dataset_items = int(os.getenv("APIFY_MAX_DATASET_ITEMS", "10000"))
        if apify_max_dataset_items < 1:
            raise ValueError("APIFY_MAX_DATASET_ITEMS must be a positive integer.")
        apify_run_timeout_seconds = int(os.getenv("APIFY_RUN_TIMEOUT_SECONDS", "1800"))
        if apify_run_timeout_seconds < 1:
            raise ValueError("APIFY_RUN_TIMEOUT_SECONDS must be a positive integer.")
        return cls(
            asset=os.getenv("ASSET", "BTC-USD").strip().upper(),
            interval=interval,
            lookback_days=lookback_days,
            database_path=Path(os.getenv("DATABASE_PATH", "data/agent.sqlite3")),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            reports_dir=Path(os.getenv("REPORTS_DIR", "reports")),
            apify_api_token=os.getenv("APIFY_API_TOKEN") or None,
            apify_actor_id=os.getenv("APIFY_ACTOR_ID") or None,
            apify_input_json=os.getenv("APIFY_INPUT_JSON") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            exa_api_key=os.getenv("EXA_API_KEY") or None,
            apify_max_dataset_items=apify_max_dataset_items,
            apify_run_timeout_seconds=apify_run_timeout_seconds,
        )

    @property
    def price_interval(self) -> str:
        return "1h" if self.interval == "hourly" else "1d"
