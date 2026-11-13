# CWT X Sentiment + Price Prediction

A Python research pipeline and Streamlit dashboard for exploring X (Twitter) sentiment
alongside cryptocurrency prices. It ingests price bars, X posts, and news; builds
time-aligned price and sentiment features; evaluates price forecasts with walk-forward
validation; and generates reports and charts.

The default configuration is BTC-USD, hourly bars, and a 180-day lookback. The model
uses **price and X sentiment features only**. Tavily/Exa news is stored for the
human-readable outlook and is not used as a model feature.

> **Research prototype:** forecasts and trading metrics are experimental, not financial
> advice. X coverage can be sparse, and the current model has not demonstrated a
> predictive or profitable edge. Review coverage and baseline comparisons in the latest
> evaluation report before drawing conclusions.

## Features

- Price ingestion and SQLite storage.
- Apify X-post ingestion, normalization, deduplication, and lookback filtering.
- Optional Tavily/Exa news ingestion and local extractive summaries.
- Hourly/daily feature construction and sentiment coverage diagnostics.
- Five walk-forward forecast horizons (`n+1` through `n+5`); `n+0` is the latest
  observed close, not a forecast.
- Persistence and direction-majority baselines, plus price-error and strategy metrics.
- Evaluation reports, sentiment summaries, charts, and a Streamlit dashboard.

## Requirements and installation

Python 3.10 or later is required. From PowerShell in the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Edit `.env` locally with your own Apify token and Actor settings. Tavily and Exa keys
are optional; if you do not configure a news provider, use `--skip-news`. **Never paste
credentials into chat or commit `.env`.** The `.env.example` file contains placeholders,
not working credentials.

For Streamlit Community Cloud, add the same settings as root-level TOML entries in the
app's **Settings → Secrets**. For example:

```toml
APIFY_API_TOKEN = "your-rotated-token"
APIFY_ACTOR_ID = "apidojo/twitter-scraper-lite"
APIFY_INPUT_JSON = '{"searchTerms":["BTC"],"sort":"Latest","maxItems":20,"includeSearchTerms":false,"start":"{{since}}","end":"{{until}}"}'
```

Do not include the example token literally. Save the secrets and reboot the app after
changing them. Streamlit Cloud does not receive your local `.env` or SQLite database;
the hosted dashboard starts with its own local database and must be populated by a data
refresh or an import workflow.

## Run the pipeline

Fetch prices and X data, skip optional news, train/evaluate, and generate reports:

```powershell
cwt-agent run --skip-news
```

If you have configured a news provider and want news too:

```powershell
cwt-agent run
```

The CLI also supports individual phases:

```powershell
cwt-agent ingest --skip-news
cwt-agent train
cwt-agent report
```

You can skip X ingestion with `--skip-x`. Historical posts can be imported from CSV,
JSON, or JSONL without calling Apify:

```powershell
cwt-agent run --skip-x --skip-news --tweets-file .\historical_posts.csv
```

The importer recognizes common text fields (`text`, `fullText`, `full_text`) and
timestamp fields (`createdAt`, `created_at`, `timestamp`, `date`). Imported posts must
have usable text and timestamps.

## Apify configuration and historical collection

Actor input schemas differ. Confirm that `APIFY_ACTOR_ID` and `APIFY_INPUT_JSON` match
the selected Actor. The example configuration targets
`apidojo/twitter-scraper-lite`:

```dotenv
APIFY_ACTOR_ID=apidojo/twitter-scraper-lite
APIFY_INPUT_JSON={"searchTerms":["BTC"],"sort":"Latest","maxItems":20,"includeSearchTerms":false,"start":"{{since}}","end":"{{until}}"}
```

When both top-level `start` and `end` fields are present, the app replaces them with
consecutive monthly windows covering `LOOKBACK_DAYS` and starts one Actor run per
window. A 180-day collection therefore typically starts about six or seven Actor runs.
It combines the datasets, normalizes posts, filters to the lookback, and deduplicates
posts by ID.

Review current Actor pricing before collecting. Test one or two date windows in Apify
Console first, confirm returned timestamps match the requested dates, and keep an
appropriate maximum-cost limit enabled. Verify whether the provider's cost limit applies
to API-triggered runs. The dashboard's **Refresh data** button starts the configured
collection, so it may incur the cost of all monthly runs. `maxItems` is the Actor-side
result cap **per window**;
`APIFY_MAX_DATASET_ITEMS` is the client-side dataset download cap **per run**. Neither
setting guarantees broad date coverage. Increase result limits only after checking the
Actor's pricing and the pilot results.

For Actors that use date operators inside search terms, the older `{{since}}` and
`{{until}}` placeholder expansion remains supported. Use the date-filter format
documented by the selected Actor; no date slicing occurs if the input has neither
top-level date fields nor both query placeholders. Sparse-coverage warnings are
intentional: they indicate that the collected rows are not sufficient for reliable
sentiment conclusions.

## Dashboard

Start the local dashboard from the project root:

```powershell
streamlit run src/cwt_agent/dashboard.py
```

The dashboard displays live asset sentiment, historical price/sentiment charts,
news summaries, six displayed price points (`n+0` plus five future horizons),
model metrics, and walk-forward validation results. It reads the configured SQLite
database and saved model outputs. **Refresh data** fetches new provider data; run
`cwt-agent train` separately to refresh forecasts and evaluation metrics.
If a provider refresh fails, the dashboard displays the error and keeps rendering any
previously stored data.

To load real X posts without starting an Apify run, open **Import historical X posts**
on the dashboard, choose a CSV, JSON, or JSONL export, then submit the form. The file
must contain post text and timestamps (for example `text` and `createdAt`; common
aliases are supported). The dashboard imports only posts within `LOOKBACK_DAYS` for
the currently selected asset, then immediately refreshes the sentiment views. Streamlit
Cloud does not share the local database, and its app-local SQLite data may be lost when
the hosted app is restarted or redeployed; keep the source export so you can re-import it.

## Evaluation notes and limitations

- `n+0` is the latest observed completed bar close; it is not a model prediction.
- `n+1` through `n+5` are forecasts one through five bars ahead. Each horizon uses a
  separate Random Forest model.
- Walk-forward validation preserves time ordering and uses a five-bar purge gap.
- Reports compare MAE/RMSE with a persistence baseline and directional accuracy with a
  majority-direction baseline computed from training data.
- Strategy metrics use non-overlapping horizon-length trades and assume a 10 bps
  round-trip cost. Fees, slippage, and market impact may differ.
- Sentiment scores use equal-weight posts. Engagement and follower counts are stored but
  excluded from model features; missing-post bars have a separate availability feature.
- Hourly bars are treated as completed at their provider timestamp; posts in each bar
  are used to predict later bars.
- A minimum of 120 usable rows is required. Performance metrics do not guarantee future
  results, and this research pipeline is not a trading recommendation.
- Historical conclusions are only as trustworthy as the X collection's date coverage.
  Check post count, active days, active periods, and coverage percentage in each new
  evaluation report.

## Tests

Run the test suite from the project root:

```powershell
python -m pytest -q
```

## Generated files

- `data/agent.sqlite3`: normalized prices, X posts, and news.
- `artifacts/`: fitted per-horizon models.
- `reports/*_evaluation.md`: metrics, forecasts, and X-coverage diagnostics.
- `reports/*_sentiment_summary.md`: current sentiment and news brief.
- `reports/sentiment_price.png` and `reports/forecast_matrix.png`: visualizations.
- `reports/*_oos_predictions.csv`: walk-forward predictions.

Generated data, reports, fitted models, virtual environments, and credentials are
excluded from Git by `.gitignore`.
