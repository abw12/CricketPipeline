# Cricket Pipeline

PySpark ETL application for IPL Cricsheet ball-by-ball data. The data flow is:

`raw Cricsheet JSON -> Bronze -> Silver -> Gold`

Bronze keeps flattened source-aligned tables, Silver normalizes and enriches entities, and Gold produces analytics-ready season and match summary tables.

## Setup

Install runtime and test dependencies with Poetry:

```powershell
poetry install --with dev --without notebooks
```

Install notebook dependencies only when needed:

```powershell
poetry install --with dev,notebooks
```

Run commands from the project root. If you run the installed package from another directory, set:

```powershell
$env:CRICKET_PIPELINE_ROOT="C:\Projects\CricketPipeline"
```

## CLI

List available steps:

```powershell
poetry run cricket-pipeline list-steps
```

Inspect raw JSON:

```powershell
poetry run cricket-pipeline inspect-raw --files 3
```

Run the full pipeline:

```powershell
poetry run cricket-pipeline run --stage all
```

Run one stage:

```powershell
poetry run cricket-pipeline run --stage bronze
poetry run cricket-pipeline run --stage silver
poetry run cricket-pipeline run --stage gold
```

Run one registered step:

```powershell
poetry run cricket-pipeline run --step bronze-deliveries
```

Preview without writing:

```powershell
poetry run cricket-pipeline run --step bronze-matches --sample --no-write
```

On local Windows runs, physical `partitionBy` writes are disabled by default to avoid local Hadoop path/permission issues. The partition columns remain in the data. Set `CRICKET_PIPELINE_FORCE_PARTITIONS=1` to force configured partitioned writes.

## Dashboard

Run the Streamlit dashboard from the project root:

```powershell
poetry run cricket-dashboard
```

You can also run Streamlit directly:

```powershell
poetry run streamlit run dashboard/app/app.py
```

The dashboard reads processed Parquet files from `data/processed/gold` and selected `data/processed/silver` tables. Run the ETL pipeline first if the processed data is missing or stale.

## Tests

```powershell
poetry run pytest
```
