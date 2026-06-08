# Phase 1 Implementation Plan

## Current Milestone

Phase 1 is data ingestion.

Do not implement feature engineering, modeling, API, dashboard, SHAP, drift monitoring, or graph features in this phase.

## Prerequisite

Place the labeled IEEE-CIS training files in:

```txt
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

The official Kaggle test files are optional and should not be used for local evaluation because their labels are hidden.

## Files To Create Or Edit

Create:

- `src/transaction_risk_engine/data/load.py`
- `src/transaction_risk_engine/data/schema.py`
- `src/transaction_risk_engine/utils/paths.py`
- `tests/test_data_load.py`

Edit if needed:

- `config/config.yaml`
- `config/config.example.yaml`
- `README.md`

Generated outputs:

- `data/interim/joined_transactions.parquet`
- `reports/data_profile.md`

## Functions To Implement

In `src/transaction_risk_engine/data/load.py`:

- `resolve_data_paths(config: dict) -> DataPaths`
- `load_transaction_data(path: Path) -> pandas.DataFrame`
- `load_identity_data(path: Path) -> pandas.DataFrame`
- `standardize_column_names(frame: pandas.DataFrame) -> pandas.DataFrame`
- `join_transaction_identity(transactions: pandas.DataFrame, identity: pandas.DataFrame) -> pandas.DataFrame`
- `add_proxy_ids(frame: pandas.DataFrame) -> pandas.DataFrame`
- `add_relative_time_columns(frame: pandas.DataFrame) -> pandas.DataFrame`
- `write_joined_data(frame: pandas.DataFrame, output_path: Path) -> None`
- `write_data_profile(frame: pandas.DataFrame, output_path: Path) -> None`
- `main() -> None`

In `src/transaction_risk_engine/data/schema.py`:

- required file names,
- required join key,
- required target,
- proxy ID source columns,
- important columns for profiling.

## Data Rules

- Use a left join from `train_transaction.csv` to `train_identity.csv`.
- Preserve every transaction row from `train_transaction.csv`.
- Preserve `isFraud`.
- Standardize raw identity column names so `id-01` and `id_01` become the same schema style.
- Do not create target-derived features.
- Do not drop columns silently.
- Do not infer real calendar dates from `TransactionDT`; it is a relative time delta.
- Treat missing identity information as expected, not as a load failure.

## Leakage Traps

- Do not compute fraud rates by card, device, email, address, or any other entity.
- Do not create global target encodings.
- Do not create train/validation/test splits in Phase 1 unless needed only for reporting; formal split belongs to Phase 2.
- Do not use Kaggle `test_transaction.csv` for metrics.
- Do not use any feature that depends on future rows.

## Tests To Add

Use small synthetic DataFrames, not the full Kaggle data.

Required tests:

- missing transaction file raises `FileNotFoundError`,
- missing identity file raises `FileNotFoundError`,
- join preserves all transaction rows,
- join preserves `isFraud`,
- join does not require every transaction to have identity data,
- proxy ID columns are created,
- proxy IDs handle missing values deterministically,
- identity columns with hyphens are standardized to underscores,
- `TransactionDT`-derived columns are created without requiring real datetimes,
- generated profile includes shape, target rate, missingness, categorical counts, and numeric summary.

## Commands To Run

Before raw data is available:

```bash
python -m pytest
```

After raw data is placed in `data/raw`:

```bash
python -m transaction_risk_engine.data.load --config config/config.yaml
python -m pytest
```

Expected generated files:

```txt
data/interim/joined_transactions.parquet
reports/data_profile.md
```

## Acceptance Criteria

Phase 1 is complete only when:

- the data ingestion CLI runs from the project root,
- missing raw files fail loudly with useful messages,
- the joined Parquet file exists,
- the data profile exists,
- tests pass,
- README commands are still accurate,
- no feature engineering or model training code has been added prematurely.
