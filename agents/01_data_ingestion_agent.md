# Agent 01 — Data Ingestion

## Mission

Create a reliable data loading pipeline for IEEE-CIS transaction and identity files.

## Inputs

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`
- `config/config.yaml`

## Outputs

- `data/interim/joined_transactions.parquet`
- `reports/data_profile.md`

## Tasks

1. Load transaction and identity CSVs.
2. Join on `TransactionID` using left join from transaction to identity.
3. Standardize column names or maintain original names consistently.
4. Create proxy IDs:
   - `card_uid`
   - `address_uid`
   - `email_uid`
   - `device_uid`
5. Create derived timestamp column from `TransactionDT`.
6. Save joined data as Parquet.
7. Generate data profile:
   - shape,
   - target rate,
   - missingness,
   - top categorical counts,
   - numeric summary.

## Acceptance criteria

- Joined output contains `isFraud`.
- No target-derived features are created.
- Missing input files produce clear errors.
- Data profile is saved.

## Suggested command

```bash
python -m transaction_risk_engine.data.load --config config/config.yaml
```

## Agent prompt

```txt
Read AGENTS.md and docs/DATASET_AND_LEAKAGE_RULES.md.
Implement the data ingestion module only.
Create functions for loading, joining, proxy ID creation, and report generation.
Add basic tests.
Do not implement feature engineering yet.
```
