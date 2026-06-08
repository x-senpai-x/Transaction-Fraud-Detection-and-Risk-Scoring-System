# Agent 02 — Feature Engineering

## Mission

Create leakage-safe features for transaction fraud classification.

## Inputs

- `data/interim/joined_transactions.parquet`
- `config/features.yaml`

## Outputs

- `data/processed/features_train.parquet`
- `data/processed/features_valid.parquet`
- `data/processed/features_test.parquet`
- `models/feature_metadata.json`

## Feature blocks

### Block A — Basic features

- `TransactionAmt_log`
- `TransactionAmt_fraction`
- hour/day features from `TransactionDT`
- missingness indicators for identity columns

### Block B — Frequency features

Train-fitted counts for:

- `card1`, `card2`, `card3`, `card5`, `card6`
- `addr1`, `addr2`
- `P_emaildomain`, `R_emaildomain`
- `DeviceType`, `DeviceInfo`
- proxy IDs

### Block C — Historical features

For each entity:

- previous transaction count,
- previous mean amount,
- previous max amount,
- amount ratio to previous mean,
- time since previous transaction.

Must be computed using only previous rows in chronological order.

### Block D — Velocity features, optional for baseline

- transactions in trailing 1h/6h/24h windows,
- unique cards per device in recent window,
- unique devices per card in recent window.

## Acceptance criteria

- No target column in feature matrix.
- Time split is chronological.
- Feature columns are identical across train/valid/test.
- Historical features exclude current and future rows.

## Suggested command

```bash
python -m transaction_risk_engine.features.build --config config/config.yaml
```

## Agent prompt

```txt
Read AGENTS.md and docs/DATASET_AND_LEAKAGE_RULES.md.
Implement basic, frequency, and historical feature blocks.
Focus on leakage-safe correctness over speed.
Save feature metadata and add tests for chronological splits.
Do not implement graph features yet.
```
