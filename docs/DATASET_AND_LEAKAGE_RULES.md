# Dataset and Leakage Rules

## Dataset

Use the IEEE-CIS Fraud Detection dataset.

Primary source:

- Kaggle competition: `ieee-fraud-detection`

The official Kaggle test labels are hidden. For this project, create train/validation/test splits from the labeled training files only. Do not use Kaggle's unlabeled test files for local model evaluation.

Download expectation:

```bash
python -m pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

If the competition download is unavailable, use a Kaggle mirror only as a fallback and keep the file names below unchanged.

Main files:

- `train_transaction.csv`
- `train_identity.csv`

Optional Kaggle files:

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Join key:

- `TransactionID`

Target:

- `isFraud`

Expected labeled training shape, before local validation:

- `train_transaction.csv`: about 590k rows and 394 columns.
- `train_identity.csv`: about 144k rows and 41 columns.
- Left-joined train table: about 590k rows and 434 columns.
- Fraud prevalence: about 3.5%.

The Phase 1 data profile must compute and report the exact local counts.

Important columns:

- `TransactionDT`
- `TransactionAmt`
- `ProductCD`
- `card1` to `card6`
- `addr1`, `addr2`
- `P_emaildomain`, `R_emaildomain`
- `DeviceType`, `DeviceInfo`
- `id_01` to `id_38`

Important caveats:

- `TransactionDT` is a time delta from an undisclosed reference datetime, not a real timestamp. Use it for chronological ordering and relative time features, but do not infer calendar dates or time zones from it.
- Not every transaction has a matching identity row. Use a left join from `train_transaction.csv` to `train_identity.csv` and keep all transactions.
- Many features are anonymized or pre-engineered by the data provider. Favor transparent baseline features first, then add broader anonymized feature blocks only after leakage checks and ablations.

## Proxy entity IDs

Because actual users/devices are anonymized, create proxy IDs:

```python
card_uid = card1 + '_' + card2 + '_' + card3 + '_' + card5 + '_' + card6
address_uid = addr1 + '_' + addr2
email_uid = P_emaildomain
receiver_email_uid = R_emaildomain
device_uid = DeviceType + '_' + DeviceInfo + '_' + id_30 + '_' + id_31
```

Do not assume these are perfect identities. Treat them as approximate entities.

## Leakage rules

### Rule 1 — Time split always

Final train/validation/test split must be chronological.

### Rule 2 — Historical features must be past-only

For transaction at time `t`, only use transactions with time `< t`.

### Rule 3 — Frequency encoding caution

Global frequency encoding can leak future distribution. Baseline can use train-fitted frequency maps, then apply to validation/test. Advanced version should use expanding historical counts.

### Rule 4 — Target encoding caution

Do not use target encoding unless implemented in a leakage-safe, time-aware way. Prefer frequency encoding first.

### Rule 5 — Entity fraud rates are dangerous

Entity fraud rates must never include the current transaction or future transactions. For baseline, avoid fraud-rate features unless carefully implemented.

### Rule 6 — Save artifacts

The exact preprocessing logic and feature columns used in training must be saved and reused for inference.
