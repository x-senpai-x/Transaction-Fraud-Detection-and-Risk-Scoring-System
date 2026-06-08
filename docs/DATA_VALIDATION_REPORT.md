# Data Validation Report

## Local Files Checked

The current `data/raw` directory contains:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Local junk files such as `.DS_Store` may appear on macOS and are ignored by `.gitignore`.

## Verdict

Pass for the planned baseline project.

The labeled training files are present and complete enough to build the end-to-end resume baseline: ingestion, chronological splitting, feature engineering, model training, evaluation, saved artifacts, API, dashboard, and SHAP explanations.

The official test files are useful for schema compatibility checks, inference demos, and future Kaggle-style scoring, but they cannot be used for local supervised training or evaluation because they do not contain true labels.

## Counts

Line counts include the header row.

| File | Lines | Data Rows | Columns | Target Present |
| --- | ---: | ---: | ---: | --- |
| `train_transaction.csv` | 590,541 | 590,540 | 394 | yes |
| `train_identity.csv` | 144,234 | 144,233 | 41 | no |
| `test_transaction.csv` | 506,692 | 506,691 | 393 | no |
| `test_identity.csv` | 141,908 | 141,907 | 41 | no |
| `sample_submission.csv` | 506,692 | 506,691 | 2 | placeholder only |

## Training Data

`train_transaction.csv`:

- has `TransactionID`,
- has `isFraud`,
- has `TransactionDT`,
- has `TransactionAmt`,
- has expected card, address, email, count, distance, and `V*` feature groups,
- has 20,663 fraud rows,
- fraud rate is about 3.499%.

`train_identity.csv`:

- has `TransactionID`,
- has `id_01` through `id_38`,
- has `DeviceType`,
- has `DeviceInfo`.

Join coverage:

- 144,233 of 590,540 training transactions have identity rows.
- Training identity match rate is about 24.42%.
- Sparse identity coverage is expected for this dataset.

Chronological 70/15/15 split viability:

| Split | Rows | Fraud Rows | Fraud Rate | `TransactionDT` Range |
| --- | ---: | ---: | ---: | --- |
| Train | 413,378 | 14,538 | 3.517% | 86,400 to 10,437,996 |
| Validation | 88,581 | 3,042 | 3.434% | 10,438,003 to 13,151,840 |
| Local test | 88,581 | 3,083 | 3.480% | 13,151,880 to 15,811,131 |

This is enough support for PR-AUC, Recall@Top-K, precision-at-threshold, threshold tuning, ablations, and final local model reporting.

Training ranges:

- `TransactionID`: 2,987,000 to 3,577,539.
- `TransactionDT`: 86,400 to 15,811,131.

## Test Data

`test_transaction.csv`:

- has expected transaction features,
- does not have `isFraud`,
- has 506,691 rows.

`test_identity.csv`:

- has expected identity/device fields,
- uses `id-01` through `id-38` with hyphens, unlike train identity's `id_01` through `id_38`.

Join coverage:

- 141,907 of 506,691 test transactions have identity rows.
- Test identity match rate is about 28.01%.

Test ranges:

- `TransactionID`: 3,663,549 to 4,170,239.
- `TransactionDT`: 18,403,224 to 34,214,345.

## Important Implementation Notes

- Phase 1 should use `train_transaction.csv` and `train_identity.csv` as the source of labeled data.
- The official test files must not be used for PR-AUC, Recall@Top-K, threshold tuning, or local validation.
- `sample_submission.csv` contains placeholder `isFraud = 0.5` values and is not a label file.
- Phase 1 should standardize identity column names by replacing hyphens with underscores after loading, so train/test schemas are compatible.
- Formal train/validation/test splits should be created from labeled training rows using chronological `TransactionDT` order.

## Whole-Project Suitability

| Project Area | Suitable? | Notes |
| --- | --- | --- |
| Data ingestion and join | yes | `TransactionID` is present in transaction and identity files. Identity is sparse but expected. |
| Chronological split | yes | `TransactionDT` is complete and ordered enough for train/validation/local-test splits. |
| Baseline features | yes | Amount, product, card, address, email, count, distance, and anonymized `V*` fields are available. |
| Historical features | yes | Relative time and proxy entities support past-only counts, amount stats, and time-since-previous features. |
| Velocity features | yes | `TransactionDT` supports trailing-window features, though it is relative time, not wall-clock time. |
| LightGBM/XGBoost model | yes | Dataset is tabular, large enough, and has a realistic imbalanced target. |
| PR-AUC and Recall@Top-K | yes | Validation and local test splits each have over 3,000 fraud rows. |
| API demo | yes | Use saved feature metadata and sampled transactions. Do not retrain inside the API. |
| Streamlit dashboard | yes | Metrics, score distributions, transaction explorer, and explanations can all be shown. |
| SHAP explanations | yes | Best suited for the boosted tree model and engineered features. |
| Drift monitoring | yes, with caveat | Feature/prediction drift can be simulated across chronological splits or unlabeled official test data. Label-based post-deployment performance drift requires labels and cannot use official test labels. |
| Graph features | yes, simple only | Proxy entity relationships are available, but they are not true customer/device IDs. Use simple graph-degree/count features before any embeddings. |
| Champion-challenger | yes | Use later chronological windows from labeled training data. |

## Main Caveats

- The official Kaggle test set has no true labels, so final project metrics must come from a chronological holdout inside `train_transaction.csv`.
- `TransactionDT` is relative time. It is valid for ordering and windows, but not for real dates, time zones, weekdays, holidays, or calendar seasonality.
- Many fields are anonymized or pre-engineered. Avoid overclaiming business interpretability for `V*`, `C*`, `D*`, and `M*` fields.
- Identity/device rows exist for only about one quarter of training transactions. Device features need missingness handling and should not be mandatory for prediction.
- Receiver email is mostly missing. It can be used with missing indicators, but should not be central to the baseline.
- The full CSVs are large. Pandas can handle them on a normal laptop, but feature generation should avoid unnecessary full copies and should save intermediate Parquet artifacts.
