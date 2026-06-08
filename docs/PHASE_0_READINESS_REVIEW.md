# Phase 0 Readiness Review

## Status

Phase 0 repository setup is complete.

Implemented:

- Python package skeleton under `src/transaction_risk_engine`.
- Checked-in baseline config at `config/config.yaml`.
- Example config at `config/config.example.yaml`.
- Dependency files: `pyproject.toml`, `requirements.txt`, `requirements.baseline.txt`.
- Artifact directories with `.gitkeep` files.
- `.gitignore` for local junk, raw data, generated features, reports, and model artifacts.
- Minimal config loader and tests.
- README setup, validation, data placement, and planned milestone commands.

Validation run:

```bash
python -m pip install pytest pyyaml
python -m pip install -e . --no-deps
python -m pytest
python -c "from transaction_risk_engine.config import load_config; print(load_config()['project']['name'])"
```

Result:

- `3 passed`
- import smoke test returned `transaction-risk-engine`

## Environment Note

The current shell uses Python 3.13.13. Phase 0 tests pass under that interpreter, but the project should still use Python 3.11 or 3.12 for the full ML stack unless dependency compatibility is verified separately.

## Dataset Fit

IEEE-CIS is a good fit for this project because it provides:

- transaction-level fraud labels in `isFraud`,
- a relative time field suitable for chronological validation,
- transaction amount, card, address, email, device, and identity fields,
- class imbalance suitable for PR-AUC and Recall@Top-K evaluation,
- enough rows for train/validation/test splitting and feature ablations.

The main limitation is that many features are anonymized, so the demo should emphasize risk-engineering discipline rather than pretending every feature is business-interpretable.

External references checked:

- Kaggle competition data page: https://www.kaggle.com/c/ieee-fraud-detection/data
- Kaggle dataset mirror showing the expected raw files: https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection
- IEEE-CIS walkthrough confirming `TransactionID` joins, sparse identity rows, `TransactionDT`, feature groups, row counts, and time-based splitting rationale: https://pcx.linkedinfo.co/post/fraud-detection/
- Academic analysis confirming merged row/column counts and approximate fraud prevalence: https://arno.uvt.nl/show.cgi?fid=161340

## Phase 1 Recommendation

Proceed to Phase 1 after the raw Kaggle files are placed in `data/raw`.

Detailed implementation plan: `docs/PHASE_1_IMPLEMENTATION_PLAN.md`.

Phase 1 should implement only:

- missing-file checks,
- loading `train_transaction.csv` and `train_identity.csv`,
- left join on `TransactionID`,
- proxy ID creation,
- derived relative timestamp fields,
- Parquet output,
- data profile report,
- tests for target preservation, row preservation, and missing-file failures.
