# Implementation Roadmap

## Phase 0 — Repository setup

Deliverables:

- project skeleton,
- requirements/pyproject,
- config YAML,
- sample data folder,
- tests folder,
- README with setup commands.

Acceptance criteria:

- `python -m pytest` runs,
- repo imports work,
- CLI entrypoints are documented.

## Phase 1 — Data ingestion

Deliverables:

- load `train_transaction.csv`,
- load `train_identity.csv`,
- join on `TransactionID`,
- create proxy IDs,
- save cleaned Parquet,
- generate data report.

Acceptance criteria:

- joined dataset has target `isFraud`,
- proxy IDs are present,
- no target-derived features are created.

## Phase 2 — Baseline features

Deliverables:

- amount features,
- time features,
- missingness flags,
- frequency encodings,
- time-based split.

Acceptance criteria:

- training/validation/test are chronological,
- no target column in feature matrix,
- feature metadata saved.

## Phase 3 — Baseline model

Deliverables:

- logistic regression baseline,
- LightGBM/XGBoost model,
- PR-AUC/ROC-AUC/Recall@Top-K report,
- saved model artifact.

Acceptance criteria:

- model training script runs from CLI,
- metrics are saved to reports,
- best model is serialized.

## Phase 4 — Behavioral features

Deliverables:

- historical counts and amount stats per entity,
- amount ratio to historical mean,
- time since previous transaction,
- ablation report comparing feature sets.

Acceptance criteria:

- historical features use only past transactions,
- ablation shows whether features help,
- feature code has tests.

## Phase 5 — API and dashboard

Deliverables:

- FastAPI `/health`, `/predict`, `/batch_predict`,
- Streamlit dashboard,
- transaction explorer,
- model metrics page.

Acceptance criteria:

- API returns valid probability and decision,
- dashboard starts locally,
- sample payload documented.

## Phase 6 — Explainability

Deliverables:

- SHAP explanations,
- global feature importance,
- readable reason strings.

Acceptance criteria:

- sample transaction explanation works,
- dashboard can display top reasons.

## Phase 7 — Advanced extensions

Pick based on remaining time:

1. velocity features,
2. graph features,
3. drift monitoring,
4. champion-challenger,
5. replay simulator.

Do not begin Phase 7 until Phase 1–6 are working.
