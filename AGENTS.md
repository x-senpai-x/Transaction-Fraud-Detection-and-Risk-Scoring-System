# AGENTS.md — Global Instructions for AI Coding Agents

This file is the central instruction document for Claude Code, Codex, Antigravity, Cursor, Copilot, or any other agent working on Transaction Risk Engine.

## Project mission

Build a resume-ready adaptive transaction risk engine. The baseline must be complete, correct, and demoable before advanced experiments are added.

Do not over-engineer. The priority order is:

1. Correct leakage-safe ML pipeline.
2. Strong model evaluation.
3. Clean API and dashboard demo.
4. Explainability.
5. Monitoring and graph extensions.

## Non-negotiable rules

1. Use time-based splits. Do not random split for final results.
2. Do not use future information while creating historical features.
3. Do not use accuracy as the primary metric.
4. Always report PR-AUC and Recall@Top-K.
5. Every major script must be runnable from CLI.
6. Every generated artifact must have a clear output path.
7. Keep all configuration in YAML/env files, not hardcoded in scripts.
8. Use deterministic seeds.
9. Prefer simple reliable implementation over clever incomplete implementation.
10. Update README whenever user-facing commands change.

## Baseline definition of done

The baseline is complete only when the following work end-to-end:

- data load and join,
- time split,
- feature matrix generation,
- baseline model training,
- LightGBM/XGBoost training,
- evaluation report,
- saved model,
- `/predict` API,
- Streamlit dashboard,
- basic SHAP explanation,
- README with commands and screenshots placeholders.

## Preferred stack

- Python 3.11+
- pandas / polars optional
- numpy
- scikit-learn
- lightgbm or xgboost
- shap
- fastapi
- pydantic
- uvicorn
- streamlit
- matplotlib
- pytest
- joblib
- pyyaml

## Coding style

- Use type hints for public functions.
- Use small modules and pure functions where possible.
- Avoid notebook-only logic. Notebooks are for EDA, not production pipeline.
- Add docstrings to feature functions explaining leakage behavior.
- Never silently drop columns without logging.
- Fail loudly when required input files are missing.

## Testing expectations

Minimum tests:

- data join keeps expected target column,
- time split preserves chronological order,
- feature generation has no target column in X,
- inference returns probability between 0 and 1,
- API `/health` works,
- API `/predict` works on a sample payload.

## What not to build in baseline

Do not build these before the baseline is working:

- real Kafka/Flink streaming,
- real Chronon/Feast feature store,
- GraphSAGE/GAT as main model,
- online learning that updates model per transaction,
- complex Next.js frontend,
- Kubernetes deployment.
