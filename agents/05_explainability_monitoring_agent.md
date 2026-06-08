# Agent 05 — Explainability and Monitoring

## Mission

Add interpretability and basic production-style monitoring.

## Inputs

- trained model,
- feature matrix,
- predictions,
- baseline train distributions.

## Outputs

- SHAP plots,
- per-transaction explanation JSON,
- drift metrics JSON,
- monitoring dashboard section.

## Explainability tasks

1. Compute SHAP values for the trained tree model.
2. Generate global feature importance plot.
3. For a transaction, return top features increasing risk.
4. Convert technical feature names into readable reasons where possible.

## Monitoring tasks

1. Track prediction score distribution.
2. Compute PSI for risk scores and top features.
3. Compute KS test for selected numeric features.
4. Show drift status:
   - stable,
   - moderate drift,
   - severe drift.

## Champion-challenger optional tasks

1. Train challenger on newer time window.
2. Compare champion vs challenger on holdout.
3. Recommend promotion only if challenger improves PR-AUC/Recall@Top-K without worsening false positives.

## Acceptance criteria

- At least one global SHAP plot saved.
- At least one individual explanation works.
- Drift report can be generated from two datasets.

## Agent prompt

```txt
Read AGENTS.md and docs/ML_CONCEPTS_TARGETED.md.
Implement SHAP explanations and simple PSI/KS drift reports.
Integrate outputs into reports/ and dashboard.
Keep it simple and robust.
```
