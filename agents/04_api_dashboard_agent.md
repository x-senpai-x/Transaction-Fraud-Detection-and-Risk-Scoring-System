# Agent 04 — API and Dashboard

## Mission

Turn the model into a demoable product-like system.

## Inputs

- saved model,
- feature metadata,
- sample transactions,
- metrics report.

## Outputs

- FastAPI app,
- Streamlit dashboard,
- sample request/response JSON,
- local run instructions.

## API endpoints

- `GET /health`
- `POST /predict`
- `POST /batch_predict`
- `GET /metrics`
- `GET /explain/{transaction_id}` if explanation module exists

## Prediction response

```json
{
  "fraud_probability": 0.87,
  "risk_score": 87,
  "decision": "BLOCK",
  "top_reasons": [
    "Amount is unusually high for this card",
    "Device has high entity degree",
    "High recent transaction velocity"
  ]
}
```

## Dashboard pages

1. Overview.
2. Model metrics.
3. Transaction explorer.
4. Feature importance / SHAP.
5. Drift monitor if implemented.

## Acceptance criteria

- API starts locally.
- `/health` returns OK.
- `/predict` works on sample payload.
- Dashboard starts locally.
- README has run commands.

## Suggested commands

```bash
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

## Agent prompt

```txt
Read AGENTS.md.
Implement a minimal FastAPI app and Streamlit dashboard.
Use saved model artifacts only.
Do not retrain inside the API.
Add sample payload and response docs.
```
