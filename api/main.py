import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionResponse, TransactionRequest
from transaction_risk_engine.models.inference import FraudRiskPredictor

app = FastAPI(
    title="Transaction Risk Engine API",
    description="Real-time fraud scoring inference service.",
    version="0.1.0"
)

# Allow all for dashboard connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor globally on startup
try:
    predictor = FraudRiskPredictor()
except Exception as e:
    print(f"Warning: Failed to load FraudRiskPredictor. Is the model trained? Error: {e}")
    predictor = None


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the baseline first.")
    
    # Model config=extra allow means we can cast it to dict with everything
    transaction_dict = transaction.model_dump(exclude_unset=True)
    
    # Run pipeline
    try:
        response = predictor.predict(transaction_dict)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Return the metrics JSON file content."""
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics report not found.")
    
    with open(metrics_path, "r") as f:
        return json.load(f)
