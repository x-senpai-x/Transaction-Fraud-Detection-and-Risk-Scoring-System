import json

from fastapi.testclient import TestClient

from unittest.mock import patch
from api.main import app



client = TestClient(app)


@patch("api.main.predictor")
def test_health_check(mock_predictor):
    mock_predictor.predict.return_value = {"status": "ok"} # dummy
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


@patch("api.main.predictor")
def test_predict_endpoint(mock_predictor):
    mock_predictor.predict.return_value = {
        "fraud_probability": 0.05,
        "risk_score": 20,
        "decision": "APPROVE",
        "top_reasons": ["test"]
    }
    # Use a dummy payload that has the required basic fields
    payload = {
        "TransactionID": 12345,
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "card1": 1000,
        "P_emaildomain": "gmail.com"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "decision" in data
    assert "top_reasons" in data
    
    # Assert probability bounds
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    assert data["decision"] in ["APPROVE", "REVIEW", "BLOCK"]
