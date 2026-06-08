from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_score: int
    decision: str
    top_reasons: list[str]


class TransactionRequest(BaseModel):
    # Accept absolutely any keys as input dictionary since raw payload is large
    model_config = ConfigDict(extra="allow")

    # We enforce just the bare minimums for validation but allow anything else
    TransactionID: int
    TransactionAmt: float
    ProductCD: str
