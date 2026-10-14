"""FastAPI scoring service for transaction monitoring."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from src.risk import score_transactions

app = FastAPI(title="AML Transaction Monitoring API", version="1.0.0", description="Transaction risk scoring API. Scores support review and are not determinations of criminal activity.")

class Transaction(BaseModel):
    amount_paid: float = Field(ge=0)
    amount_received: float = Field(ge=0)
    from_bank: str
    to_bank: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(transaction: Transaction):
    try:
        result = score_transactions(pd.DataFrame([transaction.model_dump()])).iloc[0]
        return {"risk_score": int(result.risk_score), "risk_probability": float(result.risk_probability), "risk_level": result.risk_level, "scoring_method": result.scoring_method, "review_required": result.risk_level != "Low Risk"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc
