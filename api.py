"""
Fraud Detection - Stage 2a: a web API that scores a transaction with the trained model.

Run from the project folder (fraud-detector):
    python -m uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs in your browser to try it.

Needs models/fraud_model.pkl (created by train_fraud.py).
"""

import json
import os

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import PredictionRecord, get_db, init_db

MODEL_PATH = "models/fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("models/fraud_model.pkl not found. Run `python train_fraud.py` first.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
columns = bundle["columns"]  # the exact column order the model was trained on

init_db()  # creates the predictions table if it does not exist yet

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Send one transaction and get back the fraud probability and a fraud / not-fraud decision.",
    version="1.0.0",
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields instead of ignoring them

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class Prediction(BaseModel):
    id: int
    fraud_probability: float
    is_fraud: bool
    threshold: float


@app.get("/health")
def health():
    """Quick check that the service is running."""
    return {"status": "ok", "model": type(model).__name__, "threshold": round(threshold, 4)}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, db: Session = Depends(get_db)):
    """Score one transaction and save the result in the database."""
    data = transaction.model_dump()
    row = pd.DataFrame([data])[columns]
    try:
        probability = float(model.predict_proba(row)[0][1])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model could not score this transaction: {error}")

    is_fraud = probability >= threshold
    record = PredictionRecord(
        amount=data["Amount"],
        fraud_probability=probability,
        is_fraud=is_fraud,
        threshold=threshold,
        payload=json.dumps(data),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return Prediction(
        id=record.id,
        fraud_probability=round(probability, 4),
        is_fraud=is_fraud,
        threshold=round(threshold, 4),
    )


@app.get("/predictions")
def recent_predictions(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    """The most recent scored transactions, newest first."""
    rows = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "amount": r.amount,
            "fraud_probability": round(r.fraud_probability, 4),
            "is_fraud": r.is_fraud,
        }
        for r in rows
    ]


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    """Summary of everything scored so far."""
    total = db.query(func.count(PredictionRecord.id)).scalar() or 0
    flagged = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.is_fraud.is_(True)).scalar() or 0
    return {
        "total_scored": total,
        "flagged_as_fraud": flagged,
        "fraud_rate": round(flagged / total, 4) if total else 0.0,
    }
