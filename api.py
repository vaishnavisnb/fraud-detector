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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MODEL_PATH = "models/fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("models/fraud_model.pkl not found. Run `python train_fraud.py` first.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
columns = bundle["columns"]  # the exact column order the model was trained on

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
    fraud_probability: float
    is_fraud: bool
    threshold: float


@app.get("/health")
def health():
    """Quick check that the service is running."""
    return {"status": "ok", "model": type(model).__name__, "threshold": round(threshold, 4)}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    """Score one transaction."""
    row = pd.DataFrame([transaction.model_dump()])[columns]
    try:
        probability = float(model.predict_proba(row)[0][1])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model could not score this transaction: {error}")
    return Prediction(
        fraud_probability=round(probability, 4),
        is_fraud=probability >= threshold,
        threshold=round(threshold, 4),
    )
