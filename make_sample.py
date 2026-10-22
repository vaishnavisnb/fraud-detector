"""
Creates two example transactions you can paste into the API page:
    samples/fraud_example.json   (a transaction the model flags as fraud)
    samples/normal_example.json  (a normal-looking transaction)

Run once from the project folder:  python make_sample.py
Needs data/creditcard.csv and models/fraud_model.pkl
"""

import json
import os

import joblib
import pandas as pd

df = pd.read_csv("data/creditcard.csv")
bundle = joblib.load("models/fraud_model.pkl")
model, columns = bundle["model"], bundle["columns"]

fraud_rows = df[df["Class"] == 1]
normal_rows = df[df["Class"] == 0].head(2000)

fraud_pick = fraud_rows.iloc[model.predict_proba(fraud_rows[columns])[:, 1].argmax()]
normal_pick = normal_rows.iloc[model.predict_proba(normal_rows[columns])[:, 1].argmin()]

os.makedirs("samples", exist_ok=True)
for name, row in [("fraud_example", fraud_pick), ("normal_example", normal_pick)]:
    data = {column: float(row[column]) for column in columns}
    with open(f"samples/{name}.json", "w") as file:
        json.dump(data, file, indent=2)
    print(f"Saved samples/{name}.json")
