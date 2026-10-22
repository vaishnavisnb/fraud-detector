"""
Fraud Detection - Stage 1: train and evaluate models on highly imbalanced data.

Run from the project folder (fraud-detector):
    python train_fraud.py

Expects: data/creditcard.csv  (Kaggle "Credit Card Fraud Detection" dataset, column "Class": 1 = fraud)
Creates: models/fraud_model.pkl, reports/metrics.csv, reports/pr_curve.png, reports/confusion_matrix.png
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/creditcard.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------- 1. Load and look at the data ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
fraud_count = int(df["Class"].sum())
print(f"Fraud cases: {fraud_count} of {len(df)} ({fraud_count / len(df):.3%})")
print("Missing values:", int(df.isnull().sum().sum()))

X = df.drop(columns=["Class"])
y = df["Class"]

# ---------- 2. Split: train / validation / test (all keep the same fraud ratio) ----------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
print(f"Train {len(X_train)} | Validation {len(X_val)} | Test {len(X_test)}")

# ---------- 3. Models (class weights make the rare fraud class count more) ----------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
}


def best_threshold(y_true, scores):
    """Pick the probability cut-off that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    return float(thresholds[int(np.argmax(f1))])


# ---------- 4. Train, then compare models on the VALIDATION set ----------
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    results.append(
        {
            "Model": name,
            "PR-AUC": average_precision_score(y_val, val_scores),
            "ROC-AUC": roc_auc_score(y_val, val_scores),
            "Best threshold": best_threshold(y_val, val_scores),
        }
    )

results_df = pd.DataFrame(results).round(4)
print("\nValidation results:\n", results_df.to_string(index=False))

# ---------- 5. Pick the winner by PR-AUC (the right metric for rare events) ----------
best_row = results_df.sort_values("PR-AUC", ascending=False).iloc[0]
best_name = best_row["Model"]
threshold = float(best_row["Best threshold"])
best_model = models[best_name]
print(f"\nBest model: {best_name} (validation PR-AUC {best_row['PR-AUC']:.3f}, threshold {threshold:.3f})")

# ---------- 6. Final check on the untouched TEST set ----------
test_scores = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_scores >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

final = {
    "Model": best_name,
    "Threshold": round(threshold, 4),
    "Precision": round(precision_score(y_test, test_pred), 4),
    "Recall": round(recall_score(y_test, test_pred), 4),
    "F1": round(f1_score(y_test, test_pred), 4),
    "PR-AUC": round(average_precision_score(y_test, test_scores), 4),
    "ROC-AUC": round(roc_auc_score(y_test, test_scores), 4),
    "Frauds caught": int(tp),
    "Frauds missed": int(fn),
    "False alarms": int(fp),
}
print("\nTEST set results:")
for key, value in final.items():
    print(f"  {key}: {value}")

# ---------- 7. Save the model, numbers and charts ----------
joblib.dump({"model": best_model, "threshold": threshold, "columns": list(X.columns)}, "models/fraud_model.pkl")
pd.DataFrame([final]).to_csv("reports/metrics.csv", index=False)

PrecisionRecallDisplay.from_predictions(y_test, test_scores, name=best_name)
plt.title("Precision-Recall curve (test set)")
plt.savefig("reports/pr_curve.png", bbox_inches="tight")
plt.close()

ConfusionMatrixDisplay.from_predictions(y_test, test_pred, display_labels=["Normal", "Fraud"])
plt.title(f"Confusion matrix (threshold {threshold:.2f})")
plt.savefig("reports/confusion_matrix.png", bbox_inches="tight")
plt.close()

print("\nSaved: models/fraud_model.pkl, reports/metrics.csv, reports/pr_curve.png, reports/confusion_matrix.png")
