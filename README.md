# 🕵️ Credit Card Fraud Detector

A machine learning pipeline that detects fraudulent credit card transactions in a highly imbalanced dataset, where fraud is only about 0.17% of all transactions. The focus is on evaluating the model the right way for rare events (precision, recall, PR-AUC), not on accuracy.

## Problem

If a model labels every transaction "normal" it is 99.8% accurate, yet it catches zero fraud. So accuracy is the wrong yardstick here. This project builds and compares models that actually find the rare fraud cases, and measures how many frauds are caught versus how many normal transactions are wrongly flagged.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (Machine Learning Group, ULB): about 285,000 European card transactions from two days, of which 492 are fraud. Features `V1` to `V28` are anonymised (PCA-transformed), plus `Time` and `Amount`. The target column is `Class` (1 = fraud).

The CSV is not included in this repo because of its size. Download it and place it at `data/creditcard.csv`.

## Approach

1. **Split** the data into train, validation and test sets, keeping the same fraud ratio in each (stratified).
2. **Handle the imbalance** with class weights, so the rare fraud class counts more during training.
3. **Compare three models** on the validation set: Logistic Regression, Random Forest and Histogram Gradient Boosting.
4. **Select by PR-AUC**, the metric that suits rare events, and pick the best model.
5. **Tune the decision threshold** on the validation set (the cut-off probability that gives the best F1), instead of using the default 0.5.
6. **Evaluate once on the untouched test set** and report the results below.

## Results (test set)

Best model: **Random Forest** (decision threshold 0.40)

| Metric | Value |
|---|---|
| Precision | 0.802 |
| Recall | 0.827 |
| F1 | 0.814 |
| PR-AUC | 0.821 |
| ROC-AUC | 0.966 |

Out of 98 frauds in the test set, the model **caught 81**, **missed 17**, and raised **20 false alarms**.

![Precision-Recall curve](reports/pr_curve.png)
![Confusion matrix](reports/confusion_matrix.png)

**What this means:** about 4 in 5 frauds are caught, and about 4 in 5 flagged transactions are really fraud. Lowering the threshold would catch more fraud but cause more false alarms, so in practice a bank would choose the balance based on the cost of each kind of mistake.

## Tech stack

Python, pandas, scikit-learn, matplotlib, joblib

## Project structure

```
fraud-detector/
├── data/              # creditcard.csv (download from Kaggle, not in the repo)
├── models/            # saved model (created by the script, not in the repo)
├── reports/           # metrics and charts
├── train_fraud.py     # training and evaluation pipeline
└── README.md
```

## How to run

```bash
git clone https://github.com/vaishnavisnb/fraud-detector.git
cd fraud-detector

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

python -m pip install pandas scikit-learn matplotlib joblib

# put creditcard.csv inside the data folder, then:
python train_fraud.py
```

## Limitations and next steps

- The test set contains only 98 frauds, so the metrics move noticeably if a few cases change. Cross-validation would give a steadier estimate.
- The data covers only two days of European transactions, and the features are anonymised, so it cannot be used to explain why a transaction looks fraudulent.
- Next: serve the model through a **FastAPI** service, store scored transactions in **PostgreSQL**, and package everything with **Docker**.

## Author

Vaishnavi · [GitHub](https://github.com/vaishnavisnb)
