import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
)

from load_data import load_data

_CACHE = {}
TARGET = "PlacementStatus"


def get_logistic_regression_results() -> dict:
    if "log" in _CACHE:
        return _CACHE["log"]

    df = load_data()
    df = df.drop(columns=["StudentID", "IsAnomaly", "Salary Package"], errors="ignore").copy()

    # ── Impute (no inplace) ──────────────────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── Encode ───────────────────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [c for c in df.columns if c != TARGET]
    X = df[features].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)

    model   = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]

    acc  = float(accuracy_score(y_test,  y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test,    y_pred, zero_division=0))
    f1   = float(f1_score(y_test,        y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc      = float(auc(fpr, tpr))

    # Downsample ROC to ~100 pts
    step = max(1, len(fpr) // 100)

    # Top-10 features by |coefficient|
    coef_pairs = sorted(
        zip(features, model.coef_[0].tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    result = {
        "metrics": {
            "accuracy":  round(acc,     4),
            "precision": round(prec,    4),
            "recall":    round(rec,     4),
            "f1":        round(f1,      4),
            "auc":       round(roc_auc, 4),
        },
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[::step]],
            "tpr": [round(float(v), 4) for v in tpr[::step]],
        },
        "top_features":        {k: round(float(v), 4) for k, v in coef_pairs},
        "top_feature_labels":  [k for k, _ in coef_pairs],
        "top_feature_values":  [round(float(v), 4) for _, v in coef_pairs],
        "n_train":    int(len(X_train)),
        "n_test":     int(len(X_test)),
        "features":   features,
    }

    _CACHE["log"] = result
    return result
