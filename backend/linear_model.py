import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from load_data import load_data

_CACHE = {}

FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore",
    "MockInterviewScore", "SoftSkillsRating", "AttendancePercent",
    "Internships", "Projects", "Certifications",
]
TARGET = "Salary Package"


def get_linear_regression_results() -> dict:
    if "lr" in _CACHE:
        return _CACHE["lr"]

    df = load_data()
    df = df.drop(columns=["StudentID", "IsAnomaly"], errors="ignore").copy()

    # ── Impute (no inplace) ──────────────────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── Encode ───────────────────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Only placed students carry a meaningful salary
    df = df[df["PlacementStatus"] == 1].copy()

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler   = StandardScaler()
    X_tr     = scaler.fit_transform(X_train)
    X_te     = scaler.transform(X_test)

    model    = LinearRegression()
    model.fit(X_tr, y_train)
    y_pred   = model.predict(X_te)

    r2   = float(r2_score(y_test,   y_pred))
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Sample 300 points for scatter
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(y_test), min(300, len(y_test)), replace=False)

    # Coefficients sorted by |value|
    coef_pairs = sorted(
        zip(FEATURES, model.coef_.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    result = {
        "metrics": {
            "r2":   round(r2,   4),
            "mae":  round(mae,  4),
            "rmse": round(rmse, 4),
        },
        "scatter": {
            "actual":    [round(float(v), 3) for v in y_test[idx]],
            "predicted": [round(float(v), 3) for v in y_pred[idx]],
        },
        "coefficients":    {k: round(float(v), 4) for k, v in coef_pairs},
        "coef_labels":     [k for k, _ in coef_pairs],
        "coef_values":     [round(float(v), 4) for _, v in coef_pairs],
        "features":        FEATURES,
        "n_train":         int(len(X_train)),
        "n_test":          int(len(X_test)),
        "intercept":       round(float(model.intercept_), 4),
    }

    _CACHE["lr"] = result
    return result
