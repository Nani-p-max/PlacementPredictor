import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from load_data import load_data

_CACHE = {}


def get_preprocessing_results() -> dict:
    """Run the full preprocessing pipeline and return a detailed report."""
    if "prep" in _CACHE:
        return _CACHE["prep"]

    df = load_data()
    steps = []

    # ── Step 1 : Drop irrelevant columns ──────────────────────────────────────
    DROP_COLS = ["StudentID", "IsAnomaly"]
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    steps.append({
        "icon":   "🗑️",
        "title":  "Drop Irrelevant Columns",
        "detail": f"Removed: {', '.join(existing_drops) if existing_drops else 'none'}",
        "badge":  f"{len(existing_drops)} columns removed",
        "status": "success",
    })

    # ── Step 2 : Capture missing-value counts before imputation ───────────────
    missing_before = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    # ── Step 3 : Impute missing values ────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = {col: int(df[col].isna().sum()) for col in missing_before}
    total_filled = sum(missing_before.values())

    steps.append({
        "icon":   "🩹",
        "title":  "Handle Missing Values",
        "detail": (
            f"Numeric → median ({len(num_cols)} cols)  |  "
            f"Categorical → mode ({len(cat_cols)} cols)"
        ),
        "badge":  f"{total_filled} cells filled",
        "status": "success",
    })

    # ── Step 4 : Label-encode categorical columns ─────────────────────────────
    encoding_maps = {}
    for col in cat_cols:
        le = LabelEncoder()
        unique_vals = sorted(df[col].unique().tolist())
        df[col] = le.fit_transform(df[col])
        encoding_maps[col] = {str(v): int(le.transform([v])[0]) for v in unique_vals}

    steps.append({
        "icon":   "🏷️",
        "title":  "Label Encode Categorical Columns",
        "detail": f"Encoded: {', '.join(cat_cols)}",
        "badge":  f"{len(cat_cols)} columns encoded",
        "status": "success",
    })

    # ── Step 5 : Standard-scale feature columns ───────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("PlacementStatus", "Salary Package")]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    steps.append({
        "icon":   "📐",
        "title":  "Standard Scale Feature Columns",
        "detail": f"Applied StandardScaler (μ=0, σ=1) to {len(feature_cols)} features",
        "badge":  f"{len(feature_cols)} features scaled",
        "status": "success",
    })

    # ── Build scaled-preview (5 rows, rounded to 4 dp) ────────────────────────
    scaled_preview_cols = feature_cols[:8]
    scaled_preview = df_scaled[scaled_preview_cols].head(5).round(4).to_dict(orient="records")

    result = {
        "steps":          steps,
        "missing_before": missing_before,
        "missing_after":  missing_after,
        "encoding_maps":  encoding_maps,
        "n_rows":         df.shape[0],
        "n_cols":         df.shape[1],
        "cat_cols":       cat_cols,
        "num_cols":       num_cols,
        "feature_cols":   feature_cols,
        "scaled_preview": scaled_preview,
        "scaled_preview_cols": scaled_preview_cols,
    }

    _CACHE["prep"] = result
    return result
