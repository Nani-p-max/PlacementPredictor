import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from load_data import load_data

_CACHE = {}

SCATTER_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "SoftSkillsRating", "AttendancePercent", "Internships", "Projects",
    "Certifications", "Salary Package", "PlacementStatus",
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
]

KEY_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore",
    "MockInterviewScore", "AttendancePercent", "SoftSkillsRating",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal clean for EDA — no inplace mutations."""
    df = df.drop(columns=["StudentID", "IsAnomaly"], errors="ignore").copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def get_eda_results() -> dict:
    if "eda" in _CACHE:
        return _CACHE["eda"]

    df = _clean(load_data())

    # ── 1. Placement distribution ────────────────────────────────────────────
    vc = df["PlacementStatus"].value_counts().to_dict()
    placement_dist = {
        "labels": ["Not Placed", "Placed"],
        "values": [int(vc.get(0, 0)), int(vc.get(1, 0))],
    }
    placed_count  = int(vc.get(1, 0))
    placement_rate = round(placed_count / len(df) * 100, 2)

    # ── 2. Scatter sample — any two numeric axes, coloured by PlacementStatus ─
    available_scatter = [c for c in SCATTER_FEATURES if c in df.columns]
    scatter_sample = (
        df[available_scatter]
        .sample(min(600, len(df)), random_state=42)
        .round(3)
    )
    scatter_data = scatter_sample.to_dict(orient="list")

    # ── 3. Numeric histograms (pre-binned for speed) ─────────────────────────
    distributions = {}
    for feat in KEY_FEATURES:
        if feat not in df.columns:
            continue
        hist, edges = np.histogram(df[feat].dropna(), bins=25)
        distributions[feat] = {
            "counts": hist.tolist(),
            "bins":   [round(float(b), 2) for b in edges[:-1]],
        }

    # ── 4. Categorical → placement rate ──────────────────────────────────────
    cat_analysis = {}
    for cat in ["Gender", "Stream", "CollegeTier"]:
        if cat not in df.columns:
            continue
        grp = df.groupby(cat)["PlacementStatus"].mean().mul(100).round(2)
        cat_analysis[cat] = {
            "labels":          grp.index.tolist(),
            "placement_rates": grp.values.tolist(),
        }

    # ── 5. Correlation matrix (top-13 by |corr| with Salary Package) ─────────
    num_df   = df.select_dtypes(include="number")
    corr_full = num_df.corr()
    top_feats = corr_full["Salary Package"].abs().nlargest(13).index.tolist()
    corr_sub  = corr_full.loc[top_feats, top_feats].round(3)
    correlation = {
        "labels": top_feats,
        "matrix": corr_sub.values.tolist(),
    }

    # ── 6. Descriptive stats ─────────────────────────────────────────────────
    desc = df[KEY_FEATURES + ["Salary Package"]].describe().round(3)
    stats_rows = [
        {
            "feature": feat,
            "mean":    round(float(desc.loc["mean", feat]), 3),
            "std":     round(float(desc.loc["std",  feat]), 3),
            "min":     round(float(desc.loc["min",  feat]), 3),
            "median":  round(float(desc.loc["50%",  feat]), 3),
            "max":     round(float(desc.loc["max",  feat]), 3),
        }
        for feat in desc.columns
    ]

    result = {
        "placement_dist":       placement_dist,
        "scatter_data":         scatter_data,
        "scatter_features":     [f for f in available_scatter if f not in ("PlacementStatus",)],
        "distributions":        distributions,
        "hist_features":        KEY_FEATURES,
        "cat_analysis":         cat_analysis,
        "cat_fields":           list(cat_analysis.keys()),
        "correlation":          correlation,
        "stats_rows":           stats_rows,
        "total_students":       len(df),
        "placed_count":         placed_count,
        "placement_rate":       placement_rate,
    }

    _CACHE["eda"] = result
    return result
