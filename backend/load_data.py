import os
import pandas as pd

# CSV lives one level above the backend/ folder
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "placement_predict_50k Dataset.csv",
)


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    if "PlacementStatus" in df.columns:
        status = df["PlacementStatus"]
        if status.dtype == "object":
            normalized = status.astype(str).str.strip().str.lower()
            labels = {
                "0": 0, "false": 0, "no": 0, "not placed": 0,
                "1": 1, "true": 1, "yes": 1, "placed": 1,
            }
            encoded = normalized.map(labels)
            if encoded.isna().any():
                unknown = sorted(normalized[encoded.isna()].unique().tolist())
                raise ValueError(
                    "Unsupported PlacementStatus values: "
                    + ", ".join(unknown)
                )
            df["PlacementStatus"] = encoded.astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data-Loading section summary
# ─────────────────────────────────────────────────────────────────────────────

def get_data_summary(path: str = DATA_PATH) -> dict:
    """Return shape, dtypes, missing-value info and a 10-row preview."""
    df = load_data(path)

    return {
        "n_rows":         df.shape[0],
        "n_cols":         df.shape[1],
        "columns":        list(df.columns),
        "dtypes":         {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_pct":    {
            col: round(df[col].isna().sum() / len(df) * 100, 2)
            for col in df.columns
        },
        "preview":        df.head(10).to_dict(orient="records"),
        "numeric_cols":   list(df.select_dtypes(include="number").columns),
        "cat_cols":       list(df.select_dtypes(include="object").columns),
    }
