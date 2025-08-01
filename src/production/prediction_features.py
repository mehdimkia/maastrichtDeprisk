import pandas as pd
from feature_engineering_interactions import build_feature_df

BASE_COLUMNS = [
    "sleep_minutes", "frag_index", "age", "sex",  # ← inputs from the form
    # add any other raw fields you collect
]

def build_row(payload: dict) -> pd.DataFrame:
    """Turn the JSON payload into the engineered 1-row DataFrame."""
    raw = pd.DataFrame([payload], columns=BASE_COLUMNS)
    feats = build_feature_df(raw)
    return feats.drop(columns=["LD_PHQ9depr_event"], errors="ignore")