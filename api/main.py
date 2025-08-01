"""
FastAPI inference service for the Maastricht Depression‑Risk model
------------------------------------------------------------------
• Loads the XGBoost + preprocessing pipeline (.joblib)
• Optional X‑API‑Key protection (set env MODEL_API_KEY)
• POST /score ⇢ { prob: 0.23 }

2025‑07‑31 — v1.1.2
▶ Patch‑2: widen import path so it works when training helpers sit in
  src/production/*.py. We now add <repo‑root>/src and <repo‑root>/src/production
  to sys.path before importing.
"""

from __future__ import annotations

# ── std‑lib ───────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import pathlib
from typing import Optional, Dict, Any

# ■ Make sure Python can see helper modules wherever they live.
ROOT = pathlib.Path(__file__).resolve().parent.parent  # <repo‑root>
SRC_DIR = ROOT / "src"
PROD_DIR = SRC_DIR / "production"
for p in (ROOT, SRC_DIR, PROD_DIR):
    sys.path.insert(0, str(p))

# ── third‑party ───────────────────────────────────────────────────────────────
import joblib
import pandas as pd
import numpy as np
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── local imports (feature builder) ──────────────────────────────────────────
try:
    # ① when module lives at <repo>/feature_engineering_interactions.py
    from feature_engineering_interactions import build_feature_df  # type: ignore
except ModuleNotFoundError:
    try:
        # ② when module lives at <repo>/src/production/feature_engineering_interactions.py
        from production.feature_engineering_interactions import (  # type: ignore
            build_feature_df,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "🛑 Could not import 'feature_engineering_interactions'. "
            "Ensure it is located in the project or adjust import paths in api/main.py"
        ) from e

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH", str(ROOT / "models/Week7_xgb_onehot_interactions.joblib")
)
API_KEY = os.getenv("MODEL_API_KEY")  # optional
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

warnings.filterwarnings(
    "ignore", message=".*If you are loading a serialized model.*"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)


    def _patch_xgb_attrs(obj):
        """Add legacy attrs & methods so old pickles run under XGBoost 1.7."""
        if isinstance(obj, xgb.XGBClassifier):
            if not hasattr(obj, "use_label_encoder"):
                obj.use_label_encoder = False
            if not hasattr(obj, "gpu_id"):
                obj.gpu_id = -1  # CPU
            if not hasattr(obj, "predictor"):
                obj.predictor = "auto"
            # ▸ add an empty __sklearn_tags__ so scikit-learn’s validation passes
            if not hasattr(obj, "__sklearn_tags__"):
                obj.__sklearn_tags__ = lambda self=obj: {}
        # recurse into nested transformers / pipelines
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                _patch_xgb_attrs(step)
        if hasattr(obj, "transformers"):
            for _, trf, _ in obj.transformers:
                _patch_xgb_attrs(trf)


    _patch_xgb_attrs(model)  # keep this call right after joblib.load()
except FileNotFoundError as err:
    raise RuntimeError(f"🛑  Model not found at {MODEL_PATH}") from err

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI instantiation & CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Depression‑Risk Inference API",
    version="1.1.2",
    docs_url="/docs" if os.getenv("ENV") != "prod" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request / response models
# ─────────────────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    """Payload expected from the Next.js form."""

    sleep_minutes: float = Field(..., ge=120, le=900)
    frag_index: float
    age: int = Field(..., ge=18, le=100)
    sex: str  # "M" or "F"

    # Optional future fields — leave as None for now
    bmi: Optional[float] = Field(None, ge=15, le=60)
    education: Optional[int] = Field(None, ge=1, le=3)
    marital_status: Optional[int] = Field(None, ge=0, le=3)
    smoking_cat: Optional[int] = Field(None, ge=0, le=2)
    alcohol_cat: Optional[int] = Field(None, ge=0, le=3)
    diabetes: Optional[int] = Field(None, ge=0, le=1)
    cvd: Optional[int] = Field(None, ge=0, le=3)
    dhd_sum_min_alc: Optional[float] = None
    mvpatile: Optional[int] = Field(None, ge=1, le=4)
    prior_depr: Optional[int] = Field(None, ge=0, le=1)
    med_depr: Optional[int] = Field(None, ge=0, le=1)
    neuropath: Optional[int] = Field(None, ge=0, le=1)


class ScoreResponse(BaseModel):
    prob: float


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build engineered feature row
# ─────────────────────────────────────────────────────────────────────────────
RAW_COLS = [
    # Sleep exposure & modifiers
    "sleep_dur2",
    "frag_break_q4",
    "standstep_q4",
    # Baseline covariates
    "Age.ph1",
    "Sex",
    "N_Diabetes_WHO.ph1",
    "n_education_3cat.ph1",
    "marital_status.ph1",
    "smoking_3cat.ph1",
    "N_alcohol_cat.ph1",
    "bmi.ph1",
    "N_CVD.ph1",
    "dhd_sum_min_alc.ph1",
    "mvpatile",
    # Mental-health / neuro flags
    "MINIlifedepr.ph1",
    "med_depression.ph1",
    "impaired_vibration_sense.ph1",
    # Outcome (will be dropped)
    "LD_PHQ9depr_event",
    # Raw continuous sleep minutes
    "mean_inbed_min_night_t.ph1",
]

FIELD_MAP: Dict[str, str] = {
    "sleep_minutes": "mean_inbed_min_night_t.ph1",
    "frag_index": "frag_break_q4",
    "age": "Age.ph1",
    "sex": "Sex",
    # Optional mappings
    "bmi": "bmi.ph1",
    "education": "n_education_3cat.ph1",
    "marital_status": "marital_status.ph1",
    "smoking_cat": "smoking_3cat.ph1",
    "alcohol_cat": "N_alcohol_cat.ph1",
    "diabetes": "N_Diabetes_WHO.ph1",
    "cvd": "N_CVD.ph1",
    "dhd_sum_min_alc": "dhd_sum_min_alc.ph1",
    "mvpatile": "mvpatile",
    "prior_depr": "MINIlifedepr.ph1",
    "med_depr": "med_depression.ph1",
    "neuropath": "impaired_vibration_sense.ph1",
}


def build_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    # normalise / coerce front-end values
    sex_val = payload.get("sex")
    if isinstance(sex_val, str):
        payload["sex"] = 1 if sex_val.strip().upper().startswith("M") else 0
    """Convert JSON payload ➜ engineered 1‑row DataFrame."""
    raw: Dict[str, Any] = {col: pd.NA for col in RAW_COLS}
    for key, value in payload.items():
        if key in FIELD_MAP and value is not None:
            raw[FIELD_MAP[key]] = value

    df_raw = pd.DataFrame([raw])
    feats = build_feature_df(df_raw)
    # ✱ new — ensure missing values are genuine NaNs, not pandas NAType
    feats = feats.replace({pd.NA: np.nan})
    return feats.drop(columns=["LD_PHQ9depr_event"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# API‑key guard & /score endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _check_key(header_key: str | None):
    if API_KEY and header_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, x_api_key: str | None = Header(None, alias="x-api-key")):
    _check_key(x_api_key)
    df_feats = build_feature_row(req.model_dump())
    prob = float(model.predict_proba(df_feats)[:, 1][0])
    return ScoreResponse(prob=prob)


# ─────────────────────────────────────────────────────────────────────────────
# Local dev entry‑point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
