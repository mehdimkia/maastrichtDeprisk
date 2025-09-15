"""
FastAPI inference service – Maastricht Depression-Risk model
v1.3.2 · 2025-09-15
• Fix NameError (xgboost alias)
• Resilient import path handling for feature_engineering_interactions
• Minor CORS & logging tidy-ups
"""

from __future__ import annotations

# ── std-lib ────────────────────────────────────────────────────────────────
import os
import sys
import math
import pathlib
import logging
import warnings
from typing import Optional, Dict, Any

# ── paths ──────────────────────────────────────────────────────────────────
# Resolve to directory containing this file (usually api/)
ROOT = pathlib.Path(__file__).resolve().parent

# Help Python find modules regardless of repo layout (flat vs pkg)
for p in (ROOT, ROOT.parent, ROOT / "lib", ROOT / "production"):
    sys.path.insert(0, str(p))

# ── third-party ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
import uvicorn
import xgboost as xgb
import sklearn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ── local helpers ──────────────────────────────────────────────────────────
# Try common locations so this works both locally and in Docker/Render
try:
    from lib.feature_engineering_interactions import build_feature_df  # type: ignore
except ModuleNotFoundError:
    try:
        from feature_engineering_interactions import build_feature_df  # type: ignore
    except ModuleNotFoundError:
        from production.feature_engineering_interactions import build_feature_df  # type: ignore

# ── config ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(ROOT / "models" / "Week7_xgb_onehot_interactions.joblib"),
)

API_KEY = os.getenv("MODEL_API_KEY")
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

warnings.filterwarnings("ignore", message=".*serialized model.*")

logger = logging.getLogger("uvicorn.error")
logger.info("sklearn=%s xgboost=%s", sklearn.__version__, xgb.__version__)

# --- compat shim for legacy XGBoost pickles on newer scikit-learn ----------
# Provide __sklearn_tags__ if XGB wrappers don't have it
if not hasattr(xgb.XGBClassifier, "__sklearn_tags__"):
    xgb.XGBClassifier.__sklearn_tags__ = lambda self: {}
if hasattr(xgb, "XGBRegressor") and not hasattr(xgb.XGBRegressor, "__sklearn_tags__"):
    xgb.XGBRegressor.__sklearn_tags__ = lambda self: {}

# ---- Platt recalibration --------------------------------------------------
# Original prevalence in the training cohort ≈ 19 %
P_ORIG = float(os.getenv("P_ORIG", 0.19))  # can override via env
# Desired baseline prevalence (e.g., general population) = 8 %
P_TARGET = float(os.getenv("P_TARGET", 0.08))

LOGIT_SHIFT = math.log(P_TARGET / (1 - P_TARGET)) - math.log(P_ORIG / (1 - P_ORIG))

# ── model load ─────────────────────────────────────────────────────────────
def _patch_xgb(obj):
    """Walk objects/pipelines and patch XGB attrs so old pickles work."""
    if isinstance(obj, xgb.XGBClassifier):
        obj.use_label_encoder = getattr(obj, "use_label_encoder", False)
        obj.gpu_id = getattr(obj, "gpu_id", -1)
        obj.predictor = getattr(obj, "predictor", "auto")
        obj.__sklearn_tags__ = getattr(obj, "__sklearn_tags__", lambda: {})
    for attr in ("steps", "transformers"):
        if hasattr(obj, attr):
            for _, step, *_ in getattr(obj, attr):
                _patch_xgb(step)

authorised_model_path = pathlib.Path(MODEL_PATH)
if not authorised_model_path.exists():
    raise RuntimeError(f"🛑 Model not found at {authorised_model_path}")

model = joblib.load(authorised_model_path)
_patch_xgb(model)

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Depression-Risk API",
    version="1.3.2",
    docs_url="/docs" if os.getenv("ENV") != "prod" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS if ORIGINS != ["*"] else ["*"],
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# ── schema ─────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    sleep_minutes: float = Field(..., ge=120, le=900)
    frag_index: float
    age: int = Field(..., ge=18, le=100)
    sex: str  # "M" | "F"

    bmi: Optional[float] = Field(None, ge=15, le=60)
    smoking_cat: Optional[int] = Field(None, ge=0, le=2)
    alcohol_cat: Optional[int] = Field(None, ge=0, le=3)
    diabetes: Optional[int] = Field(None, ge=0, le=1)

    standstep_q4: Optional[int] = Field(None, ge=1, le=4)
    any_vuln: Optional[int] = Field(None, ge=0, le=1)
    neuropathy: Optional[int] = Field(None, ge=0, le=1)


class ScoreResponse(BaseModel):
    prob: float


# ── raw → engineered helpers ───────────────────────────────────────────────
RAW_COLS = [
    # Sleep & fragmentation codes
    "sleep_dur2",
    "frag_break_q4",
    "standstep_q4",
    # Baseline covariates
    "Age.ph1",
    "Sex",
    "N_Diabetes_WHO.ph1",
    "bmi.ph1",
    "smoking_3cat.ph1",
    "N_alcohol_cat.ph1",
    # ↓ extras required by build_feature_df even if NA ↓
    "n_education_3cat.ph1",
    "marital_status.ph1",
    "N_CVD.ph1",
    "dhd_sum_min_alc.ph1",
    "mvpatile",
    # Mental-health / neuro flags
    "MINIlifedepr.ph1",
    "med_depression.ph1",
    "impaired_vibration_sense.ph1",
    # Continuous sleep minutes (for quadratic term)
    "mean_inbed_min_night_t.ph1",
    # Outcome to drop
    "LD_PHQ9depr_event",
]

FIELD_MAP: Dict[str, str] = {
    "sleep_minutes": "mean_inbed_min_night_t.ph1",
    "age": "Age.ph1",
    "sex": "Sex",
    # derived categorical codes
    "frag_break_q4": "frag_break_q4",
    "sleep_dur2": "sleep_dur2",
    # optional baseline
    "bmi": "bmi.ph1",
    "smoking_cat": "smoking_3cat.ph1",
    "alcohol_cat": "N_alcohol_cat.ph1",
    "diabetes": "N_Diabetes_WHO.ph1",
    # activity & comps
    "standstep_q4": "standstep_q4",
    "any_vuln": "MINIlifedepr.ph1",
    "neuropathy": "impaired_vibration_sense.ph1",
    # composite proxies
    "prior_depr": "MINIlifedepr.ph1",
    "med_depr": "med_depression.ph1",
}


def build_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert JSON payload → engineered 1-row DataFrame."""

    # ── derive categorical sleep duration (sleep_dur2) ────────────────
    if "sleep_minutes" in payload:
        mins = float(payload["sleep_minutes"])
        payload["sleep_dur2"] = 1 if mins < 420 else 3 if mins >= 540 else 2

    # ── derive fragmentation quartile (frag_break_q4) ──────────────────
    if "frag_index" in payload:
        fi = float(payload["frag_index"])
        payload["frag_break_q4"] = 1 if fi <= 1.00 else 2 if fi <= 1.86 else 3 if fi <= 2.83 else 4

    # ── normalise sex to binary ────────────────────────────────────────
    if isinstance((sv := payload.get("sex")), str):
        payload["sex"] = 1 if sv.upper().startswith("M") else 2  # 1 = Male, 2 = Female

    # ── propagate any_vuln into underlying flags ───────────────────────
    if payload.get("any_vuln") is not None:
        payload["prior_depr"] = payload["any_vuln"]
        payload["med_depr"] = payload["any_vuln"]

    # ── assemble raw row with all expected cols ───────────────────────
    raw = {col: pd.NA for col in RAW_COLS}
    for k, v in payload.items():
        if k in FIELD_MAP and v is not None:
            raw[FIELD_MAP[k]] = v

    df_raw = pd.DataFrame([raw])
    feats = build_feature_df(df_raw).replace({pd.NA: np.nan})
    return feats.drop(columns=["LD_PHQ9depr_event"], errors="ignore")


# ── security ───────────────────────────────────────────────────────────────
def _guard(key: str | None):
    if API_KEY and key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


# ── endpoints ──────────────────────────────────────────────────────────────
@app.post("/score", response_model=ScoreResponse)
def score(
    req: ScoreRequest,
    x_api_key: str | None = Header(None, alias="x-api-key"),
):
    _guard(x_api_key)

    payload = req.model_dump()
    feats = build_feature_row(payload)

    # 1) raw model probability
    p_raw = float(model.predict_proba(feats)[:, 1][0])

    # 2) global Platt recalibration (shift mean from 19% → 8%)
    logit_recal = math.log(p_raw / (1 - p_raw)) + LOGIT_SHIFT
    prob = 1 / (1 + math.exp(-logit_recal))

    # 3) optional U-shape adjustment if user gave *no* vulnerability flag
    if payload.get("any_vuln", 0) == 0:
        sleep_cat = int(payload["sleep_dur2"])  # 1, 2, or 3
        PENALTY = {1: -0.06, 2: 0.00, 3: +0.17}  # log-odds bumps
        logit_adj = math.log(prob / (1 - prob)) + PENALTY[sleep_cat]
        prob = 1 / (1 + math.exp(-logit_adj))

    return ScoreResponse(prob=prob)


# --- health & root probes (Render expects 200) -----------------------------
@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
def root():
    return {"ok": True}


@app.head("/", include_in_schema=False)
def root_head():
    return PlainTextResponse("", status_code=200)


# ── local dev ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
