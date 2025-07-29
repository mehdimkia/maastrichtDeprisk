"""
FastAPI inference service for the Maastricht Depression‑Risk model
------------------------------------------------------------------
• Loads the XGBoost + preprocessing pipeline (.joblib)
• Optional X‑API‑Key protection (set env MODEL_API_KEY)
• POST /score ➜ { prob: 0.23 }
"""

import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Week7_xgb_onehot_interactions.joblib")
API_KEY    = os.getenv("MODEL_API_KEY")            # optional
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Load model once at startup ───────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"🛑  Model not found at {MODEL_PATH}")

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Depression‑Risk Inference API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV") != "prod" else None,
)

# Allow browser calls from your Next.js site
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ── Request schema (adapt fields to your form) ───────────────────
class ScoreRequest(BaseModel):
    sleep_minutes: float = Field(..., ge=120, le=900)
    frag_index: float
    age: int = Field(..., ge=18, le=100)
    sex: str
    # add any other covariates exactly as in training ↓
    # bmi: float | None = None
    # ...

# ── Response schema ──────────────────────────────────────────────
class ScoreResponse(BaseModel):
    prob: float

# ── Helper: key check ────────────────────────────────────────────
def _check_key(header_key: str | None):
    if API_KEY and header_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

# ── Route ────────────────────────────────────────────────────────
@app.post("/score", response_model=ScoreResponse)
def score(
    req: ScoreRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    _check_key(x_api_key)

    # convert to DataFrame with column order matching training
    df = pd.DataFrame([req.model_dump()])
    prob = float(model.predict_proba(df)[:, 1][0])
    return ScoreResponse(prob=prob)


# ── Local dev entrypoint ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
