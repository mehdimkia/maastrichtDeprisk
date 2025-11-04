# Maastricht Deprisk 🧠📈

*A research prototype to estimate ****\~4-year**** risk of developing clinically relevant depressive symptoms (PHQ-9 ≥ 10) from wearable-derived sleep metrics plus clinical and sociodemographic covariates, trained on The Maastricht Study.*

&#x20;

> **Not a medical device.** Research/education use only. Do **not** use results for diagnosis or treatment. Maastricht Study data are governed by data-use agreements; this repository contains **code only** (no study data).

---

## Project status — November 2025

- **Model v2 (Week-8 feature set)** with interaction features and a spline basis for nightly in-bed minutes; exported as a single **scikit-learn pipeline** (`.joblib`) with a `.meta.json` sidecar (metrics + hyper-params).
- **FastAPI scoring service** (`/score`) for low-latency inference with an optional API key.
- **Next.js app** with:
  - **Predictor** (interactive risk calculator).
  - **Dashboard** (explore incidence by **sleep duration × fragmentation** and subgroups).
  - **Model/API** info pages.
- **Hold-out performance:** AUROC ≈ **0.71**, AUPRC ≈ **0.29**.\
  **Analytic cohort:** \~**6,004** participants; **880** incident cases.

---

## What’s inside

```
maastrichtDeprisk/
├─ api/                       # FastAPI scoring service
│  └─ main.py
├─ V2/
│  └─ src/                    # Training & feature engineering
│     ├─ data_ingest_server.py
│     ├─ feature_engineering_interactions.py
│     └─ train_xgb_server.py
├─ web/                       # Next.js app (App Router)
│  └─ src/app/
│     ├─ page.tsx            # Landing
│     ├─ predict/            # Predictor page
│     └─ powerbi/            # Cohort dashboard (embedded)
└─ README.md
```

**Model artifacts (after training)**\
`Databases/Week8_xgb_onehot_interactions.joblib`\
`Databases/Week8_xgb_onehot_interactions.meta.json`

---

## Quick start

### 1) Train (Python 3.10+)

Use your environment manager of choice (uv/venv/conda/poetry).

```bash
# 1) Ingest → intermediate pickle (paths are examples)
python V2/src/data_ingest_server.py \
  --input /secure/maastricht/Week7.sav \
  --out-pkl /secure/maastricht/Week8.pkl

# 2) Feature engineering + split (produces Week8_* parquet files)
python V2/src/feature_engineering_interactions.py

# 3) Tune & train XGBoost, export champion pipeline + meta.json
python V2/src/train_xgb_server.py
```

Artifacts are written under `Databases/` and include the full preprocessing+model pipeline suitable for direct `.predict_proba()` use.

### 2) Run the scoring API (FastAPI)

```bash
cd api
# Install deps (choose one)
pip install -r requirements.txt || poetry install

# Set env vars (example)
export MODEL_PATH="../Databases/Week8_xgb_onehot_interactions.joblib"
export MODEL_API_KEY="dev-secret"           # optional; require via x-api-key
export CORS_ORIGINS="http://localhost:3000" # comma-separated

uvicorn main:app --reload --port 8000
```

**Endpoint**

- `POST /score` → `{ "prob": <float> }`

**Minimal request body (example)**

```json
{
  "sleep_minutes": 480,
  "frag_quartile": 3,
  "age": 60,
  "sex": "M"
}
```

**Extended optional fields** (names may vary slightly depending on your schema):

- `bmi`, `education`, `marital_status`, `smoking_status`, `alcohol_use`,\
  `diabetes`, `cvd`, `dhdi`, `mvpa`,\
  `prior_depr` (history of depression), `med_depr` (antidepressant use), `neuropathy`.

**curl**

```bash
curl -X POST http://localhost:8000/score \
  -H "content-type: application/json" \
  -H "x-api-key: dev-secret" \
  -d '{"sleep_minutes":480,"frag_quartile":3,"age":60,"sex":"M"}'
```

### 3) Run the web app (Node 18+ recommended)

```bash
cd web
npm install
cp .env.local.example .env.local

# Point the frontend to your API:
# NEXT_PUBLIC_API_URL=http://localhost:8000

npm run dev   # → http://localhost:3000
```

**Routes**

- `/` – Landing
- `/predict` – Risk calculator (posts JSON to the API)
- `/powerbi` – Cohort dashboard (embedded)

---

## Model summary (v2)

- **Estimator:** Gradient-boosted trees (**XGBoost** `hist`) with class-imbalance handling.
- **Features:** one-hot encoded categoricals; **restricted cubic spline** for nightly in-bed minutes; engineered interactions (e.g., sleep × vulnerability); standardization where applicable.
- **Split:** stratified train/test with fixed seed; metrics reported on the **held-out** test set.
- **Primary metrics:** **AUROC** and **AUPRC**; calibration and utility curves monitored during experimentation.
- **Export:** `joblib` pipeline that includes preprocessing + model; sidecar `.meta.json` records AUROC/AUPRC and hyper-parameters for exact reproducibility.

**Scoring directly from Python**

```python
from joblib import load
import pandas as pd

pipe = load("Databases/Week8_xgb_onehot_interactions.joblib")
X = pd.DataFrame([{ "sleep_minutes": 480, "frag_quartile": 3, "age": 60, "sex": "M" }])
proba = pipe.predict_proba(X)[0, 1]
```

---

## Environment variables

**API**

- `MODEL_PATH` — filesystem path to the exported `.joblib`.
- `MODEL_API_KEY` — if set, requests must provide `x-api-key` header.
- `CORS_ORIGINS` — comma-separated list of allowed origins (default `*`).

**Web**

- `NEXT_PUBLIC_API_URL` — base URL for the scoring API.

---

## Privacy & data governance

- This repository ships **no raw cohort data**. Keep all demos to **synthetic or derived** inputs unless you have explicit permission under the Maastricht Study’s DUA.
- The public app is **privacy-first**: only small, derived features (not raw wearable streams) are sent to the API.
- Any real-world deployment must pass institutional review and security checks.

---

## Roadmap

-

---

## Changelog

- **2025-11-04** — Updated README for v2 (API + Next.js predictor & dashboard; Week-8 feature set; clarified env/config).
- **2025-08** — v1 pipeline frozen; internal demo shipped.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## How to cite

> *Maastricht Deprisk (v2, 2025): A reproducible ML pipeline and web demo for estimating incident depressive-symptom risk using The Maastricht Study.*

---

## Acknowledgments

We thank participants and investigators of **The Maastricht Study** and all collaborators who contributed feedback on feature engineering, model evaluation, clinical framing, and product design.

