# Maastricht Deprisk 🧠📈  
_A proof-of-concept for predicting incident depressive symptoms in the Maastricht Study_

![build](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## Project status — August 12, 2025

- **ML pipeline v1.0 frozen** using the Week-7 feature set; production training lives in `src/production/train_xgb_server.py` and exports a fully self-contained scikit-learn pipeline (`.joblib`) with a JSON sidecar of metrics and hyper-parameters.  
- **Companion web demo (internal beta)** is **included in this repository** under **`src/app`** (Next.js 15):  
  - **Predict** page: interactive risk calculator backed by the exported pipeline. We **removed “Low/Moderate/High” risk bands** and now display **absolute 7-year risk** (with short “ℹ️ info” tooltips per predictor, including Sleep Fragmentation).  
  - **Advanced predictors** (history of depression, antidepressant use, neuropathy) are **temporarily hidden** in the UI while we finalize clinical messaging and documentation.  
  - **Power BI dashboard route** is present and marked **“Under construction.”**
- **No Maastricht Study data** are contained in this repository. This repo includes **code only** and, where needed, **synthetic examples** for demonstration.
- Model artifacts are written to `Databases/Week7_xgb_onehot_interactions.joblib` with a `.meta.json` sidecar (AUROC, AUPRC, hyper-parameters).

> ⚠️ **Clinical & data governance**  
> This software is a **research prototype** and **not a medical device**. Results should **not** be used for diagnosis or treatment decisions. Use of Maastricht Study data is governed by its data-use agreements and publication procedures; do **not** redistribute study data or derived participant-level outputs. Keep the web demo strictly to synthetic/illustrative inputs unless you have explicit approval.

---

## Contents

**src/**
- **production/** ← everything the web / API will import  
  - `data_ingest_server.py`  
  - `feature_engineering_server.py`  
  - `feature_engineering_interactions.py`  
  - `qa_feature_files.py`  
  - `train_xgb_server.py` 🏆

- **experiments/** ← one-off benchmarks & notebooks  
  - `train_cat_xgb_server.py`  
  - `train_xgb_ensemble.py`  
  - `train_xgb_improved.py`  
  - `train_models_server.py`  
  - `imbalance_utils.py`, `test_setup.py`

- **app/** (Next.js) ← the **web demo** (internal beta)  
  - `/` (Home)  
  - `/predict` (risk calculator)  
  - `/powerbi` (placeholder: “Under construction”)  
  - `components/ui/*` (including `tooltip`)

**catboost_info/** ← CatBoost training logs (git-ignored)

---

## Model leaderboard (Week-7 feature set)

| Script | Features | AUROC | AUPRC | Notes |
|--------|----------|------:|------:|-------|
| **production/train_xgb_server.py** | one-hot + 5-knot spline | **0.710** | **0.294** | Production v1.0 |
| experiments/train_cat_xgb_server.py | CatBoost native cats | 0.702 | 0.291 | –0.8 pp AUROC |
| experiments/train_xgb_ensemble.py | Stacked (XGB + LR) | 0.700 | 0.295 | +0.1 pp AUPRC, but more complex |
| experiments/train_xgb_improved.py | Cost-sensitive XGB | 0.706 | 0.295 | Tied AUPRC |

> **Data split:** 4,803 training / 1,201 test participants — see `src/production/data_ingest_server.py` for details of the pipeline and data checks.

---

## Quick-start

### Train the production model (Python)

```bash
git clone https://github.com/<your-handle>/maastrichtDeprisk.git
cd maastrichtDeprisk
poetry install                    # or: pip install -r requirements.txt
poetry run python src/production/train_xgb_server.py
```
The trained pipeline is saved to `Databases/Week7_xgb_onehot_interactions.joblib` with a `.meta.json` sidecar containing AUROC, AUPRC and hyper-parameters.

### Run the web demo (Next.js)

```bash
# From the repository root
npm install        # or: pnpm install / yarn
npm run dev        # starts Next.js on http://localhost:3000
```
> Requires **Node.js 18.17+** (Node 20+ recommended). The demo uses components in `src/app` and `src/components/ui`.

### Score with the trained pipeline (Python)

```python
from joblib import load
import pandas as pd

# Load exported pipeline
pipe = load("Databases/Week7_xgb_onehot_interactions.joblib")

# Score new data (must match the training schema)
X = pd.read_parquet("path/to/new_data.parquet")  # or construct a DataFrame
probas = pipe.predict_proba(X)[:, 1]
```

---

## Evaluation protocol (summary)

- Stratified train/test split with fixed seed; metrics reported on the **held-out test set**.  
- Primary metrics: **AUROC** and **AUPRC**; we also monitor calibration and decision-curve utilities during experimentation.  
- Features include one-hot encoded categorical variables and spline-expanded continuous predictors (5-knot).

---

## Web demo notes (internal)

- Absolute risk is shown without band labels; the rationale is to avoid over-simplification in communications to clinicians and participants.  
- Each input includes a concise **“ℹ️ info” tooltip** explaining the measurement (e.g., Sleep Fragmentation definition).  
- The Power BI dashboard route is live with an “Under construction” message until the analytics view is finalized.  
- Source files live under `src/app/*` with shared components in `src/components/*`.

---

## Road-map

- [x] Export production pipeline & metadata
- [x] Web demo (internal beta) co-located in this repository
- [ ] REST/JSON scoring API (FastAPI) and/or AWS endpoint
- [ ] Calibration & drift monitoring (Evidently / MLflow)
- [ ] Model Card + documentation of fairness and external validity
- [ ] Manuscript preparation and (if approved) public data dictionary

---

## Changelog

- **2025-08-12** — Web demo (internal beta) launched **inside this repo**; removed risk band labels; added predictor tooltips; published Power BI “Under construction” page.  
- **2025-07** — Pipeline v1.0 frozen on Week-7 feature set; metrics exported alongside model artifact.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## How to cite

If you use parts of this codebase in research, please cite as:

> _Maastricht Deprisk: A proof-of-concept pipeline for predicting incident depressive symptoms in the Maastricht Study (v1.0, 2025)._

