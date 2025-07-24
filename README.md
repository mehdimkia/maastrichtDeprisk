# Maastricht Deprisk 🧠📈  
_A proof‑of‑concept for predicting incident depressive symptoms in the Maastricht Study_

![build](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## Contents

**src/**
- **production/** ← everything the web / API will import
  - data_ingest_server.py
  - feature_engineering_server.py
  - feature_engineering_interactions.py
  - qa_feature_files.py
  - train_xgb_server.py 🏆

- **experiments/** ← one-off benchmarks & notebooks
  - train_cat_xgb_server.py
  - train_xgb_ensemble.py
  - train_xgb_improved.py
  - train_models_server.py
  - imbalance_utils.py, test_setup.py

**catboost_info/** ← CatBoost training logs (git-ignored)

---

## Model leaderboard (week‑7 feature set)

| Script | Features | AUROC | AUPRC | Notes |
|--------|----------|------:|------:|-------|
| **production/train_xgb_server.py** | one‑hot + 5‑knot spline | **0.710** | **0.294** | Production v1.0 |
| experiments/train_cat_xgb_server.py | CatBoost native cats | 0.702 | 0.291 | –0.8 pp AUROC |
| experiments/train_xgb_ensemble.py | Stacked (XGB + LR) | 0.700 | 0.295 | +0.1 pp AUPRC, but more complex |
| experiments/train_xgb_improved.py | Cost‑sensitive XGB | 0.706 | 0.295 | Tied AUPRC |

> **Data:** 4 803 training / 1 201 test participants – see `src/production/data_ingest_server.py`.

---

## Quick-start

```bash
git clone https://github.com/<your-handle>/maastrichtDeprisk.git
cd maastrichtDeprisk
poetry install                    # or pip install -r requirements.txt
poetry run python src/production/train_xgb_server.py
```
The trained pipeline is saved to Databases/Week7_xgb_onehot_interactions.joblib with a .meta.json sidecar containing AUROC, AUPRC and hyper‑parameters.

## Re‑running an experiment

```bash
# Example: CatBoost benchmark
poetry run python src/experiments/train_cat_xgb_server.py \
    --n-trials 50 --seed 123
```
Results are written to the same Databases/ folder but never overwrite production artefacts.

## Road‑map

- [ ] API deployment (AWS SageMaker endpoint)
- [ ] Drift & calibration monitoring (ML S mod / Evidently)
- [ ] Streamlit / Next.js dashboard for clinicians
- [ ] Paper submission & open‑data release

Contributions welcome — open a PR or start a discussion!
