# ModelDoctor

**Intelligent Detection and Mitigation of Unreliable ML Predictions**

ModelDoctor is a machine learning reliability system for tabular classification datasets. It trains a primary model (Model 1) to make predictions, and a secondary reliability model (Model 2, "the doctor") to estimate whether each of Model 1's predictions should be **trusted**, **reviewed**, or **rejected**.

> ModelDoctor does not make predictions better; it tells you when not to believe them.

---

## Key Features

- **Automated model selection (Model 1):** cross-validates Logistic Regression, Decision Tree, Random Forest, XGBoost and KNN on the same stratified folds of the train split, picks the best one and calibrates its probabilities inside the train split.
- **Model-agnostic reliability features:** confidence, entropy, margin, bootstrap disagreement / std, kNN distance to training data, Isolation Forest score, missing-value counts, out-of-range and unseen-category counts (the last three computed from the raw input).
- **Reliability model (Model 2):** learns P(correct) on the calibration split, which Model 1 never saw. If there are too few errors to learn from (< 30), it falls back to Model 1's calibrated confidence and says so.
- **Mitigation layer:** TRUST / REVIEW / REJECT using thresholds tuned on out-of-fold calibration predictions, with human-readable warnings.
- **Honest evaluation:** on an isolated test split, always against the confidence-threshold baseline at the *same coverage*. Includes the risk–coverage curve, AUROC/AUPRC, error-detection precision/recall, calibration (ECE, reliability diagram).
- **Monitoring:** every served prediction is logged; submit real outcomes later and the dashboard compares estimated vs observed reliability and raises degradation / drift / coverage alerts.
- **Web app:** React dashboard for upload → training progress → report → predict (form + batch CSV) → monitoring.

## Scope

- Binary and multiclass **classification only**. Continuous numeric targets are rejected with: *"Only classification targets are supported in this version."*
- At least 500 rows, at least 10 rows per class, upload limit 50 MB (configurable).
- Strict, stratified 60% train / 20% calibration / 20% test split (seed 42).

---

## Quickstart

Requirements: Python 3.10+ and Node.js 18+.

### 1. Backend (FastAPI, port 8000)

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Health check: http://localhost:8000/api/health
- Swagger docs: http://localhost:8000/docs

### 2. Frontend (React + Vite, port 5173)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to `http://127.0.0.1:8000` (override with `VITE_PROXY_TARGET`, or call another origin directly with `VITE_API_URL`).

### 3. Try it

1. Drop `sample_data/loans.csv` on the upload page. The target `default` is suggested automatically.
2. Press **Start training** and watch the stages. Full models take roughly 10–40 s; tick *Fast mode* under *Advanced options* for quicker runs.
3. Read the **Report**: ModelDoctor vs the confidence baseline, risk–coverage and calibration charts, thresholds, the leaderboard.
4. On **Predict**, try typical values, then an unusual record (a huge income, an unseen category, a blank field) and compare *confidence* with *trustworthiness*.
5. Upload a batch CSV that still contains the `default` column: its values are stored as real outcomes, so the **Monitoring** page fills up immediately.

More datasets: `python sample_data/make_datasets.py` (already run; it creates `breast_cancer.csv` and `customer_tiers.csv`).

### Tests

```bash
cd backend
pytest            # 71 tests: data layer, Model 1, features, Model 2, mitigation, evaluation, monitoring, every API endpoint
```

### Reproduce the experiment table

```bash
cd backend
python scripts/run_experiments.py      # writes ../RESULTS.md (about 1 minute)
```

---

## Results (summary)

See [RESULTS.md](RESULTS.md) for the full tables, including every algorithm forced as Model 1. On the three bundled datasets ModelDoctor **matches** the calibrated-confidence baseline at equal coverage rather than beating it:

| Dataset | Model 1 | Error, no filter | Coverage | Baseline error | ModelDoctor error |
|---|---|---|---|---|---|
| Loans (synthetic, binary) | Logistic Regression | 14.8% | 35.9% | 6.1% | 5.9% |
| Customer tiers (synthetic, 3 classes) | Logistic Regression | 31.8% | 35.0% | 16.7% | 16.7% |
| Breast cancer (real, binary, 569 rows) | Logistic Regression | 3.5% | 100% | 3.5% | 3.5% (fallback: too few errors) |

This is reported honestly, as the project charter requires. Model 1 is already calibrated inside the train split, so on clean i.i.d. data its confidence is a strong error detector. The reliability layer adds the most where inputs are unusual (out of range, unseen categories, missing values, far from training data). Those cases show up as warnings and sharply lower trustworthiness for individual predictions, but they are rare in i.i.d. test splits.

---

## API

Base path `/api`. Errors are JSON `{"detail": "...", "code": "..."}` with 400 (bad input), 404 (unknown session), 409 (training not finished), 413 (too large), 422 (validation), 500 (no stack traces).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| POST | `/api/datasets/upload` | Multipart CSV → `session_id`, columns, preview rows, suggested target |
| GET | `/api/datasets/{id}/profile?target_column=` | Rows, types, missing values, class balance, task, validity |
| POST | `/api/train/{id}` | `{"target_column", "target_risk"=0.05, "min_coverage"=0.30, "force_model", "fast", "force_review_on_anomalies"}` → background training |
| GET | `/api/train/{id}/status` | `uploaded` / `pending` / `running` (stage) / `completed` / `failed` (error) |
| GET | `/api/report/{id}` | Leaderboard, selected model, comparison, AUROC/AUPRC, curves, thresholds, profile |
| GET | `/api/schema/{id}` | Feature names, types, categories, train ranges (drives the input form) |
| POST | `/api/predict/{id}` | `{"features": {...}}` (null = missing) → prediction, confidence, trustworthiness, decision, warnings |
| POST | `/api/predict/{id}/batch` | Multipart CSV → table of results (+ outcomes recorded if the target column is present) |
| POST | `/api/feedback/{id}` | `{"items": [{"prediction_id", "actual"}]}` |
| GET | `/api/monitoring/{id}` | Coverage, decision distribution, timeline, observed vs estimated, alerts, recent predictions |
| GET / DELETE | `/api/sessions`, `/api/sessions/{id}` | List / delete sessions |

Pydantic schemas live in `backend/app/schemas/models.py`.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODELDOCTOR_STORAGE_DIR` | `backend/storage` | Where sessions are saved |
| `MODELDOCTOR_MAX_UPLOAD_MB` | `50` | Upload size limit |
| `MODELDOCTOR_SESSION_TTL_HOURS` | `72` | Sessions untouched for longer are deleted at startup / on upload (0 disables) |
| `MODELDOCTOR_FAST_TRAINING` | `false` | Default for the `fast` training option |
| `MODELDOCTOR_MAX_BATCH_ROWS` | `10000` | Batch prediction row limit |
| `MODELDOCTOR_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |

---

## Project structure

```
backend/
  app/            FastAPI: main.py, api/ (routes), schemas/ (Pydantic), core/ (config, sessions, training, errors)
  ml/
    data/         loading + validation, profiling, preprocessing, 3-way split
    models/       candidate pool, CV model selection + calibration
    reliability/  features, Model 2, mitigation thresholds/warnings, evaluation, monitoring
    pipeline.py   end-to-end training (Phase A) and inference (Phase B)
  scripts/        demos and run_experiments.py
  storage/        per-session artifacts (git-ignored)
  tests/
frontend/src/     pages/ (Upload, Training, Report, Predict, Monitoring), components/, api/
sample_data/      loans.csv, customer_tiers.csv, breast_cancer.csv (+ generators)
```

## Known limitations

- Tabular classification only; no regression, images, text or time series.
- Small or very clean datasets give Model 2 too few errors to learn from; ModelDoctor then falls back to calibrated confidence (and says so).
- Trustworthiness is an estimate learned from history ("among past predictions that looked like this, about X% were correct"), not a guarantee for one case.
- Thresholds are tuned on the calibration split; the band error rates shown for them are optimistic. The test-split numbers are the honest ones.
- Under strong drift, reliability estimates degrade; monitoring flags this, and retraining means training a new session on fresh labeled data.
- Sessions are stored on local disk and training runs in the API process, so this is a single-user / single-server setup.
