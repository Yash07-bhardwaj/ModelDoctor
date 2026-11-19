# ModelDoctor

**Intelligent Detection and Mitigation of Unreliable ML Predictions**

ModelDoctor is an advanced machine learning reliability system for tabular classification datasets. It trains a primary model (Model 1) to make predictions, and a secondary reliability model (Model 2, "the doctor") to evaluate whether Model 1's predictions should be **trusted**, **reviewed**, or **rejected**.

---

## Key Features

- **Automated Model Selection (Model 1):** Explores candidate models (Logistic Regression, Decision Trees, Random Forest, XGBoost, etc.) using stratified cross-validation on the train split and calibrates class probabilities.
- **Model-Agnostic Reliability Features:** Computes entropy, margin, bootstrap disagreement, kNN distance to training points, Isolation Forest anomaly scores, missing-value counts, and out-of-range counts from raw inputs.
- **Reliability Model (Model 2):** Learns on unseen calibration data to predict the true empirical likelihood of correctness ($P(\text{correct})$).
- **Actionable Mitigation Layer:** Assigns **TRUST**, **REVIEW**, or **REJECT** decisions with transparent human-readable warning flags.
- **Rigorous Evaluation:** Evaluates on an isolated test set, benchmarking performance against conventional confidence-threshold filtering.

---

## Scope & Constraints

- **Classification Only:** Binary and multiclass classification are supported. Continuous numeric targets are rejected with a clear error message.
- **Data Splits:** Strict 60% Train / 20% Calibration / 20% Test partitioning to eliminate label and evaluation leakage.

---

## Quickstart

### Backend Setup

```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API Health Check: `http://localhost:8000/api/health`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`

### Running Tests

```bash
pytest backend/tests
```
