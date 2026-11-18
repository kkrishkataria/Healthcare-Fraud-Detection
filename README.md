# Verity Health — Healthcare Fraud Risk Detection

[![Live dashboard](https://img.shields.io/badge/live%20dashboard-Vercel-000000?logo=vercel)](https://healthcare-fraud-detection-beryl.vercel.app/)

An end-to-end machine-learning application for prioritizing potential healthcare-provider fraud risk. It combines an XGBoost classifier, a FastAPI inference API, and a React dashboard to help audit teams explore provider data, score custom scenarios, and focus review effort on unusual billing and utilization patterns.

**Live app:** [healthcare-fraud-detection-beryl.vercel.app](https://healthcare-fraud-detection-beryl.vercel.app/)

> Predictions are decision-support signals, not proof of fraud. Every flagged provider requires human investigation, relevant clinical context, and due process before an audit or enforcement decision.

## What it does

- Scores individual provider profiles for potential fraud risk.
- Classifies risk as Low, Medium, or High using a calibrated review threshold.
- Computes billing, reimbursement, utilization, and clinical-intensity features automatically.
- Supports searchable browsing across **5,410** provider records.
- Provides portfolio-level risk counts and average risk probability.
- Gives investigators an explainability view with local, SHAP-style risk-driver context in the dashboard.
- Validates input data and safely imputes the three clinical fields that may be absent.

## Architecture

```text
React + Vite dashboard (Vercel)
          |
          | HTTPS requests
          v
FastAPI risk API (Render / local Uvicorn)
          |
          +-- XGBoost classifier + scikit-learn imputer
          +-- provider_features.csv (5,410 aggregated provider records)
```

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, Vite, Lucide React, CSS |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| ML | XGBoost, scikit-learn, pandas, NumPy |
| Analysis | Jupyter, matplotlib, seaborn, SHAP |
| Deployment | Vercel frontend; remote FastAPI fallback hosted on Render |

## Repository layout

```text
.
├── app/main.py                         # FastAPI app, validation, feature engineering, inference
├── data/provider_features.csv          # Aggregated provider-level data (5,410 records)
├── frontend/
│   ├── src/App.jsx                     # Dashboard UI and API integration
│   ├── src/index.css                   # Global styling
│   └── package.json                    # Frontend scripts and dependencies
├── models/
│   ├── xgb_tuned.pkl                   # Trained XGBoost classifier
│   ├── imputer.pkl                     # Missing-value imputer
│   └── model_metadata.json             # Feature order, threshold, and risk-band metadata
├── notebooks/01_data_understanding.ipynb # EDA, feature engineering, and training work
├── requirements.txt
└── README.md
```

## Run locally

### Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer with npm

### 1. Start the API

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API is then available at:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- Portfolio summary: <http://127.0.0.1:8000/dashboard/summary>

### 2. Start the dashboard

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally <http://localhost:5173>). On localhost, the dashboard uses `http://127.0.0.1:8000` by default. In deployed environments it uses the configured API URL and falls back to the hosted API.

### Configure a different API URL

Create `frontend/.env`:

```env
VITE_API_URL=https://your-api.example.com
```

Restart Vite after changing environment variables.

## API reference

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status and documentation location |
| `GET` | `/health` | Model status, feature count, threshold, and risk bands |
| `GET` | `/features` | Required, derived, and nullable input fields |
| `POST` | `/predict` | Score a custom provider profile |
| `GET` | `/providers` | Search and paginate provider IDs |
| `GET` | `/providers/{provider_id}` | Retrieve a provider profile |
| `GET` | `/providers/{provider_id}/predict` | Score an existing provider |
| `GET` | `/dashboard/summary` | Return portfolio-level risk statistics |

### Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{
    "features": {
      "InpatientClaims": 12,
      "OutpatientClaims": 96,
      "InpatientReimbursement": 210000,
      "OutpatientReimbursement": 90000,
      "UniquePatients": 65,
      "UniquePhysicianCount": 7,
      "PhysicianConcentration": 0.28,
      "AvgPatientAge": 71.2,
      "MedianPatientAge": 72,
      "FemalePatientRatio": 0.57,
      "AvgChronicConditions": 3.4,
      "AvgLengthOfStay": 5.2,
      "MedianLengthOfStay": 4,
      "UniqueDiagnosisCount": 24,
      "UniqueProcedureCount": 9,
      "TotalDiagnosisEntries": 286,
      "TotalProcedureEntries": 104
    }
  }'
```

The response includes `fraud_probability`, `classification_threshold`, `predicted_potential_fraud`, `risk_level`, and a reminder that the result is not a fraud finding.

## Model inputs and feature engineering

The API accepts 17 direct inputs grouped as follows:

- **Claims and reimbursement:** inpatient/outpatient claims and reimbursements.
- **Provider profile:** patient volume, physician count and concentration, patient age, sex ratio, and chronic-condition rate.
- **Clinical activity:** stay length, diagnosis/procedure diversity, and diagnosis/procedure entry counts.

Before inference, the API derives 12 additional measures: total claims and reimbursement, reimbursement per claim/patient, claims per patient/physician, inpatient share, diagnosis/procedure intensity, and logarithmic volume transformations. `AvgLengthOfStay`, `MedianLengthOfStay`, and `UniqueProcedureCount` may be `null`; the saved imputer fills these values and the model receives matching missingness indicators.

Input validation rejects unknown fields, non-numeric or infinite values, disallowed missing values, negative count fields, and cases that would produce invalid ratios.

## Risk bands

| Risk level | Probability range | Suggested handling |
| --- | --- | --- |
| Low | `< 22.80%` | Continue routine monitoring. |
| Medium | `22.80%–59.99%` | Perform secondary review. |
| High | `≥ 60.00%` | Prioritize for detailed audit review. |

The review threshold is stored in [`models/model_metadata.json`](models/model_metadata.json) and is currently `0.2279669642`.

## Dashboard workflow

1. Review portfolio statistics and risk distribution.
2. Search the provider directory or select a benchmark scenario.
3. Inspect and adjust a provider profile in the assessment form.
4. Run a risk assessment.
5. Use the risk meter and feature-driver view to inform an investigator's next steps.

## Retraining and analysis

The notebook at [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) contains the exploratory analysis, feature-engineering, and training workflow. Run it in order when retraining so the model, imputer, and metadata remain compatible with the API's expected feature order.

## Responsible use

This project is designed for prioritization—not automated adjudication. Model output can reflect data quality, historical labeling, coverage gaps, and population bias. Use it with privacy protections, audit logs, representative validation, fairness testing, and qualified human review.
