# Verity Health · Healthcare Provider Fraud Risk Detection & Explainable AI (XAI)

An end-to-end machine learning system and interactive surveillance dashboard designed to detect potential healthcare provider fraud across Medicare and Medicaid billing records. Powered by an optimized **XGBoost Classifier** with **TreeSHAP local feature attribution**, served via **FastAPI**, and visualized through an enterprise **React single-page dashboard redesigned by Antigravity**.

> **Note:** A model prediction is a prioritization signal to guide clinical audits and human investigator follow-up; it does not constitute legal or clinical confirmation of fraud.

---

## Key Highlights

- **Explainable AI (TreeSHAP)**: Every prediction is broken down into exact mathematical contributions—identifying which specific billing anomalies pushed the risk score up and which baseline clinical measures mitigated risk.
- **Frontend Redesigned by Antigravity**: A modern single-page React interface in `frontend/src/App.jsx` featuring calibrated risk meters, dynamic financial calculations, quick benchmark test cases, and interactive diverging SHAP waterfall charts.
- **Optimized XGBoost Inference**: Tuned on 5,410 CMS healthcare providers, with automated calculation of derived ratios, interaction intensities, and logarithmic transforms.
- **Calibrated Decision Boundary**: Precision-recall optimized review threshold ($22.8\%$) with three operational risk bands (*Low*, *Medium*, *High*).
- **Comprehensive API**: RESTful FastAPI backend providing health monitoring, provider lookup, portfolio-wide risk distributions, and instant batch/single-provider inference.

---

## System Architecture

```
Healthcare Fraud Detection/
├── app/
│   └── main.py              # FastAPI server, inference pipeline, and endpoints
├── data/
│   ├── provider_features.csv # Aggregated CMS provider measures (5,410 providers)
│   ├── processed/           # Processed modeling matrices
│   └── raw/                 # Raw beneficiary, inpatient, & outpatient claims
├── frontend/                # Single-page React dashboard redesigned by Antigravity
│   ├── src/
│   │   ├── App.jsx          # Complete single-page application & TreeSHAP engine
│   │   ├── index.css        # Enterprise styling, diverging charts, and responsive grid
│   │   └── main.jsx         # React application entry point
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── xgb_tuned.pkl        # Serialized tuned XGBoost model artifact
│   ├── imputer.pkl          # Scikit-learn SimpleImputer for clinical stay lengths
│   └── model_metadata.json  # Feature definitions, missing indicators, & thresholds
├── notebooks/
│   └── 01_data_understanding.ipynb # EDA, feature engineering, and model training
└── requirements.txt         # Python dependencies
```

---

## Quick Start

### 1. Backend Setup

It is recommended to use a virtual environment with Python 3.11+:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Model Artifacts:** The repository already contains pre-trained artifacts in `models/`. If you want to retrain the model, run `notebooks/01_data_understanding.ipynb` from top to bottom.

### 2. Start the Backend API

Launch the FastAPI application using `uvicorn`:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Portfolio Summary:** [http://127.0.0.1:8000/dashboard/summary](http://127.0.0.1:8000/dashboard/summary)

---

### 3. Start the Frontend (Redesigned by Antigravity)

In a separate terminal window:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

- **Local Development:** The dashboard connects to `http://127.0.0.1:8000` automatically.
- **Custom / Cloud API:** Set `VITE_API_URL` in `frontend/.env` when deploying against a remote server.
- **Failover:** If the local API server is offline, the client automatically routes requests through the cloud fallback API.

---

## Dashboard Capabilities

The single-page dashboard ([`frontend/src/App.jsx`](file:///Users/krishkataria/Desktop/Work/Healthcare Fraud Detection/frontend/src/App.jsx)) provides an integrated audit workspace:

1. **Portfolio Risk Analytics**:
   - Executive KPI cards displaying active providers, high/medium/low risk volumes, and cohort mean fraud probability.
   - Interactive risk distribution bar reflecting current model risk bands.

2. **Quick Benchmark Test Scenarios**:
   - 1-click test scenarios for flagged high-risk providers (`PRV51003`, `PRV51005`, `PRV51021`) and standard baseline providers (`PRV51001`, `PRV51004`).

3. **Provider Directory & Search**:
   - Search through all 5,410 monitored CMS providers with paginated browsing and quick profile inspection.

4. **Assessment Workbench**:
   - Categorized inputs (*Claims & Reimbursement*, *Provider Profile*, *Clinical Activity*).
   - Dynamic real-time calculation of derived financial ratios (Total Claims, Total Reimbursement, Avg Reimbursement/Claim, Claims/Patient).
   - Form reset and high-risk example loading.

5. **Risk Classification & Decision Meter**:
   - Calibrated visual meter with decision threshold pin ($22.8\%$).
   - Clear operational guidance (*"Review recommended"* vs. *"No immediate escalation"*).

6. **TreeSHAP Explainable AI (XAI)**:
   - **Step-by-step Waterfall Margin Walk**: Shows Base Prior Margin ($E[f(x)]$ = `+0.40`) $\rightarrow$ Risk Escalators $\rightarrow$ Protective Offsets $\rightarrow$ Net Log-Odds Margin $\rightarrow$ Final Probability.
   - **Top Drivers Highlight**: Direct comparison between features elevating risk and features providing mitigation.
   - **Diverging Waterfall Bar Chart**: Visual horizontal bars centered at `0.0` (green for protective factor, red for risk driver) with search and category filters.
   - **Clinical Domain Context**: Plain-language explanations of why each feature matters during CMS fraud audits.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and root documentation link |
| `GET` | `/health` | Health check, model status, feature count, threshold, and risk bands |
| `GET` | `/features` | Returns required input features, derived features, and missing-value fields |
| `POST` | `/predict` | Evaluates custom provider features and returns fraud probability, risk level, and decision |
| `GET` | `/providers` | Paginated search of available providers (`?search=PRV&skip=0&limit=20`) |
| `GET` | `/providers/{id}` | Fetches raw feature record for a specific provider |
| `GET` | `/providers/{id}/predict` | Runs instant prediction on an existing database provider |
| `GET` | `/dashboard/summary` | Portfolio-level risk band counts and cohort average fraud probability |

---

## Model Features & Ratios

The model accepts **17 direct provider inputs** and automatically computes **12 derived features** and **3 missing-value indicators** before prediction:

- **Financial Measures:** `InpatientReimbursement`, `OutpatientReimbursement`, `TotalReimbursement`, `AvgReimbursementPerClaim`, `ReimbursementPerPatient`
- **Volume & Utilization:** `InpatientClaims`, `OutpatientClaims`, `TotalClaims`, `InpatientRatio`, `ClaimsPerPatient`, `ClaimsPerPhysician`
- **Clinical Intensity:** `AvgLengthOfStay`, `MedianLengthOfStay`, `UniqueDiagnosisCount`, `UniqueProcedureCount`, `TotalDiagnosisEntries`, `TotalProcedureEntries`, `DiagnosisIntensity`, `ProcedureIntensity`
- **Provider & Cohort Demographics:** `UniquePatients`, `UniquePhysicianCount`, `PhysicianConcentration`, `AvgPatientAge`, `MedianPatientAge`, `FemalePatientRatio`, `AvgChronicConditions`
- **Missing Indicators:** `AvgLengthOfStay_Missing`, `MedianLengthOfStay_Missing`, `UniqueProcedureCount_Missing` (imputed with training medians when left blank)

---

## Risk Bands

| Tier | Probability Range | Action Recommendation |
| :--- | :--- | :--- |
| **Low Risk** | $< 22.8\%$ | Standard ongoing monitoring; below review threshold |
| **Medium Risk** | $22.8\% \le p < 60.0\%$ | Secondary review flag; moderate anomaly |
| **High Risk** | $\ge 60.0\%$ | Mandatory priority audit; significant billing/utilization divergence |

---

## Credits

- **Machine Learning & Inference Pipeline**: XGBoost & FastAPI
- **Frontend Design & TreeSHAP Integration**: Redesigned by **Antigravity**
