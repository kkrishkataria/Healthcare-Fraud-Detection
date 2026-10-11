
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "provider_features.csv"

providers_df = pd.read_csv(DATA_PATH)

pickle_model_path = MODEL_DIR / "xgb_tuned.pkl"
metadata_path = MODEL_DIR / "model_metadata.json"
imputer_path = MODEL_DIR / "imputer.pkl"

if pickle_model_path.exists():
    with open(pickle_model_path, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(
        "No trained model found. Expected xgb_tuned.json or xgb_tuned.pkl "
        f"in {MODEL_DIR}."
    )

with open(imputer_path, "rb") as f:
    imputer = pickle.load(f)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold"]
FEATURES = metadata["features"]
MISSING_FEATURES = metadata["missing_features"]
RISK_BANDS = metadata["risk_bands"]

BASE_FEATURES = [
    feature for feature in FEATURES
    if not feature.endswith("_Missing")
]

DERIVED_FEATURES = {
    "TotalClaims",
    "TotalReimbursement",
    "AvgReimbursementPerClaim",
    "ClaimsPerPatient",
    "ReimbursementPerPatient",
    "InpatientRatio",
    "DiagnosisIntensity",
    "ProcedureIntensity",
    "ClaimsPerPhysician",
    "LogTotalDiagnosisEntries",
    "LogClaimsPerPhysician",
    "LogTotalProcedureEntries",
}

INPUT_FEATURES = [
    feature for feature in BASE_FEATURES
    if feature not in DERIVED_FEATURES
]

EXPECTED_INDICATOR_FEATURES = [
    f"{feature}_Missing" for feature in MISSING_FEATURES
]

if any(feature not in FEATURES for feature in MISSING_FEATURES):
    raise ValueError("Missing-value features must be present in model metadata.")

if any(feature not in FEATURES for feature in EXPECTED_INDICATOR_FEATURES):
    raise ValueError(
        "Model metadata does not include every missing-value indicator feature."
    )

if not DERIVED_FEATURES.issubset(FEATURES):
    raise ValueError("Derived features must be present in model metadata.")

if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
    raise ValueError(
        "Model feature count does not match model_metadata.json: "
        f"{getattr(model, 'n_features_in_', 'unknown')} != {len(FEATURES)}."
    )

model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None and list(model_feature_names) != FEATURES:
    raise ValueError("Model feature order does not match model_metadata.json.")

if getattr(imputer, "n_features_in_", len(MISSING_FEATURES)) != len(
    MISSING_FEATURES
):
    raise ValueError("Imputer feature count does not match model metadata.")

imputer_feature_names = getattr(imputer, "feature_names_in_", None)
if (
    imputer_feature_names is not None
    and list(imputer_feature_names) != MISSING_FEATURES
):
    raise ValueError("Imputer feature order does not match model metadata.")


app = FastAPI(
    title="Healthcare Provider Fraud Risk API",
    description="Predicts provider-level potential fraud risk.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, float | None]


@app.get("/")
def home():
    return {
        "message": "Healthcare Provider Fraud Risk API is running",
        "model": "xgb_tuned",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "xgb_tuned",
        "model_feature_count": len(FEATURES),
        "input_feature_count": len(INPUT_FEATURES),
        "classification_threshold": THRESHOLD,
        "risk_bands": RISK_BANDS,
    }


@app.get("/features")
def get_features():
    return {
        "required_features": INPUT_FEATURES,
        "missing_value_features": MISSING_FEATURES,
        "derived_features": sorted(DERIVED_FEATURES),
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    supplied = request.features

    missing_keys = [
        feature for feature in INPUT_FEATURES
        if feature not in supplied
    ]

    if missing_keys:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required features are missing",
                "missing_features": missing_keys
            }
        )

    unknown_keys = [
        key for key in supplied
        if key not in INPUT_FEATURES
    ]

    if unknown_keys:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unknown features supplied",
                "unknown_features": unknown_keys
            }
        )

    try:
        row = pd.DataFrame([
            {
                feature: supplied[feature]
                for feature in INPUT_FEATURES
            }
        ])

        for feature in INPUT_FEATURES:
            row[feature] = pd.to_numeric(row[feature], errors="raise")

        values = row.to_numpy(dtype=float)

        if np.isinf(values).any():
            raise HTTPException(
                status_code=422,
                detail="Features cannot contain infinity."
            )

        null_features = [
            feature for feature in INPUT_FEATURES
            if feature not in MISSING_FEATURES and row[feature].isna().any()
        ]

        if null_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Only configured missing-value features may be null.",
                    "invalid_null_features": null_features,
                },
            )

        invalid_denominators = [
            feature for feature in [
                "InpatientClaims",
                "OutpatientClaims",
                "UniquePatients",
                "UniquePhysicianCount",
                "TotalDiagnosisEntries",
                "TotalProcedureEntries",
            ]
            if row[feature].iloc[0] < 0
        ]

        if invalid_denominators:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Claim counts and entity counts cannot be negative.",
                    "invalid_features": invalid_denominators,
                },
            )

        total_claims = row["InpatientClaims"] + row["OutpatientClaims"]
        zero_denominators = [
            feature for feature, value in {
                "TotalClaims": total_claims.iloc[0],
                "UniquePatients": row["UniquePatients"].iloc[0],
                "UniquePhysicianCount": row["UniquePhysicianCount"].iloc[0],
            }.items()
            if value <= 0
        ]

        if zero_denominators:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Derived ratios require positive claim, patient, and physician counts.",
                    "invalid_features": zero_denominators,
                },
            )

        row["TotalClaims"] = total_claims
        row["TotalReimbursement"] = (
            row["InpatientReimbursement"] + row["OutpatientReimbursement"]
        )
        row["AvgReimbursementPerClaim"] = (
            row["TotalReimbursement"] / row["TotalClaims"]
        )
        row["ClaimsPerPatient"] = row["TotalClaims"] / row["UniquePatients"]
        row["ReimbursementPerPatient"] = (
            row["TotalReimbursement"] / row["UniquePatients"]
        )
        row["InpatientRatio"] = row["InpatientClaims"] / row["TotalClaims"]
        row["DiagnosisIntensity"] = (
            row["TotalDiagnosisEntries"] / row["TotalClaims"]
        )
        row["ProcedureIntensity"] = (
            row["TotalProcedureEntries"] / row["TotalClaims"]
        )
        row["ClaimsPerPhysician"] = (
            row["TotalClaims"] / row["UniquePhysicianCount"]
        )
        row["LogTotalDiagnosisEntries"] = np.log1p(
            row["TotalDiagnosisEntries"]
        )
        row["LogClaimsPerPhysician"] = np.log1p(
            row["ClaimsPerPhysician"]
        )
        row["LogTotalProcedureEntries"] = np.log1p(
            row["TotalProcedureEntries"]
        )

        for feature in MISSING_FEATURES:
            row[feature + "_Missing"] = (
                row[feature].isna().astype(int)
            )

        row[MISSING_FEATURES] = imputer.transform(
            row[MISSING_FEATURES]
        )

        row = row.reindex(columns=FEATURES)

        if not np.isfinite(
            row.to_numpy(dtype=float)
        ).all():
            raise HTTPException(
                status_code=422,
                detail="Features must contain finite numeric values."
            )

        probability = float(
            model.predict_proba(row)[0, 1]
        )

        predicted_fraud = probability >= THRESHOLD

        if probability < RISK_BANDS["low_max"]:
            risk_level = "Low"
        elif probability < RISK_BANDS["medium_max"]:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "fraud_probability": round(probability, 4),
            "classification_threshold": round(
                THRESHOLD, 4
            ),
            "predicted_potential_fraud": bool(
                predicted_fraud
            ),
            "risk_level": risk_level,
            "note": (
                "Model prediction only; this does not "
                "confirm fraud."
            )
        }

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}"
        ) from exc

@app.get("/providers/{provider_id}")
def get_provider(provider_id: str):
    provider = providers_df[
        providers_df["Provider"] == provider_id
    ]

    if provider.empty:
        raise HTTPException(
            status_code=404,
            detail="Provider not found"
        )

    record = provider.iloc[0].to_dict()


    record.pop("PotentialFraud", None)


    record = {
        key: None if pd.isna(value) else value
        for key, value in record.items()
    }

    return record


@app.get("/providers/{provider_id}/predict")
def predict_provider(provider_id: str):
    provider = providers_df[
        providers_df["Provider"] == provider_id
    ]

    if provider.empty:
        raise HTTPException(
            status_code=404,
            detail="Provider not found"
        )

    record = provider.iloc[0].to_dict()

    features = {
        feature: (
            None if pd.isna(record[feature]) else record[feature]
        )
        for feature in INPUT_FEATURES
        if feature in record
    }

    result = predict(PredictionRequest(features=features))

    return {
        "provider_id": provider_id,
        **result
    }


@app.get("/providers")
def list_providers(
    search: str = "",
    skip: int = 0,
    limit: int = 20
):
    if skip < 0:
        raise HTTPException(
            status_code=422,
            detail="skip cannot be negative"
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 100"
        )


    filtered = providers_df

    if search.strip():
        filtered = filtered[
            filtered["Provider"].str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    total = len(filtered)

    page = filtered.iloc[skip:skip + limit]

    provider_ids = page["Provider"].tolist()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "providers": provider_ids
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    rows = providers_df[INPUT_FEATURES].copy()

    rows["TotalClaims"] = (
        rows["InpatientClaims"] + rows["OutpatientClaims"]
    )
    rows["TotalReimbursement"] = (
        rows["InpatientReimbursement"] + rows["OutpatientReimbursement"]
    )
    rows["AvgReimbursementPerClaim"] = (
        rows["TotalReimbursement"] / rows["TotalClaims"]
    )
    rows["ClaimsPerPatient"] = (
        rows["TotalClaims"] / rows["UniquePatients"]
    )
    rows["ReimbursementPerPatient"] = (
        rows["TotalReimbursement"] / rows["UniquePatients"]
    )
    rows["InpatientRatio"] = (
        rows["InpatientClaims"] / rows["TotalClaims"]
    )
    rows["DiagnosisIntensity"] = (
        rows["TotalDiagnosisEntries"] / rows["TotalClaims"]
    )
    rows["ProcedureIntensity"] = (
        rows["TotalProcedureEntries"] / rows["TotalClaims"]
    )
    rows["ClaimsPerPhysician"] = (
        rows["TotalClaims"] / rows["UniquePhysicianCount"]
    )
    rows["LogTotalDiagnosisEntries"] = np.log1p(
        rows["TotalDiagnosisEntries"]
    )
    rows["LogClaimsPerPhysician"] = np.log1p(
        rows["ClaimsPerPhysician"]
    )
    rows["LogTotalProcedureEntries"] = np.log1p(
        rows["TotalProcedureEntries"]
    )

    for feature in MISSING_FEATURES:
        rows[f"{feature}_Missing"] = rows[feature].isna().astype(int)

    rows[MISSING_FEATURES] = imputer.transform(rows[MISSING_FEATURES])
    probabilities = model.predict_proba(rows.reindex(columns=FEATURES))[:, 1]
    low_risk = probabilities < RISK_BANDS["low_max"]
    medium_risk = (
        (probabilities >= RISK_BANDS["low_max"])
        & (probabilities < RISK_BANDS["medium_max"])
    )
    high_risk = probabilities >= RISK_BANDS["medium_max"]

    return {
        "total_providers": int(len(probabilities)),
        "low_risk_providers": int(low_risk.sum()),
        "medium_risk_providers": int(medium_risk.sum()),
        "high_risk_providers": int(high_risk.sum()),
        "average_fraud_probability": round(float(probabilities.mean()), 4),
    }
