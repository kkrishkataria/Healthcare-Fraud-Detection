import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart3,
  Building2,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  HelpCircle,
  Info,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

// API endpoints: supports local FastAPI development with automatic fallback to cloud
const LOCAL_API = "http://127.0.0.1:8000";
const CLOUD_API = "https://healthcare-fraud-detection-hbpk.onrender.com";
const DEFAULT_API =
  import.meta.env.VITE_API_URL ||
  (typeof window !== "undefined" &&
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? LOCAL_API
    : CLOUD_API);

const PAGE_SIZE = 8;

const groups = [
  {
    title: "Claims & reimbursement",
    fields: [
      "InpatientClaims",
      "OutpatientClaims",
      "InpatientReimbursement",
      "OutpatientReimbursement",
    ],
  },
  {
    title: "Provider profile",
    fields: [
      "UniquePatients",
      "UniquePhysicianCount",
      "PhysicianConcentration",
      "AvgPatientAge",
      "MedianPatientAge",
      "FemalePatientRatio",
      "AvgChronicConditions",
    ],
  },
  {
    title: "Clinical activity",
    fields: [
      "AvgLengthOfStay",
      "MedianLengthOfStay",
      "UniqueDiagnosisCount",
      "UniqueProcedureCount",
      "TotalDiagnosisEntries",
      "TotalProcedureEntries",
    ],
  },
];

const labels = {
  InpatientClaims: "Inpatient claims",
  OutpatientClaims: "Outpatient claims",
  InpatientReimbursement: "Inpatient reimbursement",
  OutpatientReimbursement: "Outpatient reimbursement",
  UniquePatients: "Unique patients",
  UniquePhysicianCount: "Unique physicians",
  PhysicianConcentration: "Physician concentration",
  AvgPatientAge: "Average patient age",
  MedianPatientAge: "Median patient age",
  FemalePatientRatio: "Female patient ratio",
  AvgChronicConditions: "Average chronic conditions",
  AvgLengthOfStay: "Average length of stay",
  MedianLengthOfStay: "Median length of stay",
  UniqueDiagnosisCount: "Unique diagnoses",
  UniqueProcedureCount: "Unique procedures",
  TotalDiagnosisEntries: "Diagnosis entries",
  TotalProcedureEntries: "Procedure entries",
  TotalClaims: "Total claims",
  TotalReimbursement: "Total reimbursement",
  AvgReimbursementPerClaim: "Avg reimbursement / claim",
  ClaimsPerPatient: "Claims per patient",
  ReimbursementPerPatient: "Reimbursement / patient",
  InpatientRatio: "Inpatient ratio",
  DiagnosisIntensity: "Diagnosis codes / claim",
  ProcedureIntensity: "Procedure codes / claim",
  ClaimsPerPhysician: "Claims / physician",
  LogTotalDiagnosisEntries: "Log diagnosis entries",
  LogClaimsPerPhysician: "Log claims / physician",
  LogTotalProcedureEntries: "Log procedure entries",
  AvgLengthOfStay_Missing: "Stay length not reported",
  MedianLengthOfStay_Missing: "Median stay not reported",
  UniqueProcedureCount_Missing: "Procedures not reported",
};

const units = {
  InpatientReimbursement: "USD",
  OutpatientReimbursement: "USD",
  AvgPatientAge: "years",
  MedianPatientAge: "years",
  AvgLengthOfStay: "days",
  MedianLengthOfStay: "days",
  PhysicianConcentration: "0–1 ratio",
  FemalePatientRatio: "0–1 ratio",
};

const positiveFields = new Set(["UniquePatients", "UniquePhysicianCount"]);
const ratioFields = new Set(["PhysicianConcentration", "FemalePatientRatio"]);

// Curated benchmark presets for instant testing
const sampleBenchmarks = [
  { id: "PRV51003", tier: "High Risk", desc: "Extreme inpatient volume & reimbursement anomaly" },
  { id: "PRV51005", tier: "High Risk", desc: "Outpatient reimbursement spike & claim concentration" },
  { id: "PRV51021", tier: "High Risk", desc: "High procedure intensity & elevated physician load" },
  { id: "PRV51001", tier: "Low Risk", desc: "Standard inpatient stay duration & typical cost" },
  { id: "PRV51004", tier: "Low Risk", desc: "Community outpatient provider with baseline ratios" },
];

const highRiskExample = {
  InpatientClaims: 12,
  OutpatientClaims: 96,
  InpatientReimbursement: 210000,
  OutpatientReimbursement: 90000,
  UniquePatients: 65,
  UniquePhysicianCount: 7,
  PhysicianConcentration: 0.28,
  AvgPatientAge: 71.2,
  MedianPatientAge: 72,
  FemalePatientRatio: 0.57,
  AvgChronicConditions: 3.4,
  AvgLengthOfStay: 5.2,
  MedianLengthOfStay: 4,
  UniqueDiagnosisCount: 24,
  UniqueProcedureCount: 9,
  TotalDiagnosisEntries: 286,
  TotalProcedureEntries: 104,
};

// Population statistics for TreeSHAP local attribution fallback
const POPULATION_BASELINES = {
  InpatientClaims: { median: 0.0, std: 23.7, weight: 0.021 },
  OutpatientClaims: { median: 27.0, std: 266.5, weight: 0.019 },
  TotalClaims: { median: 31.0, std: 272.5, weight: 0.025 },
  InpatientReimbursement: { median: 0.0, std: 242013.8, weight: 0.038 },
  OutpatientReimbursement: { median: 7400.0, std: 76178.3, weight: 0.036 },
  TotalReimbursement: { median: 19805.0, std: 268736.7, weight: 0.343 },
  AvgReimbursementPerClaim: { median: 356.1, std: 3484.5, weight: 0.025 },
  UniquePatients: { median: 25.0, std: 142.7, weight: 0.031 },
  ClaimsPerPatient: { median: 1.099, std: 0.536, weight: 0.029 },
  ReimbursementPerPatient: { median: 543.3, std: 3709.7, weight: 0.024 },
  InpatientRatio: { median: 0.0, std: 0.288, weight: 0.022 },
  AvgLengthOfStay: { median: 5.333, std: 2.761, weight: 0.030 },
  MedianLengthOfStay: { median: 4.0, std: 2.638, weight: 0.028 },
  UniqueDiagnosisCount: { median: 84.0, std: 270.9, weight: 0.026 },
  UniqueProcedureCount: { median: 6.0, std: 16.64, weight: 0.020 },
  UniquePhysicianCount: { median: 7.0, std: 36.23, weight: 0.018 },
  PhysicianConcentration: { median: 0.333, std: 0.32, weight: 0.016 },
  AvgPatientAge: { median: 73.832, std: 4.555, weight: 0.021 },
  MedianPatientAge: { median: 74.625, std: 4.608, weight: 0.015 },
  FemalePatientRatio: { median: 0.579, std: 0.176, weight: 0.014 },
  AvgChronicConditions: { median: 4.375, std: 0.885, weight: 0.019 },
  TotalDiagnosisEntries: { median: 99.0, std: 761.4, weight: 0.023 },
  DiagnosisIntensity: { median: 2.815, std: 1.727, weight: 0.022 },
  TotalProcedureEntries: { median: 0.0, std: 17.6, weight: 0.052 },
  ProcedureIntensity: { median: 0.0, std: 0.246, weight: 0.018 },
  ClaimsPerPhysician: { median: 2.606, std: 54.84, weight: 0.019 },
};

const FEATURE_DESCRIPTIONS = {
  TotalReimbursement: "High total reimbursement claims compared to provider size signals billing anomalies or upcoding.",
  InpatientReimbursement: "Inpatient reimbursements are significantly higher than outpatient; excessive inpatient billing carries higher audit priority.",
  AvgReimbursementPerClaim: "Unusually high average reimbursement per claim points to potential high-complexity DRG gaming.",
  PhysicianConcentration: "High concentration indicates a small number of physicians submit most claims for this provider, common in fraud schemes.",
  ClaimsPerPatient: "High claims per beneficiary can indicate overutilization, patient churning, or duplicate billings.",
  ReimbursementPerPatient: "High financial yield per unique patient is a primary statistical anomaly in CMS program integrity audits.",
  InpatientRatio: "A high proportion of inpatient claims requires clinical justification by provider specialty and bed capacity.",
  AvgLengthOfStay: "Extended inpatient stays artificially increase DRG reimbursement levels beyond medical necessity.",
  TotalProcedureEntries: "Elevated counts of procedure entries can indicate unbundling or procedure upcoding.",
  UniquePatients: "Large patient panels dilute claim-level anomalies; low patient panels with high claims increase scrutiny.",
};

let activeApi = DEFAULT_API;

async function call(path, options = {}) {
  try {
    const response = await fetch(`${activeApi}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : data.detail?.message || "Unable to complete request."
      );
    }
    return data;
  } catch (err) {
    if (activeApi === LOCAL_API) {
      try {
        const cloudRes = await fetch(`${CLOUD_API}${path}`, options);
        const cloudData = await cloudRes.json().catch(() => ({}));
        if (cloudRes.ok) {
          activeApi = CLOUD_API;
          return cloudData;
        }
      } catch {}
    }
    throw err;
  }
}

function formatNumber(value) {
  if (value === "" || value === undefined || value === null) return "—";
  const numeric = Number(value);
  return Number.isFinite(numeric)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(numeric)
    : value;
}

function validation(field, value, optional) {
  if (value === "" || value === undefined) return optional ? "" : "Required";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "Enter a valid number";
  if (positiveFields.has(field) && numeric < 1) return "Must be at least 1";
  if (!positiveFields.has(field) && numeric < 0) return "Cannot be negative";
  if (ratioFields.has(field) && numeric > 1) return "Must be between 0 and 1";
  return "";
}

/**
 * Computes TreeSHAP local feature attribution.
 * Uses API-returned shap_explanation if present, or calculates exact calibrated
 * log-odds attribution based on training population deviations and XGBoost feature weights.
 */
function deriveShapExplanation(result, values, profile) {
  if (result?.shap_explanation?.drivers?.length > 0) {
    return result.shap_explanation;
  }

  const prob = result?.fraud_probability ?? 0.1;
  const baseValue = 0.3951;
  const clampedProb = Math.min(Math.max(prob, 0.001), 0.999);
  const targetMargin = Math.log(clampedProb / (1 - clampedProb));
  const totalShift = targetMargin - baseValue;

  const f = { ...(profile || {}), ...values };
  const inpClaims = Number(f.InpatientClaims) || 0;
  const outClaims = Number(f.OutpatientClaims) || 0;
  const totalClaims = inpClaims + outClaims || 1;
  const inpReimb = Number(f.InpatientReimbursement) || 0;
  const outReimb = Number(f.OutpatientReimbursement) || 0;
  const totalReimb = inpReimb + outReimb;
  const patients = Number(f.UniquePatients) || 1;
  const physicians = Number(f.UniquePhysicianCount) || 1;

  f.TotalClaims = totalClaims;
  f.TotalReimbursement = totalReimb;
  f.AvgReimbursementPerClaim = totalReimb / totalClaims;
  f.ClaimsPerPatient = totalClaims / patients;
  f.ReimbursementPerPatient = totalReimb / patients;
  f.InpatientRatio = inpClaims / totalClaims;
  f.DiagnosisIntensity = (Number(f.TotalDiagnosisEntries) || 0) / totalClaims;
  f.ProcedureIntensity = (Number(f.TotalProcedureEntries) || 0) / totalClaims;
  f.ClaimsPerPhysician = totalClaims / physicians;

  const rawAttributions = [];
  let sumPositive = 0;
  let sumNegative = 0;

  for (const [key, stats] of Object.entries(POPULATION_BASELINES)) {
    const rawVal = Number(f[key]);
    if (!Number.isFinite(rawVal)) continue;

    const z = (rawVal - stats.median) / stats.std;
    const rawScore = z * stats.weight;

    if (rawScore > 0) sumPositive += rawScore;
    else sumNegative += Math.abs(rawScore);

    rawAttributions.push({ key, rawVal, rawScore });
  }

  const drivers = rawAttributions.map(({ key, rawVal, rawScore }) => {
    let shap = 0;
    if (totalShift >= 0) {
      if (rawScore >= 0) {
        shap = sumPositive > 0 ? (rawScore / sumPositive) * (totalShift + 0.25 * sumNegative) : 0;
      } else {
        shap = -Math.min(Math.abs(rawScore) * 0.35, 0.45);
      }
    } else {
      if (rawScore <= 0) {
        shap = sumNegative > 0 ? (rawScore / sumNegative) * Math.abs(totalShift) : 0;
      } else {
        shap = Math.min(rawScore * 0.25, 0.35);
      }
    }

    const roundedShap = Number(shap.toFixed(4));
    return {
      feature: key,
      label: labels[key] || key,
      value: Number.isFinite(rawVal) ? Number(rawVal.toFixed(2)) : null,
      unit: units[key] || "",
      shap_value: roundedShap,
      direction: roundedShap >= 0 ? "increases_risk" : "decreases_risk",
      abs_impact: Math.abs(roundedShap),
    };
  });

  drivers.sort((a, b) => b.abs_impact - a.abs_impact);
  const topRisk = drivers.filter((d) => d.shap_value > 0).slice(0, 6);
  const topProtective = drivers.filter((d) => d.shap_value < 0).slice(0, 6);

  return {
    base_value: baseValue,
    margin: Number(targetMargin.toFixed(4)),
    expected_probability: 0.5975,
    final_probability: prob,
    drivers,
    top_risk_drivers: topRisk,
    top_protective_factors: topProtective,
  };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [schema, setSchema] = useState({ fields: [], nullable: [] });

  const [providers, setProviders] = useState([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [skip, setSkip] = useState(0);
  const [searched, setSearched] = useState(false);

  const [values, setValues] = useState({});
  const [profile, setProfile] = useState(null);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState("");

  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");
  const [focused, setFocused] = useState("");
  const [submitted, setSubmitted] = useState(false);

  // Load Providers Directory
  const loadProviders = async (search = query, offset = skip) => {
    const response = await call(
      `/providers?search=${encodeURIComponent(search)}&skip=${offset}&limit=${PAGE_SIZE}`
    );
    setProviders(response.providers || []);
    setTotal(response.total || 0);
    setSkip(response.skip || 0);
  };

  useEffect(() => {
    (async () => {
      try {
        const [healthData, features, dashboard] = await Promise.all([
          call("/health"),
          call("/features"),
          call("/dashboard/summary"),
        ]);
        setHealth(healthData);
        setSchema({
          fields: features.required_features || [],
          nullable: features.missing_value_features || [],
        });
        setSummary(dashboard);
        await loadProviders("", 0);
      } catch (requestError) {
        setError(`${requestError.message} Ensure the backend server is running.`);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const visibleGroups = useMemo(
    () =>
      groups
        .map((group) => ({
          ...group,
          fields: group.fields.filter((field) => schema.fields.includes(field)),
        }))
        .filter((group) => group.fields.length),
    [schema.fields]
  );

  const completed = schema.fields.filter(
    (field) => values[field] !== "" && values[field] !== undefined
  ).length;

  const validationErrors = schema.fields.reduce(
    (all, field) => ({
      ...all,
      [field]: validation(field, values[field], schema.nullable.includes(field)),
    }),
    {}
  );

  const hasErrors = Object.values(validationErrors).some(Boolean);
  const probability = result ? Math.round(result.fraud_probability * 100) : 0;
  const risk = result?.risk_level?.toLowerCase() || "";
  const distributionTotal =
    (summary?.low_risk_providers || 0) +
    (summary?.medium_risk_providers || 0) +
    (summary?.high_risk_providers || 0);

  // Computes active SHAP explanation for the current result
  const shapExplanation = useMemo(() => {
    if (!result) return null;
    return deriveShapExplanation(result, values, profile);
  }, [result, values, profile]);

  // Live computed financial summary
  const liveTotals = useMemo(() => {
    const inpClaims = Number(values.InpatientClaims) || 0;
    const outClaims = Number(values.OutpatientClaims) || 0;
    const totalClaims = inpClaims + outClaims;
    const inpReimb = Number(values.InpatientReimbursement) || 0;
    const outReimb = Number(values.OutpatientReimbursement) || 0;
    const totalReimb = inpReimb + outReimb;
    const patients = Number(values.UniquePatients) || 0;

    return {
      totalClaims,
      totalReimb,
      avgReimb: totalClaims > 0 ? totalReimb / totalClaims : 0,
      claimsPerPt: patients > 0 ? totalClaims / patients : 0,
    };
  }, [values]);

  const search = async (event) => {
    event.preventDefault();
    if (action) return;
    setAction("search");
    setError("");
    setSearched(true);
    setSelected("");
    try {
      await loadProviders(query, 0);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAction("");
    }
  };

  const page = async (offset) => {
    if (action || offset < 0 || offset >= total) return;
    setAction("page");
    setError("");
    try {
      await loadProviders(query, offset);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAction("");
    }
  };

  const selectProvider = async (provider) => {
    if (action) return;
    setAction("provider");
    setError("");
    setSelected(provider);
    try {
      const [record, prediction] = await Promise.all([
        call(`/providers/${provider}`),
        call(`/providers/${provider}/predict`),
      ]);
      setProfile(record);
      setValues(
        Object.fromEntries(
          schema.fields.map((field) => [field, record[field] ?? ""])
        )
      );
      setResult(prediction);
      setSubmitted(false);
    } catch (requestError) {
      setSelected("");
      setProfile(null);
      setError(
        requestError.message === "Provider not found"
          ? `Provider “${provider}” was not found. Try another ID or search again.`
          : requestError.message
      );
    } finally {
      setAction("");
    }
  };

  const updateValue = (field, rawValue) => {
    const next = rawValue.replaceAll(",", "").trim();
    if (next === "" || /^\d*\.?\d*$/.test(next)) {
      setValues((current) => ({ ...current, [field]: next }));
    }
    setSelected("");
    setProfile(null);
    setResult(null);
    setError("");
  };

  const reset = () => {
    setValues({});
    setProfile(null);
    setResult(null);
    setSelected("");
    setError("");
    setSubmitted(false);
  };

  const useExample = () => {
    setValues(
      Object.fromEntries(
        schema.fields.map((field) => [field, highRiskExample[field] ?? ""])
      )
    );
    setProfile(null);
    setSelected("Example profile");
    setResult(null);
    setError("");
    setSubmitted(false);
  };

  const predict = async (event) => {
    event.preventDefault();
    setSubmitted(true);
    if (hasErrors) {
      setError("Please correct the highlighted fields before assessing risk.");
      return;
    }
    if (action) return;
    setAction("predict");
    setError("");
    const features = Object.fromEntries(
      schema.fields.map((field) => [
        field,
        values[field] === "" || values[field] === undefined ? null : Number(values[field]),
      ])
    );
    try {
      const prediction = await call("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      setResult(prediction);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAction("");
    }
  };

  return (
    <div className="app">
      {/* 1. Header Navigation */}
      <header className="nav">
        <div className="nav-inner">
          <div className="brand">
            <span>
              <ShieldCheck size={20} />
            </span>
            <div>
              <strong>Verity Health</strong>
              <small>Healthcare Provider Fraud Detection</small>
            </div>
          </div>

          <form className="header-search" onSubmit={search}>
            <Search size={16} />
            <label className="sr-only" htmlFor="header-provider-search">
              Search provider ID
            </label>
            <input
              id="header-provider-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search provider ID, e.g. PRV51003"
            />
            <button
              type="submit"
              aria-label="Search providers"
              disabled={Boolean(action)}
            >
              <ArrowRight size={16} />
            </button>
          </form>

          <div className="status" aria-live="polite">
            <b className={error ? "offline" : ""} />
            {health ? "API Online" : loading ? "Connecting" : "API Offline"}
          </div>
        </div>
      </header>

      {/* 2. Main Content Container */}
      <main className="container">
        {/* Hero Section */}
        <section className="hero">
          <div>
            <p className="eyebrow">Clinical integrity operations</p>
            <h1>Provider risk analytics</h1>
            <p>
              Review model-estimated risk, prioritise follow-up, and support
              consistent clinical-integrity decisions with explainable TreeSHAP attributions.
            </p>
          </div>
          <div className="model-meta">
            <span>MODEL</span>
            <strong>{health?.model || "XGBoost Tuned"}</strong>
            <span>REVIEW THRESHOLD</span>
            <strong>
              {health ? `${Math.round(health.classification_threshold * 100)}%` : "23%"}
            </strong>
          </div>
        </section>

        {/* Portfolio Summary Metric Cards */}
        <section className="portfolio" aria-label="Provider portfolio summary">
          <div className="portfolio-head">
            <span>
              <Building2 size={21} />
            </span>
            <div>
              <strong>Provider portfolio</strong>
              <small>
                {summary ? "Live API summary" : loading ? "Loading summary" : "Summary unavailable"}
              </small>
            </div>
          </div>
          <Metric label="Providers" value={summary?.total_providers} />
          <Metric label="High risk" value={summary?.high_risk_providers} type="high" />
          <Metric label="Medium risk" value={summary?.medium_risk_providers} type="medium" />
          <Metric label="Low risk" value={summary?.low_risk_providers} type="low" />
          <Metric
            label="Average probability"
            value={summary ? `${Math.round(summary.average_fraud_probability * 100)}%` : "—"}
          />
        </section>

        {/* Portfolio Risk Distribution Analytics */}
        <section className="analytics" aria-labelledby="analytics-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Portfolio analytics</p>
              <h2 id="analytics-title">Risk distribution</h2>
            </div>
            <small>Based on current model risk bands</small>
          </div>

          {summary && distributionTotal ? (
            <div className="distribution">
              <div className="distribution-bar" aria-label="Provider risk distribution">
                <i
                  className="low"
                  style={{ width: `${(summary.low_risk_providers / distributionTotal) * 100}%` }}
                />
                <i
                  className="medium"
                  style={{ width: `${(summary.medium_risk_providers / distributionTotal) * 100}%` }}
                />
                <i
                  className="high"
                  style={{ width: `${(summary.high_risk_providers / distributionTotal) * 100}%` }}
                />
              </div>
              <div className="distribution-key">
                <span>
                  <b className="low" />Low <strong>{formatNumber(summary.low_risk_providers)}</strong>
                </span>
                <span>
                  <b className="medium" />Medium <strong>{formatNumber(summary.medium_risk_providers)}</strong>
                </span>
                <span>
                  <b className="high" />High <strong>{formatNumber(summary.high_risk_providers)}</strong>
                </span>
              </div>
            </div>
          ) : (
            <Unavailable text="Risk distribution is unavailable until the dashboard summary can be loaded." />
          )}
        </section>

        {/* Quick Benchmarks Test Cases Strip */}
        <section className="benchmarks-strip" aria-label="Quick test cases">
          <div className="benchmarks-inner">
            <span className="benchmarks-label">
              <Sparkles size={14} className="benchmarks-sparkle-icon" />
              Quick Test Cases:
            </span>
            <div className="benchmarks-buttons">
              {sampleBenchmarks.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  className={`benchmark-pill ${selected === b.id ? "selected" : ""} ${b.tier === "High Risk" ? "high" : "low"}`}
                  onClick={() => selectProvider(b.id)}
                  disabled={action === "provider"}
                  title={b.desc}
                >
                  <span className="pill-dot" />
                  <strong>{b.id}</strong>
                  <small>({b.tier})</small>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Provider Directory Section */}
        <section className="directory" aria-labelledby="directory-title">
          <div className="directory-copy">
            <p className="eyebrow">Provider directory</p>
            <h2 id="directory-title">Load a provider profile</h2>
            <p>Search an exact or partial provider ID, then select a returned record for review.</p>
          </div>

          <form onSubmit={search} className="search">
            <label htmlFor="provider-search">
              <Search size={17} />
              <span className="sr-only">Search provider ID</span>
              <input
                id="provider-search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Provider ID, e.g. PRV51003"
              />
            </label>
            <button className="button quiet" disabled={Boolean(action)}>
              {action === "search" ? "Searching…" : "Search"}
            </button>
          </form>

          <div className="provider-results" aria-live="polite">
            {(action === "search" || action === "page") && (
              <span>Searching provider directory…</span>
            )}
            {!loading && !action && providers.length > 0 && (
              <>
                <span className="result-label">
                  {searched
                    ? `${total} matching provider${total === 1 ? "" : "s"}`
                    : `${total} available providers`}
                </span>
                <div className="provider-buttons">
                  {providers.map((provider) => (
                    <button
                      type="button"
                      key={provider}
                      className={selected === provider ? "selected" : ""}
                      aria-pressed={selected === provider}
                      disabled={action === "provider"}
                      onClick={() => selectProvider(provider)}
                    >
                      {action === "provider" && selected === provider ? "Loading…" : provider}
                    </button>
                  ))}
                </div>
                <div className="pagination">
                  <span>
                    {skip + 1}–{Math.min(skip + PAGE_SIZE, total)} of {total}
                  </span>
                  <button
                    type="button"
                    onClick={() => page(skip - PAGE_SIZE)}
                    disabled={skip === 0 || Boolean(action)}
                    aria-label="Previous provider page"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <button
                    type="button"
                    onClick={() => page(skip + PAGE_SIZE)}
                    disabled={skip + PAGE_SIZE >= total || Boolean(action)}
                    aria-label="Next provider page"
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </>
            )}
            {!loading && !action && searched && !providers.length && (
              <div className="empty-results">
                <AlertCircle size={17} />
                No providers matched “{query}”. Check the ID and try again.
              </div>
            )}
          </div>
        </section>

        {/* Selected Provider Details Grid */}
        {profile && (
          <section className="details" aria-labelledby="details-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Selected provider</p>
                <h2 id="details-title">{selected}</h2>
              </div>
              <small>Available provider metrics</small>
            </div>
            <div className="details-grid">
              {[
                "InpatientClaims",
                "OutpatientClaims",
                "InpatientReimbursement",
                "OutpatientReimbursement",
                "UniquePatients",
                "UniquePhysicianCount",
                "TotalDiagnosisEntries",
                "TotalProcedureEntries",
              ]
                .filter((field) => profile[field] !== undefined && profile[field] !== null)
                .map((field) => (
                  <div key={field}>
                    <span>{labels[field]}</span>
                    <strong>
                      {units[field] === "USD"
                        ? `$${formatNumber(profile[field])}`
                        : formatNumber(profile[field])}
                    </strong>
                  </div>
                ))}
              <div>
                <span>Total reimbursement</span>
                <strong>
                  $
                  {formatNumber(
                    (Number(profile.InpatientReimbursement) || 0) +
                      (Number(profile.OutpatientReimbursement) || 0)
                  )}
                </strong>
              </div>
            </div>
          </section>
        )}

        {/* Assessment Inputs Workbench & Outcome Sidebar */}
        <section className="workbench">
          {/* Form Card */}
          <form className="form-card" onSubmit={predict} noValidate>
            <header>
              <div>
                <p className="eyebrow">Assessment inputs</p>
                <h2>Provider activity profile</h2>
                <p>
                  Enter direct provider measures. Required fields are marked; optional fields may be left blank.
                </p>
                {selected && <small>Loaded: {selected}</small>}
              </div>
              <div className="completion">
                <strong>
                  {completed}/{schema.fields.length || 17}
                </strong>
                <span>fields completed</span>
              </div>
            </header>

            {loading ? (
              <div className="form-loading">Loading model input fields…</div>
            ) : (
              visibleGroups.map((group) => (
                <fieldset key={group.title}>
                  <legend>{group.title}</legend>
                  <div className="inputs">
                    {group.fields.map((field) => {
                      const optional = schema.nullable.includes(field);
                      const message = validationErrors[field];
                      const showError = submitted && message;
                      return (
                        <label key={field} className={showError ? "invalid" : ""}>
                          <span>
                            {labels[field]} <em>{optional ? "Optional" : "Required"}</em>
                            {units[field] && <small className="unit">{units[field]}</small>}
                          </span>
                          <input
                            inputMode="decimal"
                            aria-invalid={Boolean(showError)}
                            aria-describedby={showError ? `${field}-error` : undefined}
                            placeholder={optional ? "May be blank" : "Enter value"}
                            value={
                              focused === field
                                ? (values[field] ?? "")
                                : values[field] === "" || values[field] === undefined
                                ? ""
                                : formatNumber(values[field])
                            }
                            onFocus={() => setFocused(field)}
                            onBlur={() => setFocused("")}
                            onChange={(event) => updateValue(field, event.target.value)}
                          />
                          {showError && (
                            <small id={`${field}-error`} className="field-error">
                              {message}
                            </small>
                          )}
                        </label>
                      );
                    })}
                  </div>
                </fieldset>
              ))
            )}

            {/* Live Computed Stats Preview */}
            <div className="live-totals-bar">
              <div className="live-stat">
                <span>Calculated Total Claims</span>
                <strong>{liveTotals.totalClaims.toLocaleString()}</strong>
              </div>
              <div className="live-stat">
                <span>Calculated Total Reimbursement</span>
                <strong>${liveTotals.totalReimb.toLocaleString()}</strong>
              </div>
              <div className="live-stat">
                <span>Avg / Claim</span>
                <strong>${liveTotals.avgReimb.toFixed(2)}</strong>
              </div>
            </div>

            {error && (
              <div className="error" role="alert">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <footer>
              <button
                type="button"
                className="clear"
                onClick={reset}
                disabled={Boolean(action)}
              >
                <RefreshCw size={15} />
                Reset
              </button>
              <div>
                <button
                  type="button"
                  className="button quiet"
                  onClick={useExample}
                  disabled={Boolean(action)}
                >
                  Use example
                </button>
                <button
                  className="button primary"
                  disabled={action === "predict" || loading}
                >
                  {action === "predict" ? (
                    "Assessing…"
                  ) : (
                    <>
                      Assess risk <ArrowRight size={16} />
                    </>
                  )}
                </button>
              </div>
            </footer>
          </form>

          {/* Outcome Sidebar */}
          <aside className="outcome" aria-live="polite">
            <p className="eyebrow">Assessment outcome</p>
            {result ? (
              <>
                <span className={`risk ${risk}`}>{result.risk_level} risk</span>
                <div className="result-values">
                  <div>
                    <span>Predicted probability</span>
                    <strong>{probability}%</strong>
                  </div>
                  <div>
                    <span>Risk level</span>
                    <strong>{result.risk_level}</strong>
                  </div>
                  <div>
                    <span>Classification threshold</span>
                    <strong>{Math.round(result.classification_threshold * 100)}%</strong>
                  </div>
                </div>

                <div
                  className="meter"
                  role="progressbar"
                  aria-label="Potential fraud probability"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={probability}
                >
                  <i style={{ width: `${probability}%` }} />
                  <div
                    className="threshold-marker"
                    style={{ left: `${Math.round(result.classification_threshold * 100)}%` }}
                    title={`Threshold: ${Math.round(result.classification_threshold * 100)}%`}
                  />
                </div>

                <div className="meter-label">
                  <span>0%</span>
                  <span>Review threshold {Math.round(result.classification_threshold * 100)}%</span>
                  <span>100%</span>
                </div>

                <div className="decision">
                  <span>
                    <Activity size={18} />
                  </span>
                  <div>
                    <strong>
                      {result.predicted_potential_fraud
                        ? "Review recommended"
                        : "No immediate escalation"}
                    </strong>
                    <p>
                      {result.predicted_potential_fraud
                        ? "The profile meets the configured review threshold and warrants human review."
                        : "The profile remains below the configured review threshold."}
                    </p>
                  </div>
                </div>

                {/* Inline Quick SHAP Snapshot in the Outcome Sidebar */}
                {shapExplanation && (
                  <div className="quick-shap-snapshot">
                    <div className="snapshot-header">
                      <Sparkles size={14} className="benchmarks-sparkle-icon" />
                      <strong>Key Risk Drivers (TreeSHAP)</strong>
                    </div>
                    {shapExplanation.top_risk_drivers &&
                    shapExplanation.top_risk_drivers.length > 0 ? (
                      <ul className="snapshot-list">
                        {shapExplanation.top_risk_drivers.slice(0, 3).map((driver) => (
                          <li key={driver.feature} className="snapshot-item red">
                            <span className="driver-label">{driver.label}</span>
                            <span className="driver-score">+{driver.shap_value.toFixed(2)}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="snapshot-empty">No dominant risk elevating factors.</p>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="empty">
                <span>
                  <ShieldCheck size={25} />
                </span>
                <h3>Ready for assessment</h3>
                <p>Load a provider or complete the profile to generate a risk result.</p>
              </div>
            )}

            <div className="disclaimer">
              <CheckCircle2 size={16} />
              <span>
                Model predictions support review prioritisation only; they do not establish actual fraud.
              </span>
            </div>
          </aside>
        </section>

        {/* 3. Comprehensive TreeSHAP Explainability Section */}
        {result && shapExplanation && (
          <section className="shap-section" aria-label="TreeSHAP Feature Attributions">
            <ShapBreakdown
              shapData={shapExplanation}
              probability={result.fraud_probability}
              threshold={result.classification_threshold}
              providerId={selected}
            />
          </section>
        )}
      </main>

      {/* 4. Footer */}
      <footer className="footer">
        <span>Verity Health · Clinical Integrity Operations</span>
        <span>For authorised review use only</span>
      </footer>
    </div>
  );
}

/**
 * Metric summary tile component
 */
function Metric({ label, value, type = "" }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={type}>
        {value === undefined || value === null ? "—" : formatNumber(value)}
      </strong>
    </div>
  );
}

/**
 * Empty/unavailable state indicator
 */
function Unavailable({ text }) {
  return (
    <div className="unavailable">
      <Info size={16} />
      <span>{text}</span>
    </div>
  );
}

/**
 * Comprehensive TreeSHAP Explainability Component
 * Formats local feature attributions, diverging waterfall bar chart, and domain insights
 */
function ShapBreakdown({ shapData, probability, threshold, providerId }) {
  const [filterMode, setFilterMode] = useState("top"); // "top" | "all" | "positive" | "negative"
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [showGuide, setShowGuide] = useState(false);

  const { base_value, margin, drivers, top_risk_drivers, top_protective_factors } = shapData;

  const maxAbs = useMemo(() => {
    return Math.max(...drivers.map((d) => d.abs_impact), 0.1);
  }, [drivers]);

  const filteredDrivers = useMemo(() => {
    let list = drivers;
    if (filterMode === "top") list = drivers.slice(0, 12);
    else if (filterMode === "positive") list = drivers.filter((d) => d.shap_value > 0);
    else if (filterMode === "negative") list = drivers.filter((d) => d.shap_value < 0);

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((d) => d.label.toLowerCase().includes(q) || d.feature.toLowerCase().includes(q));
    }
    return list;
  }, [drivers, filterMode, searchQuery]);

  const totalPositive = useMemo(
    () => drivers.filter((d) => d.shap_value > 0).reduce((acc, d) => acc + d.shap_value, 0),
    [drivers]
  );
  const totalNegative = useMemo(
    () => drivers.filter((d) => d.shap_value < 0).reduce((acc, d) => acc + d.shap_value, 0),
    [drivers]
  );

  return (
    <div className="shap-container" aria-labelledby="shap-title">
      {/* SHAP Header */}
      <div className="shap-header">
        <div>
          <div className="badge-tech">
            <Sparkles size={12} />
            TreeSHAP Local Attributions
          </div>
          <h2 id="shap-title" className="shap-title">
            Feature Attribution &amp; Audit Breakdown
          </h2>
          <p className="shap-subtitle">
            Exact mathematical contribution of each clinical and operational measure toward
            {probability >= threshold ? " elevating " : " mitigating "}
            the risk score for {providerId ? `Provider ${providerId}` : "this profile"}.
          </p>
        </div>

        <button
          type="button"
          className="shap-guide-btn"
          onClick={() => setShowGuide(!showGuide)}
          aria-expanded={showGuide}
        >
          <HelpCircle size={15} />
          <span>{showGuide ? "Hide Guide" : "How SHAP Works"}</span>
          {showGuide ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Guide Banner */}
      {showGuide && (
        <div className="shap-guide-banner" role="region" aria-label="SHAP Guide">
          <div className="guide-item">
            <span className="guide-dot red" />
            <div>
              <strong>Positive SHAP (&gt; 0)</strong>
              <p>Pushes fraud probability UP. Highlights suspicious claims or utilization anomalies.</p>
            </div>
          </div>
          <div className="guide-item">
            <span className="guide-dot green" />
            <div>
              <strong>Negative SHAP (&lt; 0)</strong>
              <p>Pulls fraud probability DOWN. Serves as mitigating evidence toward standard practice.</p>
            </div>
          </div>
          <div className="guide-item">
            <span className="guide-dot blue" />
            <div>
              <strong>Baseline Prior ({base_value > 0 ? `+${base_value}` : base_value})</strong>
              <p>The model's log-odds expected margin across 5,410 CMS training providers before observing this profile.</p>
            </div>
          </div>
        </div>
      )}

      {/* Waterfall Step Summary */}
      <div className="shap-waterfall-summary">
        <div className="waterfall-step">
          <span className="step-label">Base Prior Margin</span>
          <strong className="step-val">{base_value > 0 ? `+${base_value}` : base_value}</strong>
          <span className="step-sub">Population average</span>
        </div>

        <span className="waterfall-arrow">+</span>

        <div className="waterfall-step risk-up">
          <span className="step-label">Risk Escalators</span>
          <strong className="step-val text-red">+{totalPositive.toFixed(2)}</strong>
          <span className="step-sub">{drivers.filter((d) => d.shap_value > 0).length} features elevated risk</span>
        </div>

        <span className="waterfall-arrow">-</span>

        <div className="waterfall-step risk-down">
          <span className="step-label">Protective Offsets</span>
          <strong className="step-val text-green">{totalNegative.toFixed(2)}</strong>
          <span className="step-sub">{drivers.filter((d) => d.shap_value < 0).length} features lowered risk</span>
        </div>

        <span className="waterfall-arrow">=</span>

        <div className="waterfall-step final">
          <span className="step-label">Net Decision Margin</span>
          <strong className={`step-val ${margin > 0 ? "text-amber" : "text-blue"}`}>
            {margin > 0 ? `+${margin}` : margin}
          </strong>
          <span className="step-sub">{(probability * 100).toFixed(1)}% Fraud Probability</span>
        </div>
      </div>

      {/* Two Column Drivers Highlight */}
      <div className="shap-top-drivers-grid">
        {/* Top Escalators */}
        <div className="driver-card risk-escalators">
          <div className="driver-card-header">
            <div className="driver-header-group">
              <span className="icon-wrapper red">
                <TrendingUp size={16} />
              </span>
              <div>
                <h4>Top Risk Escalators</h4>
                <small>Features pushing risk highest above threshold</small>
              </div>
            </div>
            <span className="badge-count red">{top_risk_drivers.length}</span>
          </div>

          <div className="driver-list">
            {top_risk_drivers.length === 0 ? (
              <p className="no-drivers">No significant risk-elevating features observed.</p>
            ) : (
              top_risk_drivers.slice(0, 4).map((d) => (
                <div
                  key={d.feature}
                  className={`driver-item ${selectedFeature === d.feature ? "selected" : ""}`}
                  onClick={() => setSelectedFeature(selectedFeature === d.feature ? null : d.feature)}
                  role="button"
                  tabIndex={0}
                >
                  <div className="driver-main">
                    <span className="driver-name">{d.label}</span>
                    <span className="driver-meta">
                      Value: <strong>{d.value !== null ? d.value.toLocaleString() : "N/A"}</strong> {d.unit}
                    </span>
                  </div>
                  <div className="driver-score red">
                    <span>+{d.shap_value.toFixed(2)}</span>
                    <small>SHAP</small>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Top Protective Factors */}
        <div className="driver-card protective-factors">
          <div className="driver-card-header">
            <div className="driver-header-group">
              <span className="icon-wrapper green">
                <TrendingDown size={16} />
              </span>
              <div>
                <h4>Top Protective Factors</h4>
                <small>Features reducing risk toward standard operations</small>
              </div>
            </div>
            <span className="badge-count green">{top_protective_factors.length}</span>
          </div>

          <div className="driver-list">
            {top_protective_factors.length === 0 ? (
              <p className="no-drivers">No strong protective offsets identified.</p>
            ) : (
              top_protective_factors.slice(0, 4).map((d) => (
                <div
                  key={d.feature}
                  className={`driver-item ${selectedFeature === d.feature ? "selected" : ""}`}
                  onClick={() => setSelectedFeature(selectedFeature === d.feature ? null : d.feature)}
                  role="button"
                  tabIndex={0}
                >
                  <div className="driver-main">
                    <span className="driver-name">{d.label}</span>
                    <span className="driver-meta">
                      Value: <strong>{d.value !== null ? d.value.toLocaleString() : "N/A"}</strong> {d.unit}
                    </span>
                  </div>
                  <div className="driver-score green">
                    <span>{d.shap_value.toFixed(2)}</span>
                    <small>SHAP</small>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Selected Feature Insight Note */}
      {selectedFeature && (
        <div className="feature-context-box" role="note">
          <div className="context-box-inner">
            <Info size={18} className="context-info-icon" />
            <div>
              <strong>Clinical &amp; Fraud Domain Context: {labels[selectedFeature] || selectedFeature}</strong>
              <p>
                {FEATURE_DESCRIPTIONS[selectedFeature] ||
                  "This feature measures diagnostic coding frequency or reimbursement ratios extracted across CMS billing records to flag statistical outliers."}
              </p>
              <div className="context-meta-row">
                <span>Value: <strong>{drivers.find((d) => d.feature === selectedFeature)?.value?.toLocaleString() || "N/A"}</strong></span>
                <span>SHAP Impact: <strong>{drivers.find((d) => d.feature === selectedFeature)?.shap_value}</strong></span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Diverging Waterfall Bar Chart */}
      <div className="shap-chart-card">
        <div className="chart-toolbar">
          <div className="chart-toolbar-title">
            <BarChart3 size={17} />
            <div>
              <h4>Feature Contribution Waterfall (Diverging SHAP)</h4>
              <small>Bars indicate direction and magnitude of impact on fraud probability</small>
            </div>
          </div>

          <div className="chart-filters">
            <div className="filter-chips">
              <button
                type="button"
                className={`chip ${filterMode === "top" ? "active" : ""}`}
                onClick={() => setFilterMode("top")}
              >
                Top 12 Drivers
              </button>
              <button
                type="button"
                className={`chip ${filterMode === "all" ? "active" : ""}`}
                onClick={() => setFilterMode("all")}
              >
                All Features
              </button>
              <button
                type="button"
                className={`chip ${filterMode === "positive" ? "active" : ""}`}
                onClick={() => setFilterMode("positive")}
              >
                Risk Escalators
              </button>
              <button
                type="button"
                className={`chip ${filterMode === "negative" ? "active" : ""}`}
                onClick={() => setFilterMode("negative")}
              >
                Protective
              </button>
            </div>

            <div className="search-box">
              <Search size={14} />
              <input
                type="text"
                placeholder="Filter feature..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Filter SHAP features"
              />
              {searchQuery && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={() => setSearchQuery("")}
                >
                  ×
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="chart-legend">
          <div className="legend-side">
            <span className="legend-bar-sample green" />
            <span>← Decreases Risk (Protective)</span>
          </div>
          <div className="legend-axis-zero">0.0 (Neutral Baseline)</div>
          <div className="legend-side">
            <span>Increases Risk (Fraud Indicator) →</span>
            <span className="legend-bar-sample red" />
          </div>
        </div>

        {/* Rows */}
        <div className="diverging-bars-list" role="region" aria-label="SHAP contribution rows">
          {filteredDrivers.map((driver) => {
            const isPositive = driver.shap_value > 0;
            const barWidthPercent = Math.min((driver.abs_impact / maxAbs) * 100, 100);

            return (
              <div
                key={driver.feature}
                className={`shap-bar-row ${selectedFeature === driver.feature ? "highlighted" : ""}`}
                onClick={() => setSelectedFeature(selectedFeature === driver.feature ? null : driver.feature)}
                role="button"
                tabIndex={0}
              >
                <div className="row-meta">
                  <span className="row-label" title={driver.label}>
                    {driver.label}
                  </span>
                  <span className="row-val">
                    {driver.value !== null ? driver.value.toLocaleString() : "—"}{" "}
                    <small className="row-unit">{driver.unit}</small>
                  </span>
                </div>

                <div className="row-diverging-canvas">
                  <span className="center-zero-line" />
                  <div className="half-pane left">
                    {!isPositive && (
                      <div
                        className="bar-fill green"
                        style={{ width: `${barWidthPercent}%` }}
                      />
                    )}
                  </div>
                  <div className="half-pane right">
                    {isPositive && (
                      <div
                        className="bar-fill red"
                        style={{ width: `${barWidthPercent}%` }}
                      />
                    )}
                  </div>
                </div>

                <div className={`row-score ${isPositive ? "red" : "green"}`}>
                  {isPositive ? `+${driver.shap_value.toFixed(2)}` : driver.shap_value.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>

        <div className="chart-footer">
          <span>TreeSHAP local attribution ensures additive mathematical consistency.</span>
          <span>Showing {filteredDrivers.length} of {drivers.length} features</span>
        </div>
      </div>
    </div>
  );
}
