import { useEffect, useMemo, useState } from "react";
import { Activity, ArrowRight, Building2, CheckCircle2, RefreshCw, Search, ShieldCheck } from "lucide-react";

const API = "https://healthcare-fraud-detection-hbpk.onrender.com";

const groups = [
  { title: "Claims & reimbursement", fields: ["InpatientClaims", "OutpatientClaims", "InpatientReimbursement", "OutpatientReimbursement"] },
  { title: "Provider profile", fields: ["UniquePatients", "UniquePhysicianCount", "PhysicianConcentration", "AvgPatientAge", "MedianPatientAge", "FemalePatientRatio", "AvgChronicConditions"] },
  { title: "Clinical activity", fields: ["AvgLengthOfStay", "MedianLengthOfStay", "UniqueDiagnosisCount", "UniqueProcedureCount", "TotalDiagnosisEntries", "TotalProcedureEntries"] }
];

const labels = {
  InpatientClaims: "Inpatient claims", OutpatientClaims: "Outpatient claims", InpatientReimbursement: "Inpatient reimbursement", OutpatientReimbursement: "Outpatient reimbursement", UniquePatients: "Unique patients", UniquePhysicianCount: "Unique physicians", PhysicianConcentration: "Physician concentration", AvgPatientAge: "Average patient age", MedianPatientAge: "Median patient age", FemalePatientRatio: "Female patient ratio", AvgChronicConditions: "Average chronic conditions", AvgLengthOfStay: "Average length of stay", MedianLengthOfStay: "Median length of stay", UniqueDiagnosisCount: "Unique diagnoses", UniqueProcedureCount: "Unique procedures", TotalDiagnosisEntries: "Diagnosis entries", TotalProcedureEntries: "Procedure entries"
};

const highRiskExample = { InpatientClaims: 12, OutpatientClaims: 96, InpatientReimbursement: 210000, OutpatientReimbursement: 90000, UniquePatients: 65, UniquePhysicianCount: 7, PhysicianConcentration: 0.28, AvgPatientAge: 71.2, MedianPatientAge: 72, FemalePatientRatio: 0.57, AvgChronicConditions: 3.4, AvgLengthOfStay: 5.2, MedianLengthOfStay: 4, UniqueDiagnosisCount: 24, UniqueProcedureCount: 9, TotalDiagnosisEntries: 286, TotalProcedureEntries: 104 };

async function call(path, options) {
  const response = await fetch(`${API}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || "Unable to complete request.");
  return data;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [schema, setSchema] = useState({ fields: [], nullable: [] });
  const [providers, setProviders] = useState([]);
  const [query, setQuery] = useState("");
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");

  const findProviders = async (search = "") => {
    const response = await call(`/providers?search=${encodeURIComponent(search)}&limit=8`);
    setProviders(response.providers);
  };

  useEffect(() => {
    (async () => {
      try {
        const [healthData, features, dashboard] = await Promise.all([call("/health"), call("/features"), call("/dashboard/summary")]);
        setHealth(healthData);
        setSchema({ fields: features.required_features, nullable: features.missing_value_features });
        setSummary(dashboard);
        await findProviders();
      } catch (requestError) {
        setError(`${requestError.message} Ensure the FastAPI server is running.`);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const visibleGroups = useMemo(() => groups.map((group) => ({ ...group, fields: group.fields.filter((field) => schema.fields.includes(field)) })).filter((group) => group.fields.length), [schema.fields]);
  const completed = schema.fields.filter((field) => values[field] !== "" && values[field] !== undefined).length;

  const selectProvider = async (provider) => {
    setAction("provider"); setError(""); setSelected(provider);
    try {
      const [profile, prediction] = await Promise.all([call(`/providers/${provider}`), call(`/providers/${provider}/predict`)]);
      setValues(Object.fromEntries(schema.fields.map((field) => [field, profile[field] ?? ""])));
      setResult(prediction);
    } catch (requestError) { setSelected(""); setError(requestError.message); }
    finally { setAction(""); }
  };

  const submitSearch = async (event) => {
    event.preventDefault(); setAction("search"); setError("");
    try { await findProviders(query); } catch (requestError) { setError(requestError.message); } finally { setAction(""); }
  };

  const predict = async (event) => {
    event.preventDefault();
    const missing = schema.fields.filter((field) => (values[field] === "" || values[field] === undefined) && !schema.nullable.includes(field));
    if (missing.length) { setError(`Please complete ${missing.length} required field${missing.length === 1 ? "" : "s"}.`); return; }
    setAction("predict"); setError("");
    const features = Object.fromEntries(schema.fields.map((field) => [field, values[field] === "" || values[field] === undefined ? null : Number(values[field])]));
    try { setResult(await call("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }) })); }
    catch (requestError) { setError(requestError.message); } finally { setAction(""); }
  };

  const risk = result?.risk_level?.toLowerCase() || "";
  const probability = result ? Math.round(result.fraud_probability * 100) : 0;

  return <div className="app">
    <header className="nav"><div className="nav-inner"><div className="brand"><span><ShieldCheck size={20} /></span><strong>Verity Health</strong><i /><small>Provider risk review</small></div><div className="status"><b className={error ? "offline" : ""} />{health ? "API online" : "Connecting"}</div></div></header>
    <main className="container">
      <section className="hero"><div><p className="eyebrow">Clinical integrity operations</p><h1>Provider risk review</h1><p>Assess provider activity, prioritise review, and support a consistent clinical-integrity workflow.</p></div><div className="model-meta"><span>MODEL</span><strong>{health?.model || "Loading"}</strong><span>REVIEW THRESHOLD</span><strong>{health ? `${Math.round(health.classification_threshold * 100)}%` : "—"}</strong></div></section>
      <section className="portfolio"><div className="portfolio-head"><span><Building2 size={21} /></span><div><strong>Provider portfolio</strong><small>Live model summary</small></div></div><Metric label="Providers" value={summary?.total_providers?.toLocaleString() || "—"} /><Metric label="High risk" value={summary?.high_risk_providers?.toLocaleString() || "—"} type="high" /><Metric label="Medium risk" value={summary?.medium_risk_providers?.toLocaleString() || "—"} /><Metric label="Average probability" value={summary ? `${Math.round(summary.average_fraud_probability * 100)}%` : "—"} /></section>
      <section className="directory"><div className="directory-copy"><p className="eyebrow">Provider directory</p><h2>Load a provider profile</h2><p>Select a saved provider to populate the form and return its current assessment.</p></div><form onSubmit={submitSearch} className="search"><label><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search provider ID, e.g. PRV51003" /></label><button className="button quiet" disabled={action === "search"}>{action === "search" ? "Searching…" : "Search"}</button></form><div className="provider-buttons">{providers.map((provider) => <button type="button" key={provider} className={selected === provider ? "selected" : ""} disabled={action === "provider"} onClick={() => selectProvider(provider)}>{provider}</button>)}{!loading && !providers.length && <span>No providers found.</span>}</div></section>
      <section className="workbench">
        <form className="form-card" onSubmit={predict}><header><div><p className="eyebrow">Assessment inputs</p><h2>Provider activity profile</h2><p>Enter direct provider measures. The API calculates necessary totals, ratios, and transformations.</p>{selected && <small>Loaded: {selected}</small>}</div><div className="completion"><strong>{completed}/{schema.fields.length || 17}</strong><span>fields completed</span></div></header>{loading ? <div className="form-loading">Loading model input fields…</div> : visibleGroups.map((group) => <fieldset key={group.title}><legend>{group.title}</legend><div className="inputs">{group.fields.map((field) => <label key={field}><span>{labels[field]}{schema.nullable.includes(field) && <em>Optional</em>}</span><input type="number" step="any" placeholder={schema.nullable.includes(field) ? "May be blank" : "Enter value"} value={values[field] ?? ""} onChange={(event) => { setValues({ ...values, [field]: event.target.value }); setSelected(""); setError(""); }} /></label>)}</div></fieldset>)}{error && <div className="error">{error}</div>}<footer><button type="button" className="clear" onClick={() => { setValues({}); setResult(null); setSelected(""); setError(""); }}><RefreshCw size={15} />Clear form</button><div><button type="button" className="button quiet" onClick={() => { setValues(Object.fromEntries(schema.fields.map((field) => [field, highRiskExample[field] ?? ""]))); setSelected("High-risk example"); setResult(null); }}>Use example</button><button className="button primary" disabled={action === "predict"}>{action === "predict" ? "Assessing…" : <>Assess risk <ArrowRight size={16} /></>}</button></div></footer></form>
        <aside className="outcome"><p className="eyebrow">Assessment outcome</p>{result ? <><span className={`risk ${risk}`}>{result.risk_level} risk</span><div className="score"><strong>{probability}<small>%</small></strong><span>potential fraud probability</span></div><div className="meter"><i style={{ width: `${probability}%` }} /></div><div className="meter-label"><span>0%</span><span>Threshold {Math.round(result.classification_threshold * 100)}%</span><span>100%</span></div><div className="decision"><span><Activity size={18} /></span><div><strong>{result.predicted_potential_fraud ? "Review recommended" : "No immediate escalation"}</strong><p>{result.predicted_potential_fraud ? "The provider profile meets the configured review threshold." : "The provider profile remains below the review threshold."}</p></div></div></> : <div className="empty"><span><ShieldCheck size={25} /></span><h3>Ready for assessment</h3><p>Load a provider or complete the profile to generate a risk result.</p></div>}<div className="disclaimer"><CheckCircle2 size={16} /><span>Model output supports review prioritisation; it does not confirm fraud.</span></div></aside>
      </section>
    </main><footer className="footer"><span>Verity Health · Clinical Integrity Operations</span><span>For authorised review use only</span></footer>
  </div>;
}

function Metric({ label, value, type = "" }) { return <div className="metric"><span>{label}</span><strong className={type}>{value}</strong></div>; }
