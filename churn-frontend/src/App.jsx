// src/App.jsx
import { useState } from "react";

const DEFAULT_API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [form, setForm] = useState({
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 12,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 70.35,
    TotalCharges: 845.5,
  });

  function setField(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${apiUrl}/predict_with_explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 font-sans">
      <header className="mb-6">
        <h1 className="text-3xl font-bold mb-2">📊 Telco Churn Prediction</h1>
        <details className="text-sm text-gray-600">
          <summary className="cursor-pointer">API Settings</summary>
          <input
            className="border px-2 py-1 mt-2 w-full rounded"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
          />
        </details>
      </header>

      <form onSubmit={submit} className="space-y-6">
        {/* Personal Info */}
        <section className="p-4 border rounded-lg shadow-sm bg-gray-50">
          <h2 className="font-semibold mb-3">👤 Personal Info</h2>
          <div className="grid grid-cols-2 gap-4">
            <select
              value={form.gender}
              onChange={(e) => setField("gender", e.target.value)}
              className="border p-2 rounded"
            >
              <option>Female</option>
              <option>Male</option>
            </select>
            <select
              value={form.SeniorCitizen}
              onChange={(e) =>
                setField("SeniorCitizen", Number(e.target.value))
              }
              className="border p-2 rounded"
            >
              <option value={0}>Not Senior</option>
              <option value={1}>Senior Citizen</option>
            </select>
            <select
              value={form.Partner}
              onChange={(e) => setField("Partner", e.target.value)}
              className="border p-2 rounded"
            >
              <option>Yes</option>
              <option>No</option>
            </select>
            <select
              value={form.Dependents}
              onChange={(e) => setField("Dependents", e.target.value)}
              className="border p-2 rounded"
            >
              <option>Yes</option>
              <option>No</option>
            </select>

            <div>
              <label className="text-sm text-gray-600">Tenure (months)</label>
              <input
                type="range"
                min={0}
                max={72}
                value={form.tenure}
                onChange={(e) => setField("tenure", Number(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500">{form.tenure} months</p>
            </div>
          </div>
        </section>

        {/* Services */}
        <section className="p-4 border rounded-lg shadow-sm bg-gray-50">
          <h2 className="font-semibold mb-3">📡 Services</h2>
          <div className="grid grid-cols-2 gap-4">
            {[
              { key: "PhoneService", opts: ["Yes", "No"] },
              { key: "MultipleLines", opts: ["Yes", "No", "No phone service"] },
              { key: "InternetService", opts: ["DSL", "Fiber optic", "No"] },
              { key: "OnlineSecurity", opts: ["Yes", "No", "No internet service"] },
              { key: "OnlineBackup", opts: ["Yes", "No", "No internet service"] },
              { key: "DeviceProtection", opts: ["Yes", "No", "No internet service"] },
              { key: "TechSupport", opts: ["Yes", "No", "No internet service"] },
              { key: "StreamingTV", opts: ["Yes", "No", "No internet service"] },
              { key: "StreamingMovies", opts: ["Yes", "No", "No internet service"] },
            ].map(({ key, opts }) => (
              <select
                key={key}
                value={form[key]}
                onChange={(e) => setField(key, e.target.value)}
                className="border p-2 rounded"
              >
                {opts.map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            ))}
          </div>
        </section>

        {/* Billing */}
        <section className="p-4 border rounded-lg shadow-sm bg-gray-50">
          <h2 className="font-semibold mb-3">💳 Billing</h2>
          <div className="grid grid-cols-2 gap-4">
            <select
              value={form.Contract}
              onChange={(e) => setField("Contract", e.target.value)}
              className="border p-2 rounded"
            >
              <option>Month-to-month</option>
              <option>One year</option>
              <option>Two year</option>
            </select>
            <select
              value={form.PaperlessBilling}
              onChange={(e) => setField("PaperlessBilling", e.target.value)}
              className="border p-2 rounded"
            >
              <option>Yes</option>
              <option>No</option>
            </select>
            <select
              value={form.PaymentMethod}
              onChange={(e) => setField("PaymentMethod", e.target.value)}
              className="border p-2 rounded col-span-2"
            >
              <option>Electronic check</option>
              <option>Mailed check</option>
              <option>Bank transfer (automatic)</option>
              <option>Credit card (automatic)</option>
            </select>

            <div>
              <label className="text-sm text-gray-600">Monthly Charges</label>
              <input
                type="number"
                value={form.MonthlyCharges}
                onChange={(e) =>
                  setField("MonthlyCharges", Number(e.target.value))
                }
                className="border p-2 rounded w-full"
              />
            </div>
            <div>
              <label className="text-sm text-gray-600">Total Charges</label>
              <input
                type="number"
                value={form.TotalCharges}
                onChange={(e) =>
                  setField("TotalCharges", Number(e.target.value))
                }
                className="border p-2 rounded w-full"
              />
            </div>
          </div>
        </section>

        <button
          type="submit"
          className="bg-blue-600 text-white px-6 py-2 rounded-lg shadow hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
          disabled={loading}
        >
          {loading ? (
            <span className="animate-spin border-2 border-white border-t-transparent rounded-full w-5 h-5 mr-2"></span>
          ) : null}
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {/* Results */}
      <section className="mt-6 p-4 border rounded-lg shadow-sm bg-white">
        <h2 className="font-semibold mb-2">Result</h2>
        {result ? (
          result.error ? (
            <p className="text-red-600">{result.error}</p>
          ) : (
            <div>
              <p className="mb-2">
                <strong>Churn probability:</strong> {result.churn_probability}
              </p>
              <div className="w-full bg-gray-200 rounded h-4 mb-2">
                <div
                  className={`h-4 rounded ${
                    result.churn_label === "Yes"
                      ? "bg-red-500"
                      : "bg-green-500"
                  }`}
                  style={{
                    width: `${(result.churn_probability * 100).toFixed(0)}%`,
                  }}
                ></div>
              </div>
              <p>
                <strong>Label:</strong>{" "}
                <span
                  className={`px-2 py-1 rounded ${
                    result.churn_label === "Yes"
                      ? "bg-red-100 text-red-700"
                      : "bg-green-100 text-green-700"
                  }`}
                >
                  {result.churn_label}
                </span>
              </p>

              <h3 className="font-semibold mt-4">Top Explanations</h3>
              {result.explanation ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
                  {result.explanation.map((e, i) => (
                    <div
                      key={i}
                      className="p-3 border rounded-lg shadow-sm bg-gray-50"
                    >
                      <b>{e.feature}</b>:{" "}
                      {"shap_value" in e
                        ? e.shap_value.toFixed(4)
                        : e.importance ?? JSON.stringify(e)}
                    </div>
                  ))}
                </div>
              ) : (
                <p>No explanation available</p>
              )}
            </div>
          )
        ) : (
          <p>No prediction yet</p>
        )}
      </section>
    </div>
  );
}
