// Regression views. Everything regression-specific lives here so the classification pages
// only branch on `task === "regression"` / `task_type === "regression"`.
//
// Meaning of the numbers shown: `estimated_abs_error` is ModelDoctor's EXPECTED absolute error of
// a prediction (target units). Larger = less reliable. It is not a probability and not a bound.
import React from "react";
import { Link } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, errorMessage } from "../api/client.js";
import { DeployControl } from "./status.jsx";
import { DecisionSummary, ExplanationPanel, RiskBadge, TrafficLight } from "./decision.jsx";
import { Button, Card, DECISION_COLORS, DecisionBadge, Empty, Notice, Stat, Table, pct } from "./ui.jsx";

/** Compact number for any target scale (1e-9 … 1e12, negative, zero). */
export const fmtNum = (v) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const x = Number(v);
  if (x !== 0 && (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3)) return x.toExponential(3);
  return String(Number(x.toPrecision(5)));
};

const MEANING = "Expected absolute error of this prediction (target units) — not a guaranteed bound.";

export function RegressionResultCard({ result, target }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Prediction · {target}</div>
          <div className="tabular mt-1 text-3xl font-semibold" data-testid="reg-prediction">{fmtNum(result.prediction)}</div>
          <div className="mt-1"><RiskBadge level={result.risk_level} /></div>
        </div>
        <DecisionSummary result={result} />
      </div>
      <div className="mt-5 rounded-lg bg-slate-50 p-4">
        <div className="text-sm font-medium text-slate-700">Estimated error</div>
        <div className="tabular mt-1 text-2xl font-semibold text-indigo-700" data-testid="reg-error">
          ≈ {fmtNum(result.estimated_abs_error)}
        </div>
        <div className="mt-1 text-xs text-slate-500">
          {MEANING}{" "}
          {result.error_estimate_source === "calibration_mean_error"
            ? "No per-record estimate is available, so the average calibration error is shown."
            : "Estimated by the reliability model from signals such as model disagreement and distance from the training data."}
        </div>
      </div>
      <div className="mt-5"><ExplanationPanel result={result} /></div>
      {result.reliability_features && Object.keys(result.reliability_features).length > 0 && (
        <details className="mt-5 text-sm">
          <summary className="cursor-pointer text-slate-600">Reliability signals</summary>
          <div className="mt-2 text-xs text-slate-600">
            {Object.entries(result.reliability_features).map(([k, v]) => (
              <div key={k} className="tabular flex justify-between"><span>{k}</span><span>{fmtNum(v)}</span></div>
            ))}
          </div>
        </details>
      )}
      <div className="mt-3 text-xs text-slate-400">Prediction ID {result.prediction_id} · add the real value later on the Monitoring page.</div>
    </Card>
  );
}

function bandRows(bands) {
  return Object.fromEntries((bands || []).map((b) => [b.decision, b]));
}

function SelectiveMaeChart({ md, baselines }) {
  const colors = ["#94a3b8", "#0ea5e9", "#f59e0b"];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="coverage" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12}
          label={{ value: "Coverage (lowest estimated error accepted first)", position: "insideBottom", offset: -10, fontSize: 12 }} />
        <YAxis tickFormatter={fmtNum} fontSize={12} width={64} />
        <Tooltip formatter={(v) => fmtNum(v)} labelFormatter={(v) => `Coverage ${pct(v)}`} />
        <Legend verticalAlign="top" height={28} />
        {baselines.map((b, i) => (
          <Line key={b.baseline} data={b.risk_coverage.curve} dataKey="selective_mae" name={b.baseline}
            stroke={colors[i % colors.length]} strokeWidth={1.5} dot={false} />
        ))}
        <Line data={md} dataKey="selective_mae" name="ModelDoctor" stroke="#4f46e5" strokeWidth={2.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function RegressionReport({ r, id }) {
  const te = r.test_evaluation || {};
  const m1 = te.model1 || {};
  const m2 = te.model2 || {};
  const ev = r.mitigation_evaluation || {};
  const md = bandRows(ev.modeldoctor_bands);
  const tol = r.error_tolerance || {};
  const fallback = r.reliability_mode !== "model2";
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">System report · regression</h1>
          <p className="mt-1 text-sm text-slate-600">
            Target <b>{r.target_column}</b> · Model 1: <b>{r.selected_model}</b> (selected by cross-validated{" "}
            {String(r.selection_metric).toUpperCase()}) · all numbers below are on the held-out test split ({ev.n_test} rows)
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <DeployControl sessionId={id} />
          <Link to={`/s/${id}/predict`} className="rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white hover:bg-indigo-500">
            Make predictions →
          </Link>
        </div>
      </div>

      {fallback && (
        <Notice tone="warn"><b>No per-record error model.</b> {r.model2?.message} Every prediction gets the average calibration error.</Notice>
      )}
      <Notice tone="info">
        A prediction counts as a <b>failure</b> when its absolute error exceeds <b>{fmtNum(tol.value)}</b> ({tol.source}).
        Estimated errors are expected values, not guaranteed bounds. This model does not provide prediction intervals.
      </Notice>

      <div>
        <h2 className="mb-2 text-lg font-semibold">Model 1 · prediction quality</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="MAE" value={fmtNum(m1.mae)} hint={`median predictor: ${fmtNum(m1.median_baseline?.mae)}`} tone="accent" />
          <Stat label="RMSE" value={fmtNum(m1.rmse)} />
          <Stat label="R²" value={m1.r2 === null ? "undefined" : fmtNum(m1.r2)} hint={m1.r2_note || ""} />
          <Stat label="MAE skill vs median" value={fmtNum(m1.mae_skill_vs_median)} hint="0 = no better than the median, 1 = perfect" />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Model 1 candidates" subtitle="Cross-validation on the train split only">
          <Table rows={r.leaderboard} rowKey={(x) => x.model} columns={[
            { key: "model", label: "Model", render: (x) => (x.model === r.selected_model ? <b>{x.model}</b> : x.model) },
            { key: "mae", label: "MAE", align: "right", render: (x) => fmtNum(x.mae_mean) },
            { key: "rmse", label: "RMSE", align: "right", render: (x) => fmtNum(x.rmse_mean) },
            { key: "r2", label: "R²", align: "right", render: (x) => fmtNum(x.r2_mean) },
          ]} />
        </Card>
        <Card title="Model 2 · error estimation" subtitle={`Status: ${r.model2?.status}${r.model2?.winner ? ` · winner ${r.model2.winner}` : ""}`}>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Rank agreement (Spearman)" value={fmtNum(m2.spearman)} hint="estimated vs actual error, test split" />
            <Stat label="MAE of the estimate" value={fmtNum(m2.mae)} hint={`constant estimate: ${fmtNum(m2.constant_reference?.mae)}`} />
          </div>
          {m2.ranking_baselines && (
            <div className="mt-3 text-xs text-slate-600">
              Spearman of single signals:{" "}
              {Object.entries(m2.ranking_baselines).map(([k, v]) => `${k} ${fmtNum(v)}`).join(" · ")}
            </div>
          )}
        </Card>
      </div>

      <div>
        <h2 className="mb-2 text-lg font-semibold">Mitigation · TRUST / REVIEW / REJECT</h2>
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="ModelDoctor decisions" subtitle={r.thresholds?.message}>
            <Table rows={ev.modeldoctor_bands || []} rowKey={(b) => b.decision} columns={[
              { key: "decision", label: "Decision", render: (b) => <DecisionBadge decision={b.decision} /> },
              { key: "count", label: "Count", align: "right" },
              { key: "coverage", label: "Share", align: "right", render: (b) => pct(b.coverage) },
              { key: "mae", label: "MAE", align: "right", render: (b) => fmtNum(b.mae) },
              { key: "failure_rate", label: "Failure rate", align: "right", render: (b) => pct(b.failure_rate) },
            ]} />
            <div className="mt-2 text-xs text-slate-500">
              No filtering: MAE {fmtNum(ev.no_filtering?.mae)} · failure rate {pct(ev.no_filtering?.failure_rate)}
            </div>
          </Card>
          <Card title="Baselines at the same coverage" subtitle={`Each baseline TRUSTs ${md.TRUST?.count ?? 0} rows and REJECTs ${md.REJECT?.count ?? 0}`}>
            <Table rows={[{ baseline: "ModelDoctor", bands: ev.modeldoctor_bands }, ...(ev.matched_coverage_baselines || [])]}
              rowKey={(c) => c.baseline} columns={[
                { key: "baseline", label: "Method", render: (c) => (c.baseline === "ModelDoctor" ? <b>ModelDoctor</b> : c.baseline) },
                { key: "mae", label: "TRUST MAE", align: "right", render: (c) => fmtNum(bandRows(c.bands).TRUST?.mae) },
                { key: "fail", label: "TRUST failures", align: "right", render: (c) => pct(bandRows(c.bands).TRUST?.failure_rate) },
              ]} />
          </Card>
        </div>
        <Card className="mt-6" title="Selective MAE vs coverage" subtitle="Lower is better: accepting the predictions each method rates most reliable first">
          <SelectiveMaeChart md={ev.risk_coverage?.modeldoctor?.curve || []} baselines={ev.matched_coverage_baselines || []} />
        </Card>
      </div>

      <Card title="Leakage audit" subtitle={r.leakage_audit?.passed ? "All checks passed" : "Some checks FAILED"}>
        <ul className="space-y-1 text-sm">
          {(r.leakage_audit?.checks || []).map((c) => (
            <li key={c.check}>
              <span className={c.passed === false ? "text-rose-600" : "text-emerald-700"}>{c.passed === false ? "✗" : "✓"}</span>{" "}
              <b>{c.check}</b> <span className="text-slate-500">— {c.evidence}</span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function DecisionTiles({ m }) {
  const by = Object.fromEntries(m.decision_distribution.map((d) => [d.decision, d]));
  return (
    <div className="grid grid-cols-3 gap-3">
      {["TRUST", "REVIEW", "REJECT"].map((d) => (
        <div key={d} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">{d}</span>
            <TrafficLight decision={d} size="sm" />
          </div>
          <div className="tabular mt-1 text-2xl font-semibold" style={{ color: DECISION_COLORS[d] }}>{(by[d]?.count ?? 0).toLocaleString()}</div>
        </div>
      ))}
    </div>
  );
}

export function RegressionOverview({ m, id }) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Overview · regression</h1>
          <p className="mt-1 text-sm text-slate-600">Served predictions and their estimated errors.</p>
        </div>
        <DeployControl sessionId={id} />
      </div>
      {m.total_predictions === 0 ? (
        <Card><Empty title="No predictions served yet">Use the <Link className="text-indigo-600 underline" to={`/s/${id}/predict`}>Predict page</Link>.</Empty></Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Predictions served" value={m.total_predictions.toLocaleString()} hint={`${m.labeled_predictions} with real values`} />
            <Stat label="Coverage (TRUST)" value={pct(m.coverage)} hint={`test split: ${pct(m.baseline?.test_coverage)}`} tone="accent" />
            <Stat label="Mean estimated error" value={fmtNum(m.mean_estimated_error)} hint="expected absolute error" />
            <Stat label="Observed MAE" value={fmtNum(m.observed_vs_estimated?.observed_mae)} hint="labeled predictions" />
          </div>
          <DecisionTiles m={m} />
        </>
      )}
    </div>
  );
}

export function RegressionMonitoring({ m, id, reload }) {
  const [pending, setPending] = React.useState({});
  const [saving, setSaving] = React.useState(false);
  const [msg, setMsg] = React.useState(null);
  const ove = m.observed_vs_estimated || {};
  const entries = Object.entries(pending).filter(([, v]) => v !== "" && v !== undefined);

  const submit = async () => {
    setSaving(true);
    setMsg(null);
    try {
      const items = entries.map(([prediction_id, v]) => ({ prediction_id, actual: Number(v) }));
      const res = await api.feedback(id, items);
      setMsg({ tone: res.rejected.length ? "warn" : "good", text: `${res.accepted} value(s) saved${res.rejected.length ? `, ${res.rejected.length} rejected` : ""}.` });
      setPending({});
      reload();
    } catch (e) {
      setMsg({ tone: "bad", text: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Monitoring · regression</h1>
          <p className="mt-1 text-sm text-slate-600">When real values arrive, compare estimated errors with actual errors.</p>
        </div>
        <Button variant="secondary" onClick={reload}>Refresh</Button>
      </div>
      {m.total_predictions === 0 ? (
        <Card><Empty title="No predictions served yet">Use the <Link className="text-indigo-600 underline" to={`/s/${id}/predict`}>Predict page</Link>.</Empty></Card>
      ) : (
        <>
          {m.alerts.length > 0
            ? m.alerts.map((a) => <Notice key={a.code} tone={a.level === "critical" ? "bad" : "warn"}><b>{a.code.replace(/_/g, " ")}:</b> {a.message}</Notice>)
            : <Notice tone="good">No degradation or drift detected{m.labeled_predictions < 20 ? " (alerts need at least 20 labeled predictions)" : ""}.</Notice>}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Predictions served" value={m.total_predictions.toLocaleString()} hint={`${m.labeled_predictions} with real values`} />
            <Stat label="Coverage (TRUST)" value={pct(m.coverage)} hint={`test split: ${pct(m.baseline?.test_coverage)}`} tone="accent" />
            <Stat label="Estimated error (labeled)" value={fmtNum(ove.mean_estimated_abs_error)} hint="mean expected absolute error" />
            <Stat label="Observed MAE" value={fmtNum(ove.observed_mae)} hint={`failure rate ${pct(ove.observed_failure_rate)} (> ${fmtNum(ove.error_tolerance)})`} />
          </div>
          <Card title="Decisions" subtitle="Observed values are shown where real outcomes are known">
            <Table rows={m.decision_distribution} rowKey={(d) => d.decision} columns={[
              { key: "decision", label: "Decision", render: (d) => <DecisionBadge decision={d.decision} /> },
              { key: "count", label: "Count", align: "right" },
              { key: "labeled", label: "Labeled", align: "right" },
              { key: "est", label: "Mean estimated error", align: "right", render: (d) => fmtNum(d.mean_estimated_error) },
              { key: "mae", label: "Observed MAE", align: "right", render: (d) => fmtNum(d.observed_mae) },
              { key: "fail", label: "Failure rate", align: "right", render: (d) => pct(d.observed_failure_rate) },
            ]} />
          </Card>
          <Card title="Recent predictions" subtitle="Enter the real value of a prediction to monitor its error"
            actions={<Button onClick={submit} disabled={!entries.length || saving}>{saving ? "Saving…" : `Save ${entries.length || ""} value(s)`}</Button>}>
            {msg && <div className="mb-3"><Notice tone={msg.tone}>{msg.text}</Notice></div>}
            <Table rows={m.recent_predictions} rowKey={(p) => p.prediction_id} columns={[
              { key: "timestamp", label: "Time", render: (p) => <span className="text-xs">{p.timestamp}</span> },
              { key: "prediction", label: "Prediction", align: "right", render: (p) => fmtNum(p.prediction) },
              { key: "est", label: "Estimated error", align: "right", render: (p) => fmtNum(p.estimated_abs_error) },
              { key: "decision", label: "Decision", render: (p) => <DecisionBadge decision={p.decision} /> },
              { key: "actual", label: "Real value", render: (p) => (p.actual !== null && p.actual !== undefined ? fmtNum(p.actual) : (
                <input type="number" step="any" aria-label={`Real value for ${p.prediction_id}`} value={pending[p.prediction_id] ?? ""}
                  onChange={(e) => setPending((s) => ({ ...s, [p.prediction_id]: e.target.value }))}
                  className="w-28 rounded-md border border-slate-300 px-2 py-1 text-xs" />
              )) },
            ]} />
          </Card>
        </>
      )}
    </div>
  );
}
