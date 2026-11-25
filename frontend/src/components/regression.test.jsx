import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../api/client.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    api: {
      schema: vi.fn(), predict: vi.fn(), predictBatch: vi.fn(), monitoring: vi.fn(), feedback: vi.fn(),
      report: vi.fn(), upload: vi.fn(), profile: vi.fn(), train: vi.fn(), listSessions: vi.fn(),
      activateModel: vi.fn(), modelInfo: vi.fn(),
    },
  };
});

import { api } from "../api/client.js";
import { fmtNum } from "./regression.jsx";
import PredictPage from "../pages/PredictPage.jsx";
import ReportPage from "../pages/ReportPage.jsx";
import MonitoringPage from "../pages/MonitoringPage.jsx";
import UploadPage from "../pages/UploadPage.jsx";

const SCHEMA = {
  session_id: "s1",
  task_type: "regression",
  target_column: "price",
  features: [{ name: "x1", type: "numeric", missing_rate: 0, integer: false, min: -3, max: 3, median: 0 }],
};

const RESULT = {
  prediction_id: "p1", task_type: "regression", prediction: -12.3456, estimated_abs_error: 0.8123,
  error_estimate_source: "model2", decision: "REVIEW", risk_level: "MEDIUM",
  explanation: "Needs human review: a person should verify this prediction before it is used.",
  reasons: ["The estimate is an expected error, not a guaranteed bound."],
  warnings: [], reliability_mode: "model2", reliability_features: { bootstrap_std: 0.1 },
};

const band = (decision, count, mae, failure_rate) => ({ decision, count, coverage: count / 100, failures: 0, failure_rate, mae, rmse: mae });
const REPORT = {
  session_id: "s1", task: "regression", target_column: "price", selected_model: "XGBoost", selection_metric: "mae",
  leaderboard: [{ model: "XGBoost", mae_mean: 1.1, rmse_mean: 1.5, r2_mean: 0.8 }],
  reliability_mode: "model2",
  model2: { status: "success", winner: "Linear (all features)", message: "ok" },
  thresholds: { message: "Target failure rate 5.0% met." },
  error_tolerance: { value: 2.5, scaled: 0.7, source: "train_oof_abs_error_q0.9" },
  test_evaluation: {
    model1: { n: 100, mae: 1.0, rmse: 1.4, r2: 0.8, mae_skill_vs_median: 0.6, median_baseline: { mae: 2.5 } },
    model2: { spearman: 0.44, mae: 0.6, constant_reference: { mae: 0.9 }, ranking_baselines: { "Model 1 bootstrap spread": 0.3 } },
    uncertainty: { prediction_intervals: null },
  },
  mitigation_evaluation: {
    n_test: 100,
    no_filtering: { mae: 1.0, failure_rate: 0.1 },
    modeldoctor_bands: [band("TRUST", 80, 0.7, 0.03), band("REVIEW", 20, 2.4, 0.33), band("REJECT", 0, null, null)],
    matched_coverage_baselines: [{ baseline: "Model 1 bootstrap spread", bands: [band("TRUST", 80, 0.8, 0.05)], risk_coverage: { curve: [] } }],
    risk_coverage: { modeldoctor: { curve: [{ coverage: 1, selective_mae: 1.0 }] } },
  },
  leakage_audit: { passed: true, checks: [{ check: "partitions are disjoint", passed: true, evidence: "no overlap" }] },
  reproducibility: {},
};

function renderAt(path, element, route) {
  return render(<MemoryRouter initialEntries={[path]}><Routes><Route path={route} element={element} /></Routes></MemoryRouter>);
}

beforeEach(() => vi.clearAllMocks());

describe("fmtNum", () => {
  it("formats any target scale", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(0)).toBe("0");
    expect(fmtNum(-12.34567)).toBe("-12.346");
    expect(fmtNum(5e9)).toBe("5.000e+9");
    expect(fmtNum(2e-7)).toBe("2.000e-7");
  });
});

describe("regression prediction", () => {
  it("shows a numeric prediction and an estimated error, never class probabilities or trustworthiness", async () => {
    api.schema.mockResolvedValue(SCHEMA);
    api.predict.mockResolvedValue(RESULT);
    renderAt("/s/s1/predict", <PredictPage />, "/s/:id/predict");
    fireEvent.click(await screen.findByRole("button", { name: "Predict" }));
    expect((await screen.findByTestId("reg-prediction")).textContent).toContain("-12.346");
    expect(screen.getByTestId("reg-error").textContent).toContain("0.8123");
    expect(screen.getAllByText(/not a guaranteed bound/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Trustworthiness/)).toBeNull();
    expect(screen.queryByText(/Class probabilities/)).toBeNull();
    expect(api.predict).toHaveBeenCalledWith("s1", { x1: 0 });
  });

  it("handles a missing error estimate gracefully", async () => {
    api.schema.mockResolvedValue(SCHEMA);
    api.predict.mockResolvedValue({ ...RESULT, estimated_abs_error: null, error_estimate_source: "calibration_mean_error", reliability_features: {} });
    renderAt("/s/s1/predict", <PredictPage />, "/s/:id/predict");
    fireEvent.click(await screen.findByRole("button", { name: "Predict" }));
    expect((await screen.findByTestId("reg-error")).textContent).toContain("—");
    expect(screen.getByText(/average calibration error/)).toBeTruthy();
  });
});

describe("regression report", () => {
  it("renders regression sections and no classification charts", async () => {
    api.report.mockResolvedValue(REPORT);
    renderAt("/s/s1/report", <ReportPage />, "/s/:id/report");
    expect(await screen.findByText("System report · regression")).toBeTruthy();
    expect(screen.getByText(/does not provide prediction intervals/)).toBeTruthy();
    expect(screen.getByText("Leakage audit")).toBeTruthy();
    expect(screen.queryByText(/Confidence baseline/)).toBeNull();
    expect(screen.queryByText(/calibrated\)/)).toBeNull();
  });
});

describe("regression monitoring", () => {
  it("submits numeric real values", async () => {
    api.schema.mockResolvedValue(SCHEMA);
    api.monitoring.mockResolvedValue({
      task_type: "regression", total_predictions: 1, labeled_predictions: 0, coverage: 0, mean_estimated_error: 0.8,
      decision_distribution: [{ decision: "TRUST", count: 0 }, { decision: "REVIEW", count: 1 }, { decision: "REJECT", count: 0 }],
      timeline: [], observed_vs_estimated: { labeled: 0, error_tolerance: 2.5 }, alerts: [], retraining_recommended: false,
      baseline: { test_coverage: 0.8 },
      recent_predictions: [{ prediction_id: "p1", timestamp: "t", source: "single", task_type: "regression", prediction: 1.5, estimated_abs_error: 0.8, decision: "REVIEW", actual: null }],
    });
    api.feedback.mockResolvedValue({ accepted: 1, rejected: [], labeled_total: 1 });
    renderAt("/s/s1/monitor", <MonitoringPage />, "/s/:id/monitor");
    fireEvent.change(await screen.findByLabelText("Real value for p1"), { target: { value: "2.25" } });
    fireEvent.click(screen.getByRole("button", { name: /Save 1 value/ }));
    await waitFor(() => expect(api.feedback).toHaveBeenCalledWith("s1", [{ prediction_id: "p1", actual: 2.25 }]));
  });
});

describe("upload task selection", () => {
  it("profiles and trains with the explicitly chosen task", async () => {
    api.listSessions.mockResolvedValue([]);
    api.upload.mockResolvedValue({
      session_id: "s9", filename: "d.csv", rows: 600, preview: [], suggested_target: null,
      columns: [{ name: "x1", dtype: "numeric", missing_count: 0, unique_count: 600, is_target_candidate: false, is_regression_candidate: true },
        { name: "price", dtype: "numeric", missing_count: 0, unique_count: 600, is_target_candidate: false, is_regression_candidate: true }],
    });
    api.profile.mockImplementation((id, target, task) => Promise.resolve(task === "regression"
      ? { valid: true, row_count: 600, feature_count: 1, detected_task: "regression", target_summary: { min: -5, median: 0, max: 5 }, warnings: [] }
      : { valid: false, validation_error: "Only classification targets are supported in this version.", suggested_task: "regression", row_count: 600, warnings: [] }));
    api.train.mockResolvedValue({});
    const { container } = render(<MemoryRouter><UploadPage /></MemoryRouter>);
    const file = new File(["x1,price\n1,2\n"], "d.csv", { type: "text/csv" });
    fireEvent.change(container.querySelector('input[type="file"]'), { target: { files: [file] } });

    fireEvent.click(await screen.findByRole("button", { name: /train it as regression/ }));
    expect(await screen.findByTestId("target-summary")).toBeTruthy();
    expect(api.profile).toHaveBeenLastCalledWith("s9", "price", "regression");
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(api.train).toHaveBeenCalled());
    expect(api.train.mock.calls[0][1]).toMatchObject({ task_type: "regression", target_column: "price" });
    expect(api.train.mock.calls[0][1].error_tolerance).toBeUndefined();
  });
});
