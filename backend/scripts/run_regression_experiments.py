"""Run ModelDoctor REGRESSION end to end on seeded synthetic datasets and write a report.

    cd backend
    python scripts/run_regression_experiments.py            # full models
    python scripts/run_regression_experiments.py --fast     # smaller ensembles

Every run writes to its OWN directory, reports/regression/<UTC time>_<fingerprint>/:
    summary.md          human-readable report (Model 1 / Model 2 / uncertainty / mitigation / leakage)
    <dataset>.json      full machine-readable report per dataset
and appends one line to reports/regression/manifest.jsonl. An existing run directory is
never overwritten (unless --overwrite), so results cannot be silently replaced by a stale
or unrelated experiment.

The datasets are synthetic because the project ships no real regression dataset; each
generator and its seed are recorded in the report. Nothing here tunes anything: all choices
are made inside the pipeline on train / calibration data, and each test split is scored once.
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.data.loader import load_dataset  # noqa: E402
from ml.pipeline import TrainingConfig, train_modeldoctor  # noqa: E402

DEFAULT_OUT = BASE_DIR.parent / "reports" / "regression"


def _base(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    df = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.uniform(-3, 3, n),
        "x3": rng.gamma(2, 2, n),
        "region": rng.choice(["north", "south", "east", "online"], n),
    })
    df.loc[rng.rand(n) < 0.05, "x3"] = np.nan
    return df


def gen_homoscedastic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = _base(n, rng)
    df["y"] = 3 * df.x1 - 2 * df.x2 + 0.5 * df.x3.fillna(4) + rng.randn(n)
    return df


def gen_hetero_region(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = _base(n, rng)
    noise = rng.randn(n) * np.where(df.region == "online", 3.0, 0.5)
    df["y"] = 2 * df.x1 + df.x2 ** 2 + noise
    return df


def gen_hetero_continuous(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = _base(n, rng)
    df["y"] = 2 * df.x1 + 3 * np.sin(df.x2) + rng.randn(n) * (0.2 + 0.6 * np.abs(df.x2))
    return df


def gen_large_offset_negative(n: int, seed: int) -> pd.DataFrame:
    """Same structure as hetero_continuous, but targets around -5e9, and exactly 0 when x1 < -1.5 (~7%)."""
    df = gen_hetero_continuous(n, seed)
    df["y"] = df["y"] * 1e6 - 5e9
    df.loc[df["x1"] < -1.5, "y"] = 0.0
    return df


DATASETS: List[Dict[str, Any]] = [
    {"name": "homoscedastic", "generator": gen_homoscedastic, "n": 3000, "seed": 101,
     "description": "Linear signal, the same noise everywhere (Model 2 should find no signal)."},
    {"name": "heteroscedastic_region", "generator": gen_hetero_region, "n": 3000, "seed": 102,
     "description": "Non-linear signal; noise 6x larger in one categorical region."},
    {"name": "heteroscedastic_continuous", "generator": gen_hetero_continuous, "n": 3000, "seed": 103,
     "description": "Non-linear signal; noise grows with |x2|."},
    {"name": "large_offset_negative", "generator": gen_large_offset_negative, "n": 3000, "seed": 104,
     "description": "As heteroscedastic_continuous, scaled x1e6, shifted to -5e9, exact zeros when x1 < -1.5."},
]


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (abs(v) >= 1e5 or (0 < abs(v) < 1e-3)):
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def build_summary(results: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> str:
    """Markdown summary separating Model 1 / Model 2 / uncertainty / mitigation / leakage."""
    L = [f"# ModelDoctor regression evaluation — run `{meta['run_id']}`", "",
         f"Generated {meta['created_at']} by `backend/scripts/run_regression_experiments.py`"
         f"{' --fast' if meta['fast'] else ''}. Every number below is on each dataset's held-out "
         "**test split** (20%), scored once after Model 1, Model 2 and the thresholds were fixed on "
         "train / calibration data. Datasets are seeded synthetic generators (no real regression "
         "dataset ships with the project).", ""]

    L += ["## 1. Model 1 (prediction quality)", "",
          "| Dataset | Model 1 | n test | MAE | RMSE | R² | MAE skill vs train median |", "|---|---|---|---|---|---|---|"]
    for name, rep in results.items():
        m = rep["test_evaluation"]["model1"]
        L.append(f"| {name} | {rep['selected_model']} | {m['n']} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
                 f"{_fmt(m['r2'])} | {_fmt(m['mae_skill_vs_median'])} |")

    L += ["", "## 2. Model 2 (error estimation)", "",
          "Estimated vs observed absolute error. Constant = the calibration mean error for every row.", "",
          "| Dataset | Model 2 status / winner | Spearman | Pearson | MAE of estimate | MAE constant | "
          "Spearman: bootstrap spread / local train error / kNN distance |", "|---|---|---|---|---|---|---|"]
    for name, rep in results.items():
        m = rep["test_evaluation"]["model2"]
        rb = m["ranking_baselines"]
        const = m["constant_reference"]["mae"] if m["constant_reference"] else None
        L.append(f"| {name} | {rep['model2']['status']} / {rep['model2']['winner'] or '—'} | {_fmt(m['spearman'], 3)} | "
                 f"{_fmt(m['pearson'], 3)} | {_fmt(m['mae'])} | {_fmt(const)} | "
                 + " / ".join(_fmt(v, 3) for v in rb.values()) + " |")

    L += ["", "## 3. Uncertainty / calibration", "",
          "This version reports an estimated absolute error per prediction, **not** a prediction interval; "
          "no interval coverage is claimed. Calibration of the error estimate (equal-count bins of the "
          "estimate, test split):", ""]
    for name, rep in results.items():
        bins = rep["test_evaluation"]["model2"]["calibration_bins"]
        if bins:
            pairs = ", ".join(f"{_fmt(b['mean_estimated'], 3)}→{_fmt(b['mean_observed'], 3)}" for b in bins)
            L.append(f"- **{name}** (estimated→observed mean error per decile): {pairs}")

    L += ["", "## 4. Mitigation (TRUST / REVIEW / REJECT)", "",
          "Failure = absolute error above the tolerance (default: 90th percentile of Model 1's out-of-fold "
          "train errors). Baselines TRUST/REJECT the same number of rows (matched coverage).", "",
          "| Dataset | Tolerance | TRUST cov. | TRUST MAE / fail | REVIEW n / fail | REJECT n / fail | "
          "No filter MAE / fail | Matched TRUST fail: spread / local error / distance |",
          "|---|---|---|---|---|---|---|---|"]
    for name, rep in results.items():
        ev = rep["mitigation_evaluation"]
        b = {x["decision"]: x for x in ev["modeldoctor_bands"]}
        nf = ev["no_filtering"]
        base = " / ".join(_fmt(c["bands"][0]["failure_rate"], 3) for c in ev["matched_coverage_baselines"])
        L.append(f"| {name} | {_fmt(rep['error_tolerance']['value'], 3)} | {_fmt(b['TRUST']['coverage'], 3)} | "
                 f"{_fmt(b['TRUST']['mae'], 3)} / {_fmt(b['TRUST']['failure_rate'], 3)} | "
                 f"{b['REVIEW']['count']} / {_fmt(b['REVIEW']['failure_rate'], 3)} | "
                 f"{b['REJECT']['count']} / {_fmt(b['REJECT']['failure_rate'], 3)} | "
                 f"{_fmt(nf['mae'], 3)} / {_fmt(nf['failure_rate'], 3)} | {base} |")
    L += ["", "Calibration-split threshold messages:", ""]
    for name, rep in results.items():
        L.append(f"- **{name}**: {rep['thresholds']['message']}")

    L += ["", "## 5. Leakage audit (evidence from the fitted objects)", ""]
    for name, rep in results.items():
        a = rep["leakage_audit"]
        L.append(f"- **{name}**: {'PASSED' if a['passed'] else 'FAILED'} — "
                 + "; ".join(f"{c['check']}: {c['evidence']}" for c in a["checks"]))

    L += ["", "## 6. Reproducibility", "",
          f"Library versions: {json.dumps(meta['library_versions'])}", "",
          "| Dataset | Generator seed | Dataset sha256 | Test index sha256 | Run id | Behaviour fingerprint |",
          "|---|---|---|---|---|---|"]
    for name, rep in results.items():
        r = rep["reproducibility"]
        L.append(f"| {name} | {meta['datasets'][name]['seed']} | `{r['dataset']['fingerprint_sha256'][:16]}` | "
                 f"`{r['split']['index_sha256']['test'][:16]}` | `{r['run_id'][:12]}` | "
                 f"`{r['behaviour_fingerprint']['sha256'][:16]}` |")

    L += ["", "## 7. Dataset limitations", "",
          "- Synthetic data only; results show the mechanism works when error structure exists, not real-world performance.",
          "- Calibration band rates are optimistic (thresholds are tuned on them); test rates above are the honest ones.",
          "- 600 test rows per dataset: differences of a few failures between methods are within noise.", ""]
    return "\n".join(L)


def run(out_dir: Path = DEFAULT_OUT, fast: bool = False, overwrite: bool = False,
        datasets: Optional[List[Dict[str, Any]]] = None,
        on_progress: Callable[[str], None] = print) -> Path:
    datasets = datasets or DATASETS
    warnings.filterwarnings("ignore")
    frames = {d["name"]: d["generator"](d["n"], d["seed"]) for d in datasets}
    combo = hashlib.sha256()
    for name, df in frames.items():
        combo.update(name.encode())
        combo.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    combo.update(b"fast" if fast else b"full")
    created = datetime.now(timezone.utc)
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}_{combo.hexdigest()[:8]}"
    run_dir = Path(out_dir) / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory {run_dir} already exists; refusing to overwrite.")

    results: Dict[str, Dict[str, Any]] = {}
    for d in datasets:
        raw = frames[d["name"]]
        df = load_dataset(raw.to_csv(index=False), "y", task_type="regression")
        _, rep = train_modeldoctor(df, "y", TrainingConfig(task_type="regression", fast=fast))
        rep["dataset_generator"] = {k: v for k, v in d.items() if k != "generator"}
        results[d["name"]] = rep
        on_progress(f"{d['name']}: Model 1 {rep['selected_model']}, Model 2 {rep['model2']['status']}, "
                    f"leakage audit {'passed' if rep['leakage_audit']['passed'] else 'FAILED'}")

    first = next(iter(results.values()))
    meta = {"run_id": run_id, "created_at": created.isoformat(timespec="seconds"), "fast": fast,
            "library_versions": first["reproducibility"]["library_versions"],
            "datasets": {d["name"]: {k: v for k, v in d.items() if k != "generator"} for d in datasets}}
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    for name, rep in results.items():
        (run_dir / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (run_dir / "summary.md").write_text(build_summary(results, meta), encoding="utf-8")
    with open(Path(out_dir) / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**meta, "path": str(run_dir)}) + "\n")
    on_progress(f"Wrote {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args.out, fast=args.fast, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
