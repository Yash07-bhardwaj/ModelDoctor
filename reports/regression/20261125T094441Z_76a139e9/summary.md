# ModelDoctor regression evaluation — run `20261125T094441Z_76a139e9`

Generated 2026-11-25T09:44:41+00:00 by `backend/scripts/run_regression_experiments.py`. Every number below is on each dataset's held-out **test split** (20%), scored once after Model 1, Model 2 and the thresholds were fixed on train / calibration data. Datasets are seeded synthetic generators (no real regression dataset ships with the project).

## 1. Model 1 (prediction quality)

| Dataset | Model 1 | n test | MAE | RMSE | R² | MAE skill vs train median |
|---|---|---|---|---|---|---|
| homoscedastic | Ridge Regression | 600 | 0.7815 | 0.9993 | 0.9590 | 0.8064 |
| heteroscedastic_region | XGBoost | 600 | 0.9631 | 1.5866 | 0.8076 | 0.6717 |
| heteroscedastic_continuous | Random Forest | 600 | 0.9404 | 1.2945 | 0.8291 | 0.6336 |
| large_offset_negative | Decision Tree | 600 | 9.358e+06 | 2.041e+08 | 0.9797 | 0.9793 |

## 2. Model 2 (error estimation)

Estimated vs observed absolute error. Constant = the calibration mean error for every row.

| Dataset | Model 2 status / winner | Spearman | Pearson | MAE of estimate | MAE constant | Spearman: bootstrap spread / local train error / kNN distance |
|---|---|---|---|---|---|---|
| homoscedastic | no_signal / — | — | — | 0.4943 | 0.4943 | 0.041 / -0.023 / -0.004 |
| heteroscedastic_region | success / Linear (all features) | 0.437 | 0.614 | 0.6322 | 0.8749 | 0.383 / 0.432 / 0.093 |
| heteroscedastic_continuous | success / Linear (all features) | 0.482 | 0.444 | 0.5837 | 0.6747 | 0.372 / 0.483 / 0.034 |
| large_offset_negative | success / Gradient Boosting | 0.392 | 0.075 | 9.000e+06 | 9.073e+06 | 0.284 / 0.402 / 0.004 |

## 3. Uncertainty / calibration

This version reports an estimated absolute error per prediction, **not** a prediction interval; no interval coverage is claimed. Calibration of the error estimate (equal-count bins of the estimate, test split):

- **homoscedastic** (estimated→observed mean error per decile): 0.801→0.783, 0.801→0.669, 0.801→0.722, 0.801→0.730, 0.801→0.823, 0.801→0.759, 0.801→0.797, 0.801→0.811, 0.801→0.802, 0.801→0.919
- **heteroscedastic_region** (estimated→observed mean error per decile): 0.262→0.433, 0.388→0.479, 0.459→0.452, 0.518→0.520, 0.589→0.525, 0.670→0.529, 0.760→0.485, 1.444→1.556, 2.450→2.118, 2.987→2.534
- **heteroscedastic_continuous** (estimated→observed mean error per decile): 0.300→0.328, 0.445→0.408, 0.575→0.553, 0.748→0.646, 0.891→0.984, 1.026→1.026, 1.135→1.333, 1.256→1.416, 1.373→1.235, 1.627→1.475
- **large_offset_negative** (estimated→observed mean error per decile): 1.195e+05→1.371e+05, 5.075e+05→8.851e+05, 6.505e+05→8.317e+05, 7.820e+05→8.884e+05, 9.138e+05→9.790e+05, 1.043e+06→1.239e+06, 1.173e+06→1.051e+06, 1.316e+06→1.405e+06, 1.487e+06→1.354e+06, 1.894e+06→8.481e+07

## 4. Mitigation (TRUST / REVIEW / REJECT)

Failure = absolute error above the tolerance (default: 90th percentile of Model 1's out-of-fold train errors). Baselines TRUST/REJECT the same number of rows (matched coverage).

| Dataset | Tolerance | TRUST cov. | TRUST MAE / fail | REVIEW n / fail | REJECT n / fail | No filter MAE / fail | Matched TRUST fail: spread / local error / distance |
|---|---|---|---|---|---|---|---|
| homoscedastic | 1.622 | 0.000 | — / — | 600 / 0.107 | 0 / — | 0.782 / 0.107 | — / — / — |
| heteroscedastic_region | 2.792 | 0.820 | 0.664 / 0.030 | 0 / — | 108 / 0.333 | 0.963 / 0.085 | 0.043 / 0.022 / 0.085 |
| heteroscedastic_continuous | 2.170 | 0.752 | 0.793 / 0.067 | 149 / 0.222 | 0 / — | 0.940 / 0.105 | 0.075 / 0.060 / 0.102 |
| large_offset_negative | 2.237e+06 | 0.610 | 8.184e+05 / 0.052 | 234 / 0.209 | 0 / — | 9.358e+06 / 0.113 | 0.090 / 0.068 / 0.087 |

Calibration-split threshold messages:

- **homoscedastic**: Error estimates do not vary (no separation) and the overall calibration failure rate 11.7% exceeds the 5.0% target; TRUST is disabled. No band of >= 30 rows reaches a 30% failure rate; REJECT is disabled.
- **heteroscedastic_region**: Target failure rate 5.0% met at estimated error <= 0.609 (scaled) with 83.5% calibration coverage. REJECT when estimated error > 0.609 (scaled; band failure rate >= 30%).
- **heteroscedastic_continuous**: Target failure rate 5.0% met at estimated error <= 0.3688 (scaled) with 75.7% calibration coverage. No band of >= 30 rows reaches a 30% failure rate; REJECT is disabled.
- **large_offset_negative**: Target failure rate 5.0% met at estimated error <= 0.3147 (scaled) with 62.2% calibration coverage. No band of >= 30 rows reaches a 30% failure rate; REJECT is disabled.

## 5. Leakage audit (evidence from the fitted objects)

- **homoscedastic**: PASSED — partitions are disjoint: train 1800, calibration 600, test 600 rows; pairwise intersections empty; Model 1 cross-validation used train rows only: selection received the 1800 train rows only (out-of-fold vector length 1800); calibration / test rows are never passed to it; Model 1 feature preprocessing fitted on train only: scaler saw 1800 rows (train = 1800); Model 1 target scaling fitted on train only: target scaler saw 1800 rows (train = 1800); Reliability feature extractor fitted on train only: extractor scaler saw 1800 rows (train = 1800); Model 2 trained on calibration rows only (out-of-fold Model 1 errors): 600 Model 2 rows == 600 calibration rows; Model 1 never saw them; Thresholds tuned on calibration out-of-fold estimates only: thresholds were tuned on 600 calibration out-of-fold estimates (of 600 calibration rows; rows without an out-of-fold estimate are excluded); Error tolerance derived without calibration / test labels: source: train_oof_abs_error_q0.9; Test partition used once, for final evaluation: 600 rows scored after thresholds were fixed (test = 600)
- **heteroscedastic_region**: PASSED — partitions are disjoint: train 1800, calibration 600, test 600 rows; pairwise intersections empty; Model 1 cross-validation used train rows only: selection received the 1800 train rows only (out-of-fold vector length 1800); calibration / test rows are never passed to it; Model 1 feature preprocessing fitted on train only: scaler saw 1800 rows (train = 1800); Model 1 target scaling fitted on train only: target scaler saw 1800 rows (train = 1800); Reliability feature extractor fitted on train only: extractor scaler saw 1800 rows (train = 1800); Model 2 trained on calibration rows only (out-of-fold Model 1 errors): 600 Model 2 rows == 600 calibration rows; Model 1 never saw them; Thresholds tuned on calibration out-of-fold estimates only: thresholds were tuned on 600 calibration out-of-fold estimates (of 600 calibration rows; rows without an out-of-fold estimate are excluded); Error tolerance derived without calibration / test labels: source: train_oof_abs_error_q0.9; Test partition used once, for final evaluation: 600 rows scored after thresholds were fixed (test = 600)
- **heteroscedastic_continuous**: PASSED — partitions are disjoint: train 1800, calibration 600, test 600 rows; pairwise intersections empty; Model 1 cross-validation used train rows only: selection received the 1800 train rows only (out-of-fold vector length 1800); calibration / test rows are never passed to it; Model 1 feature preprocessing fitted on train only: scaler saw 1800 rows (train = 1800); Model 1 target scaling fitted on train only: target scaler saw 1800 rows (train = 1800); Reliability feature extractor fitted on train only: extractor scaler saw 1800 rows (train = 1800); Model 2 trained on calibration rows only (out-of-fold Model 1 errors): 600 Model 2 rows == 600 calibration rows; Model 1 never saw them; Thresholds tuned on calibration out-of-fold estimates only: thresholds were tuned on 600 calibration out-of-fold estimates (of 600 calibration rows; rows without an out-of-fold estimate are excluded); Error tolerance derived without calibration / test labels: source: train_oof_abs_error_q0.9; Test partition used once, for final evaluation: 600 rows scored after thresholds were fixed (test = 600)
- **large_offset_negative**: PASSED — partitions are disjoint: train 1800, calibration 600, test 600 rows; pairwise intersections empty; Model 1 cross-validation used train rows only: selection received the 1800 train rows only (out-of-fold vector length 1800); calibration / test rows are never passed to it; Model 1 feature preprocessing fitted on train only: scaler saw 1800 rows (train = 1800); Model 1 target scaling fitted on train only: target scaler saw 1800 rows (train = 1800); Reliability feature extractor fitted on train only: extractor scaler saw 1800 rows (train = 1800); Model 2 trained on calibration rows only (out-of-fold Model 1 errors): 600 Model 2 rows == 600 calibration rows; Model 1 never saw them; Thresholds tuned on calibration out-of-fold estimates only: thresholds were tuned on 600 calibration out-of-fold estimates (of 600 calibration rows; rows without an out-of-fold estimate are excluded); Error tolerance derived without calibration / test labels: source: train_oof_abs_error_q0.9; Test partition used once, for final evaluation: 600 rows scored after thresholds were fixed (test = 600)

## 6. Reproducibility

Library versions: {"python": "3.12.10", "numpy": "2.5.5", "pandas": "3.1.1", "scikit_learn": "1.9.1", "scipy": "1.18.1", "xgboost": "3.4.1"}

| Dataset | Generator seed | Dataset sha256 | Test index sha256 | Run id | Behaviour fingerprint |
|---|---|---|---|---|---|
| homoscedastic | 101 | `f047145f8d427052` | `d2586bfd25a7121b` | `b38f5ae3fc04` | `ab8d617ab5fb6bd3` |
| heteroscedastic_region | 102 | `addb1e41fa3e4dde` | `d2586bfd25a7121b` | `f46707554923` | `f52e6dc5f0294bee` |
| heteroscedastic_continuous | 103 | `1d6b403b2dffc39a` | `d2586bfd25a7121b` | `f856b3043e2a` | `f6efd6ae4b2f01c8` |
| large_offset_negative | 104 | `9f3f36afcec3fef7` | `d2586bfd25a7121b` | `3b2659acd45d` | `86bcb7b4314c655e` |

## 7. Dataset limitations

- Synthetic data only; results show the mechanism works when error structure exists, not real-world performance.
- Calibration band rates are optimistic (thresholds are tuned on them); test rates above are the honest ones.
- 600 test rows per dataset: differences of a few failures between methods are within noise.
