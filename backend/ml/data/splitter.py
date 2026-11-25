"""Stratified 3-way dataset splitting module for ModelDoctor."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.data.loader import DatasetValidationError


@dataclass
class DataSplits:
    """
    Container for raw data splits, encoded targets, and label metadata.
    Preserves original index on raw DataFrames.
    """
    X_train_raw: pd.DataFrame
    X_calib_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_calib: np.ndarray
    y_test: np.ndarray
    target_encoder: LabelEncoder
    class_mapping: Dict[str, int]
    inverse_class_mapping: Dict[int, str]
    classes: List[str]
    feature_names: List[str]
    target_column: str

    def get_split_summary(self) -> Dict[str, Any]:
        """Return counts and class distributions across all three splits."""
        total = len(self.X_train_raw) + len(self.X_calib_raw) + len(self.X_test_raw)
        return {
            "total_rows": total,
            "train": {
                "rows": len(self.X_train_raw),
                "percentage": round(len(self.X_train_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_train == c).sum())
                    for c in range(len(self.classes))
                },
            },
            "calibration": {
                "rows": len(self.X_calib_raw),
                "percentage": round(len(self.X_calib_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_calib == c).sum())
                    for c in range(len(self.classes))
                },
            },
            "test": {
                "rows": len(self.X_test_raw),
                "percentage": round(len(self.X_test_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_test == c).sum())
                    for c in range(len(self.classes))
                },
            },
        }


def split_data(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> DataSplits:
    """
    Perform a strict 60/20/20 stratified split into Train, Calibration,
    and Test sets.
    
    DataFrames returned are RAW (untransformed) with original indices preserved.
    The target column is completely removed from feature matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Validated input DataFrame.
    target_column : str
        Target column name.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    DataSplits
        Dataclass containing raw feature splits and encoded targets.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # 1. Separate raw features and target
    X_raw = df.drop(columns=[target_column]).copy()
    y_raw = df[target_column].copy()

    # 2. Fit label encoder on target column
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw.astype(str))
    classes = [str(cls) for cls in encoder.classes_]
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    inverse_class_mapping = {idx: cls for idx, cls in enumerate(classes)}

    # 3. First split: 60% Train, 40% Temp (Calibration + Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw,
        y_encoded,
        train_size=0.60,
        stratify=y_encoded,
        random_state=random_state,
    )

    # 4. Second split: Split Temp 50/50 -> 20% Calibration, 20% Test of total
    X_calib_raw, X_test_raw, y_calib, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return DataSplits(
        X_train_raw=X_train_raw,
        X_calib_raw=X_calib_raw,
        X_test_raw=X_test_raw,
        y_train=np.asarray(y_train),
        y_calib=np.asarray(y_calib),
        y_test=np.asarray(y_test),
        target_encoder=encoder,
        class_mapping=class_mapping,
        inverse_class_mapping=inverse_class_mapping,
        classes=classes,
        feature_names=list(X_raw.columns),
        target_column=target_column,
    )


# ---------------------------------------------------------------------------
# Regression splitting (classification `split_data` above is unchanged)
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
# Every partition must be able to support 5-fold CV with several rows per fold
# (Model 1 CV runs on train, Model 2 / threshold CV on calibration).
MIN_PARTITION_ROWS = 30
SPLIT_STRATEGIES = ("random", "group", "time")
# Train (Model 1 CV) and calibration (Model 2 CV) need at least 2 groups / time points each.
MIN_KEYS_FOR_CV = 2


class SplitValidationError(DatasetValidationError):
    """Raised when a train / calibration / test split is invalid or leaks (a dataset validation error)."""


@dataclass
class RegressionSplits:
    """
    Raw regression splits. y arrays are float; there is no label encoder or class map.
    Original DataFrame indices are preserved on the X frames.
    """
    X_train_raw: pd.DataFrame
    X_calib_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_calib: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_column: str
    strategy: str
    fractions: Tuple[float, float, float]
    random_state: int
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    boundaries: Dict[str, Any] = field(default_factory=dict)
    duplicate_rows_across_splits: int = 0
    # Group labels or (coerced) timestamps for ALL rows, indexed like the original data;
    # None for the random strategy. Used to keep cross-validation inside a partition
    # group- / time-aware. Never a model feature.
    split_keys: Optional[pd.Series] = None

    def get_split_summary(self) -> Dict[str, Any]:
        """Row counts and target summaries per split (JSON-serializable)."""
        total = len(self.X_train_raw) + len(self.X_calib_raw) + len(self.X_test_raw)

        def part(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
            return {
                "rows": int(len(X)),
                "percentage": round(len(X) / total * 100.0, 1) if total else 0.0,
                "target": {
                    "mean": round(float(np.mean(y)), 6),
                    "std": round(float(np.std(y)), 6),
                    "min": float(np.min(y)),
                    "median": float(np.median(y)),
                    "max": float(np.max(y)),
                },
            }

        return {
            "task_type": "regression",
            "strategy": self.strategy,
            "fractions": list(self.fractions),
            "random_state": self.random_state,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "boundaries": self.boundaries,
            "duplicate_rows_across_splits": self.duplicate_rows_across_splits,
            "total_rows": total,
            "train": part(self.X_train_raw, self.y_train),
            "calibration": part(self.X_calib_raw, self.y_calib),
            "test": part(self.X_test_raw, self.y_test),
        }


def validate_split_fractions(fractions: Sequence[float]) -> Tuple[float, float, float]:
    """Three positive fractions summing to 1 (train, calibration, test)."""
    try:
        fr = tuple(float(f) for f in fractions)
    except (TypeError, ValueError) as exc:
        raise SplitValidationError("Split fractions must be three numbers.") from exc
    if len(fr) != 3:
        raise SplitValidationError("Split fractions must be exactly (train, calibration, test).")
    if any(not np.isfinite(f) or f <= 0 or f >= 1 for f in fr):
        raise SplitValidationError(f"Each split fraction must be strictly between 0 and 1; got {fr}.")
    if abs(sum(fr) - 1.0) > 1e-6:
        raise SplitValidationError(f"Split fractions must sum to 1; got {fr} (sum {sum(fr):.4f}).")
    return fr  # type: ignore[return-value]


def validate_splits(
    train_index: pd.Index,
    calib_index: pd.Index,
    test_index: pd.Index,
    min_rows: int = MIN_PARTITION_ROWS,
    groups: Optional[pd.Series] = None,
    times: Optional[pd.Series] = None,
) -> None:
    """
    Raise SplitValidationError if partitions are empty / too small, contain duplicate
    index labels, overlap, share a group (when `groups` given), or are not in strict
    temporal order train < calibration < test (when `times` given).
    `groups` / `times` are Series indexed like the original data.
    """
    parts = {"train": pd.Index(train_index), "calibration": pd.Index(calib_index), "test": pd.Index(test_index)}
    for name, idx in parts.items():
        if len(idx) == 0:
            raise SplitValidationError(f"The {name} partition is empty.")
        if len(idx) < min_rows:
            raise SplitValidationError(
                f"The {name} partition has {len(idx)} rows; at least {min_rows} are required."
            )
        if idx.has_duplicates:
            raise SplitValidationError(f"The {name} partition contains duplicate index labels.")
    names = list(parts)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = parts[names[i]].intersection(parts[names[j]])
            if len(overlap):
                raise SplitValidationError(
                    f"{names[i]} and {names[j]} partitions share {len(overlap)} row(s) (index leakage)."
                )
    if groups is not None:
        sets = {n: set(groups.loc[idx].tolist()) for n, idx in parts.items()}
        for i in range(3):
            for j in range(i + 1, 3):
                shared = sets[names[i]] & sets[names[j]]
                if shared:
                    raise SplitValidationError(
                        f"{names[i]} and {names[j]} partitions share {len(shared)} group(s) (group leakage)."
                    )
    if times is not None:
        t = {n: times.loc[idx] for n, idx in parts.items()}
        if not (t["train"].max() < t["calibration"].min() and t["calibration"].max() < t["test"].min()):
            raise SplitValidationError(
                "Temporal leakage: partitions are not strictly ordered in time (train < calibration < test)."
            )


def _coerce_time(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        out = series.astype(float)
    else:
        out = pd.to_datetime(series, errors="coerce")
    bad = out.isna() & series.notna()
    if bad.any():
        raise SplitValidationError(
            f"Time column '{name}' has {int(bad.sum())} value(s) that are neither numeric nor parseable dates."
        )
    if out.isna().any():
        raise SplitValidationError(f"Time column '{name}' has {int(out.isna().sum())} missing value(s).")
    return out


def _time_boundaries(times: pd.Series, fractions: Tuple[float, float, float]) -> Tuple[Any, Any]:
    """
    Cut points on the sorted unique timestamps so that no timestamp is split across two
    partitions (rows that share a time always land in the same partition).
    """
    order = times.sort_values(kind="mergesort")
    n = len(order)
    uniq = order.drop_duplicates()
    cum = order.groupby(order, sort=True).size().cumsum()  # rows with time <= each unique time

    def cut(target_rows: float) -> Any:
        # first unique time whose cumulative count reaches the target
        pos = int(np.searchsorted(cum.to_numpy(), target_rows, side="left"))
        pos = min(pos, len(cum) - 1)
        return cum.index[pos]

    t1 = cut(fractions[0] * n)
    t2 = cut((fractions[0] + fractions[1]) * n)
    if not (t1 < t2 < uniq.iloc[-1]):
        raise SplitValidationError(
            "Not enough distinct time points to form ordered train / calibration / test partitions."
        )
    return t1, t2


def split_regression_data(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    min_partition_rows: int = MIN_PARTITION_ROWS,
) -> RegressionSplits:
    """
    Split a validated regression DataFrame into train / calibration / test (default 60/20/20).

    Strategies (chosen only from explicit configuration, never from data names):
    - random (default): shuffled split. Continuous targets are NOT stratified (stratifying
      on a continuous value is undefined); IID rows are assumed.
    - group (`group_column` set): whole groups go to one partition (dependent observations,
      e.g. several rows per customer). Proportions apply to groups, so row shares are approximate.
    - time (`time_column` set): ordered by time; train = earliest, test = latest. Rows sharing a
      timestamp are never split across partitions.

    The group / time columns are split metadata, not features: they are removed from X.
    Every split is checked with `validate_splits` (sizes, overlap, group / temporal leakage).
    """
    fr = validate_split_fractions(fractions)
    if group_column is not None and time_column is not None:
        raise SplitValidationError("Use either group_column or time_column, not both.")
    for col, label in ((group_column, "Group"), (time_column, "Time")):
        if col is not None:
            if col not in df.columns:
                raise SplitValidationError(f"{label} column '{col}' not found.")
            if col == target_column:
                raise SplitValidationError(f"{label} column cannot be the target column.")
    if df.index.has_duplicates:
        raise SplitValidationError("Input data has duplicate index labels; reset the index before splitting.")
    if len(df) < 3 * min_partition_rows:
        raise SplitValidationError(
            f"Dataset has {len(df)} rows; at least {3 * min_partition_rows} are needed for three partitions."
        )

    meta_cols = [c for c in (group_column, time_column) if c is not None]
    X_all = df.drop(columns=[target_column] + meta_cols)
    y_all = df[target_column].astype(float)
    groups = times = None
    boundaries: Dict[str, Any] = {}

    if time_column is not None:
        strategy = "time"
        times = _coerce_time(df[time_column], time_column)
        t1, t2 = _time_boundaries(times, fr)
        train_idx = times.index[times <= t1]
        calib_idx = times.index[(times > t1) & (times <= t2)]
        test_idx = times.index[times > t2]
        boundaries = {"train_end": str(t1), "calibration_end": str(t2)}
    elif group_column is not None:
        strategy = "group"
        groups = df[group_column]
        if groups.isna().any():
            raise SplitValidationError(f"Group column '{group_column}' has {int(groups.isna().sum())} missing value(s).")
        uniq = np.asarray(pd.unique(groups))
        if len(uniq) < 3:
            raise SplitValidationError(f"Group column '{group_column}' needs at least 3 distinct groups.")
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(uniq))
        n_ca = max(MIN_KEYS_FOR_CV, int(round(fr[1] * len(uniq))))
        # Train gets its share, but never so many groups that test would be left empty.
        n_tr = max(MIN_KEYS_FOR_CV, min(int(round(fr[0] * len(uniq))), len(uniq) - n_ca - 1))
        if n_tr + n_ca >= len(uniq):
            raise SplitValidationError(
                f"Group column '{group_column}' has {len(uniq)} groups; at least {2 * MIN_KEYS_FOR_CV + 1} are "
                f"needed ({MIN_KEYS_FOR_CV} train, {MIN_KEYS_FOR_CV} calibration, 1 test)."
            )
        g_tr, g_ca = set(uniq[perm[:n_tr]]), set(uniq[perm[n_tr:n_tr + n_ca]])
        train_idx = groups.index[groups.isin(g_tr)]
        calib_idx = groups.index[groups.isin(g_ca)]
        test_idx = groups.index[~groups.isin(g_tr | g_ca)]
        boundaries = {"groups": {"train": len(g_tr), "calibration": len(g_ca),
                                 "test": int(len(uniq) - len(g_tr) - len(g_ca))}}
    else:
        strategy = "random"
        idx = df.index
        train_idx, temp_idx = train_test_split(idx, train_size=fr[0], random_state=random_state, shuffle=True)
        calib_share = fr[1] / (fr[1] + fr[2])
        calib_idx, test_idx = train_test_split(temp_idx, train_size=calib_share, random_state=random_state,
                                               shuffle=True)

    validate_splits(train_idx, calib_idx, test_idx, min_rows=min_partition_rows, groups=groups, times=times)
    keys = groups if groups is not None else times
    if keys is not None:
        what = "groups" if groups is not None else "distinct time points"
        col = group_column if groups is not None else time_column
        for name, idx, need in (("train", train_idx, MIN_KEYS_FOR_CV), ("calibration", calib_idx, MIN_KEYS_FOR_CV),
                                ("test", test_idx, 1)):
            found = int(keys.loc[idx].nunique())
            if found < need:
                raise SplitValidationError(
                    f"The {name} partition has {found} {what} of '{col}'; at least {need} are needed "
                    f"(train and calibration are cross-validated). Use a column with more {what}."
                )

    # Exact duplicate records (features + target) present in more than one partition inflate
    # test scores; they can be legitimate, so they are reported rather than rejected.
    key = pd.util.hash_pandas_object(df.drop(columns=meta_cols), index=False)
    tr_keys = set(key.loc[train_idx])
    dup = int(key.loc[calib_idx].isin(tr_keys).sum() + key.loc[test_idx].isin(tr_keys | set(key.loc[calib_idx])).sum())

    return RegressionSplits(
        X_train_raw=X_all.loc[train_idx],
        X_calib_raw=X_all.loc[calib_idx],
        X_test_raw=X_all.loc[test_idx],
        y_train=y_all.loc[train_idx].to_numpy(),
        y_calib=y_all.loc[calib_idx].to_numpy(),
        y_test=y_all.loc[test_idx].to_numpy(),
        feature_names=list(X_all.columns),
        target_column=target_column,
        strategy=strategy,
        fractions=fr,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
        boundaries=boundaries,
        duplicate_rows_across_splits=dup,
        split_keys=groups if groups is not None else times,
    )
