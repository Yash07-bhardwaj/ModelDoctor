"""Dataset loader and validation module for ModelDoctor."""

from typing import Union, IO
import io
import numpy as np
import pandas as pd

from ml.task import CLASSIFICATION, REGRESSION, resolve_task_type


class DatasetValidationError(ValueError):
    """Base exception for all dataset validation failures."""
    pass


class MissingTargetError(DatasetValidationError):
    """Raised when the specified target column is not present in the dataset."""
    pass


class InsufficientRowsError(DatasetValidationError):
    """Raised when the dataset has fewer rows than the required minimum (500)."""
    pass


class InvalidTargetError(DatasetValidationError):
    """Raised when target values are invalid (missing, single-class, or rare classes)."""
    pass


class UnsupportedTaskError(DatasetValidationError):
    """Raised when a non-classification (continuous/regression) target is detected."""
    pass


MIN_REQUIRED_ROWS = 500
MIN_SAMPLES_PER_CLASS = 10
MAX_DISCRETE_NUMERIC_CLASSES = 20


def is_continuous_target(series: pd.Series) -> bool:
    """
    Determine if a series represents a continuous numeric target (regression).
    
    Heuristic:
    - Non-numeric dtypes (object, category, bool, string) are classification.
    - Floating point series with non-integer values are continuous.
    - Numeric series with more than 20 unique values are continuous.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False

    # Check for floats with non-integer values
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    unique_count = non_null.nunique()

    # If it is a floating-point type and has fractional components
    if pd.api.types.is_float_dtype(series):
        # Check if values have fractional parts
        fractional = ~np.isclose(non_null, np.round(non_null))
        if np.any(fractional):
            return True

    # Numeric targets with high cardinality are continuous regression targets
    if unique_count > MAX_DISCRETE_NUMERIC_CLASSES:
        return True

    return False


def load_dataset(
    filepath_or_buffer: Union[str, IO[bytes], IO[str]],
    target_column: str,
    task_type: str = CLASSIFICATION,
) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame and strictly validate it
    for the configured task (classification by default, or regression).

    Parameters
    ----------
    filepath_or_buffer : str, Path, or file-like object
        File path or buffer containing CSV data.
    target_column : str
        The name of the target/label column.
    task_type : str, default="classification"
        "classification" (unchanged behaviour) or "regression" (see validate_regression_target).

    Returns
    -------
    pd.DataFrame
        Validated pandas DataFrame.

    Raises
    ------
    DatasetValidationError
        If dataset cannot be parsed or fails validation criteria.
    """
    task_type = resolve_task_type(task_type)
    try:
        if isinstance(filepath_or_buffer, str) and "\n" in filepath_or_buffer:
            df = pd.read_csv(io.StringIO(filepath_or_buffer))
        elif isinstance(filepath_or_buffer, bytes):
            df = pd.read_csv(io.BytesIO(filepath_or_buffer))
        else:
            df = pd.read_csv(filepath_or_buffer)
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(f"Failed to read CSV file: {str(exc)}") from exc

    # 1. Validate target column presence
    if target_column not in df.columns:
        raise MissingTargetError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    # 2. Validate row count
    row_count = len(df)
    if row_count < MIN_REQUIRED_ROWS:
        raise InsufficientRowsError(
            f"Dataset has {row_count} rows. At least {MIN_REQUIRED_ROWS} rows are required "
            "for meaningful train, calibration, and test splits."
        )

    if task_type == REGRESSION:
        validate_regression_target(df, target_column)
        return df

    target_series = df[target_column]

    # 3. Check for continuous target (Regression is strictly out of scope)
    if is_continuous_target(target_series):
        raise UnsupportedTaskError(
            "Only classification targets are supported in this version. "
            f"Target '{target_column}' appears to be a continuous numeric variable. "
            "To model it as a continuous value, set task_type='regression'."
        )

    # 4. Validate target column has no missing values
    missing_targets = target_series.isna().sum()
    if missing_targets > 0:
        raise InvalidTargetError(
            f"Target column '{target_column}' contains {missing_targets} missing values. "
            "Target column cannot contain missing values."
        )

    # 5. Validate at least 2 distinct classes
    class_counts = target_series.value_counts()
    unique_classes = len(class_counts)
    if unique_classes < 2:
        raise InvalidTargetError(
            f"Target column '{target_column}' has only {unique_classes} class. "
            "At least 2 distinct classes are required for classification."
        )

    # 6. Validate minimum samples per class for stratified splitting
    rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if len(rare_classes) > 0:
        rare_desc = ", ".join(f"'{k}': {v}" for k, v in rare_classes.items())
        raise InvalidTargetError(
            f"Target column '{target_column}' has class(es) with fewer than "
            f"{MIN_SAMPLES_PER_CLASS} samples ({rare_desc}). Each class must have at least "
            f"{MIN_SAMPLES_PER_CLASS} samples to allow stratified train/calibration/test splits."
        )

    return df


# ---------------------------------------------------------------------------
# Regression target validation
# ---------------------------------------------------------------------------

def validate_regression_target(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate a continuous numeric regression target. Invalid values are rejected,
    never coerced or turned into categorical labels.

    Checks (in order): numeric dtype (booleans and text are rejected), no missing
    values, all values finite, not constant, at least one feature column, and no
    feature column that is an exact copy of the target (target leakage).
    Integer-valued, zero, negative, tiny and huge targets are all valid.
    """
    target = df[target_column]

    if pd.api.types.is_bool_dtype(target) or not pd.api.types.is_numeric_dtype(target):
        sample = [str(v) for v in target.dropna().astype(str).unique()[:3]]
        raise InvalidTargetError(
            f"Regression target '{target_column}' must be numeric; found non-numeric values "
            f"(e.g. {sample}). Use task_type='classification' for categorical targets."
        )

    n_missing = int(target.isna().sum())
    if n_missing:
        raise InvalidTargetError(
            f"Regression target '{target_column}' contains {n_missing} missing values. "
            "Target column cannot contain missing values."
        )

    values = target.to_numpy(dtype=float)
    n_nonfinite = int((~np.isfinite(values)).sum())
    if n_nonfinite:
        raise InvalidTargetError(
            f"Regression target '{target_column}' contains {n_nonfinite} infinite or non-finite values."
        )

    if np.unique(values).size < 2:
        raise InvalidTargetError(
            f"Regression target '{target_column}' is constant; a regression target must vary."
        )

    features = [c for c in df.columns if c != target_column]
    if not features:
        raise DatasetValidationError("Dataset has no feature columns besides the target.")

    leaking = []
    for col in features:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            col_vals = series.to_numpy(dtype=float)
            if np.array_equal(col_vals, values, equal_nan=False):
                leaking.append(str(col))
    if leaking:
        raise DatasetValidationError(
            f"Feature column(s) {leaking} are identical to the target '{target_column}' "
            "(target leakage); remove them before training."
        )
