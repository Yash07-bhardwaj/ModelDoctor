"""Explicit task-type handling for ModelDoctor.

The task type is a user-supplied configuration value, never inferred from file or
column names:

- "classification" (default, unchanged behaviour): categorical / low-cardinality
  target; continuous numeric targets are rejected with a hint to use regression.
- "regression": continuous numeric target. Must be requested explicitly; the
  profiler may *suggest* it, but nothing switches to it automatically, so an existing
  classification workflow can never silently become a regression run.
"""

from typing import Optional

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = (CLASSIFICATION, REGRESSION)


class InvalidTaskTypeError(ValueError):
    """Raised for an unknown task type value."""


def resolve_task_type(value: Optional[str]) -> str:
    """Normalise a configured task type; None means the default (classification)."""
    if value is None:
        return CLASSIFICATION
    norm = str(value).strip().lower()
    if norm not in TASK_TYPES:
        raise InvalidTaskTypeError(
            f"Unknown task type '{value}'. Expected one of: {', '.join(TASK_TYPES)}."
        )
    return norm
