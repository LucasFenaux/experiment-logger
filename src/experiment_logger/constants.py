"""
ML Research Logging & Plotting Framework — Phase 1 (Lightweight Logger).

Shared constants across logger, reader, and environment capture modules.
Governed by ADR 12, ADR 16, ADR 24, ADR 29, ADR 31, ADR 32, ADR 33, ADR 36, ADR 40, ADR 41, ADR 42.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping

# -----------------------------------------------------------------------------
# 1. Standard Filesystem Artifact Names
# -----------------------------------------------------------------------------
LOG_FILE_NAME: Final[str] = "log.jsonl"
LOG_FILE: Final[str] = LOG_FILE_NAME

META_FILE_NAME: Final[str] = "meta.json"
META_FILE: Final[str] = META_FILE_NAME

PATCH_FILE_NAME: Final[str] = "diff.patch"
DIFF_FILE: Final[str] = PATCH_FILE_NAME

DEFAULT_LOG_DIR: Final[str] = "logs"

# -----------------------------------------------------------------------------
# 2. Schema Versioning & Line-1 Header (ADR 32)
# -----------------------------------------------------------------------------
MLVIZ_VERSION: Final[str] = "1.0"
HEADER_TYPE: Final[str] = "header"
VERSION_HEADER_KEY: Final[str] = "mlviz_version"
RECORD_TYPE_KEY: Final[str] = "type"

DEFAULT_HEADER: Final[dict[str, str]] = {
    VERSION_HEADER_KEY: MLVIZ_VERSION,
    RECORD_TYPE_KEY: HEADER_TYPE,
}


def make_header(
    run_name: str | None = None,
    created_at: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Constructs the line-1 metadata header dictionary for log.jsonl.

    Parameters
    ----------
    run_name : str | None
        Name of the run. If provided, included in header.
    created_at : str | None
        ISO-8601 formatted timestamp string.
    extra_fields : Mapping[str, Any] | None
        Optional additional header fields.

    Returns
    -------
    dict[str, Any]
        Line-1 metadata header dictionary.
    """
    header: dict[str, Any] = {
        VERSION_HEADER_KEY: MLVIZ_VERSION,
        RECORD_TYPE_KEY: HEADER_TYPE,
    }
    if run_name is not None:
        header["run_name"] = run_name
    if created_at is not None:
        header["created_at"] = created_at
    if extra_fields:
        for k, v in extra_fields.items():
            if k not in header:
                header[k] = v
    return header


def is_header_record(record: Mapping[str, Any]) -> bool:
    """
    Determines whether a parsed JSON record is a line-1 metadata header.

    Parameters
    ----------
    record : Mapping[str, Any]
        Parsed JSON dictionary from log.jsonl.

    Returns
    -------
    bool
        True if record is a metadata header line, False otherwise.
    """
    return (
        record.get(RECORD_TYPE_KEY) == HEADER_TYPE
        or VERSION_HEADER_KEY in record
    )


# -----------------------------------------------------------------------------
# 3. Special Float String Representations (ADR 42)
# -----------------------------------------------------------------------------
NAN_STR: Final[str] = "NaN"
NAN_STRING: Final[str] = NAN_STR

POS_INF_STR: Final[str] = "+Infinity"
POS_INF_STRING: Final[str] = POS_INF_STR
INF_STR: Final[str] = POS_INF_STR

NEG_INF_STR: Final[str] = "-Infinity"
NEG_INF_STRING: Final[str] = NEG_INF_STR

BARE_INF_STR: Final[str] = "Infinity"

SPECIAL_FLOAT_STRINGS: Final[frozenset[str]] = frozenset({
    NAN_STR,
    "nan",
    "NAN",
    POS_INF_STR,
    BARE_INF_STR,
    "inf",
    "+inf",
    NEG_INF_STR,
    "-inf",
})

SPECIAL_STR_TO_FLOAT: Final[dict[str, float]] = {
    NAN_STR: float("nan"),
    "nan": float("nan"),
    "NAN": float("nan"),
    POS_INF_STR: float("inf"),
    BARE_INF_STR: float("inf"),
    "inf": float("inf"),
    "+inf": float("inf"),
    NEG_INF_STR: float("-inf"),
    "-inf": float("-inf"),
}


def serialize_special_floats(val: Any) -> Any:
    """
    Recursively converts float('nan'), float('inf'), and -float('inf') into
    RFC-compliant JSON strings per ADR 42 ('NaN', '+Infinity', '-Infinity').

    Operates on scalars and 1D sequences (lists/tuples).
    """
    if isinstance(val, float):
        if math.isnan(val):
            return NAN_STR
        if math.isinf(val):
            return POS_INF_STR if val > 0 else NEG_INF_STR
        return val
    if isinstance(val, (list, tuple)):
        return [serialize_special_floats(item) for item in val]
    if isinstance(val, dict):
        return {k: serialize_special_floats(v) for k, v in val.items()}
    return val


def deserialize_special_floats(val: Any) -> Any:
    """
    Recursively converts special float strings ('NaN', '+Infinity', '-Infinity')
    back to numeric float representations per ADR 42.

    Operates on scalars and 1D sequences (lists/tuples).
    """
    if isinstance(val, str) and val in SPECIAL_STR_TO_FLOAT:
        return SPECIAL_STR_TO_FLOAT[val]
    if isinstance(val, (list, tuple)):
        return [deserialize_special_floats(item) for item in val]
    if isinstance(val, dict):
        return {k: deserialize_special_floats(v) for k, v in val.items()}
    return val


# -----------------------------------------------------------------------------
# 4. Runtime Timestamp and Step Keys (ADR 16, ADR 40)
# -----------------------------------------------------------------------------
TIMESTAMP_KEY: Final[str] = "_timestamp"
TIME_SINCE_START_KEY: Final[str] = "_time_since_start"
INJECTED_SYSTEM_KEYS: Final[tuple[str, ...]] = (TIMESTAMP_KEY, TIME_SINCE_START_KEY)

STEP_KEY: Final[str] = "step"
EPOCH_KEY: Final[str] = "epoch"
REQUIRED_STEP_KEYS: Final[tuple[str, ...]] = (STEP_KEY, EPOCH_KEY)

# -----------------------------------------------------------------------------
# 5. Collision Timestamp Formatting (ADR 41)
# -----------------------------------------------------------------------------
COLLISION_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
COLLISION_TIMESTAMP_MICRO_FORMAT: Final[str] = "%Y%m%d_%H%M%S_%f"

# -----------------------------------------------------------------------------
# 6. View DSL & Metadata Conventions (ADR 20, ADR 36, View DSL)
# -----------------------------------------------------------------------------
HYPERPARAMETER_PREFIX: Final[str] = "hyperparameters."
RUN_NAME_KEY: Final[str] = "run_name"
TAGS_KEY: Final[str] = "tags"
HYPERPARAMETERS_KEY: Final[str] = "hyperparameters"
ENVIRONMENT_KEY: Final[str] = "environment"
CREATED_AT_KEY: Final[str] = "created_at"
