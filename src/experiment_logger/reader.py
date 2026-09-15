"""
ML Research Logging & Plotting Framework — Phase 1 (Lightweight Logger).

LogReader module for parsing JSONL experiment logs and metadata into pandas DataFrames.
Governed by ADR 12, 16, 20, 24, 29, 32, 40, 42, 43, 44.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
import warnings

import pandas as pd

from experiment_logger import constants

# -----------------------------------------------------------------------------
# 1. Special Float Normalization Mapping (ADR 42)
# -----------------------------------------------------------------------------
_FLOAT_STR_MAP: dict[str, float] = {
    # NaN variants
    "nan": float("nan"),
    "+nan": float("nan"),
    "-nan": float("nan"),
    # +Infinity variants
    "inf": float("inf"),
    "+inf": float("inf"),
    "infinity": float("inf"),
    "+infinity": float("inf"),
    # -Infinity variants
    "-inf": float("-inf"),
    "-infinity": float("-inf"),
}


def _deserialize_special_floats(val: Any) -> Any:
    """Recursively converts special float strings ('NaN', '+Infinity', '-Infinity')

    back to native IEEE 754 float representations per ADR 42.
    Preserves regular strings, numbers, booleans, and 1D sequences.
    """
    if isinstance(val, str):
        if val in constants.SPECIAL_STR_TO_FLOAT:
            return constants.SPECIAL_STR_TO_FLOAT[val]
        val_clean = val.strip().lower()
        if val_clean in _FLOAT_STR_MAP:
            return _FLOAT_STR_MAP[val_clean]
        return val
    if isinstance(val, list):
        return [_deserialize_special_floats(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_deserialize_special_floats(item) for item in val)
    if isinstance(val, dict):
        return {k: _deserialize_special_floats(v) for k, v in val.items()}
    return val


def _flatten_dict(
    d: Mapping[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flattens a nested dictionary into dot-separated keys, preserving parent dict keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict) and v:
            items.update(_flatten_dict(v, new_key, sep=sep))
            items[new_key] = v
        else:
            items[new_key] = v
    return items


def _normalize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures columns containing only numeric values and IEEE floats (NaN, +/-Inf)

    are typed as float64/int64 rather than object, while strictly preserving
    text/string columns and 1D array/sequence columns as object.
    """
    if df.empty:
        return df

    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                all_numeric = all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in non_null
                )
                if all_numeric:
                    df[col] = pd.to_numeric(df[col])
            else:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
    return df


class LogReader:
    """Standalone reader and DataFrame ingestion engine for experiment logs."""

    def __init__(self, logs_dir: str | Path = "logs") -> None:
        """Initialize LogReader with base directory path."""
        self.logs_dir: Path = Path(logs_dir)

    def _resolve_run_dir(self, run_dir: str | Path) -> Path:
        """Resolve run directory against cwd or logs_dir."""
        p = Path(run_dir)
        if p.is_dir():
            return p
        alt = self.logs_dir / p
        if alt.is_dir():
            return alt
        raise FileNotFoundError(f"Run directory not found: '{run_dir}'")

    def get_metadata(self, run_dir: str | Path) -> dict[str, Any]:
        """Load and return run metadata from meta.json."""
        target_dir = self._resolve_run_dir(run_dir)
        meta_file = target_dir / constants.META_FILE

        if not meta_file.exists():
            fallback_file = target_dir / "hyperparameters.json"
            if fallback_file.exists():
                try:
                    with open(fallback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data if isinstance(data, dict) else {}
                except Exception:
                    return {}
            return {}

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                warnings.warn(
                    f"Metadata in '{meta_file}' is not a JSON object.",
                    category=UserWarning,
                    stacklevel=2,
                )
                return {}
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            warnings.warn(
                f"Failed to read metadata file '{meta_file}': {exc}",
                category=UserWarning,
                stacklevel=2,
            )
            return {}

    def get_hyperparameters(self, run_dir: str | Path) -> dict[str, Any]:
        """Load and return hyperparameters dictionary for a run."""
        metadata = self.get_metadata(run_dir)
        if constants.HYPERPARAMETERS_KEY in metadata and isinstance(
            metadata[constants.HYPERPARAMETERS_KEY], dict
        ):
            return metadata[constants.HYPERPARAMETERS_KEY]
        if metadata and constants.RUN_NAME_KEY not in metadata and constants.CREATED_AT_KEY not in metadata:
            return metadata
        return {}

    def read_jsonl(
        self,
        file_path: str | Path,
        run_name: str | None = None,
    ) -> pd.DataFrame:
        """Parse a standalone JSONL metrics log file into a pandas DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: '{file_path}'")
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file, but found a directory: '{file_path}'")

        if path.stat().st_size == 0:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        inferred_run_name: str | None = None

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    # Check for restart crash boundary: corrupted prefix concatenated with valid record
                    recovered_dict: dict[str, Any] | None = None
                    first_brace = stripped.find("{")
                    if first_brace != -1:
                        next_brace = stripped.find("{", first_brace + 1)
                        while next_brace != -1:
                            candidate = stripped[next_brace:]
                            try:
                                cand_data = json.loads(candidate)
                                if isinstance(cand_data, dict):
                                    recovered_dict = cand_data
                                    break
                            except json.JSONDecodeError:
                                pass
                            next_brace = stripped.find("{", next_brace + 1)

                    warnings.warn(
                        f"Corrupted JSONL line {line_idx} in '{path.name}': {exc}. Skipping corrupted content.",
                        category=UserWarning,
                        stacklevel=2,
                    )

                    if recovered_dict is not None:
                        if constants.is_header_record(recovered_dict):
                            if inferred_run_name is None and constants.RUN_NAME_KEY in recovered_dict:
                                inferred_run_name = str(recovered_dict[constants.RUN_NAME_KEY])
                        else:
                            records.append(_deserialize_special_floats(recovered_dict))
                    continue

                if not isinstance(parsed, dict):
                    warnings.warn(
                        f"Line {line_idx} in '{path.name}' is not a JSON object (got {type(parsed).__name__}). Skipping line.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    continue

                # Header detection (line-1 or mid-file resumed run headers)
                if constants.is_header_record(parsed):
                    if inferred_run_name is None and constants.RUN_NAME_KEY in parsed:
                        inferred_run_name = str(parsed[constants.RUN_NAME_KEY])
                    continue

                deserialized = _deserialize_special_floats(parsed)
                records.append(deserialized)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        effective_run_name = (
            str(run_name)
            if run_name is not None
            else (inferred_run_name or (path.parent.name if path.parent.name and path.parent.name != "." else path.stem))
        )

        if constants.RUN_NAME_KEY not in df.columns:
            df.insert(0, constants.RUN_NAME_KEY, effective_run_name)
        else:
            if run_name is not None:
                df[constants.RUN_NAME_KEY] = run_name
            else:
                df[constants.RUN_NAME_KEY] = df[constants.RUN_NAME_KEY].fillna(effective_run_name)

        # Reorder system columns to front if present
        system_cols = [
            constants.RUN_NAME_KEY,
            constants.STEP_KEY,
            constants.EPOCH_KEY,
            constants.TIMESTAMP_KEY,
            constants.TIME_SINCE_START_KEY,
        ]
        first_cols = [c for c in system_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + other_cols]

        return _normalize_dataframe_dtypes(df)

    def read_run(self, run_dir: str | Path) -> pd.DataFrame:
        """Parse a single run directory containing log.jsonl and meta.json."""
        target_dir = self._resolve_run_dir(run_dir)
        log_file = target_dir / constants.LOG_FILE
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found in run directory: '{log_file}'")

        metadata = self.get_metadata(target_dir)
        run_name = metadata.get(constants.RUN_NAME_KEY) or target_dir.name

        df = self.read_jsonl(log_file, run_name=run_name)
        if df.empty:
            return df

        hp = self.get_hyperparameters(target_dir)
        if hp:
            flattened_hp = _flatten_dict(hp)
            for k, v in flattened_hp.items():
                sanitized_val = _deserialize_special_floats(v)
                col_name = f"{constants.HYPERPARAMETER_PREFIX}{k}"
                if isinstance(sanitized_val, (list, tuple, dict)):
                    df[col_name] = [sanitized_val] * len(df)
                else:
                    df[col_name] = sanitized_val

        return df

    def read_all(self, logs_dir: str | Path | None = None) -> pd.DataFrame:
        """Recursively discover and parse all runs under logs_dir."""
        target_dir = Path(logs_dir) if logs_dir is not None else self.logs_dir
        if not target_dir.exists() or not target_dir.is_dir():
            return pd.DataFrame()

        candidate_files = sorted(target_dir.rglob(constants.LOG_FILE))
        valid_run_dirs: list[Path] = []
        for log_file in candidate_files:
            if not log_file.is_file():
                continue
            try:
                rel_parts = log_file.relative_to(target_dir).parts
            except ValueError:
                rel_parts = log_file.parts
            if any(
                p.startswith(".") or p in ("__pycache__", "venv", ".venv", "site-packages")
                for p in rel_parts[:-1]
            ):
                continue
            valid_run_dirs.append(log_file.parent)

        seen: set[Path] = set()
        unique_run_dirs: list[Path] = []
        for d in valid_run_dirs:
            if d not in seen:
                seen.add(d)
                unique_run_dirs.append(d)

        if not unique_run_dirs:
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for rd in unique_run_dirs:
            try:
                run_df = self.read_run(rd)
                if not run_df.empty:
                    dfs.append(run_df)
            except Exception as exc:
                warnings.warn(
                    f"Failed to read run in '{rd}': {exc}. Skipping run.",
                    category=UserWarning,
                    stacklevel=2,
                )

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=0, ignore_index=True, sort=False)
