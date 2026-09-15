"""
ML Research Logging & Plotting Framework — Phase 1 (Lightweight Logger).

Core Logger module for recording hyperparameters, environment state,
and streaming JSONL experiment metrics.
Governed by ADR 15, 16, 19, 24, 26, 29, 30, 32, 33, 34, 36, 39, 40, 41, 42, 44.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, TextIO

from experiment_logger import constants


def _is_foreign_tensor(obj: Any) -> bool:
    """Detects foreign tensor types (torch, tensorflow, numpy) without requiring imports."""
    if obj is None:
        return False
    cls = obj.__class__
    mod = getattr(cls, "__module__", "") or ""
    if mod.startswith(("torch", "tensorflow", "numpy")):
        return True
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        return True
    if hasattr(obj, "detach") and callable(getattr(obj, "detach")):
        return True
    if hasattr(obj, "shape") and hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return True
    return False


def _sanitize_scalar_float(val: float) -> Any:
    """Converts special float values (NaN, +/-Inf) to strings per ADR 42."""
    if math.isnan(val):
        return constants.NAN_STR
    if math.isinf(val):
        return constants.POS_INF_STR if val > 0 else constants.NEG_INF_STR
    return val


def _sanitize_hyperparameters(hp: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize hyperparameters per ADR 44 and ADR 42.

    Ensures all keys are strings and all values are JSON-serializable primitives.
    Converts NaN, +Infinity, -Infinity to standardized strings.
    Raises TypeError on foreign objects (PyTorch/TF tensors, custom classes).
    """
    if not isinstance(hp, dict):
        raise TypeError(f"hyperparameters must be a dict, got {type(hp).__name__}")

    def _transform(val: Any) -> Any:
        if _is_foreign_tensor(val):
            raise TypeError(
                f"Hyperparameters contains tensor/foreign object of type {type(val).__name__} per ADR 44."
            )
        if val is None or isinstance(val, (bool, str, int)):
            return val
        if isinstance(val, float):
            return _sanitize_scalar_float(val)
        if isinstance(val, dict):
            res_dict: dict[str, Any] = {}
            for k, v in val.items():
                if not isinstance(k, str):
                    raise TypeError(f"Hyperparameter keys must be strings, got {type(k).__name__}: {k!r}")
                res_dict[k] = _transform(v)
            return res_dict
        if isinstance(val, (list, tuple)):
            return [_transform(item) for item in val]
        raise TypeError(f"Hyperparameters contains non-serializable object of type {type(val).__name__}: {val!r}")

    result: dict[str, Any] = {}
    for k, v in hp.items():
        if not isinstance(k, str):
            raise TypeError(f"Hyperparameter keys must be strings, got {type(k).__name__}: {k!r}")
        result[k] = _transform(v)

    # Secondary verification via standard json.dumps with allow_nan=False
    json.dumps(result, allow_nan=False)
    return result


class Logger:
    """Lightweight, zero-overhead experiment logger writing to JSONL."""

    def __init__(
        self,
        run_name: str,
        hyperparameters: dict[str, Any] | None = None,
        log_dir: str | Path = "logs",
        restart: bool = False,
        tags: list[str] | None = None,
        capture_env: bool = True,
    ) -> None:
        """Initialize experiment run directory, validate configurations, and open log stream.

        Args:
            run_name: Explicit non-empty identifier for the experiment run (ADR 33).
            hyperparameters: Optional dictionary of JSON-serializable configurations (ADR 44).
            log_dir: Base directory path for storing runs (default: "logs").
            restart: If True, resume existing run if hyperparameters match (ADR 24).
            tags: Optional list of string tags stored in meta.json (ADR 36).
            capture_env: If True, capture Python and git environment state (ADR 31).

        Raises:
            TypeError: If run_name, hyperparameters, or tags have invalid types.
            ValueError: If run_name is empty/invalid or restart hyperparameters mismatch.
        """
        # 1. Validate run_name (ADR 33)
        if not isinstance(run_name, str):
            raise TypeError(f"run_name must be a string, got {type(run_name).__name__}")
        clean_run_name = run_name.strip()
        if not clean_run_name:
            raise ValueError("run_name cannot be empty or whitespace")
        if "\0" in clean_run_name:
            raise ValueError("run_name cannot contain null bytes")
        path_check = Path(clean_run_name)
        if (
            path_check.is_absolute()
            or ".." in path_check.parts
            or "/" in clean_run_name
            or "\\" in clean_run_name
        ):
            raise ValueError(
                f"Invalid run_name '{run_name}': cannot contain path separators or traversal"
            )

        self.original_run_name: str = clean_run_name

        # 2. Validate tags (ADR 36)
        if tags is None:
            self.tags: list[str] = []
        else:
            if not isinstance(tags, list):
                raise TypeError(f"tags must be a list of strings, got {type(tags).__name__}")
            for t in tags:
                if not isinstance(t, str):
                    raise TypeError(f"All tags must be strings, got {type(t).__name__}: {t!r}")
            self.tags = list(tags)

        # 3. State variables
        self.restart: bool = bool(restart)
        self.capture_env: bool = bool(capture_env)
        self.log_dir: Path = Path(log_dir)
        self._start_time: float = time.time()
        self._created_at_iso: str = datetime.now(timezone.utc).isoformat()
        self._closed: bool = False
        self._file: TextIO | None = None

        # 4. Resolve base directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self.log_dir / self.original_run_name

        is_resumed_run = False

        # 5. Directory Collision & Restart Resolution (ADR 41, ADR 24)
        if target_dir.exists():
            if self.restart:
                meta_file_path = target_dir / constants.META_FILE
                if not meta_file_path.exists():
                    meta_file_path = target_dir / "hyperparameters.json"
                if not meta_file_path.exists():
                    raise ValueError(
                        f"Cannot restart run '{self.original_run_name}': existing run directory missing {constants.META_FILE}"
                    )
                with open(meta_file_path, "r", encoding="utf-8") as mf:
                    existing_meta = json.load(mf)
                existing_hp = existing_meta.get("hyperparameters", {})

                if hyperparameters is None:
                    # Inherit existing hyperparameters without error
                    self.hyperparameters = existing_hp
                else:
                    sanitized_hp = _sanitize_hyperparameters(hyperparameters)
                    if existing_hp != sanitized_hp:
                        raise ValueError(
                            f"Cannot restart run '{self.original_run_name}': hyperparameters mismatch.\n"
                            f"Existing: {existing_hp}\nProvided: {sanitized_hp}"
                        )
                    self.hyperparameters = sanitized_hp

                self.run_name = self.original_run_name
                self.run_dir = target_dir
                is_resumed_run = True
            else:
                # Collision: Auto-append timestamp per ADR 41
                timestamp_str = datetime.now().strftime(constants.COLLISION_TIMESTAMP_FORMAT)
                candidate_name = f"{self.original_run_name}_{timestamp_str}"
                candidate_dir = self.log_dir / candidate_name
                if candidate_dir.exists():
                    counter = 1
                    while (self.log_dir / f"{candidate_name}_{counter}").exists():
                        counter += 1
                    candidate_name = f"{candidate_name}_{counter}"
                    candidate_dir = self.log_dir / candidate_name

                self.run_name = candidate_name
                self.run_dir = candidate_dir
                self.run_dir.mkdir(parents=True, exist_ok=True)
                if hyperparameters is None:
                    self.hyperparameters = {}
                else:
                    self.hyperparameters = _sanitize_hyperparameters(hyperparameters)
        else:
            self.run_name = self.original_run_name
            self.run_dir = target_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if hyperparameters is None:
                self.hyperparameters = {}
            else:
                self.hyperparameters = _sanitize_hyperparameters(hyperparameters)

        self.log_file = self.run_dir / constants.LOG_FILE
        self.meta_file = self.run_dir / constants.META_FILE

        # 6. Environment Capture (ADR 31)
        env_metadata: dict[str, Any] = {}
        if self.capture_env and not is_resumed_run:
            env_metadata = self._capture_environment_state()

        # 7. Persist meta.json (ADR 36)
        if not is_resumed_run:
            git_section = {
                "git_commit": env_metadata.get("git_commit"),
                "git_branch": env_metadata.get("git_branch"),
                "is_dirty": env_metadata.get("is_dirty", False),
                "git_dirty": env_metadata.get("is_dirty", False),
            } if self.capture_env else {}
            meta_record = {
                "run_name": self.run_name,
                "original_run_name": self.original_run_name,
                "created_at": self._created_at_iso,
                "hyperparameters": self.hyperparameters,
                "tags": self.tags,
                "environment": env_metadata,
                "git": git_section,
            }
            with open(self.meta_file, "w", encoding="utf-8") as mf:
                json.dump(meta_record, mf, indent=2)
                mf.write("\n")
                mf.flush()

        # 8. Open log.jsonl and inject line-1 header (ADR 32)
        if is_resumed_run:
            self._file = open(self.log_file, "a", encoding="utf-8")
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                header = constants.make_header(
                    run_name=self.run_name,
                    created_at=self._created_at_iso,
                )
                self._file.write(json.dumps(header) + "\n")
                self._file.flush()
        else:
            self._file = open(self.log_file, "w", encoding="utf-8")
            header = constants.make_header(
                run_name=self.run_name,
                created_at=self._created_at_iso,
            )
            self._file.write(json.dumps(header) + "\n")
            self._file.flush()

    def _capture_environment_state(self) -> dict[str, Any]:
        """Capture Python environment and git state with graceful fallback."""
        try:
            from experiment_logger.env import capture_environment, capture_git_info, create_diff_patch

            env_info = capture_environment() if callable(capture_environment) else {}
            git_info = capture_git_info(cwd=Path.cwd()) if callable(capture_git_info) else {}
            is_dirty = bool(git_info.get("is_dirty") or git_info.get("git_dirty"))
            if is_dirty and callable(create_diff_patch):
                create_diff_patch(self.run_dir / constants.DIFF_FILE, cwd=Path.cwd())
            return {
                **env_info,
                **git_info,
                "is_dirty": is_dirty,
                "git_dirty": is_dirty,
            }
        except Exception:
            return {
                "python_version": platform.python_version(),
                "packages": {},
                "git_commit": None,
                "git_branch": None,
                "is_dirty": False,
                "git_dirty": False,
            }

    def log(self, metrics: dict[str, Any]) -> None:
        """Synchronously log a single dictionary of metrics to log.jsonl.

        Args:
            metrics: Dictionary of metrics. Must contain 'step' or 'epoch'.
                Keys must be non-empty strings.
                Values must be native Python scalars (int, float, str, bool)
                or 1D lists/tuples of scalars.

        Raises:
            RuntimeError: If Logger is closed.
            TypeError: If metrics is not a dict or contains foreign tensors / unsupported types.
            ValueError: If metrics is empty, lacks step/epoch, or contains invalid step/epoch values.
        """
        # 1. State check
        if self._closed or self._file is None or self._file.closed:
            raise RuntimeError(f"Cannot log to closed Logger for run '{self.run_name}'")

        # 2. Container check
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")
        if not metrics:
            raise ValueError("metrics dictionary cannot be empty")

        # 3. Explicit step or epoch tracking (ADR 16)
        has_step = constants.STEP_KEY in metrics
        has_epoch = constants.EPOCH_KEY in metrics
        if not (has_step or has_epoch):
            raise ValueError(
                "Explicit step or epoch tracking is required in metrics "
                "(must contain 'step' or 'epoch') per ADR 16"
            )

        if has_step:
            s = metrics[constants.STEP_KEY]
            if isinstance(s, bool) or not isinstance(s, (int, float)):
                raise TypeError(f"'step' must be an integer or float, got {type(s).__name__}")
            if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
                raise ValueError(f"'step' must be a finite number, got {s}")
            if s < 0:
                raise ValueError(f"'step' must be non-negative, got {s}")

        if has_epoch:
            e = metrics[constants.EPOCH_KEY]
            if isinstance(e, bool) or not isinstance(e, (int, float)):
                raise TypeError(f"'epoch' must be an int or float, got {type(e).__name__}")
            if isinstance(e, float) and (math.isnan(e) or math.isinf(e)):
                raise ValueError(f"'epoch' must be a finite number, got {e}")
            if e < 0:
                raise ValueError(f"'epoch' must be non-negative, got {e}")

        # 4. Process keys and values (ADR 19, ADR 29, ADR 39, ADR 42)
        record: dict[str, Any] = {}
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"Metric key must be a string, got {type(k).__name__}: {k!r}")
            clean_k = k.strip()
            if not clean_k:
                raise ValueError(f"Metric key cannot be empty or whitespace: {k!r}")

            # User cannot tamper with injected system keys (ADR 40)
            if k in constants.INJECTED_SYSTEM_KEYS:
                continue

            if _is_foreign_tensor(v):
                raise TypeError(
                    f"Value for key '{k}' is a tensor/foreign object of type {type(v).__name__}. "
                    "Per ADR 39, foreign tensors are not auto-cast. Call .item() or float() before logging."
                )

            if isinstance(v, dict):
                raise TypeError(f"Nested dictionary is not allowed for key '{k}' per ADR 19.")

            if isinstance(v, (list, tuple)):
                sanitized_seq: list[Any] = []
                for idx, item in enumerate(v):
                    if _is_foreign_tensor(item):
                        raise TypeError(
                            f"Element at index {idx} in list '{k}' is a tensor/foreign object of type {type(item).__name__} per ADR 39."
                        )
                    if isinstance(item, (list, tuple, dict, set)):
                        raise TypeError(
                            f"Multi-dimensional or nested structure at index {idx} in list '{k}' "
                            "is not allowed per ADR 19 and ADR 29."
                        )
                    if isinstance(item, bool):
                        sanitized_seq.append(item)
                    elif isinstance(item, (int, str)):
                        sanitized_seq.append(item)
                    elif isinstance(item, float):
                        sanitized_seq.append(_sanitize_scalar_float(item))
                    else:
                        raise TypeError(
                            f"Element at index {idx} in list '{k}' has unsupported type {type(item).__name__} per ADR 19."
                        )
                record[k] = sanitized_seq
            elif isinstance(v, bool):
                record[k] = v
            elif isinstance(v, (int, str)):
                record[k] = v
            elif isinstance(v, float):
                record[k] = _sanitize_scalar_float(v)
            else:
                raise TypeError(f"Value for key '{k}' has unsupported type {type(v).__name__} per ADR 19.")

        # 5. Inject runtime timestamps (ADR 40)
        current_time = time.time()
        record[constants.TIMESTAMP_KEY] = current_time
        record[constants.TIME_SINCE_START_KEY] = max(0.0, current_time - self._start_time)

        # 6. Synchronous write & immediate flush (ADR 34)
        line = json.dumps(record, allow_nan=False) + "\n"
        self._file.write(line)
        self._file.flush()

    def close(self) -> None:
        """Flush and close underlying file descriptors (idempotent)."""
        if not self._closed and self._file is not None:
            try:
                self._file.flush()
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
            self._closed = True

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
