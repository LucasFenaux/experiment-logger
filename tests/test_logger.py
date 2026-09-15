"""
Comprehensive unit test suite for Logger core API (Milestone 1).

Covers:
- ADR 15 (single-process)
- ADR 16 (explicit step/epoch tracking)
- ADR 19 & 39 (strict native types, tensor rejection)
- ADR 24 (crashed run resumption & hyperparameter equality)
- ADR 26 (single log.jsonl append)
- ADR 29 (1D array logging)
- ADR 30 (separate from stdout/stderr)
- ADR 32 (line-1 metadata header)
- ADR 33 (explicit non-empty run_name)
- ADR 34 (synchronous write & immediate flush)
- ADR 36 (tags & metadata persistence in meta.json)
- ADR 40 (runtime timestamp injection & tamper prevention)
- ADR 41 (directory collision auto-timestamping)
- ADR 42 (special float NaN / +/- Infinity serialization)
- ADR 44 (strict JSON-serializable hyperparameters)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Any

import pytest

from experiment_logger import constants
from experiment_logger.logger import Logger


# =============================================================================
# Helper Mock Objects for Foreign Tensor & Custom Class Testing
# =============================================================================
class MockTorchTensor:
    __module__ = "torch"

    def __init__(self, val: float = 1.0) -> None:
        self.val = val

    def item(self) -> float:
        return self.val


class MockNumpyArray:
    __module__ = "numpy"

    def __init__(self, shape: tuple[int, ...] = (2, 2)) -> None:
        self.shape = shape

    def tolist(self) -> list[list[int]]:
        return [[1, 2], [3, 4]]


class MockNumpyScalar:
    __module__ = "numpy"

    def __init__(self, val: float = 3.14) -> None:
        self.val = val

    def item(self) -> float:
        return self.val


class MockDuckTensor:
    def __init__(self, val: float = 42.0) -> None:
        self.val = val

    def item(self) -> float:
        return self.val


class CustomObject:
    def __init__(self, name: str) -> None:
        self.name = name


# =============================================================================
# Part 1: Lifecycle, Directory Resolution, & Metadata Tests (ADR 32, 33, 36, 41, 44)
# =============================================================================
class TestLoggerLifecycleAndInit:
    def test_init_fresh_run_success(self, tmp_path: Path):
        """T-LIF-01: Validates standard run directory creation, files, and header."""
        logger = Logger("run1", log_dir=tmp_path)
        assert logger.run_name == "run1"
        assert logger.original_run_name == "run1"
        assert logger.run_dir == tmp_path / "run1"
        assert logger.log_file == tmp_path / "run1" / "log.jsonl"
        assert logger.meta_file == tmp_path / "run1" / "meta.json"
        assert logger.run_dir.is_dir()
        assert logger.log_file.exists()
        assert logger.meta_file.exists()

        # Check line 1 header
        with open(logger.log_file, "r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())
        assert first_line["mlviz_version"] == "1.0"
        assert first_line["type"] == "header"
        assert first_line["run_name"] == "run1"
        assert "created_at" in first_line
        logger.close()

    def test_run_name_type_error(self, tmp_path: Path):
        """T-LIF-02: Enforces string type for run_name."""
        with pytest.raises(TypeError, match="run_name must be a string"):
            Logger(12345, log_dir=tmp_path)  # type: ignore[arg-type]

    def test_run_name_empty_value_error(self, tmp_path: Path):
        """T-LIF-03: Enforces non-empty string for run_name."""
        with pytest.raises(ValueError, match="run_name cannot be empty"):
            Logger("", log_dir=tmp_path)

    def test_run_name_whitespace_value_error(self, tmp_path: Path):
        """T-LIF-04: Enforces non-whitespace string."""
        with pytest.raises(ValueError, match="run_name cannot be empty"):
            Logger("   \t\n", log_dir=tmp_path)

    def test_run_name_path_traversal_rejected(self, tmp_path: Path):
        """T-LIF-05: Rejects .. path traversal in run name."""
        with pytest.raises(ValueError, match="cannot contain"):
            Logger("../escape", log_dir=tmp_path)

    def test_run_name_absolute_path_rejected(self, tmp_path: Path):
        """T-LIF-06: Rejects absolute path in run name."""
        with pytest.raises(ValueError, match="cannot contain"):
            Logger("/tmp/bad_run", log_dir=tmp_path)

    def test_run_name_null_byte_rejected(self, tmp_path: Path):
        """T-LIF-07: Rejects null byte in run name."""
        with pytest.raises(ValueError, match="null bytes"):
            Logger("bad\0run", log_dir=tmp_path)

    def test_hyperparameters_type_error(self, tmp_path: Path):
        """T-LIF-08: Enforces dict type for hyperparameters."""
        with pytest.raises(TypeError, match="hyperparameters must be a dict"):
            Logger("r", hyperparameters=[1, 2], log_dir=tmp_path)  # type: ignore[arg-type]

    def test_hyperparameters_key_type_error(self, tmp_path: Path):
        """T-LIF-09: Enforces string keys in hyperparameters."""
        with pytest.raises(TypeError, match="Hyperparameter keys must be strings"):
            Logger("r", hyperparameters={1: "a"}, log_dir=tmp_path)  # type: ignore[dict-item]

    def test_hyperparameters_custom_class_rejected(self, tmp_path: Path):
        """T-LIF-10: Enforces ADR 44 rejection of non-serializable objects in hyperparameters."""
        with pytest.raises(TypeError, match="non-serializable object"):
            Logger("r", hyperparameters={"ds": CustomObject("obj")}, log_dir=tmp_path)

    def test_hyperparameters_nan_inf_serialization(self, tmp_path: Path):
        """T-LIF-11: Verifies ADR 42 string casting in meta.json."""
        logger = Logger(
            "r",
            hyperparameters={
                "lr": float("nan"),
                "inf": float("inf"),
                "ninf": float("-inf"),
                "nested": {"nested_nan": float("nan")},
                "list_special": [1.0, float("nan"), float("inf")],
            },
            log_dir=tmp_path,
        )
        logger.close()

        with open(logger.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        hp = meta["hyperparameters"]
        assert hp["lr"] == "NaN"
        assert hp["inf"] == "+Infinity"
        assert hp["ninf"] == "-Infinity"
        assert hp["nested"]["nested_nan"] == "NaN"
        assert hp["list_special"] == [1.0, "NaN", "+Infinity"]

    def test_tags_type_error(self, tmp_path: Path):
        """T-LIF-12: Enforces list type for tags."""
        with pytest.raises(TypeError, match="tags must be a list"):
            Logger("r", tags="baseline", log_dir=tmp_path)  # type: ignore[arg-type]

    def test_tags_item_type_error(self, tmp_path: Path):
        """T-LIF-13: Enforces string elements in tags."""
        with pytest.raises(TypeError, match="All tags must be strings"):
            Logger("r", tags=["good", 123], log_dir=tmp_path)  # type: ignore[list-item]

    def test_tags_stored_in_meta(self, tmp_path: Path):
        """T-LIF-14: Verifies tags stored in meta.json."""
        logger = Logger("r", tags=["tag1", "tag2"], log_dir=tmp_path)
        logger.close()

        with open(logger.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["tags"] == ["tag1", "tag2"]

    def test_directory_collision_timestamp(self, tmp_path: Path):
        """T-LIF-15: Verifies ADR 41 auto-timestamping on collision."""
        l1 = Logger("base", hyperparameters={"v": 1}, log_dir=tmp_path)
        l1.close()

        l2 = Logger("base", hyperparameters={"v": 2}, log_dir=tmp_path, restart=False)
        l2.close()

        assert l1.run_dir.name == "base"
        assert l2.run_dir.name != "base"
        assert re.match(r"^base_\d{8}_\d{6}", l2.run_dir.name) is not None

        # Verify first run was not overwritten
        with open(l1.meta_file, "r", encoding="utf-8") as f:
            meta1 = json.load(f)
        assert meta1["hyperparameters"]["v"] == 1

        with open(l2.meta_file, "r", encoding="utf-8") as f:
            meta2 = json.load(f)
        assert meta2["hyperparameters"]["v"] == 2

    def test_directory_collision_rapid_monotonic(self, tmp_path: Path):
        """T-LIF-16: Verifies sub-second collision resolution producing distinct directories."""
        loggers = []
        for _ in range(3):
            l = Logger("rapid", log_dir=tmp_path, restart=False)
            l.close()
            loggers.append(l)

        dir_names = {l.run_dir.name for l in loggers}
        assert len(dir_names) == 3
        for d in dir_names:
            assert d.startswith("rapid")

    def test_restart_fresh_if_not_exists(self, tmp_path: Path):
        """T-LIF-17: Verifies restart=True creates fresh run if directory does not exist."""
        logger = Logger("res", restart=True, log_dir=tmp_path)
        assert logger.run_dir.is_dir()
        assert logger.log_file.exists()
        assert logger.meta_file.exists()
        logger.close()

    def test_restart_matching_hyperparameters(self, tmp_path: Path):
        """T-LIF-18: Verifies ADR 24 resumption when hyperparameters match."""
        hp = {"lr": 0.01, "batch_size": 32}
        with Logger("res_match", hyperparameters=hp, log_dir=tmp_path) as l1:
            l1.log({"step": 1, "loss": 0.5})

        with Logger("res_match", hyperparameters=hp, log_dir=tmp_path, restart=True) as l2:
            assert l2.run_dir == l1.run_dir
            l2.log({"step": 2, "loss": 0.4})

        # Check log lines: 1 header line + 2 metric lines
        with open(l1.log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3
        assert json.loads(lines[0])["type"] == "header"
        assert json.loads(lines[1])["step"] == 1
        assert json.loads(lines[2])["step"] == 2

    def test_restart_with_none_hyperparameters_inherits(self, tmp_path: Path):
        """Verifies restarting with hyperparameters=None inherits existing run hyperparameters."""
        hp = {"lr": 0.05}
        with Logger("res_inherit", hyperparameters=hp, log_dir=tmp_path) as l1:
            l1.log({"step": 1, "loss": 0.9})

        with Logger("res_inherit", hyperparameters=None, log_dir=tmp_path, restart=True) as l2:
            assert l2.hyperparameters == {"lr": 0.05}
            l2.log({"step": 2, "loss": 0.8})

    def test_restart_hyperparameters_mismatch_error(self, tmp_path: Path):
        """T-LIF-19: Verifies ADR 24 ValueError on hyperparameter mismatch."""
        with Logger("res_mismatch", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
            l1.log({"step": 1, "loss": 0.5})

        with pytest.raises(ValueError, match="hyperparameters mismatch"):
            Logger("res_mismatch", hyperparameters={"lr": 0.001}, log_dir=tmp_path, restart=True)

    def test_restart_missing_meta_error(self, tmp_path: Path):
        """T-LIF-20: Verifies error when existing directory has no meta.json."""
        target = tmp_path / "corrupt_res"
        target.mkdir(parents=True)
        # Directory exists but no meta.json
        with pytest.raises(ValueError, match="missing meta.json"):
            Logger("corrupt_res", log_dir=tmp_path, restart=True)

    def test_context_manager_lifecycle(self, tmp_path: Path):
        """T-LIF-21: Verifies __enter__ and __exit__ close file."""
        with Logger("cm", log_dir=tmp_path) as logger:
            assert not logger._closed
            logger.log({"step": 1, "loss": 0.1})
        assert logger._closed

    def test_close_idempotent(self, tmp_path: Path):
        """T-LIF-22: Verifies calling close() multiple times is safe."""
        logger = Logger("idem", log_dir=tmp_path)
        logger.close()
        logger.close()
        assert logger._closed

    def test_log_after_close_raises_runtime_error(self, tmp_path: Path):
        """T-LIF-23: Verifies logging to closed logger is blocked."""
        logger = Logger("closed_test", log_dir=tmp_path)
        logger.close()
        with pytest.raises(RuntimeError, match="Cannot log to closed Logger"):
            logger.log({"step": 1})

    def test_capture_env_disabled(self, tmp_path: Path):
        """T-LIF-24: Verifies capture_env=False leaves environment empty."""
        logger = Logger("no_env", capture_env=False, log_dir=tmp_path)
        logger.close()

        with open(logger.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["environment"] == {}


# =============================================================================
# Part 2: Step & Epoch Explicit Tracking Tests (ADR 16)
# =============================================================================
class TestStepEpochTracking:
    def test_log_with_step_success(self, tmp_path: Path):
        with Logger("t_step", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "loss": 0.5})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["step"] == 1
        assert data["loss"] == 0.5

    def test_log_with_epoch_int_success(self, tmp_path: Path):
        with Logger("t_epoch_int", log_dir=tmp_path) as logger:
            logger.log({"epoch": 2, "acc": 0.95})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["epoch"] == 2
        assert data["acc"] == 0.95

    def test_log_with_epoch_float_success(self, tmp_path: Path):
        with Logger("t_epoch_float", log_dir=tmp_path) as logger:
            logger.log({"epoch": 1.5, "val_loss": 0.3})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["epoch"] == 1.5

    def test_log_with_both_step_and_epoch(self, tmp_path: Path):
        with Logger("t_both", log_dir=tmp_path) as logger:
            logger.log({"step": 100, "epoch": 5, "lr": 1e-4})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["step"] == 100
        assert data["epoch"] == 5

    def test_log_missing_step_and_epoch_raises_value_error(self, tmp_path: Path):
        with Logger("t_missing", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="Explicit step or epoch tracking is required"):
                logger.log({"loss": 0.42})

    def test_log_step_type_error_on_string(self, tmp_path: Path):
        with Logger("t_step_str", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="'step' must be an integer or float"):
                logger.log({"step": "1", "loss": 0.5})  # type: ignore[dict-item]

    def test_log_step_type_error_on_bool(self, tmp_path: Path):
        with Logger("t_step_bool", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="'step' must be an integer or float"):
                logger.log({"step": True, "loss": 0.5})  # type: ignore[dict-item]

    def test_log_step_float_allowed(self, tmp_path: Path):
        """Float step (e.g. fractional epoch / fractional step) is allowed per E2E spec."""
        with Logger("t_step_float", log_dir=tmp_path) as logger:
            logger.log({"step": 1.5, "loss": 0.5})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["step"] == 1.5

    def test_log_step_value_error_on_negative(self, tmp_path: Path):
        with Logger("t_step_neg", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="non-negative"):
                logger.log({"step": -1, "loss": 0.5})

    def test_log_step_value_error_on_nan_inf(self, tmp_path: Path):
        with Logger("t_step_nan", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="finite number"):
                logger.log({"step": float("nan"), "loss": 0.5})
            with pytest.raises(ValueError, match="finite number"):
                logger.log({"step": float("inf"), "loss": 0.5})

    def test_log_epoch_type_error_on_string(self, tmp_path: Path):
        with Logger("t_epoch_str", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="'epoch' must be an int or float"):
                logger.log({"epoch": "1", "loss": 0.5})  # type: ignore[dict-item]

    def test_log_epoch_type_error_on_bool(self, tmp_path: Path):
        with Logger("t_epoch_bool", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="'epoch' must be an int or float"):
                logger.log({"epoch": False, "loss": 0.5})  # type: ignore[dict-item]

    def test_log_epoch_value_error_on_nan_inf(self, tmp_path: Path):
        with Logger("t_epoch_nan", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="finite number"):
                logger.log({"epoch": float("nan"), "loss": 0.5})
            with pytest.raises(ValueError, match="finite number"):
                logger.log({"epoch": float("-inf"), "loss": 0.5})

    def test_log_epoch_value_error_on_negative(self, tmp_path: Path):
        with Logger("t_epoch_neg", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="non-negative"):
                logger.log({"epoch": -0.5, "loss": 0.5})


# =============================================================================
# Part 3: Strict Native Types & Foreign Tensor Rejection Tests (ADR 19, 39)
# =============================================================================
class TestStrictTypeEnforcement:
    def test_log_native_scalars_success(self, tmp_path: Path):
        with Logger("t_scalars", log_dir=tmp_path) as logger:
            logger.log({
                "step": 1,
                "float_val": 3.1415,
                "int_val": 42,
                "str_val": "train",
                "bool_val": True,
            })

        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[1])
        assert data["float_val"] == 3.1415
        assert data["int_val"] == 42
        assert data["str_val"] == "train"
        assert data["bool_val"] is True

    def test_log_rejects_nested_dict(self, tmp_path: Path):
        with Logger("t_nested_dict", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="Nested dictionary is not allowed"):
                logger.log({"step": 1, "nested": {"sub": 1.0}})

    def test_log_rejects_set(self, tmp_path: Path):
        with Logger("t_set", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="unsupported type"):
                logger.log({"step": 1, "my_set": {1, 2, 3}})

    def test_log_rejects_none(self, tmp_path: Path):
        with Logger("t_none", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="unsupported type"):
                logger.log({"step": 1, "val": None})

    def test_log_rejects_custom_object(self, tmp_path: Path):
        with Logger("t_custom", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="unsupported type"):
                logger.log({"step": 1, "custom": CustomObject("test")})

    def test_log_rejects_complex_number(self, tmp_path: Path):
        with Logger("t_complex", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="unsupported type"):
                logger.log({"step": 1, "comp": complex(1, 2)})

    def test_log_rejects_foreign_tensor_torch(self, tmp_path: Path):
        with Logger("t_torch", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="tensor/foreign object"):
                logger.log({"step": 1, "loss": MockTorchTensor(0.5)})

    def test_log_rejects_foreign_tensor_numpy(self, tmp_path: Path):
        with Logger("t_numpy", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="tensor/foreign object"):
                logger.log({"step": 1, "arr": MockNumpyArray()})

    def test_log_rejects_numpy_scalar(self, tmp_path: Path):
        with Logger("t_numpy_scalar", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="tensor/foreign object"):
                logger.log({"step": 1, "val": MockNumpyScalar(1.23)})

    def test_log_rejects_duck_tensor(self, tmp_path: Path):
        with Logger("t_duck", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="tensor/foreign object"):
                logger.log({"step": 1, "duck": MockDuckTensor(42.0)})


# =============================================================================
# Part 4: 1D Array Logging Tests (ADR 29)
# =============================================================================
class Test1DArrayLogging:
    def test_log_1d_list_of_floats(self, tmp_path: Path):
        with Logger("t_arr_float", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "losses": [0.5, 0.4, 0.3]})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["losses"] == [0.5, 0.4, 0.3]

    def test_log_1d_list_of_ints(self, tmp_path: Path):
        with Logger("t_arr_int", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "counts": [10, 20, 30]})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["counts"] == [10, 20, 30]

    def test_log_1d_list_of_strings(self, tmp_path: Path):
        with Logger("t_arr_str", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "labels": ["cat", "dog", "bird"]})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["labels"] == ["cat", "dog", "bird"]

    def test_log_1d_list_of_bools(self, tmp_path: Path):
        with Logger("t_arr_bool", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "flags": [True, False, True]})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["flags"] == [True, False, True]

    def test_log_1d_tuple_of_scalars(self, tmp_path: Path):
        with Logger("t_arr_tuple", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "coords": (1.1, 2.2, 3.3)})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["coords"] == [1.1, 2.2, 3.3]

    def test_log_empty_list(self, tmp_path: Path):
        with Logger("t_arr_empty", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "empty": []})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["empty"] == []

    def test_log_rejects_2d_list_matrix(self, tmp_path: Path):
        with Logger("t_arr_2d", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="Multi-dimensional or nested structure"):
                logger.log({"step": 1, "matrix": [[1, 2], [3, 4]]})

    def test_log_rejects_list_with_dict(self, tmp_path: Path):
        with Logger("t_arr_dict", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="Multi-dimensional or nested structure"):
                logger.log({"step": 1, "bad_items": [{"a": 1}]})

    def test_log_rejects_list_with_tensor(self, tmp_path: Path):
        with Logger("t_arr_tensor", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="tensor/foreign object"):
                logger.log({"step": 1, "tensors": [MockTorchTensor(1.0)]})

    def test_log_rejects_list_with_invalid_type(self, tmp_path: Path):
        with Logger("t_arr_invalid", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="unsupported type"):
                logger.log({"step": 1, "items": [1, CustomObject("bad")]})


# =============================================================================
# Part 5: Special Float Serialization Tests (ADR 42)
# =============================================================================
class TestSpecialFloatSerialization:
    def test_log_scalar_nan_converts_to_string(self, tmp_path: Path):
        with Logger("t_nan", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "loss": float("nan")})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["loss"] == "NaN"

    def test_log_scalar_pos_inf_converts_to_string(self, tmp_path: Path):
        with Logger("t_pos_inf", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "grad": float("inf")})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["grad"] == "+Infinity"

    def test_log_scalar_neg_inf_converts_to_string(self, tmp_path: Path):
        with Logger("t_neg_inf", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "bound": float("-inf")})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["bound"] == "-Infinity"

    def test_log_math_nan_and_inf_module(self, tmp_path: Path):
        with Logger("t_math_floats", log_dir=tmp_path) as logger:
            logger.log({
                "step": 1,
                "n": math.nan,
                "p": math.inf,
                "m": -math.inf,
            })

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["n"] == "NaN"
        assert data["p"] == "+Infinity"
        assert data["m"] == "-Infinity"

    def test_log_nan_and_inf_in_1d_list(self, tmp_path: Path):
        with Logger("t_arr_specials", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "vals": [0.5, math.nan, math.inf, -math.inf]})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["vals"] == [0.5, "NaN", "+Infinity", "-Infinity"]

    def test_log_preserves_user_string_nan(self, tmp_path: Path):
        with Logger("t_user_str_nan", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "name": "NaN"})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["name"] == "NaN"

    def test_json_rfc8259_strict_parsing(self, tmp_path: Path):
        """Verifies json.loads succeeds and lines are strictly valid RFC-8259 JSON."""
        with Logger("t_rfc8259", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "loss": float("nan"), "inf": float("inf")})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            for line in f:
                parsed = json.loads(line)
                assert isinstance(parsed, dict)


# =============================================================================
# Part 6: Timestamp Injection Tests (ADR 40)
# =============================================================================
class TestTimestampInjection:
    def test_injected_timestamps_present(self, tmp_path: Path):
        t0 = time.time()
        with Logger("t_ts_present", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "loss": 0.5})
        t1 = time.time()

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert "_timestamp" in data
        assert "_time_since_start" in data
        assert t0 - 1.0 <= data["_timestamp"] <= t1 + 1.0
        assert data["_time_since_start"] >= 0.0

    def test_timestamp_monotonicity(self, tmp_path: Path):
        with Logger("t_ts_mono", log_dir=tmp_path) as logger:
            for i in range(3):
                logger.log({"step": i, "metric": i})
                time.sleep(0.01)

        with open(logger.log_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f.readlines()[1:]]

        assert records[0]["_timestamp"] <= records[1]["_timestamp"] <= records[2]["_timestamp"]
        assert records[0]["_time_since_start"] <= records[1]["_time_since_start"] <= records[2]["_time_since_start"]

    def test_tamper_protection_overrides_user_timestamp(self, tmp_path: Path):
        with Logger("t_tamper_ts", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "_timestamp": 12345.0})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["_timestamp"] != 12345.0
        assert data["_timestamp"] > 1000000.0

    def test_tamper_protection_overrides_user_time_since_start(self, tmp_path: Path):
        with Logger("t_tamper_tss", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "_time_since_start": -999.0})

        with open(logger.log_file, "r", encoding="utf-8") as f:
            data = json.loads(f.readlines()[1])
        assert data["_time_since_start"] != -999.0
        assert data["_time_since_start"] >= 0.0

    def test_input_dict_not_mutated(self, tmp_path: Path):
        input_metrics = {"step": 1, "loss": 0.5}
        with Logger("t_no_mutate", log_dir=tmp_path) as logger:
            logger.log(input_metrics)

        assert input_metrics == {"step": 1, "loss": 0.5}
        assert "_timestamp" not in input_metrics
        assert "_time_since_start" not in input_metrics


# =============================================================================
# Part 7: Synchronous Flush Tests (ADR 34)
# =============================================================================
class TestSynchronousFlush:
    def test_immediate_flush_to_disk(self, tmp_path: Path):
        logger = Logger("t_sync_flush", log_dir=tmp_path)
        logger.log({"step": 1, "val": 100})

        # Read directly from disk without closing logger
        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["val"] == 100
        logger.close()

    def test_multiple_synchronous_writes(self, tmp_path: Path):
        logger = Logger("t_sync_mult", log_dir=tmp_path)
        for i in range(1, 6):
            logger.log({"step": i, "val": i * 10})
            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 1 + i
        logger.close()

    def test_trailing_newline_per_record(self, tmp_path: Path):
        with Logger("t_nl", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "val": 1})
            logger.log({"step": 2, "val": 2})

        with open(logger.log_file, "rb") as f:
            raw_lines = f.readlines()
        for rl in raw_lines:
            assert rl.endswith(b"\n")


# =============================================================================
# Part 8: Input Container Validation Tests
# =============================================================================
class TestInputValidationAndLifecycle:
    def test_log_non_dict_raises_type_error(self, tmp_path: Path):
        with Logger("t_bad_container", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="metrics must be a dict"):
                logger.log([("step", 1), ("loss", 0.5)])  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="metrics must be a dict"):
                logger.log("step=1, loss=0.5")  # type: ignore[arg-type]

    def test_log_empty_dict_raises_value_error(self, tmp_path: Path):
        with Logger("t_empty_dict", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="metrics dictionary cannot be empty"):
                logger.log({})

    def test_log_non_string_key_raises_type_error(self, tmp_path: Path):
        with Logger("t_bad_key", log_dir=tmp_path) as logger:
            with pytest.raises(TypeError, match="Metric key must be a string"):
                logger.log({123: "val", "step": 1})  # type: ignore[dict-item]

    def test_log_blank_key_raises_value_error(self, tmp_path: Path):
        with Logger("t_blank_key", log_dir=tmp_path) as logger:
            with pytest.raises(ValueError, match="cannot be empty or whitespace"):
                logger.log({"": 0.5, "step": 1})
            with pytest.raises(ValueError, match="cannot be empty or whitespace"):
                logger.log({"   ": 0.5, "step": 1})
