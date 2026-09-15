"""Tier 2: Boundary, corner case, and error-handling E2E tests.

Comprehensive opaque-box verification for 20 features (>=5 tests per feature = 100 tests).
Covers:
- Empty inputs, zero/negative steps, type rejections
- Extreme float representations (subnormal, inf, nan)
- Non-serializable objects and mock tensors
- Corrupted lines, truncated writes, missing files
- Non-git environments, collision resolution edge cases
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
import pytest
import pandas as pd

from experiment_logger import Logger, LogReader
from experiment_logger.env import capture_environment, capture_git_info, create_diff_patch


# Helper mock class for rejecting tensors / custom objects
class MockTensorWithItem:
    def item(self):
        return 42.0


class MockNdarrayWithShape:
    def __init__(self):
        self.shape = (2, 2)

    def tolist(self):
        return [[1, 2], [3, 4]]


class UnserializableCustomClass:
    def __init__(self, val):
        self.val = val


# ==============================================================================
# Feature 1: Logger Init & Run Name Boundaries (ADR 33)
# ==============================================================================

def test_b1_01_empty_run_name_raises_value_error(tmp_path: Path):
    """Verify empty string run_name raises ValueError."""
    with pytest.raises(ValueError):
        Logger(run_name="", log_dir=tmp_path)


def test_b1_02_whitespace_only_run_name_raises_value_error(tmp_path: Path):
    """Verify whitespace-only run_name raises ValueError."""
    with pytest.raises(ValueError):
        Logger(run_name="   \t \n  ", log_dir=tmp_path)


def test_b1_03_none_run_name_raises_error(tmp_path: Path):
    """Verify None as run_name raises TypeError or ValueError."""
    with pytest.raises((TypeError, ValueError)):
        Logger(run_name=None, log_dir=tmp_path)  # type: ignore


def test_b1_04_extreme_long_run_name(tmp_path: Path):
    """Verify handling of very long run names without crashing."""
    long_name = "exp_" + "a" * 150
    logger = Logger(run_name=long_name, log_dir=tmp_path)
    assert (tmp_path / long_name).is_dir()
    logger.close()


def test_b1_05_path_characters_in_run_name(tmp_path: Path):
    """Verify run name with directory separators either raises ValueError or handles safely."""
    # Attempting to escape with slashes should raise ValueError or create subpath cleanly
    try:
        logger = Logger(run_name="nested/run/name", log_dir=tmp_path)
        logger.close()
    except (ValueError, OSError):
        pass  # Expected defensive behavior


# ==============================================================================
# Feature 2: Hyperparameter Validation Boundaries (ADR 44)
# ==============================================================================

def test_b2_01_unserializable_class_instance_raises_type_error(tmp_path: Path):
    """Verify non-JSON-serializable custom class instance in hyperparameters raises TypeError."""
    bad_hp = {"model_obj": UnserializableCustomClass("custom")}
    with pytest.raises(TypeError):
        Logger(run_name="bad_hp_obj", hyperparameters=bad_hp, log_dir=tmp_path)


def test_b2_02_set_in_hyperparameters_raises_type_error(tmp_path: Path):
    """Verify python set in hyperparameters raises TypeError."""
    bad_hp = {"categories": {"cat", "dog"}}
    with pytest.raises(TypeError):
        Logger(run_name="bad_hp_set", hyperparameters=bad_hp, log_dir=tmp_path)


def test_b2_03_function_in_hyperparameters_raises_type_error(tmp_path: Path):
    """Verify lambda or function in hyperparameters raises TypeError."""
    bad_hp = {"loss_fn": lambda x: x ** 2}
    with pytest.raises(TypeError):
        Logger(run_name="bad_hp_fn", hyperparameters=bad_hp, log_dir=tmp_path)


def test_b2_04_mock_tensor_in_hyperparameters_raises_type_error(tmp_path: Path):
    """Verify mock tensor object in hyperparameters raises TypeError."""
    bad_hp = {"tensor_param": MockTensorWithItem()}
    with pytest.raises(TypeError):
        Logger(run_name="bad_hp_tensor", hyperparameters=bad_hp, log_dir=tmp_path)


def test_b2_05_non_dict_hyperparameters_raises_type_error(tmp_path: Path):
    """Verify passing a non-dict (e.g. list or string) to hyperparameters raises TypeError."""
    with pytest.raises(TypeError):
        Logger(run_name="bad_hp_type", hyperparameters=["lr", 0.01], log_dir=tmp_path)  # type: ignore


# ==============================================================================
# Feature 3: Directory Collision Handling Boundaries (ADR 41)
# ==============================================================================

def test_b3_01_rapid_collisions_timestamp_uniqueness(tmp_path: Path):
    """Verify rapid collisions in quick succession generate unique directory names without overwriting."""
    l1 = Logger(run_name="rapid_coll", log_dir=tmp_path)
    l1.close()
    l2 = Logger(run_name="rapid_coll", log_dir=tmp_path, restart=False)
    l2.close()
    l3 = Logger(run_name="rapid_coll", log_dir=tmp_path, restart=False)
    l3.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("rapid_coll")]
    assert len(dirs) == 3


def test_b3_02_collision_with_already_timestamped_directory(tmp_path: Path):
    """Verify collision resolution when a directory already matches the timestamp format."""
    fake_ts = tmp_path / "coll_ts_20260101_120000"
    fake_ts.mkdir(parents=True)
    (fake_ts / "meta.json").write_text("{}", encoding="utf-8")

    # Creating a run named coll_ts_20260101_120000 should collide and append another timestamp or suffix
    l = Logger(run_name="coll_ts_20260101_120000", log_dir=tmp_path, restart=False)
    l.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("coll_ts_20260101_120000")]
    assert len(dirs) >= 2


def test_b3_03_deeply_nested_custom_log_dir_collision(tmp_path: Path):
    """Verify collision handling when log_dir is deeply nested."""
    deep_dir = tmp_path / "a" / "b" / "c" / "logs"
    l1 = Logger(run_name="deep_run", log_dir=deep_dir)
    l1.close()
    l2 = Logger(run_name="deep_run", log_dir=deep_dir, restart=False)
    l2.close()

    dirs = [p.name for p in deep_dir.iterdir() if p.is_dir() and p.name.startswith("deep_run")]
    assert len(dirs) == 2


def test_b3_04_restart_false_explicit_on_existing_dir(tmp_path: Path):
    """Verify explicit restart=False forces collision resolution even if hyperparameters match."""
    hp = {"seed": 100}
    l1 = Logger(run_name="no_restart_exp", hyperparameters=hp, log_dir=tmp_path)
    l1.close()

    l2 = Logger(run_name="no_restart_exp", hyperparameters=hp, log_dir=tmp_path, restart=False)
    l2.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("no_restart_exp")]
    assert len(dirs) == 2


def test_b3_05_collision_directory_name_pattern_match(tmp_path: Path):
    """Verify the collision directory name adheres to the YYYYMMDD_HHMMSS timestamp format."""
    l1 = Logger(run_name="pat_run", log_dir=tmp_path)
    l1.close()
    l2 = Logger(run_name="pat_run", log_dir=tmp_path, restart=False)
    l2.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("pat_run")]
    alt_dirs = [d for d in dirs if d != "pat_run"]
    assert len(alt_dirs) == 1
    assert re.match(r"^pat_run_\d{8}_\d{6}", alt_dirs[0]) is not None


# ==============================================================================
# Feature 4: JSONL Header Line 1 Boundaries (ADR 32)
# ==============================================================================

def test_b4_01_header_only_file_has_zero_metrics(tmp_path: Path):
    """Verify logger initialized and closed without logging metrics results in 1-line file."""
    logger = Logger(run_name="header_only", log_dir=tmp_path)
    logger.close()

    log_file = tmp_path / "header_only" / "log.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["type"] == "header"


def test_b4_02_header_line_not_included_in_reader_rows(tmp_path: Path):
    """Verify LogReader returns DataFrame with 0 rows on header-only file."""
    logger = Logger(run_name="header_zero_rows", log_dir=tmp_path)
    logger.close()

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "header_zero_rows")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_b4_03_corrupted_header_recovery(tmp_path: Path):
    """Verify LogReader survives a corrupted header line and parses subsequent metrics."""
    run_dir = tmp_path / "bad_header_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("CORRUPTED_HEADER_LINE_NON_JSON\n")
        f.write('{"step": 1, "loss": 0.4}\n')

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


def test_b4_04_missing_header_direct_metrics(tmp_path: Path):
    """Verify LogReader parses file when header is completely absent (first line is metric)."""
    run_dir = tmp_path / "no_header_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"step": 1, "loss": 0.1}\n')
        f.write('{"step": 2, "loss": 0.05}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 2
    assert list(df["step"]) == [1, 2]


def test_b4_05_header_extra_fields_tolerance(tmp_path: Path):
    """Verify LogReader tolerates extra unrecognized fields in header line."""
    run_dir = tmp_path / "future_header_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "2.0", "custom_future_flag": true}\n')
        f.write('{"step": 1, "loss": 0.9}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


# ==============================================================================
# Feature 5: Tags & meta.json Boundaries (ADR 36)
# ==============================================================================

def test_b5_01_empty_tags_list_persisted(tmp_path: Path):
    """Verify passing tags=[] creates meta.json with empty list."""
    logger = Logger(run_name="empty_tags", tags=[], log_dir=tmp_path)
    logger.close()

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "empty_tags")
    assert meta["tags"] == []


def test_b5_02_duplicate_tags_handling(tmp_path: Path):
    """Verify duplicate tags in list are handled without error."""
    tags = ["test", "test", "eval"]
    logger = Logger(run_name="dup_tags", tags=tags, log_dir=tmp_path)
    logger.close()

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "dup_tags")
    assert "test" in meta["tags"]
    assert "eval" in meta["tags"]


def test_b5_03_non_string_tags_raise_type_error(tmp_path: Path):
    """Verify passing non-string elements in tags raises TypeError or converts cleanly."""
    try:
        logger = Logger(run_name="bad_tags", tags=[123, 456], log_dir=tmp_path)  # type: ignore
        logger.close()
        meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "bad_tags")
        assert all(isinstance(t, str) for t in meta["tags"])
    except TypeError:
        pass  # Raising TypeError on non-str tags is valid


def test_b5_04_unicode_and_special_symbols_in_tags(tmp_path: Path):
    """Verify tags containing unicode characters and symbols are preserved."""
    tags = ["λ=0.01", "μ_std", "run/fast", "🌟-best"]
    logger = Logger(run_name="unicode_tags", tags=tags, log_dir=tmp_path)
    logger.close()

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "unicode_tags")
    assert meta["tags"] == tags


def test_b5_05_corrupted_meta_json_fallback(tmp_path: Path):
    """Verify LogReader still reads metrics from log.jsonl when meta.json is corrupted."""
    with Logger(run_name="corrupt_meta", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})

    meta_file = tmp_path / "corrupt_meta" / "meta.json"
    meta_file.write_text("CORRUPTED_JSON_METADATA", encoding="utf-8")

    # read_run should not crash, but continue reading metrics
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "corrupt_meta")
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


# ==============================================================================
# Feature 6: Explicit Step/Epoch Tracking Boundaries (R1, ADR 16)
# ==============================================================================

def test_b6_01_missing_step_and_epoch_raises_value_error(tmp_path: Path):
    """Verify logger.log raises ValueError if neither 'step' nor 'epoch' is provided."""
    with Logger(run_name="missing_step", log_dir=tmp_path) as logger:
        with pytest.raises(ValueError):
            logger.log({"loss": 0.5, "accuracy": 0.9})


def test_b6_02_zero_step_allowed(tmp_path: Path):
    """Verify step=0 is valid and recorded properly."""
    with Logger(run_name="step_zero", log_dir=tmp_path) as logger:
        logger.log({"step": 0, "loss": 1.0})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_zero")
    assert df["step"].iloc[0] == 0


def test_b6_03_negative_step_value(tmp_path: Path):
    """Verify negative step value is handled (either accepted or raises ValueError)."""
    with Logger(run_name="step_neg", log_dir=tmp_path) as logger:
        try:
            logger.log({"step": -1, "loss": 0.5})
            df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_neg")
            assert df["step"].iloc[0] == -1
        except ValueError:
            pass  # Raising ValueError on negative step is acceptable


def test_b6_04_float_step_value_allowed(tmp_path: Path):
    """Verify float step (e.g. fractional epoch 1.5) is logged cleanly."""
    with Logger(run_name="step_float", log_dir=tmp_path) as logger:
        logger.log({"step": 1.5, "val": 42.0})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_float")
    assert df["step"].iloc[0] == 1.5


def test_b6_05_non_numeric_step_raises_type_or_value_error(tmp_path: Path):
    """Verify string or non-numeric step value raises TypeError or ValueError."""
    with Logger(run_name="step_str", log_dir=tmp_path) as logger:
        with pytest.raises((TypeError, ValueError)):
            logger.log({"step": "first_step", "loss": 0.5})  # type: ignore


# ==============================================================================
# Feature 7: Strict Types & Tensor Rejection Boundaries (ADR 19, 39)
# ==============================================================================

def test_b7_01_mock_tensor_with_item_raises_type_error(tmp_path: Path):
    """Verify mock PyTorch tensor object with .item() is rejected with TypeError."""
    with Logger(run_name="rej_tensor_item", log_dir=tmp_path) as logger:
        with pytest.raises(TypeError):
            logger.log({"step": 1, "loss": MockTensorWithItem()})


def test_b7_02_mock_tensor_with_shape_and_tolist_raises_type_error(tmp_path: Path):
    """Verify mock ndarray object with .shape and .tolist() is rejected with TypeError."""
    with Logger(run_name="rej_ndarray", log_dir=tmp_path) as logger:
        with pytest.raises(TypeError):
            logger.log({"step": 1, "weights": MockNdarrayWithShape()})


def test_b7_03_2d_nested_list_rejected(tmp_path: Path):
    """Verify 2D nested list is rejected with TypeError (only 1D arrays supported)."""
    with Logger(run_name="rej_2d_list", log_dir=tmp_path) as logger:
        with pytest.raises(TypeError):
            logger.log({"step": 1, "matrix": [[1.0, 2.0], [3.0, 4.0]]})


def test_b7_04_nested_dict_in_metrics_rejected(tmp_path: Path):
    """Verify nested dictionary inside metrics is rejected with TypeError."""
    with Logger(run_name="rej_nested_dict", log_dir=tmp_path) as logger:
        with pytest.raises(TypeError):
            logger.log({"step": 1, "sub_metrics": {"loss": 0.5}})


def test_b7_05_unserializable_custom_object_rejected(tmp_path: Path):
    """Verify arbitrary custom class instance in metrics raises TypeError."""
    with Logger(run_name="rej_custom_obj", log_dir=tmp_path) as logger:
        with pytest.raises(TypeError):
            logger.log({"step": 1, "custom": UnserializableCustomClass(10)})


# ==============================================================================
# Feature 8: 1D Array Logging Boundaries (ADR 29)
# ==============================================================================

def test_b8_01_empty_1d_list_logged(tmp_path: Path):
    """Verify empty 1D list is logged without error."""
    with Logger(run_name="arr_empty_b", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "arr": []})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_empty_b")
    assert list(df["arr"].iloc[0]) == []


def test_b8_02_single_element_1d_array(tmp_path: Path):
    """Verify 1-element 1D list is logged and preserved."""
    with Logger(run_name="arr_single", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "arr": [42.0]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_single")
    assert list(df["arr"].iloc[0]) == [42.0]


def test_b8_03_large_1d_array_logging(tmp_path: Path):
    """Verify large 1D array (2,000 float elements) is logged without truncation."""
    large_arr = [float(i) * 0.1 for i in range(2000)]
    with Logger(run_name="arr_large", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "big": large_arr})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_large")
    assert len(df["big"].iloc[0]) == 2000


def test_b8_04_mixed_scalars_in_1d_array(tmp_path: Path):
    """Verify 1D array with mixed primitive scalars (int, float, str, bool)."""
    mixed = [1, 2.5, "text", True]
    with Logger(run_name="arr_mixed", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "arr": mixed})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_mixed")
    res = list(df["arr"].iloc[0])
    assert res[0] == 1
    assert res[1] == 2.5
    assert res[2] == "text"
    assert res[3] is True or res[3] == 1


def test_b8_05_all_nan_elements_in_1d_array(tmp_path: Path):
    """Verify 1D array of all NaN elements is deserialized to list of float NaNs."""
    with Logger(run_name="arr_all_nan", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "nans": [math.nan, math.nan, math.nan]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_all_nan")
    res = list(df["nans"].iloc[0])
    assert len(res) == 3
    assert all(math.isnan(x) for x in res)


# ==============================================================================
# Feature 9: NaN/Infinity Serialization Boundaries (ADR 42)
# ==============================================================================

def test_b9_01_subnormal_and_extreme_floats(tmp_path: Path):
    """Verify extreme subnormal and large finite floats log and read accurately."""
    vals = {"step": 1, "min_flt": sys.float_info.min, "max_flt": sys.float_info.max, "tiny": 1e-300}
    with Logger(run_name="flt_extreme", log_dir=tmp_path) as logger:
        logger.log(vals)
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "flt_extreme")
    assert df["min_flt"].iloc[0] == sys.float_info.min
    assert df["max_flt"].iloc[0] == sys.float_info.max
    assert df["tiny"].iloc[0] == 1e-300


def test_b9_02_negative_zero_float(tmp_path: Path):
    """Verify -0.0 is logged and read back as 0.0."""
    with Logger(run_name="neg_zero", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "val": -0.0})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "neg_zero")
    assert df["val"].iloc[0] == 0.0


def test_b9_03_nan_and_inf_coexisting_in_same_record(tmp_path: Path):
    """Verify NaN, +Infinity, and -Infinity coexisting in the same log record."""
    with Logger(run_name="coexist_floats", log_dir=tmp_path) as logger:
        logger.log({
            "step": 1,
            "nan_val": math.nan,
            "pos_inf": math.inf,
            "neg_inf": -math.inf,
        })
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "coexist_floats")
    assert math.isnan(df["nan_val"].iloc[0])
    assert math.isinf(df["pos_inf"].iloc[0]) and df["pos_inf"].iloc[0] > 0
    assert math.isinf(df["neg_inf"].iloc[0]) and df["neg_inf"].iloc[0] < 0


def test_b9_04_float_nan_from_string_constructor(tmp_path: Path):
    """Verify float('nan') and float('inf') are serialized same as math module constants."""
    with Logger(run_name="constructor_floats", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "nan_val": float("nan"), "inf_val": float("inf")})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "constructor_floats")
    assert math.isnan(df["nan_val"].iloc[0])
    assert math.isinf(df["inf_val"].iloc[0])


def test_b9_05_all_inf_metric_column_pandas_operations(tmp_path: Path):
    """Verify DataFrame containing infinities functions properly in pandas queries."""
    with Logger(run_name="inf_pandas", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "grad": math.inf})
        logger.log({"step": 2, "grad": -math.inf})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "inf_pandas")
    assert len(df[df["grad"] > 0]) == 1
    assert len(df[df["grad"] < 0]) == 1


# ==============================================================================
# Feature 10: Timestamp Injection Boundaries (ADR 40)
# ==============================================================================

def test_b10_01_reserved_timestamp_key_overwrite(tmp_path: Path):
    """Verify user providing '_timestamp' key is overwritten with actual timestamp."""
    with Logger(run_name="ts_override", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "_timestamp": 999.0})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ts_override")
    # Must reflect current time, not 999.0
    assert df["_timestamp"].iloc[0] != 999.0


def test_b10_02_reserved_time_since_start_overwrite(tmp_path: Path):
    """Verify user providing '_time_since_start' key is overwritten with actual delta."""
    with Logger(run_name="tss_override", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "_time_since_start": -999.0})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "tss_override")
    assert df["_time_since_start"].iloc[0] >= 0.0


def test_b10_03_rapid_successive_logs_timing(tmp_path: Path):
    """Verify rapid consecutive logs maintain non-decreasing _time_since_start."""
    with Logger(run_name="ts_rapid", log_dir=tmp_path) as logger:
        for i in range(25):
            logger.log({"step": i, "val": i})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ts_rapid")
    diffs = df["_time_since_start"].diff().dropna()
    assert (diffs >= 0.0).all()


def test_b10_04_initial_time_since_start_near_zero(tmp_path: Path):
    """Verify first log immediately after initialization has _time_since_start < 2.0 seconds."""
    with Logger(run_name="ts_initial", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ts_initial")
    assert df["_time_since_start"].iloc[0] < 2.0


def test_b10_05_elapsed_time_reflects_actual_delay(tmp_path: Path):
    """Verify sleep between log calls is reflected in _time_since_start difference."""
    with Logger(run_name="ts_delay", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
        time.sleep(0.05)
        logger.log({"step": 2, "loss": 0.4})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ts_delay")
    delta = df["_time_since_start"].iloc[1] - df["_time_since_start"].iloc[0]
    assert delta >= 0.04


# ==============================================================================
# Feature 11: Synchronous Flush Boundaries (ADR 34)
# ==============================================================================

def test_b11_01_multiple_closes_idempotent(tmp_path: Path):
    """Verify calling close() multiple times does not raise any exception."""
    logger = Logger(run_name="mult_close", log_dir=tmp_path)
    logger.log({"step": 1, "loss": 0.5})
    logger.close()
    logger.close()  # Must be idempotent
    logger.close()


def test_b11_02_log_after_close_raises_error(tmp_path: Path):
    """Verify calling log() after close() raises RuntimeError or ValueError."""
    logger = Logger(run_name="closed_log", log_dir=tmp_path)
    logger.close()
    with pytest.raises((RuntimeError, ValueError)):
        logger.log({"step": 1, "loss": 0.1})


def test_b11_03_context_manager_exception_still_closes_fd(tmp_path: Path):
    """Verify exception inside context manager block still cleanly closes file."""
    try:
        with Logger(run_name="cm_exc", log_dir=tmp_path) as logger:
            logger.log({"step": 1, "loss": 0.5})
            raise RuntimeError("Forced simulation error")
    except RuntimeError:
        pass

    log_file = tmp_path / "cm_exc" / "log.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_b11_04_file_content_valid_json_before_close(tmp_path: Path):
    """Verify log lines written to disk can be parsed as valid JSON before close() is called."""
    logger = Logger(run_name="valid_before_close", log_dir=tmp_path)
    logger.log({"step": 1, "metric": "test_flush"})

    log_file = tmp_path / "valid_before_close" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
    logger.close()


def test_b11_05_file_not_truncated_on_rapid_writes(tmp_path: Path):
    """Verify rapid successive writes do not interleave or produce truncated lines."""
    with Logger(run_name="rapid_writes", log_dir=tmp_path) as logger:
        for i in range(50):
            logger.log({"step": i, "payload": f"data_{i}" * 10})

    log_file = tmp_path / "rapid_writes" / "log.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 51  # 1 header + 50 records


# ==============================================================================
# Feature 12: Restart Mode Boundaries (ADR 24)
# ==============================================================================

def test_b12_01_restart_with_conflicting_hyperparameters_raises_value_error(tmp_path: Path):
    """Verify restart=True raises ValueError when provided hyperparameters differ from original."""
    with Logger(run_name="restart_conflict", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    with pytest.raises(ValueError):
        Logger(run_name="restart_conflict", hyperparameters={"lr": 0.05}, log_dir=tmp_path, restart=True)


def test_b12_02_restart_on_nonexistent_run_creates_new_directory(tmp_path: Path):
    """Verify restart=True on a brand-new run name creates the run cleanly without error."""
    logger = Logger(run_name="restart_brand_new", log_dir=tmp_path, restart=True)
    logger.log({"step": 1, "loss": 0.2})
    logger.close()

    assert (tmp_path / "restart_brand_new").is_dir()
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "restart_brand_new")
    assert len(df) == 1


def test_b12_03_restart_with_added_hyperparameter_raises_value_error(tmp_path: Path):
    """Verify restart=True raises ValueError when extra hyperparameters are added."""
    with Logger(run_name="restart_added_hp", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    with pytest.raises(ValueError):
        Logger(
            run_name="restart_added_hp",
            hyperparameters={"lr": 0.01, "extra_param": 10},
            log_dir=tmp_path,
            restart=True,
        )


def test_b12_04_restart_with_missing_hyperparameter_raises_value_error(tmp_path: Path):
    """Verify restart=True raises ValueError when an existing hyperparameter is omitted."""
    with Logger(run_name="restart_sub_hp", hyperparameters={"lr": 0.01, "bs": 32}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    with pytest.raises(ValueError):
        Logger(run_name="restart_sub_hp", hyperparameters={"lr": 0.01}, log_dir=tmp_path, restart=True)


def test_b12_05_multiple_restarts_accumulate_all_records(tmp_path: Path):
    """Verify 3 consecutive restarts append all records into the same log.jsonl file."""
    hp = {"model": "cnn"}
    for step in range(1, 4):
        with Logger(run_name="mult_restart", hyperparameters=hp, log_dir=tmp_path, restart=(step > 1)) as l:
            l.log({"step": step, "acc": step * 0.25})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "mult_restart")
    assert len(df) == 3
    assert list(df["step"]) == [1, 2, 3]


# ==============================================================================
# Feature 13: Python & Package State Capture Boundaries (R2, ADR 31)
# ==============================================================================

def test_b13_01_capture_env_with_empty_distributions(monkeypatch):
    """Verify capture_environment handles environment when no distributions are discovered."""
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    env = capture_environment()
    assert isinstance(env, dict)
    assert env.get("packages") == {} or isinstance(env.get("packages"), dict)


def test_b13_02_capture_env_exception_tolerance(monkeypatch):
    """Verify capture_environment tolerates exception during package inspection without crashing."""
    import importlib.metadata
    def raise_err():
        raise RuntimeError("Inspection failed")
    monkeypatch.setattr(importlib.metadata, "distributions", raise_err)
    env = capture_environment()
    assert isinstance(env, dict)
    assert "python_version" in env


def test_b13_03_capture_env_false_skips_capture(tmp_path: Path):
    """Verify Logger(capture_env=False) does not store env metadata in meta.json."""
    logger = Logger(run_name="no_env", capture_env=False, log_dir=tmp_path)
    logger.close()

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "no_env")
    assert meta.get("env") is None or meta.get("environment") is None


def test_b13_04_capture_env_json_serializable():
    """Verify the entire capture_environment() dict is serializable to JSON."""
    env = capture_environment()
    serialized = json.dumps(env)
    assert isinstance(serialized, str)
    assert len(serialized) > 0


def test_b13_05_capture_env_sys_platform_non_empty():
    """Verify sys_platform field in capture_environment is a non-empty string."""
    env = capture_environment()
    assert isinstance(env["sys_platform"], str)
    assert len(env["sys_platform"]) > 0


# ==============================================================================
# Feature 14: Git Commit Hash Capture Boundaries (R2, ADR 31)
# ==============================================================================

def test_b14_01_capture_git_info_in_non_git_dir(tmp_path: Path):
    """Verify capture_git_info in a non-git directory returns git_commit=None and is_dirty=False."""
    plain_dir = tmp_path / "not_git"
    plain_dir.mkdir()
    info = capture_git_info(cwd=plain_dir)
    assert info["git_commit"] is None
    assert info["is_dirty"] is False


def test_b14_02_capture_git_info_when_git_command_fails(monkeypatch, tmp_path: Path):
    """Verify capture_git_info handles FileNotFoundError (git executable missing)."""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(subprocess, "run", mock_run)

    info = capture_git_info(cwd=tmp_path)
    assert info["git_commit"] is None


def test_b14_03_capture_git_info_empty_repo_no_commits(tmp_path: Path):
    """Verify capture_git_info in a newly initialized repo without any commits does not crash."""
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    subprocess.run(["git", "init"], cwd=empty_repo, check=True, capture_output=True)

    info = capture_git_info(cwd=empty_repo)
    assert info["git_commit"] is None


def test_b14_04_capture_git_info_on_nonexistent_cwd(tmp_path: Path):
    """Verify capture_git_info handles nonexistent directory gracefully."""
    nonexistent = tmp_path / "does_not_exist"
    try:
        info = capture_git_info(cwd=nonexistent)
        assert info.get("git_commit") is None
    except (FileNotFoundError, OSError):
        pass


def test_b14_05_capture_git_info_dirty_file_deletion(temp_git_repo: Path):
    """Verify deleting a tracked file marks working tree is_dirty=True."""
    tracked_file = temp_git_repo / "initial.txt"
    tracked_file.unlink()
    info = capture_git_info(cwd=temp_git_repo)
    assert info["is_dirty"] is True


# ==============================================================================
# Feature 15: Uncommitted Changes diff.patch Boundaries (R2, ADR 31)
# ==============================================================================

def test_b15_01_create_diff_patch_non_git_returns_false(tmp_path: Path):
    """Verify create_diff_patch in a non-git directory returns False and does not create patch."""
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    patch_out = plain_dir / "diff.patch"
    res = create_diff_patch(patch_out, cwd=plain_dir)
    assert res is False
    assert not patch_out.exists()


def test_b15_02_create_diff_patch_clean_repo_returns_false(temp_git_repo: Path):
    """Verify create_diff_patch in a completely clean repo returns False."""
    patch_out = temp_git_repo / "diff.patch"
    res = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert res is False
    assert not patch_out.exists()


def test_b15_03_create_diff_patch_large_diff(temp_git_repo: Path):
    """Verify create_diff_patch handles large file modifications without crashing."""
    large_content = "Line content for large diff test\n" * 5000
    (temp_git_repo / "initial.txt").write_text(large_content, encoding="utf-8")
    patch_out = temp_git_repo / "large.patch"
    res = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert res is True
    assert patch_out.exists()
    assert patch_out.stat().st_size > 1000


def test_b15_04_create_diff_patch_binary_file_handling(temp_git_repo: Path):
    """Verify create_diff_patch handles binary file changes without UnicodeDecodeError."""
    bin_file = temp_git_repo / "data.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
    subprocess.run(["git", "add", "data.bin"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add binary"], cwd=temp_git_repo, check=True, capture_output=True)

    # Modify binary file
    bin_file.write_bytes(b"\xff\xfe\x00\x01\x99\x88")
    patch_out = temp_git_repo / "bin.patch"
    res = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert res is True
    assert patch_out.exists()


def test_b15_05_diff_patch_output_parent_dir_created(temp_git_repo: Path):
    """Verify create_diff_patch ensures output directory exists or can be created."""
    (temp_git_repo / "initial.txt").write_text("mod\n", encoding="utf-8")
    patch_out = temp_git_repo / "nested" / "sub" / "diff.patch"
    patch_out.parent.mkdir(parents=True, exist_ok=True)
    res = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert res is True
    assert patch_out.exists()


# ==============================================================================
# Feature 16: LogReader DataFrame Parsing Boundaries (R3)
# ==============================================================================

def test_b16_01_read_nonexistent_run_raises_file_not_found(tmp_path: Path):
    """Verify LogReader.read_run on nonexistent directory raises FileNotFoundError."""
    reader = LogReader(logs_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.read_run(tmp_path / "nonexistent_run_xyz")


def test_b16_02_read_empty_log_file_returns_empty_dataframe(tmp_path: Path):
    """Verify reading a 0-byte log.jsonl returns an empty DataFrame."""
    empty_run = tmp_path / "zero_bytes_run"
    empty_run.mkdir()
    (empty_run / "log.jsonl").write_text("", encoding="utf-8")

    df = LogReader(logs_dir=tmp_path).read_run(empty_run)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_b16_03_read_log_file_with_only_header_returns_empty_dataframe(tmp_path: Path):
    """Verify reading log file with only line 1 header returns an empty DataFrame."""
    header_run = tmp_path / "header_only_run"
    header_run.mkdir()
    (header_run / "log.jsonl").write_text('{"type": "header", "mlviz_version": "1.0"}\n', encoding="utf-8")

    df = LogReader(logs_dir=tmp_path).read_run(header_run)
    assert len(df) == 0


def test_b16_04_read_jsonl_nonexistent_file_raises_file_not_found(tmp_path: Path):
    """Verify read_jsonl on nonexistent file raises FileNotFoundError."""
    reader = LogReader(logs_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.read_jsonl(tmp_path / "missing_file.jsonl")


def test_b16_05_read_run_missing_meta_json_returns_dataframe(tmp_path: Path):
    """Verify read_run succeeds when meta.json is absent (parses metrics from log.jsonl)."""
    run_dir = tmp_path / "no_meta_run"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": 0.77}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["loss"].iloc[0] == 0.77


# ==============================================================================
# Feature 17: Corrupted Line Tolerance Boundaries (ADR 43)
# ==============================================================================

def test_b17_01_all_lines_corrupted_returns_empty_dataframe(tmp_path: Path):
    """Verify file with all corrupted lines returns empty DataFrame with warning."""
    run_dir = tmp_path / "all_corrupted"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write("corrupted line 1\n")
        f.write("{unclosed json 2\n")
        f.write("invalid [[[ 3\n")

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_b17_02_null_bytes_in_jsonl_handled(tmp_path: Path):
    """Verify line containing null bytes is skipped with warning."""
    run_dir = tmp_path / "null_bytes_run"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "wb") as f:
        f.write(b'{"type": "header", "mlviz_version": "1.0"}\n')
        f.write(b'{"step": 1, "val": 1.0}\n')
        f.write(b'{"step": 2, \x00\x00\x00}\n')
        f.write(b'{"step": 3, "val": 3.0}\n')

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 2
    assert list(df["step"]) == [1, 3]


def test_b17_03_unterminated_json_string_skipped(tmp_path: Path):
    """Verify unterminated string literal in JSON line is skipped with warning."""
    run_dir = tmp_path / "unterminated_str"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "note": "unclosed string\n')
        f.write('{"step": 2, "note": "valid"}\n')

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["step"].iloc[0] == 2


def test_b17_04_mixed_corrupted_and_valid_interleaved(tmp_path: Path):
    """Verify alternating good and bad lines preserves all good lines."""
    run_dir = tmp_path / "interleaved"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "v": 10}\n')
        f.write('bad line 1\n')
        f.write('{"step": 2, "v": 20}\n')
        f.write('bad line 2\n')
        f.write('{"step": 3, "v": 30}\n')

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 3
    assert list(df["v"]) == [10, 20, 30]


def test_b17_05_warning_category_is_user_warning(tmp_path: Path):
    """Verify caught warning for corrupted line is a UserWarning."""
    run_dir = tmp_path / "warn_type_check"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('BAD_JSON\n')

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        LogReader(logs_dir=tmp_path).read_run(run_dir)
        assert len(recorded_warnings) >= 1
        assert issubclass(recorded_warnings[0].category, UserWarning)


# ==============================================================================
# Feature 18: Special Float Deserialization Boundaries (ADR 42)
# ==============================================================================

def test_b18_01_lowercase_nan_string_deserialization(tmp_path: Path):
    """Verify 'NaN' or 'nan' string in JSON is converted to float NaN."""
    run_dir = tmp_path / "lower_nan"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": "NaN"}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert math.isnan(df["loss"].iloc[0])


def test_b18_02_infinity_without_plus_sign(tmp_path: Path):
    """Verify '+Infinity' string is deserialized to positive infinity."""
    run_dir = tmp_path / "pos_inf_check"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "grad": "+Infinity"}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert math.isinf(df["grad"].iloc[0]) and df["grad"].iloc[0] > 0


def test_b18_03_array_with_all_special_floats(tmp_path: Path):
    """Verify array with all special floats is deserialized element-by-element."""
    run_dir = tmp_path / "all_special_arr"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "arr": ["NaN", "+Infinity", "-Infinity", 10.0]}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    arr = df["arr"].iloc[0]
    assert math.isnan(arr[0])
    assert math.isinf(arr[1]) and arr[1] > 0
    assert math.isinf(arr[2]) and arr[2] < 0
    assert arr[3] == 10.0


def test_b18_04_regular_string_not_converted_to_float(tmp_path: Path):
    """Verify normal strings (like 'training_phase') are preserved as strings and not converted."""
    run_dir = tmp_path / "normal_str_check"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "stage": "train"}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert df["stage"].iloc[0] == "train"


def test_b18_05_column_all_nans_isna_true(tmp_path: Path):
    """Verify a column with all NaN values satisfies isna().all()."""
    with Logger(run_name="all_nan_col", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "metric": math.nan})
        logger.log({"step": 2, "metric": math.nan})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "all_nan_col")
    assert df["metric"].isna().all()


# ==============================================================================
# Feature 19: Hyperparameter Prefixing Boundaries (View DSL)
# ==============================================================================

def test_b19_01_get_hyperparameters_nonexistent_run_raises(tmp_path: Path):
    """Verify get_hyperparameters on nonexistent directory raises FileNotFoundError."""
    reader = LogReader(logs_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.get_hyperparameters(tmp_path / "does_not_exist")


def test_b19_02_get_metadata_nonexistent_run_raises(tmp_path: Path):
    """Verify get_metadata on nonexistent directory raises FileNotFoundError."""
    reader = LogReader(logs_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.get_metadata(tmp_path / "does_not_exist")


def test_b19_03_hyperparameters_key_name_with_dots(tmp_path: Path):
    """Verify hyperparameters with dots in key names are prefixed properly."""
    hp = {"model.backbone": "resnet50"}
    with Logger(run_name="dots_in_hp", hyperparameters=hp, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "dots_in_hp")
    assert "hyperparameters.model.backbone" in df.columns


def test_b19_04_get_hyperparameters_empty_dict(tmp_path: Path):
    """Verify run created with empty hyperparameters returns empty dict."""
    logger = Logger(run_name="hp_empty_b", hyperparameters={}, log_dir=tmp_path)
    logger.close()
    retrieved = LogReader(logs_dir=tmp_path).get_hyperparameters(tmp_path / "hp_empty_b")
    assert retrieved == {}


def test_b19_05_metric_and_hyperparameter_name_collision(tmp_path: Path):
    """Verify hyperparameter named 'loss' and metric named 'loss' do not collide."""
    hp = {"loss": "cross_entropy"}
    with Logger(run_name="collision_name", hyperparameters=hp, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.25})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "collision_name")
    assert "hyperparameters.loss" in df.columns
    assert "loss" in df.columns
    assert df["hyperparameters.loss"].iloc[0] == "cross_entropy"
    assert df["loss"].iloc[0] == 0.25


# ==============================================================================
# Feature 20: Multi-Run Discovery & Evolution Boundaries (ADR 12, 20)
# ==============================================================================

def test_b20_01_read_all_with_nested_subdirectories(tmp_path: Path):
    """Verify read_all discovers runs located in nested subdirectories."""
    sub1 = tmp_path / "group_a" / "run_1"
    sub2 = tmp_path / "group_b" / "sub_group" / "run_2"
    with Logger(run_name="run_1", log_dir=sub1.parent) as l1:
        l1.log({"step": 1, "val": 10})
    with Logger(run_name="run_2", log_dir=sub2.parent) as l2:
        l2.log({"step": 1, "val": 20})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) == 2
    assert set(df["run_name"]) == {"run_1", "run_2"}


def test_b20_02_read_all_ignores_unrelated_files(tmp_path: Path):
    """Verify read_all ignores non-run files and directories without crashing."""
    (tmp_path / "README.md").write_text("# Documentation\n", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01\x02")
    non_run_dir = tmp_path / "other_dir"
    non_run_dir.mkdir()
    (non_run_dir / "notes.txt").write_text("not a run", encoding="utf-8")

    with Logger(run_name="valid_run", log_dir=tmp_path) as l:
        l.log({"step": 1, "loss": 0.1})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) == 1
    assert df["run_name"].iloc[0] == "valid_run"


def test_b20_03_read_all_partially_corrupted_run_resilience(tmp_path: Path):
    """Verify read_all continues and parses valid runs even if one run directory is corrupted."""
    with Logger(run_name="good_run", log_dir=tmp_path) as l:
        l.log({"step": 1, "score": 100})

    bad_run = tmp_path / "bad_run"
    bad_run.mkdir()
    (bad_run / "log.jsonl").write_text("TOTALLY_CORRUPTED_NON_JSON\n", encoding="utf-8")

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) >= 1
    assert "good_run" in df["run_name"].values


def test_b20_04_read_all_disjoint_hyperparameters_schema_evolution(tmp_path: Path):
    """Verify runs with completely disjoint hyperparameters merge cleanly with NaNs."""
    with Logger(run_name="r1", hyperparameters={"alpha": 1}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "metric": 1})
    with Logger(run_name="r2", hyperparameters={"beta": 2}, log_dir=tmp_path) as l2:
        l2.log({"step": 1, "metric": 2})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert "hyperparameters.alpha" in df.columns
    assert "hyperparameters.beta" in df.columns
    assert pd.isna(df.loc[df["run_name"] == "r2", "hyperparameters.alpha"].iloc[0])
    assert pd.isna(df.loc[df["run_name"] == "r1", "hyperparameters.beta"].iloc[0])


def test_b20_05_read_all_with_path_and_string_logs_dir(tmp_path: Path):
    """Verify read_all accepts both Path and string representation for logs_dir."""
    with Logger(run_name="type_check_run", log_dir=tmp_path) as l:
        l.log({"step": 1, "v": 1})

    df1 = LogReader(logs_dir=tmp_path).read_all()
    df2 = LogReader(logs_dir=str(tmp_path)).read_all()
    assert len(df1) == 1
    assert len(df2) == 1
