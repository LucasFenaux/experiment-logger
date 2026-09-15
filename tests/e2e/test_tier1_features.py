"""Tier 1: Feature coverage happy-path E2E tests.

Comprehensive opaque-box verification for 20 features (>=5 tests per feature = 100 tests).
Tests strictly exercise public interfaces:
- Logger (experiment_logger.Logger)
- LogReader (experiment_logger.LogReader)
- Environment capture (experiment_logger.env)
- Output filesystem artifacts (log.jsonl, meta.json, diff.patch)
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


# ==============================================================================
# Feature 1: Logger Init & Run Name (ADR 33)
# ==============================================================================

def test_f1_01_alphanumeric_run_name_creates_directory(tmp_path: Path):
    """Verify Logger initializes with standard alphanumeric name and creates run directory."""
    logger = Logger(run_name="experiment01", log_dir=tmp_path)
    run_dir = tmp_path / "experiment01"
    assert run_dir.exists()
    assert run_dir.is_dir()
    logger.close()


def test_f1_02_run_name_with_hyphens_and_underscores(tmp_path: Path):
    """Verify Logger accepts run names with hyphens and underscores."""
    name = "resnet50_v2-baseline_run-01"
    logger = Logger(run_name=name, log_dir=tmp_path)
    run_dir = tmp_path / name
    assert run_dir.exists()
    assert (run_dir / "log.jsonl").exists()
    logger.close()


def test_f1_03_custom_log_dir_as_path_object(tmp_path: Path):
    """Verify Logger accepts log_dir as a pathlib.Path object and nests properly."""
    custom_dir = tmp_path / "nested" / "logs"
    logger = Logger(run_name="path_run", log_dir=custom_dir)
    assert (custom_dir / "path_run").is_dir()
    assert (custom_dir / "path_run" / "meta.json").exists()
    logger.close()


def test_f1_04_custom_log_dir_as_string(tmp_path: Path):
    """Verify Logger accepts log_dir as a string path."""
    str_dir = str(tmp_path / "str_logs")
    logger = Logger(run_name="str_run", log_dir=str_dir)
    assert (Path(str_dir) / "str_run").is_dir()
    logger.close()


def test_f1_05_logger_context_manager_lifecycle(tmp_path: Path):
    """Verify Logger works cleanly with context manager protocol (__enter__ / __exit__)."""
    with Logger(run_name="cm_run", log_dir=tmp_path) as logger:
        assert isinstance(logger, Logger)
        logger.log({"step": 1, "loss": 0.42})
    run_dir = tmp_path / "cm_run"
    assert (run_dir / "log.jsonl").exists()


# ==============================================================================
# Feature 2: Hyperparameter Validation & Persistence (ADR 44)
# ==============================================================================

def test_f2_01_primitive_types_serialized_in_meta_json(tmp_path: Path):
    """Verify primitive types (int, float, str, bool, None) in hyperparameters are serialized."""
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_name": "transformer",
        "use_dropout": True,
        "warmup_steps": None,
    }
    logger = Logger(run_name="hparam_primitives", hyperparameters=hparams, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "hparam_primitives" / "meta.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hyperparameters"] == hparams


def test_f2_02_nested_dictionary_hyperparameters(tmp_path: Path):
    """Verify nested dictionary structures in hyperparameters are serialized correctly."""
    hparams = {
        "optimizer": {"type": "adamw", "lr": 1e-4, "betas": [0.9, 0.999]},
        "scheduler": {"type": "cosine", "t_max": 100},
    }
    logger = Logger(run_name="hparam_nested", hyperparameters=hparams, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "hparam_nested" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hyperparameters"]["optimizer"]["type"] == "adamw"
    assert data["hyperparameters"]["optimizer"]["betas"] == [0.9, 0.999]


def test_f2_03_list_hyperparameters(tmp_path: Path):
    """Verify list values in hyperparameters are preserved."""
    hparams = {"layer_sizes": [256, 128, 64], "class_names": ["cat", "dog"]}
    logger = Logger(run_name="hparam_lists", hyperparameters=hparams, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "hparam_lists" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hyperparameters"]["layer_sizes"] == [256, 128, 64]
    assert data["hyperparameters"]["class_names"] == ["cat", "dog"]


def test_f2_04_none_hyperparameters_default(tmp_path: Path):
    """Verify initializing with hyperparameters=None creates meta.json with empty/None dict."""
    logger = Logger(run_name="hparam_none", hyperparameters=None, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "hparam_none" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("hyperparameters") in ({}, None)


def test_f2_05_empty_dictionary_hyperparameters(tmp_path: Path):
    """Verify initializing with hyperparameters={} creates meta.json with empty dict."""
    logger = Logger(run_name="hparam_empty", hyperparameters={}, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "hparam_empty" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("hyperparameters") == {}


# ==============================================================================
# Feature 3: Directory Collision Handling (ADR 41)
# ==============================================================================

def test_f3_01_collision_creates_timestamped_directory(tmp_path: Path):
    """Verify that colliding run names create a new timestamped directory (run_name_YYYYMMDD_HHMMSS)."""
    l1 = Logger(run_name="run_coll", log_dir=tmp_path)
    l1.close()

    time.sleep(0.01)
    l2 = Logger(run_name="run_coll", log_dir=tmp_path, restart=False)
    l2.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("run_coll")]
    assert len(dirs) == 2
    assert "run_coll" in dirs
    timestamped = [d for d in dirs if d != "run_coll"]
    assert len(timestamped) == 1
    assert re.match(r"^run_coll_\d{8}_\d{6}", timestamped[0]) is not None


def test_f3_02_collision_preserves_first_run_content(tmp_path: Path):
    """Verify first run's files remain intact when collision creates second run directory."""
    with Logger(run_name="preserve_test", hyperparameters={"v": 1}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 1.0})

    with Logger(run_name="preserve_test", hyperparameters={"v": 2}, log_dir=tmp_path, restart=False) as l2:
        l2.log({"step": 1, "loss": 2.0})

    # First run must retain its own metadata and logs
    with open(tmp_path / "preserve_test" / "meta.json", "r", encoding="utf-8") as f:
        meta1 = json.load(f)
    assert meta1["hyperparameters"]["v"] == 1


def test_f3_03_multiple_collisions_produce_distinct_directories(tmp_path: Path):
    """Verify multiple collisions create separate distinct directories."""
    l1 = Logger(run_name="multi_coll", log_dir=tmp_path)
    l1.close()
    time.sleep(1.0)
    l2 = Logger(run_name="multi_coll", log_dir=tmp_path, restart=False)
    l2.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("multi_coll")]
    assert len(dirs) == 2


def test_f3_04_collision_with_different_hyperparameters(tmp_path: Path):
    """Verify collision without restart safely initializes even when hyperparameters differ."""
    l1 = Logger(run_name="diff_hp", hyperparameters={"lr": 0.1}, log_dir=tmp_path)
    l1.close()

    # Different hyperparameters with restart=False triggers collision avoidance, not error
    l2 = Logger(run_name="diff_hp", hyperparameters={"lr": 0.01}, log_dir=tmp_path, restart=False)
    l2.close()

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("diff_hp")]
    assert len(dirs) == 2


def test_f3_05_collision_preserves_custom_log_dir(tmp_path: Path):
    """Verify collision creates the timestamped directory within the specified custom log_dir."""
    custom = tmp_path / "custom_sub"
    l1 = Logger(run_name="custom_coll", log_dir=custom)
    l1.close()
    l2 = Logger(run_name="custom_coll", log_dir=custom, restart=False)
    l2.close()

    dirs = [p.name for p in custom.iterdir() if p.is_dir() and p.name.startswith("custom_coll")]
    assert len(dirs) == 2


# ==============================================================================
# Feature 4: JSONL Header Line 1 (ADR 32)
# ==============================================================================

def test_f4_01_header_is_first_line(tmp_path: Path):
    """Verify very first line of log.jsonl is a valid header record."""
    logger = Logger(run_name="header_test", log_dir=tmp_path)
    logger.log({"step": 1, "loss": 0.5})
    logger.close()

    log_file = tmp_path / "header_test" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    header = json.loads(first_line)
    assert header.get("type") == "header"


def test_f4_02_header_contains_version_and_type(tmp_path: Path):
    """Verify line 1 header contains mlviz_version and type: header."""
    logger = Logger(run_name="header_ver", log_dir=tmp_path)
    logger.close()

    log_file = tmp_path / "header_ver" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
    assert "mlviz_version" in header
    assert header["type"] == "header"


def test_f4_03_header_written_immediately_on_init(tmp_path: Path):
    """Verify header line is written to disk immediately upon Logger initialization."""
    logger = Logger(run_name="header_immediate", log_dir=tmp_path)
    log_file = tmp_path / "header_immediate" / "log.jsonl"
    assert log_file.exists()
    assert log_file.stat().st_size > 0
    with open(log_file, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
    assert header.get("type") == "header"
    logger.close()


def test_f4_04_header_is_valid_json_with_trailing_newline(tmp_path: Path):
    """Verify header line ends with a newline character."""
    logger = Logger(run_name="header_nl", log_dir=tmp_path)
    logger.close()

    log_file = tmp_path / "header_nl" / "log.jsonl"
    with open(log_file, "rb") as f:
        raw_first_line = f.readline()
    assert raw_first_line.endswith(b"\n")


def test_f4_05_metrics_appended_after_header_line(tmp_path: Path):
    """Verify metrics records are appended starting at line 2."""
    with Logger(run_name="header_follow", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "accuracy": 0.88})

    log_file = tmp_path / "header_follow" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "header"
    assert json.loads(lines[1])["step"] == 1
    assert json.loads(lines[1])["accuracy"] == 0.88


# ==============================================================================
# Feature 5: Tags & meta.json Persistence (ADR 36)
# ==============================================================================

def test_f5_01_tags_saved_to_meta_json(tmp_path: Path):
    """Verify custom tags list is saved to meta.json."""
    tags = ["baseline", "resnet", "production"]
    logger = Logger(run_name="tags_run", tags=tags, log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "tags_run" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["tags"] == tags


def test_f5_02_empty_tags_default(tmp_path: Path):
    """Verify default tags is empty list or empty representation."""
    logger = Logger(run_name="tags_default", log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "tags_default" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("tags") in ([], None)


def test_f5_03_tags_retrievable_via_reader_get_metadata(tmp_path: Path):
    """Verify LogReader.get_metadata retrieves tags cleanly."""
    tags = ["v1", "seed-42"]
    logger = Logger(run_name="tags_reader", tags=tags, log_dir=tmp_path)
    logger.close()

    reader = LogReader(logs_dir=tmp_path)
    metadata = reader.get_metadata(tmp_path / "tags_reader")
    assert metadata["tags"] == tags


def test_f5_04_meta_json_records_run_name_and_timestamp(tmp_path: Path):
    """Verify meta.json contains run_name and a creation timestamp or time field."""
    logger = Logger(run_name="meta_fields", log_dir=tmp_path)
    logger.close()

    meta_file = tmp_path / "meta_fields" / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_name"] == "meta_fields"
    assert "created_at" in data or "timestamp" in data or "start_time" in data


def test_f5_05_meta_json_records_exact_hyperparameters(tmp_path: Path):
    """Verify meta.json stores the exact hyperparameters dictionary provided."""
    hp = {"epochs": 10, "lr": 3e-4}
    logger = Logger(run_name="meta_hp", hyperparameters=hp, log_dir=tmp_path)
    logger.close()

    reader = LogReader(logs_dir=tmp_path)
    assert reader.get_hyperparameters(tmp_path / "meta_hp") == hp


# ==============================================================================
# Feature 6: Explicit Step/Epoch Tracking (R1, ADR 16)
# ==============================================================================

def test_f6_01_log_with_integer_step(tmp_path: Path):
    """Verify logging with explicit integer step succeeds."""
    with Logger(run_name="step_int", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_int")
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


def test_f6_02_log_with_integer_epoch(tmp_path: Path):
    """Verify logging with explicit integer epoch succeeds."""
    with Logger(run_name="epoch_int", log_dir=tmp_path) as logger:
        logger.log({"epoch": 0, "val_acc": 0.85})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "epoch_int")
    assert len(df) == 1
    assert df["epoch"].iloc[0] == 0


def test_f6_03_log_with_both_step_and_epoch(tmp_path: Path):
    """Verify logging with both step and epoch succeeds and preserves both."""
    with Logger(run_name="step_and_epoch", log_dir=tmp_path) as logger:
        logger.log({"step": 100, "epoch": 2, "loss": 0.25})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_and_epoch")
    assert df["step"].iloc[0] == 100
    assert df["epoch"].iloc[0] == 2


def test_f6_04_sequential_steps_logged_in_order(tmp_path: Path):
    """Verify sequential steps are recorded in the exact order logged."""
    with Logger(run_name="step_seq", log_dir=tmp_path) as logger:
        for s in range(5):
            logger.log({"step": s, "metric": s * 10})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_seq")
    assert list(df["step"]) == [0, 1, 2, 3, 4]
    assert list(df["metric"]) == [0, 10, 20, 30, 40]


def test_f6_05_non_consecutive_steps_logged_correctly(tmp_path: Path):
    """Verify non-consecutive steps (e.g. periodic evaluations) are logged accurately."""
    with Logger(run_name="step_nonconsec", log_dir=tmp_path) as logger:
        logger.log({"step": 10, "eval": 0.1})
        logger.log({"step": 50, "eval": 0.05})
        logger.log({"step": 100, "eval": 0.02})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "step_nonconsec")
    assert list(df["step"]) == [10, 50, 100]


# ==============================================================================
# Feature 7: Strict Types & Native Scalar Acceptance (ADR 19, 39)
# ==============================================================================

def test_f7_01_native_float_metrics_accepted(tmp_path: Path):
    """Verify standard native Python float metrics are accepted and written."""
    with Logger(run_name="scalar_float", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "val": 3.14159265})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "scalar_float")
    assert pytest.approx(df["val"].iloc[0], rel=1e-5) == 3.14159265


def test_f7_02_native_int_metrics_accepted(tmp_path: Path):
    """Verify standard native Python integer metrics are accepted."""
    with Logger(run_name="scalar_int", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "samples_processed": 50000})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "scalar_int")
    assert df["samples_processed"].iloc[0] == 50000


def test_f7_03_native_str_metrics_accepted(tmp_path: Path):
    """Verify native Python string metrics are accepted."""
    with Logger(run_name="scalar_str", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "phase": "validation"})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "scalar_str")
    assert df["phase"].iloc[0] == "validation"


def test_f7_04_native_bool_metrics_accepted(tmp_path: Path):
    """Verify native Python boolean metrics are accepted."""
    with Logger(run_name="scalar_bool", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "early_stop": False, "checkpoint_saved": True})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "scalar_bool")
    assert df["early_stop"].iloc[0] is False or df["early_stop"].iloc[0] == 0
    assert df["checkpoint_saved"].iloc[0] is True or df["checkpoint_saved"].iloc[0] == 1


def test_f7_05_mixed_primitive_scalar_types_accepted(tmp_path: Path):
    """Verify logging a record with a mix of float, int, str, and bool."""
    with Logger(run_name="scalar_mixed", log_dir=tmp_path) as logger:
        logger.log({
            "step": 42,
            "loss": 0.123,
            "batches": 100,
            "status": "ok",
            "is_best": True,
        })
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "scalar_mixed")
    assert df["step"].iloc[0] == 42
    assert pytest.approx(df["loss"].iloc[0]) == 0.123
    assert df["batches"].iloc[0] == 100
    assert df["status"].iloc[0] == "ok"


# ==============================================================================
# Feature 8: 1D Array Logging (ADR 29)
# ==============================================================================

def test_f8_01_1d_list_of_floats_logged(tmp_path: Path):
    """Verify 1D list of floats is correctly written to JSONL."""
    with Logger(run_name="arr_floats", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "batch_losses": [0.6, 0.4, 0.3]})

    log_file = tmp_path / "arr_floats" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["batch_losses"] == [0.6, 0.4, 0.3]


def test_f8_02_1d_list_of_ints_logged(tmp_path: Path):
    """Verify 1D list of ints is correctly written to JSONL."""
    with Logger(run_name="arr_ints", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "counts": [10, 20, 30]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_ints")
    assert list(df["counts"].iloc[0]) == [10, 20, 30]


def test_f8_03_1d_tuple_of_scalars_logged(tmp_path: Path):
    """Verify 1D tuple of scalars is accepted and logged as list."""
    with Logger(run_name="arr_tuple", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "coords": (1.5, 2.5, 3.5)})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_tuple")
    assert list(df["coords"].iloc[0]) == [1.5, 2.5, 3.5]


def test_f8_04_empty_1d_list_logged(tmp_path: Path):
    """Verify empty 1D list is logged without error."""
    with Logger(run_name="arr_empty", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "empty_arr": []})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_empty")
    assert list(df["empty_arr"].iloc[0]) == []


def test_f8_05_1d_array_preserved_in_dataframe_without_exploding_rows(tmp_path: Path):
    """Verify 1D list is stored as object in DataFrame and does not explode row count."""
    with Logger(run_name="arr_no_explode", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "losses": [0.5, 0.4, 0.3, 0.2]})
        logger.log({"step": 2, "losses": [0.19, 0.18, 0.17, 0.16]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_no_explode")
    assert len(df) == 2
    assert len(df["losses"].iloc[0]) == 4
    assert len(df["losses"].iloc[1]) == 4


# ==============================================================================
# Feature 9: NaN/Infinity Serialization (ADR 42)
# ==============================================================================

def test_f9_01_math_nan_serialized_as_string_nan(tmp_path: Path):
    """Verify math.nan is serialized as string 'NaN' in raw JSONL file."""
    with Logger(run_name="nan_raw", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": math.nan})

    log_file = tmp_path / "nan_raw" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["loss"] == "NaN"


def test_f9_02_math_inf_serialized_as_pos_inf_string(tmp_path: Path):
    """Verify math.inf is serialized as '+Infinity' in raw JSONL file."""
    with Logger(run_name="pos_inf_raw", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "grad_norm": math.inf})

    log_file = tmp_path / "pos_inf_raw" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["grad_norm"] == "+Infinity"


def test_f9_03_negative_inf_serialized_as_neg_inf_string(tmp_path: Path):
    """Verify -math.inf is serialized as '-Infinity' in raw JSONL file."""
    with Logger(run_name="neg_inf_raw", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "bound": -math.inf})

    log_file = tmp_path / "neg_inf_raw" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["bound"] == "-Infinity"


def test_f9_04_nan_inside_1d_array_serialized(tmp_path: Path):
    """Verify math.nan inside 1D list is serialized as string 'NaN'."""
    with Logger(run_name="arr_nan_raw", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "vals": [1.0, math.nan, 2.0]})

    log_file = tmp_path / "arr_nan_raw" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["vals"] == [1.0, "NaN", 2.0]


def test_f9_05_infinities_inside_1d_array_serialized(tmp_path: Path):
    """Verify math.inf and -math.inf inside 1D list are serialized to '+Infinity' and '-Infinity'."""
    with Logger(run_name="arr_infs_raw", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "limits": [math.inf, -math.inf]})

    log_file = tmp_path / "arr_infs_raw" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    assert data["limits"] == ["+Infinity", "-Infinity"]


# ==============================================================================
# Feature 10: Timestamp Injection (ADR 40)
# ==============================================================================

def test_f10_01_timestamp_present_in_logged_record(tmp_path: Path):
    """Verify _timestamp is automatically injected and is a positive epoch float."""
    start = time.time()
    with Logger(run_name="ts_inject", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    end = time.time()

    log_file = tmp_path / "ts_inject" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.loads(f.readlines()[1])
    assert "_timestamp" in data
    assert isinstance(data["_timestamp"], (int, float))
    assert start - 1.0 <= data["_timestamp"] <= end + 1.0


def test_f10_02_time_since_start_present_in_logged_record(tmp_path: Path):
    """Verify _time_since_start is automatically injected and is a non-negative float."""
    with Logger(run_name="tss_inject", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})

    log_file = tmp_path / "tss_inject" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.loads(f.readlines()[1])
    assert "_time_since_start" in data
    assert isinstance(data["_time_since_start"], (int, float))
    assert data["_time_since_start"] >= 0.0


def test_f10_03_timestamp_monotonicity(tmp_path: Path):
    """Verify _timestamp is monotonically non-decreasing across consecutive calls."""
    with Logger(run_name="ts_mono", log_dir=tmp_path) as logger:
        for i in range(3):
            logger.log({"step": i, "val": i})
            time.sleep(0.01)

    log_file = tmp_path / "ts_mono" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()[1:]]
    timestamps = [r["_timestamp"] for r in records]
    assert timestamps[0] <= timestamps[1] <= timestamps[2]


def test_f10_04_time_since_start_monotonicity(tmp_path: Path):
    """Verify _time_since_start is monotonically non-decreasing across calls."""
    with Logger(run_name="tss_mono", log_dir=tmp_path) as logger:
        for i in range(3):
            logger.log({"step": i, "val": i})
            time.sleep(0.01)

    log_file = tmp_path / "tss_mono" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()[1:]]
    deltas = [r["_time_since_start"] for r in records]
    assert deltas[0] <= deltas[1] <= deltas[2]


def test_f10_05_timestamps_reflected_in_dataframe(tmp_path: Path):
    """Verify LogReader outputs DataFrame with _timestamp and _time_since_start columns."""
    with Logger(run_name="ts_df", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.1})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ts_df")
    assert "_timestamp" in df.columns
    assert "_time_since_start" in df.columns


# ==============================================================================
# Feature 11: Synchronous Flush (ADR 34)
# ==============================================================================

def test_f11_01_file_exists_immediately_on_init(tmp_path: Path):
    """Verify log.jsonl exists immediately upon Logger init without calling close()."""
    logger = Logger(run_name="sync_init", log_dir=tmp_path)
    log_file = tmp_path / "sync_init" / "log.jsonl"
    assert log_file.exists()
    logger.close()


def test_f11_02_metric_written_immediately_without_close(tmp_path: Path):
    """Verify metric line is written and readable on disk immediately after log(), before close()."""
    logger = Logger(run_name="sync_metric", log_dir=tmp_path)
    logger.log({"step": 1, "loss": 0.99})

    log_file = tmp_path / "sync_metric" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["loss"] == 0.99
    logger.close()


def test_f11_03_multiple_logs_immediately_flushed(tmp_path: Path):
    """Verify every consecutive log call flushes to disk synchronously."""
    logger = Logger(run_name="sync_mult", log_dir=tmp_path)
    log_file = tmp_path / "sync_mult" / "log.jsonl"

    for step in range(1, 4):
        logger.log({"step": step, "metric": step * 100})
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1 + step  # 1 header + step lines
    logger.close()


def test_f11_04_large_payload_synchronously_persisted(tmp_path: Path):
    """Verify a large array metric is immediately flushed to disk synchronously."""
    large_arr = list(range(200))
    logger = Logger(run_name="sync_large", log_dir=tmp_path)
    logger.log({"step": 1, "arr": large_arr})

    log_file = tmp_path / "sync_large" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert json.loads(lines[1])["arr"] == large_arr
    logger.close()


def test_f11_05_file_descriptor_closed_on_close(tmp_path: Path):
    """Verify close() cleanly closes resources and file remains readable."""
    logger = Logger(run_name="sync_close", log_dir=tmp_path)
    logger.log({"step": 1, "acc": 0.5})
    logger.close()

    log_file = tmp_path / "sync_close" / "log.jsonl"
    assert log_file.exists()
    assert len(log_file.read_text(encoding="utf-8").splitlines()) == 2


# ==============================================================================
# Feature 12: Restart Mode (restart=True) (ADR 24)
# ==============================================================================

def test_f12_01_restart_appends_to_existing_run(tmp_path: Path):
    """Verify restart=True appends records to existing log.jsonl rather than creating new folder."""
    with Logger(run_name="restart_run", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    with Logger(run_name="restart_run", hyperparameters={"lr": 0.01}, log_dir=tmp_path, restart=True) as l2:
        l2.log({"step": 2, "loss": 0.4})

    dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    assert dirs[0] == "restart_run"

    log_file = tmp_path / "restart_run" / "log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    data_lines = [json.loads(line) for line in lines if json.loads(line).get("type") != "header"]
    assert len(data_lines) == 2
    assert data_lines[0]["step"] == 1
    assert data_lines[1]["step"] == 2


def test_f12_02_restart_preserves_original_metadata(tmp_path: Path):
    """Verify restart=True preserves original meta.json configuration."""
    with Logger(run_name="restart_meta", hyperparameters={"model": "vit"}, tags=["t1"], log_dir=tmp_path) as l1:
        l1.log({"step": 1, "acc": 0.8})

    with Logger(run_name="restart_meta", hyperparameters={"model": "vit"}, log_dir=tmp_path, restart=True) as l2:
        l2.log({"step": 2, "acc": 0.9})

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "restart_meta")
    assert meta["hyperparameters"]["model"] == "vit"
    assert "t1" in meta["tags"]


def test_f12_03_restart_with_exact_same_hyperparameters(tmp_path: Path):
    """Verify restarting with identical hyperparameters dict succeeds."""
    hp = {"lr": 1e-3, "bs": 64}
    l1 = Logger(run_name="restart_same_hp", hyperparameters=hp, log_dir=tmp_path)
    l1.log({"step": 1, "loss": 0.7})
    l1.close()

    l2 = Logger(run_name="restart_same_hp", hyperparameters=dict(hp), log_dir=tmp_path, restart=True)
    l2.log({"step": 2, "loss": 0.6})
    l2.close()


def test_f12_04_restart_with_none_hyperparameters_inherits(tmp_path: Path):
    """Verify restarting with hyperparameters=None inherits existing run hyperparameters without error."""
    l1 = Logger(run_name="restart_none_hp", hyperparameters={"lr": 0.05}, log_dir=tmp_path)
    l1.log({"step": 1, "loss": 0.9})
    l1.close()

    l2 = Logger(run_name="restart_none_hp", hyperparameters=None, log_dir=tmp_path, restart=True)
    l2.log({"step": 2, "loss": 0.8})
    l2.close()

    hp = LogReader(logs_dir=tmp_path).get_hyperparameters(tmp_path / "restart_none_hp")
    assert hp["lr"] == 0.05


def test_f12_05_restart_session_reflected_in_reader(tmp_path: Path):
    """Verify LogReader parses multiple sessions of a restarted run into a contiguous DataFrame."""
    with Logger(run_name="restart_df", log_dir=tmp_path) as l1:
        l1.log({"step": 1, "val": 10})
        l1.log({"step": 2, "val": 20})

    with Logger(run_name="restart_df", log_dir=tmp_path, restart=True) as l2:
        l2.log({"step": 3, "val": 30})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "restart_df")
    assert list(df["step"]) == [1, 2, 3]
    assert list(df["val"]) == [10, 20, 30]


# ==============================================================================
# Feature 13: Python & Package State Capture (R2, ADR 31)
# ==============================================================================

def test_f13_01_capture_environment_returns_dict():
    """Verify capture_environment() returns a dictionary."""
    env = capture_environment()
    assert isinstance(env, dict)


def test_f13_02_capture_environment_python_version():
    """Verify python_version in capture_environment() matches sys.version."""
    env = capture_environment()
    assert "python_version" in env
    assert str(sys.version_info.major) in env["python_version"]


def test_f13_03_capture_environment_packages_dict():
    """Verify packages in capture_environment() is a mapping of package names to versions."""
    env = capture_environment()
    assert "packages" in env
    assert isinstance(env["packages"], dict)


def test_f13_04_capture_environment_sys_platform():
    """Verify sys_platform in capture_environment() matches sys.platform."""
    env = capture_environment()
    assert "sys_platform" in env
    assert env["sys_platform"] == sys.platform


def test_f13_05_environment_captured_in_meta_json(tmp_path: Path):
    """Verify Logger(capture_env=True) saves environment dictionary in meta.json."""
    logger = Logger(run_name="env_meta_test", capture_env=True, log_dir=tmp_path)
    logger.close()

    meta = LogReader(logs_dir=tmp_path).get_metadata(tmp_path / "env_meta_test")
    assert "env" in meta or "environment" in meta
    env_data = meta.get("env") or meta.get("environment")
    assert "python_version" in env_data


# ==============================================================================
# Feature 14: Git Commit Hash Capture (R2, ADR 31)
# ==============================================================================

def test_f14_01_capture_git_info_returns_dict(temp_git_repo: Path):
    """Verify capture_git_info returns a dict with git_commit, git_branch, and is_dirty."""
    git_info = capture_git_info(cwd=temp_git_repo)
    assert isinstance(git_info, dict)
    assert "git_commit" in git_info
    assert "git_branch" in git_info
    assert "is_dirty" in git_info


def test_f14_02_capture_git_commit_sha_format(temp_git_repo: Path):
    """Verify git_commit is a 40-character hex string when in a valid git repository."""
    git_info = capture_git_info(cwd=temp_git_repo)
    sha = git_info["git_commit"]
    assert isinstance(sha, str)
    assert len(sha) == 40
    assert re.match(r"^[0-9a-f]{40}$", sha) is not None


def test_f14_03_capture_git_branch_name(temp_git_repo: Path):
    """Verify git_branch is non-empty string in active git repository."""
    git_info = capture_git_info(cwd=temp_git_repo)
    branch = git_info["git_branch"]
    assert isinstance(branch, str)
    assert len(branch) > 0


def test_f14_04_capture_git_clean_status(temp_git_repo: Path):
    """Verify is_dirty is False in clean working tree."""
    git_info = capture_git_info(cwd=temp_git_repo)
    assert git_info["is_dirty"] is False


def test_f14_05_git_info_stored_in_meta_json(temp_git_repo: Path):
    """Verify Logger initialized inside git repo records git metadata in meta.json."""
    log_dir = temp_git_repo / "logs"
    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        logger = Logger(run_name="git_meta_run", log_dir=log_dir, capture_env=True)
        logger.close()
    finally:
        os.chdir(old_cwd)

    meta = LogReader(logs_dir=log_dir).get_metadata(log_dir / "git_meta_run")
    git_data = meta.get("git") or meta
    assert git_data.get("git_commit") is not None


# ==============================================================================
# Feature 15: Uncommitted Changes diff.patch (R2, ADR 31)
# ==============================================================================

def test_f15_01_diff_patch_created_when_dirty(temp_git_repo: Path):
    """Verify create_diff_patch writes patch file and returns True when working tree is dirty."""
    # Modify tracked file
    (temp_git_repo / "initial.txt").write_text("modified content\n", encoding="utf-8")
    patch_out = temp_git_repo / "test_diff.patch"

    result = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert result is True
    assert patch_out.exists()
    assert patch_out.stat().st_size > 0


def test_f15_02_diff_patch_contains_diff_content(temp_git_repo: Path):
    """Verify diff.patch contains unified diff format indicators."""
    (temp_git_repo / "initial.txt").write_text("added uncommitted line\n", encoding="utf-8")
    patch_out = temp_git_repo / "diff.patch"
    create_diff_patch(patch_out, cwd=temp_git_repo)

    content = patch_out.read_text(encoding="utf-8")
    assert "diff --git" in content or "initial.txt" in content


def test_f15_03_diff_patch_not_created_when_clean(temp_git_repo: Path):
    """Verify create_diff_patch returns False and does not create patch when tree is clean."""
    patch_out = temp_git_repo / "clean.patch"
    result = create_diff_patch(patch_out, cwd=temp_git_repo)
    assert result is False
    assert not patch_out.exists()


def test_f15_04_logger_auto_generates_diff_patch_on_init(temp_git_repo: Path):
    """Verify Logger auto-generates <run_dir>/diff.patch when repository is dirty."""
    (temp_git_repo / "initial.txt").write_text("dirty tree modification\n", encoding="utf-8")
    log_dir = temp_git_repo / "logs"

    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        logger = Logger(run_name="dirty_run", log_dir=log_dir, capture_env=True)
        logger.close()
    finally:
        os.chdir(old_cwd)

    patch_file = log_dir / "dirty_run" / "diff.patch"
    assert patch_file.exists()
    assert patch_file.stat().st_size > 0


def test_f15_05_logger_omits_diff_patch_when_clean(temp_git_repo: Path):
    """Verify Logger does not generate diff.patch when working tree is clean."""
    log_dir = temp_git_repo / "logs"
    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        logger = Logger(run_name="clean_run", log_dir=log_dir, capture_env=True)
        logger.close()
    finally:
        os.chdir(old_cwd)

    patch_file = log_dir / "clean_run" / "diff.patch"
    assert not patch_file.exists()


# ==============================================================================
# Feature 16: LogReader DataFrame Parsing (R3)
# ==============================================================================

def test_f16_01_read_run_returns_dataframe(tmp_path: Path):
    """Verify LogReader.read_run returns a pandas DataFrame."""
    with Logger(run_name="read_df", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "read_df")
    assert isinstance(df, pd.DataFrame)


def test_f16_02_read_jsonl_direct_file(tmp_path: Path):
    """Verify LogReader.read_jsonl can parse a standalone JSONL file path."""
    with Logger(run_name="read_file", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.3})
    jsonl_path = tmp_path / "read_file" / "log.jsonl"
    df = LogReader(logs_dir=tmp_path).read_jsonl(jsonl_path, run_name="read_file")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_f16_03_dataframe_row_count_matches_logged_steps(tmp_path: Path):
    """Verify parsed DataFrame row count matches exact number of logged records."""
    count = 7
    with Logger(run_name="read_count", log_dir=tmp_path) as logger:
        for i in range(count):
            logger.log({"step": i, "acc": i / count})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "read_count")
    assert len(df) == count


def test_f16_04_dataframe_contains_metrics_columns(tmp_path: Path):
    """Verify all logged metric keys are present as columns in the DataFrame."""
    with Logger(run_name="read_cols", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "train_loss": 0.4, "val_loss": 0.6, "custom_metric": 42})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "read_cols")
    for col in ["step", "train_loss", "val_loss", "custom_metric"]:
        assert col in df.columns


def test_f16_05_dataframe_contains_run_name_column(tmp_path: Path):
    """Verify run_name column is injected into the DataFrame."""
    name = "named_run_xyz"
    with Logger(run_name=name, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.1})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / name)
    assert "run_name" in df.columns
    assert (df["run_name"] == name).all()


# ==============================================================================
# Feature 17: Corrupted Line Tolerance (ADR 43)
# ==============================================================================

def test_f17_01_skips_corrupted_json_line(tmp_path: Path):
    """Verify LogReader skips corrupted JSON line and parses remaining valid lines."""
    run_dir = tmp_path / "corrupt_test"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": 0.5}\n')
        f.write('{"step": 2, CORRUPTED_GARBAGE\n')
        f.write('{"step": 3, "loss": 0.3}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 2
    assert list(df["step"]) == [1, 3]


def test_f17_02_emits_warning_on_corrupted_line(tmp_path: Path):
    """Verify LogReader emits a warning when encountering a corrupted line."""
    run_dir = tmp_path / "warn_corrupt"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": 0.5}\n')
        f.write('{"step": 2, INVALID\n')

    with pytest.warns(UserWarning):
        LogReader(logs_dir=tmp_path).read_run(run_dir)


def test_f17_03_skips_empty_and_whitespace_lines(tmp_path: Path):
    """Verify blank and whitespace lines are skipped cleanly without error."""
    run_dir = tmp_path / "blank_lines"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('\n')
        f.write('{"step": 1, "loss": 0.5}\n')
        f.write('   \t  \n')
        f.write('{"step": 2, "loss": 0.4}\n')
        f.write('\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 2


def test_f17_04_skips_truncated_final_line(tmp_path: Path):
    """Verify truncated final line (simulating Ctrl+C write interruption) is safely skipped."""
    run_dir = tmp_path / "trunc_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": 0.5}\n')
        f.write('{"step": 2, "loss":')  # partial write

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


def test_f17_05_skips_non_dict_json_record(tmp_path: Path):
    """Verify non-dict JSON records (e.g. array or raw string) are skipped."""
    run_dir = tmp_path / "nondict_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('[1, 2, 3]\n')
        f.write('"simple string"\n')
        f.write('{"step": 1, "loss": 0.2}\n')

    df = LogReader(logs_dir=tmp_path).read_run(run_dir)
    assert len(df) == 1
    assert df["step"].iloc[0] == 1


# ==============================================================================
# Feature 18: Special Float Deserialization (ADR 42)
# ==============================================================================

def test_f18_01_nan_string_deserialized_to_float_nan(tmp_path: Path):
    """Verify string 'NaN' in JSONL is deserialized to float nan in DataFrame."""
    with Logger(run_name="deser_nan", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": math.nan})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "deser_nan")
    assert math.isnan(df["loss"].iloc[0])


def test_f18_02_pos_inf_string_deserialized_to_pos_inf(tmp_path: Path):
    """Verify string '+Infinity' in JSONL is deserialized to positive infinity."""
    with Logger(run_name="deser_pos_inf", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "val": math.inf})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "deser_pos_inf")
    val = df["val"].iloc[0]
    assert math.isinf(val) and val > 0


def test_f18_03_neg_inf_string_deserialized_to_neg_inf(tmp_path: Path):
    """Verify string '-Infinity' in JSONL is deserialized to negative infinity."""
    with Logger(run_name="deser_neg_inf", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "val": -math.inf})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "deser_neg_inf")
    val = df["val"].iloc[0]
    assert math.isinf(val) and val < 0


def test_f18_04_nan_inside_1d_array_deserialized(tmp_path: Path):
    """Verify 'NaN' inside 1D array is deserialized to float nan."""
    with Logger(run_name="deser_arr_nan", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "arr": [1.0, math.nan, 2.0]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "deser_arr_nan")
    arr = df["arr"].iloc[0]
    assert arr[0] == 1.0
    assert math.isnan(arr[1])
    assert arr[2] == 2.0


def test_f18_05_infs_inside_1d_array_deserialized(tmp_path: Path):
    """Verify '+Infinity' and '-Infinity' inside 1D array are deserialized to inf floats."""
    with Logger(run_name="deser_arr_infs", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "arr": [math.inf, -math.inf]})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "deser_arr_infs")
    arr = df["arr"].iloc[0]
    assert math.isinf(arr[0]) and arr[0] > 0
    assert math.isinf(arr[1]) and arr[1] < 0


# ==============================================================================
# Feature 19: Hyperparameter Prefixing (View DSL)
# ==============================================================================

def test_f19_01_hyperparameters_prefixed_in_dataframe(tmp_path: Path):
    """Verify hyperparameters are prefixed with 'hyperparameters.<key>' in DataFrame."""
    hp = {"lr": 0.001, "batch_size": 64}
    with Logger(run_name="prefix_test", hyperparameters=hp, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "prefix_test")
    assert "hyperparameters.lr" in df.columns
    assert "hyperparameters.batch_size" in df.columns
    assert df["hyperparameters.lr"].iloc[0] == 0.001
    assert df["hyperparameters.batch_size"].iloc[0] == 64


def test_f19_02_hyperparameters_broadcast_to_all_rows(tmp_path: Path):
    """Verify hyperparameters are broadcast to all rows in the parsed DataFrame."""
    hp = {"optimizer": "adam"}
    with Logger(run_name="prefix_broadcast", hyperparameters=hp, log_dir=tmp_path) as logger:
        for i in range(4):
            logger.log({"step": i, "loss": 1.0 / (i + 1)})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "prefix_broadcast")
    assert len(df) == 4
    assert (df["hyperparameters.optimizer"] == "adam").all()


def test_f19_03_nested_hyperparameters_flattened_or_prefixed(tmp_path: Path):
    """Verify nested hyperparameters are accessible with prefixed names."""
    hp = {"model": {"arch": "resnet50", "pretrained": True}}
    with Logger(run_name="prefix_nested", hyperparameters=hp, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.4})
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "prefix_nested")
    # Either hyperparameters.model.arch or hyperparameters.model dict
    assert (
        "hyperparameters.model.arch" in df.columns
        or "hyperparameters.model" in df.columns
    )


def test_f19_04_get_hyperparameters_helper(tmp_path: Path):
    """Verify LogReader.get_hyperparameters returns dictionary directly."""
    hp = {"hidden_dim": 256, "dropout": 0.1}
    logger = Logger(run_name="helper_hp", hyperparameters=hp, log_dir=tmp_path)
    logger.close()

    reader = LogReader(logs_dir=tmp_path)
    retrieved = reader.get_hyperparameters(tmp_path / "helper_hp")
    assert retrieved == hp


def test_f19_05_get_metadata_helper(tmp_path: Path):
    """Verify LogReader.get_metadata returns the entire metadata dict."""
    logger = Logger(run_name="helper_meta", hyperparameters={"seed": 42}, tags=["prod"], log_dir=tmp_path)
    logger.close()

    reader = LogReader(logs_dir=tmp_path)
    metadata = reader.get_metadata(tmp_path / "helper_meta")
    assert metadata["run_name"] == "helper_meta"
    assert metadata["hyperparameters"] == {"seed": 42}
    assert metadata["tags"] == ["prod"]


# ==============================================================================
# Feature 20: Multi-Run Discovery & Evolution (ADR 12, 20)
# ==============================================================================

def test_f20_01_read_all_discovers_multiple_runs(tmp_path: Path):
    """Verify LogReader.read_all aggregates multiple runs in the logs directory."""
    for i in range(3):
        with Logger(run_name=f"run_{i}", hyperparameters={"seed": i}, log_dir=tmp_path) as l:
            l.log({"step": 1, "loss": 0.1 * i})

    reader = LogReader(logs_dir=tmp_path)
    df = reader.read_all()
    assert len(df) == 3
    assert set(df["run_name"]) == {"run_0", "run_1", "run_2"}


def test_f20_02_read_all_preserves_distinct_run_names(tmp_path: Path):
    """Verify read_all retains correct run_name association for each row."""
    with Logger(run_name="alpha", log_dir=tmp_path) as l:
        l.log({"step": 1, "metric": 100})
    with Logger(run_name="beta", log_dir=tmp_path) as l:
        l.log({"step": 1, "metric": 200})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert df.loc[df["run_name"] == "alpha", "metric"].iloc[0] == 100
    assert df.loc[df["run_name"] == "beta", "metric"].iloc[0] == 200


def test_f20_03_schema_evolution_missing_hyperparameters_become_nan(tmp_path: Path):
    """Verify schema evolution: runs missing a hyperparameter get NaN in merged DataFrame."""
    with Logger(run_name="v1_run", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    with Logger(run_name="v2_run", hyperparameters={"lr": 0.01, "weight_decay": 1e-4}, log_dir=tmp_path) as l2:
        l2.log({"step": 1, "loss": 0.4})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert "hyperparameters.weight_decay" in df.columns
    v1_wd = df.loc[df["run_name"] == "v1_run", "hyperparameters.weight_decay"].iloc[0]
    assert pd.isna(v1_wd)
    v2_wd = df.loc[df["run_name"] == "v2_run", "hyperparameters.weight_decay"].iloc[0]
    assert v2_wd == 1e-4


def test_f20_04_schema_evolution_disjoint_metrics(tmp_path: Path):
    """Verify schema evolution for metrics: missing metrics across runs get NaN."""
    with Logger(run_name="train_only", log_dir=tmp_path) as l1:
        l1.log({"step": 1, "train_loss": 0.5})

    with Logger(run_name="eval_only", log_dir=tmp_path) as l2:
        l2.log({"step": 1, "eval_accuracy": 0.92})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert "train_loss" in df.columns
    assert "eval_accuracy" in df.columns
    assert pd.isna(df.loc[df["run_name"] == "eval_only", "train_loss"].iloc[0])
    assert pd.isna(df.loc[df["run_name"] == "train_only", "eval_accuracy"].iloc[0])


def test_f20_05_read_all_empty_directory_returns_empty_dataframe(tmp_path: Path):
    """Verify read_all on empty directory returns an empty DataFrame without raising exception."""
    empty_dir = tmp_path / "empty_logs"
    empty_dir.mkdir()
    df = LogReader(logs_dir=empty_dir).read_all()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
