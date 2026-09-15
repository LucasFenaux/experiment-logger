"""
Unit test suite for LogReader and DataFrame engine in experiment_logger.

Governed by ADR 12, 16, 20, 24, 29, 32, 40, 42, 43, 44.
Enforces strict TDD (Red-Green-Refactor) for Milestone 3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import pytest

from experiment_logger import constants
from experiment_logger.reader import LogReader


# -----------------------------------------------------------------------------
# 1. Initialization Tests
# -----------------------------------------------------------------------------
class TestLogReaderInit:
    def test_init_default_logs_dir(self) -> None:
        reader = LogReader()
        assert reader.logs_dir == Path("logs")
        assert isinstance(reader.logs_dir, Path)

    def test_init_custom_str_logs_dir(self) -> None:
        reader = LogReader(logs_dir="my_custom_logs")
        assert reader.logs_dir == Path("my_custom_logs")
        assert isinstance(reader.logs_dir, Path)

    def test_init_custom_path_logs_dir(self, tmp_path: Path) -> None:
        custom_path = tmp_path / "experiments"
        reader = LogReader(logs_dir=custom_path)
        assert reader.logs_dir == custom_path


# -----------------------------------------------------------------------------
# 2. Metadata Helper Tests
# -----------------------------------------------------------------------------
class TestGetMetadata:
    def test_get_metadata_valid_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_01"
        run_dir.mkdir(parents=True)
        meta_payload = {
            "run_name": "run_01",
            "created_at": "2026-09-14T12:00:00Z",
            "hyperparameters": {"lr": 0.001, "batch_size": 32},
            "tags": ["baseline", "vision"],
        }
        (run_dir / constants.META_FILE).write_text(json.dumps(meta_payload), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        meta = reader.get_metadata(run_dir)
        assert meta["run_name"] == "run_01"
        assert meta["hyperparameters"]["lr"] == 0.001
        assert meta["tags"] == ["baseline", "vision"]

    def test_get_metadata_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        reader = LogReader(logs_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_metadata(tmp_path / "does_not_exist")

    def test_get_metadata_missing_meta_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir(parents=True)
        reader = LogReader(logs_dir=tmp_path)
        meta = reader.get_metadata(run_dir)
        assert meta == {}

    def test_get_metadata_corrupted_meta_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "corrupt_meta_run"
        run_dir.mkdir(parents=True)
        (run_dir / constants.META_FILE).write_text("CORRUPTED { INVALID JSON", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with pytest.warns(UserWarning):
            meta = reader.get_metadata(run_dir)
        assert meta == {}

    def test_get_metadata_fallback_hyperparameters_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "legacy_run"
        run_dir.mkdir(parents=True)
        (run_dir / "hyperparameters.json").write_text(json.dumps({"lr": 0.01}), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        meta = reader.get_metadata(run_dir)
        assert meta == {"lr": 0.01}

    def test_get_metadata_relative_to_logs_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logs" / "relative_run"
        run_dir.mkdir(parents=True)
        (run_dir / constants.META_FILE).write_text(json.dumps({"run_name": "rel"}), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path / "logs")
        meta = reader.get_metadata("relative_run")
        assert meta["run_name"] == "rel"


# -----------------------------------------------------------------------------
# 3. Hyperparameters Helper Tests
# -----------------------------------------------------------------------------
class TestGetHyperparameters:
    def test_get_hyperparameters_valid(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_hp"
        run_dir.mkdir(parents=True)
        meta_payload = {
            "run_name": "run_hp",
            "hyperparameters": {"optimizer": "AdamW", "lr": 1e-4},
        }
        (run_dir / constants.META_FILE).write_text(json.dumps(meta_payload), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        hp = reader.get_hyperparameters(run_dir)
        assert hp == {"optimizer": "AdamW", "lr": 1e-4}

    def test_get_hyperparameters_empty(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_empty_hp"
        run_dir.mkdir(parents=True)
        (run_dir / constants.META_FILE).write_text(json.dumps({"hyperparameters": {}}), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        hp = reader.get_hyperparameters(run_dir)
        assert hp == {}

    def test_get_hyperparameters_nested(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_nested_hp"
        run_dir.mkdir(parents=True)
        nested = {"model": {"arch": "resnet50", "layers": 50}, "lr": 0.01}
        (run_dir / constants.META_FILE).write_text(json.dumps({"hyperparameters": nested}), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        hp = reader.get_hyperparameters(run_dir)
        assert hp == nested
        assert hp["model"]["arch"] == "resnet50"

    def test_get_hyperparameters_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        reader = LogReader(logs_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_hyperparameters(tmp_path / "missing_dir")

    def test_get_hyperparameters_missing_meta(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_meta"
        run_dir.mkdir(parents=True)
        reader = LogReader(logs_dir=tmp_path)
        hp = reader.get_hyperparameters(run_dir)
        assert hp == {}

    def test_get_hyperparameters_corrupted_meta(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "corrupt_meta"
        run_dir.mkdir(parents=True)
        (run_dir / constants.META_FILE).write_text("{bad json...", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with pytest.warns(UserWarning):
            hp = reader.get_hyperparameters(run_dir)
        assert hp == {}

    def test_get_hyperparameters_fallback_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "fallback_hp"
        run_dir.mkdir(parents=True)
        (run_dir / "hyperparameters.json").write_text(json.dumps({"batch_size": 64}), encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        hp = reader.get_hyperparameters(run_dir)
        assert hp == {"batch_size": 64}


# -----------------------------------------------------------------------------
# 4. Hyperparameter Prefixing and Broadcasting Tests
# -----------------------------------------------------------------------------
class TestHyperparameterPrefixing:
    def test_prefix_flat_hyperparameters(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_flat"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"lr": 0.001, "batch_size": 32}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 100.0, "_time_since_start": 0.1}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.4, "_timestamp": 101.0, "_time_since_start": 1.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "hyperparameters.lr" in df.columns
        assert "hyperparameters.batch_size" in df.columns
        assert (df["hyperparameters.lr"] == 0.001).all()
        assert (df["hyperparameters.batch_size"] == 32).all()

    def test_prefix_broadcast_all_rows(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_broadcast"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"optimizer": "adam"}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            for step in range(1, 6):
                f.write(json.dumps({"step": step, "loss": 1.0 / step, "_timestamp": float(step), "_time_since_start": float(step)}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert len(df) == 5
        assert (df["hyperparameters.optimizer"] == "adam").all()

    def test_prefix_nested_dict_flattening(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_nested"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"model": {"arch": "resnet50", "depth": 50}}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "hyperparameters.model.arch" in df.columns or "hyperparameters.model" in df.columns
        if "hyperparameters.model.arch" in df.columns:
            assert df["hyperparameters.model.arch"].iloc[0] == "resnet50"
            assert df["hyperparameters.model.depth"].iloc[0] == 50

    def test_prefix_deeply_nested_flattening(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_deep"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"a": {"b": {"c": 42}}}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "hyperparameters.a.b.c" in df.columns
        assert df["hyperparameters.a.b.c"].iloc[0] == 42

    def test_prefix_keys_with_dots(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_dots"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"model.backbone": "resnet50"}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "hyperparameters.model.backbone" in df.columns
        assert df["hyperparameters.model.backbone"].iloc[0] == "resnet50"

    def test_prefix_list_hyperparameters(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_list_hp"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"layer_dims": [64, 128, 256]}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.4, "_timestamp": 2.0, "_time_since_start": 1.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "hyperparameters.layer_dims" in df.columns
        assert len(df) == 2
        assert df["hyperparameters.layer_dims"].iloc[0] == [64, 128, 256]
        assert df["hyperparameters.layer_dims"].iloc[1] == [64, 128, 256]

    def test_prefix_special_floats_deserialization(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_special_hp"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"clip_grad": "+Infinity", "threshold": "NaN"}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert math.isinf(df["hyperparameters.clip_grad"].iloc[0])
        assert df["hyperparameters.clip_grad"].iloc[0] > 0
        assert math.isnan(df["hyperparameters.threshold"].iloc[0])

    def test_prefix_name_collision_immunity(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_collision"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {"loss": "cross_entropy"}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.42, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert "loss" in df.columns
        assert "hyperparameters.loss" in df.columns
        assert df["loss"].iloc[0] == 0.42
        assert df["hyperparameters.loss"].iloc[0] == "cross_entropy"

    def test_prefix_empty_hyperparameters_no_cols(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_empty_meta"
        run_dir.mkdir(parents=True)
        meta = {"hyperparameters": {}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        hp_cols = [c for c in df.columns if c.startswith(constants.HYPERPARAMETER_PREFIX)]
        assert len(hp_cols) == 0

    def test_prefix_corrupted_meta_resilience(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_bad_meta"
        run_dir.mkdir(parents=True)
        (run_dir / constants.META_FILE).write_text("NOT VALID JSON", encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        with pytest.warns(UserWarning):
            df = reader.read_run(run_dir)
        assert len(df) == 1
        assert df["loss"].iloc[0] == 0.5


# -----------------------------------------------------------------------------
# 5. Standalone JSONL Parsing Core Tests
# -----------------------------------------------------------------------------
class TestReadJsonlCore:
    def test_read_jsonl_basic(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "test.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.3, "_timestamp": 2.0, "_time_since_start": 1.1}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]
        assert "run_name" in df.columns

    def test_read_jsonl_run_name_injection(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "experiment.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl, run_name="custom_run")
        assert (df["run_name"] == "custom_run").all()

    def test_read_jsonl_header_line_skipped(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "with_header.jsonl"
        header = {"mlviz_version": "1.0", "type": "header", "run_name": "header_run"}
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.8}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert len(df) == 1
        assert "mlviz_version" not in df.columns
        assert "type" not in df.columns
        assert df["run_name"].iloc[0] == "header_run"

    def test_read_jsonl_midfile_headers_skipped(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "resumed.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"mlviz_version": "1.0", "type": "header", "run_name": "resumed"}) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.9}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.8}) + "\n")
            # Resumed header in middle of file
            f.write(json.dumps({"mlviz_version": "1.0", "type": "header", "run_name": "resumed"}) + "\n")
            f.write(json.dumps({"step": 3, "loss": 0.7}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert len(df) == 3
        assert list(df["step"]) == [1, 2, 3]

    def test_read_jsonl_corrupted_line_skip_and_warn(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "corrupt.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            f.write('{"step": 2, CORRUPTED_GARBAGE\n')
            f.write(json.dumps({"step": 3, "loss": 0.3}) + "\n")

        reader = LogReader()
        with pytest.warns(UserWarning):
            df = reader.read_jsonl(jsonl)
        assert len(df) == 2
        assert list(df["step"]) == [1, 3]

    def test_read_jsonl_non_dict_skip_and_warn(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "non_dict.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            f.write("[1, 2, 3]\n")
            f.write('"simple string"\n')
            f.write("42\n")
            f.write(json.dumps({"step": 2, "loss": 0.4}) + "\n")

        reader = LogReader()
        with pytest.warns(UserWarning):
            df = reader.read_jsonl(jsonl)
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]

    def test_read_jsonl_whitespace_lines_ignored(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "whitespace.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write("   \t  \n")
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            f.write("    \n")
            f.write(json.dumps({"step": 2, "loss": 0.4}) + "\n")
            f.write("\n\n")

        reader = LogReader()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            df = reader.read_jsonl(jsonl)
            # Whitespace must not trigger warnings
            assert len(record) == 0
        assert len(df) == 2
        assert list(df["step"]) == [1, 2]

    def test_read_jsonl_special_floats_deserialization(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "special_floats.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "nan_val": "NaN", "pos_inf": "+Infinity", "neg_inf": "-Infinity"}) + "\n")
            f.write(json.dumps({"step": 2, "nan_val": "nan", "pos_inf": "Infinity", "neg_inf": "-inf"}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert math.isnan(df["nan_val"].iloc[0])
        assert math.isnan(df["nan_val"].iloc[1])
        assert math.isinf(df["pos_inf"].iloc[0]) and df["pos_inf"].iloc[0] > 0
        assert math.isinf(df["pos_inf"].iloc[1]) and df["pos_inf"].iloc[1] > 0
        assert math.isinf(df["neg_inf"].iloc[0]) and df["neg_inf"].iloc[0] < 0
        assert math.isinf(df["neg_inf"].iloc[1]) and df["neg_inf"].iloc[1] < 0
        assert df["nan_val"].dtype == float
        assert df["pos_inf"].dtype == float
        assert df["neg_inf"].dtype == float

    def test_read_jsonl_1d_array_special_floats(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "array_floats.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "arr": [1.0, "NaN", "+Infinity", "-Infinity"]}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert len(df) == 1
        arr = df["arr"].iloc[0]
        assert len(arr) == 4
        assert arr[0] == 1.0
        assert math.isnan(arr[1])
        assert math.isinf(arr[2]) and arr[2] > 0
        assert math.isinf(arr[3]) and arr[3] < 0

    def test_read_jsonl_plain_strings_preserved(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "strings.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "arch": "resnet50", "split": "train"}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert df["arch"].iloc[0] == "resnet50"
        assert df["split"].iloc[0] == "train"
        assert isinstance(df["arch"].iloc[0], str)

    def test_read_jsonl_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.touch()

        reader = LogReader()
        df = reader.read_jsonl(empty)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_jsonl_header_only_file(self, tmp_path: Path) -> None:
        header_file = tmp_path / "header_only.jsonl"
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(header_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_jsonl_nonexistent_file_raises(self, tmp_path: Path) -> None:
        reader = LogReader()
        with pytest.raises(FileNotFoundError):
            reader.read_jsonl(tmp_path / "nonexistent.jsonl")

    def test_read_jsonl_null_bytes_handled(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "null_bytes.jsonl"
        with open(jsonl, "wb") as f:
            f.write(b'{"step": 1, "loss": 0.5}\n')
            f.write(b'{"step": 2, \x00\x00\x00}\n')
            f.write(b'{"step": 3, "loss": 0.2}\n')

        reader = LogReader()
        with pytest.warns(UserWarning):
            df = reader.read_jsonl(jsonl)
        assert len(df) == 2
        assert list(df["step"]) == [1, 3]

    def test_read_jsonl_restart_crash_subline_recovery(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "restart_crash.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            # Unterminated line concatenated directly with session 2 line without newline
            f.write('{"step": 2, "loss": 0.4, "chec{"step": 2, "loss": 0.38}\n')
            f.write(json.dumps({"step": 3, "loss": 0.25}) + "\n")

        reader = LogReader()
        with pytest.warns(UserWarning):
            df = reader.read_jsonl(jsonl)
        assert len(df) == 3
        assert list(df["step"]) == [1, 2, 3]
        assert df["loss"].iloc[1] == 0.38


# -----------------------------------------------------------------------------
# 6. Single Run Directory Parsing Tests (read_run)
# -----------------------------------------------------------------------------
class TestReadRun:
    def test_read_run_basic(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)
        meta = {"run_name": "run_test", "hyperparameters": {"lr": 0.001}}
        (run_dir / constants.META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.5, "_timestamp": 10.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert len(df) == 1
        assert df["run_name"].iloc[0] == "run_test"
        assert df["hyperparameters.lr"].iloc[0] == 0.001

    def test_read_run_missing_meta_success(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_no_meta"
        run_dir.mkdir(parents=True)

        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert len(df) == 1
        assert df["run_name"].iloc[0] == "run_no_meta"
        assert df["loss"].iloc[0] == 0.5

    def test_read_run_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        reader = LogReader(logs_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.read_run(tmp_path / "missing_dir")

    def test_read_run_missing_log_file_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_log_file"
        run_dir.mkdir(parents=True)

        reader = LogReader(logs_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.read_run(run_dir)

    def test_read_run_empty_log_returns_empty_df(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_log_run"
        run_dir.mkdir(parents=True)
        (run_dir / constants.LOG_FILE).touch()

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_run_header_only_returns_empty_df(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "header_only_run"
        run_dir.mkdir(parents=True)
        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_run_path_formats(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "logs" / "format_run"
        run_dir.mkdir(parents=True)
        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.1}) + "\n")

        reader = LogReader(logs_dir=tmp_path / "logs")
        df1 = reader.read_run(str(run_dir))
        df2 = reader.read_run(run_dir)
        df3 = reader.read_run("format_run")
        assert len(df1) == len(df2) == len(df3) == 1


# -----------------------------------------------------------------------------
# 7. Multi-Run Discovery Tests (read_all)
# -----------------------------------------------------------------------------
class TestReadAllMultiRun:
    def test_read_all_flat_directory(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)

        for i in (1, 2, 3):
            rd = logs_dir / f"run_{i}"
            rd.mkdir()
            (rd / constants.META_FILE).write_text(json.dumps({"run_name": f"run_{i}", "hyperparameters": {"run_id": i}}), encoding="utf-8")
            with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
                f.write(json.dumps({"step": 1, "loss": 1.0 / i, "_timestamp": 1.0, "_time_since_start": 0.1}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 3
        assert set(df["run_name"]) == {"run_1", "run_2", "run_3"}
        assert set(df["hyperparameters.run_id"]) == {1, 2, 3}

    def test_read_all_nested_subdirectories(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        r1 = logs_dir / "group_a" / "run_1"
        r2 = logs_dir / "group_b" / "sub" / "run_2"
        for rd in (r1, r2):
            rd.mkdir(parents=True)
            with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 2
        assert set(df["run_name"]) == {"run_1", "run_2"}

    def test_read_all_preserves_distinct_run_names(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        for name in ("alpha", "beta"):
            rd = logs_dir / name
            rd.mkdir(parents=True)
            with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 1, "metric": 42}) + "\n")
                f.write(json.dumps({"step": 2, "metric": 84}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 4
        assert (df[df["run_name"] == "alpha"]["metric"] == [42, 84]).all()
        assert (df[df["run_name"] == "beta"]["metric"] == [42, 84]).all()

    def test_read_all_ignores_unrelated_files(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "README.md").write_text("# Logs", encoding="utf-8")
        (logs_dir / ".DS_Store").write_text("dummy", encoding="utf-8")
        (logs_dir / "empty_dir").mkdir()

        run_dir = logs_dir / "actual_run"
        run_dir.mkdir()
        with open(run_dir / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.1}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 1
        assert df["run_name"].iloc[0] == "actual_run"

    def test_read_all_skips_hidden_dirs(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        git_run = logs_dir / ".git" / "dummy_run"
        git_run.mkdir(parents=True)
        with open(git_run / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.1}) + "\n")

        real_run = logs_dir / "real_run"
        real_run.mkdir(parents=True)
        with open(real_run / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.2}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 1
        assert df["run_name"].iloc[0] == "real_run"

    def test_read_all_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_logs"
        empty_dir.mkdir(parents=True)
        reader = LogReader(logs_dir=empty_dir)
        df = reader.read_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_all_no_runs_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "no_runs"
        d.mkdir(parents=True)
        (d / "note.txt").write_text("hello", encoding="utf-8")
        reader = LogReader(logs_dir=d)
        df = reader.read_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_all_nonexistent_directory(self, tmp_path: Path) -> None:
        reader = LogReader(logs_dir=tmp_path / "missing")
        df = reader.read_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_all_corrupted_run_resilience(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        good_run = logs_dir / "good"
        good_run.mkdir(parents=True)
        with open(good_run / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")

        bad_run = logs_dir / "bad"
        bad_run.mkdir(parents=True)
        with open(bad_run / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write("COMPLETELY CORRUPTED GARBAGE NOT JSON\n")

        reader = LogReader(logs_dir=logs_dir)
        with pytest.warns(UserWarning):
            df = reader.read_all()
        assert len(df) == 1
        assert df["run_name"].iloc[0] == "good"

    def test_read_all_accepts_str_and_path(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        rd = logs_dir / "r1"
        rd.mkdir(parents=True)
        with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "val": 10}) + "\n")

        reader = LogReader()
        df_str = reader.read_all(str(logs_dir))
        df_path = reader.read_all(logs_dir)
        assert len(df_str) == len(df_path) == 1


# -----------------------------------------------------------------------------
# 8. Schema Evolution Tests (ADR 20)
# -----------------------------------------------------------------------------
class TestSchemaEvolution:
    def test_schema_evolution_disjoint_hyperparameters(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        r1 = logs_dir / "r1"
        r1.mkdir(parents=True)
        (r1 / constants.META_FILE).write_text(json.dumps({"hyperparameters": {"lr": 0.01}}), encoding="utf-8")
        with open(r1 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")

        r2 = logs_dir / "r2"
        r2.mkdir(parents=True)
        (r2 / constants.META_FILE).write_text(json.dumps({"hyperparameters": {"weight_decay": 1e-4}}), encoding="utf-8")
        with open(r2 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 0.4}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 2
        assert "hyperparameters.lr" in df.columns
        assert "hyperparameters.weight_decay" in df.columns
        r1_row = df[df["run_name"] == "r1"].iloc[0]
        r2_row = df[df["run_name"] == "r2"].iloc[0]
        assert r1_row["hyperparameters.lr"] == 0.01
        assert pd.isna(r1_row["hyperparameters.weight_decay"])
        assert pd.isna(r2_row["hyperparameters.lr"])
        assert r2_row["hyperparameters.weight_decay"] == 1e-4

    def test_schema_evolution_disjoint_metrics(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        r1 = logs_dir / "r1"
        r1.mkdir(parents=True)
        with open(r1 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "train_loss": 0.5}) + "\n")

        r2 = logs_dir / "r2"
        r2.mkdir(parents=True)
        with open(r2 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "val_acc": 0.92}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert "train_loss" in df.columns
        assert "val_acc" in df.columns
        r1_row = df[df["run_name"] == "r1"].iloc[0]
        r2_row = df[df["run_name"] == "r2"].iloc[0]
        assert r1_row["train_loss"] == 0.5
        assert pd.isna(r1_row["val_acc"])
        assert pd.isna(r2_row["train_loss"])
        assert r2_row["val_acc"] == 0.92

    def test_schema_evolution_incremental_sweep(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        for i in range(1, 4):
            rd = logs_dir / f"sweep_{i}"
            rd.mkdir(parents=True)
            hp: dict[str, Any] = {"base_param": 1.0}
            if i >= 2:
                hp["param_v2"] = "added_in_v2"
            if i >= 3:
                hp["param_v3"] = 999
            (rd / constants.META_FILE).write_text(json.dumps({"hyperparameters": hp}), encoding="utf-8")
            with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 1, "metric": i * 10}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 3
        assert (df["hyperparameters.base_param"] == 1.0).all()
        assert pd.isna(df[df["run_name"] == "sweep_1"]["hyperparameters.param_v2"].iloc[0])
        assert df[df["run_name"] == "sweep_2"]["hyperparameters.param_v2"].iloc[0] == "added_in_v2"
        assert pd.isna(df[df["run_name"] == "sweep_2"]["hyperparameters.param_v3"].iloc[0])
        assert df[df["run_name"] == "sweep_3"]["hyperparameters.param_v3"].iloc[0] == 999

    def test_schema_evolution_dtype_integrity(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        r1 = logs_dir / "r1"
        r1.mkdir(parents=True)
        with open(r1 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "float_metric": 1.5}) + "\n")

        r2 = logs_dir / "r2"
        r2.mkdir(parents=True)
        with open(r2 / constants.LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "other_metric": 2.5}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert df["float_metric"].dtype == float

    def test_schema_evolution_index_reset(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        for name in ("r1", "r2"):
            rd = logs_dir / name
            rd.mkdir(parents=True)
            with open(rd / constants.LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 1, "val": 1}) + "\n")
                f.write(json.dumps({"step": 2, "val": 2}) + "\n")

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()
        assert len(df) == 4
        assert list(df.index) == [0, 1, 2, 3]


# -----------------------------------------------------------------------------
# 9. Adversarial Stress & Resilience Tests
# -----------------------------------------------------------------------------
class TestAdversarialStress:
    def test_large_log_file_parsing(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "large_log.jsonl"
        num_records = 2000
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            for i in range(1, num_records + 1):
                f.write(json.dumps({"step": i, "loss": 1.0 / i, "acc": i / num_records, "_timestamp": float(i), "_time_since_start": float(i)}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert len(df) == num_records
        assert df["step"].iloc[0] == 1
        assert df["step"].iloc[-1] == num_records

    def test_all_corrupted_lines_resilience(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "all_bad.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(f"GARBAGE LINE {i} NOT JSON {{{{ }}}}\n")

        reader = LogReader()
        with pytest.warns(UserWarning):
            df = reader.read_jsonl(jsonl)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_restarted_run_session_continuity(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "restarted_session.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            # Session 1
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 1, "loss": 0.8, "_timestamp": 100.0, "_time_since_start": 0.1}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.6, "_timestamp": 101.0, "_time_since_start": 1.1}) + "\n")
            # Session 2 append (restart=True)
            f.write(json.dumps(constants.DEFAULT_HEADER) + "\n")
            f.write(json.dumps({"step": 3, "loss": 0.4, "_timestamp": 200.0, "_time_since_start": 0.2}) + "\n")
            f.write(json.dumps({"step": 4, "loss": 0.2, "_timestamp": 201.0, "_time_since_start": 1.2}) + "\n")

        reader = LogReader()
        df = reader.read_jsonl(jsonl)
        assert len(df) == 4
        assert list(df["step"]) == [1, 2, 3, 4]
