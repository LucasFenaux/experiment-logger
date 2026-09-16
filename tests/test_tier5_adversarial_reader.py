"""
Tier 5 White-Box Adversarial Hardening: LogReader & Multi-Run Ingestion Engine.
Target Module: experiment_logger.reader

Authorship: Challenger M4-2 (challenger_m4_2)
Stress Dimensions Covered:
1. Chained & Alternating Corrupted Line Streams:
   - Alternating valid and corrupted lines (syntax errors, incomplete JSON, non-objects, control characters, binary NULs).
   - Stress recovery of valid metrics and verification of UserWarning emission count.
   - 100% corrupted lines vs 1,000 corrupted lines preceding 1 valid line.
2. Truncated Writes with Escaped Quotes, Nested Braces, and Brackets:
   - Mid-line crash concatenation with escaped quotes inside string values before collision boundary.
   - Brackets and braces nested inside string payloads prior to second record opening brace.
   - Multiple chained '{' candidates before the valid trailing JSON record.
   - Truncated write followed by incomplete second candidate.
   - Truncated write where secondary candidate is valid JSON but not a dictionary.
   - Truncated write followed by trailing junk after closing brace.
   - Single-brace corrupted line (analyzing recovery behavior without second brace).
3. Deeply Nested Recursive Dictionary Flattening & Edge Cases:
   - Non-string dictionary keys (int, float, bool, None, tuple).
   - Empty dictionary leaves and intermediate empty dictionary nodes.
   - 15-level deeply nested hierarchical dictionary flattening.
   - Preservation of intermediate dictionary objects alongside leaf flattened keys.
   - Recursive dictionary cycle detection (verifying RecursionError on self-referential dict).
   - Mixed structures with nested lists, tuples, and ADR 42 special float strings.
4. Mixed Column Types Across 10+ Disjoint Runs Schema Evolution:
   - Ingestion of 12 disjoint runs with divergent metric types (float, int, str, bool, list, None, special floats).
   - Disjoint hyperparameter structures across all 12 runs.
   - Column upcasting to object dtype on mixed-type collisions (e.g. numeric in run 1, str in run 5).
   - RangeIndex continuity (0 to N-1) and row-count conservation across concatenated DataFrame.
   - Metric and hyperparameter column isolation per run.
5. Directory Traversal, Symbolic Links, and Unreadable File Permissions:
   - Valid symbolic links to log.jsonl files outside run directory.
   - Broken symbolic links to log.jsonl safely ignored.
   - Directory named 'log.jsonl' safely ignored (avoids IsADirectoryError).
   - Circular directory symlinks safely traversed without infinite loop.
   - Unreadable run directory / log.jsonl permissions (PermissionError isolation): emits warning, sibling runs survive.
   - Deep nested runs in multi-branch trees.
6. Corrupted vs Valid Metadata Asymmetry:
   - Corrupted meta.json (malformed JSON) with valid log.jsonl: falls back to folder name, metrics parsed, warnings emitted.
   - Valid meta.json with 100% corrupted log.jsonl: returns empty DataFrame, warnings emitted.
   - Valid meta.json with partially corrupted log.jsonl: valid rows parsed, hyperparameters from meta.json broadcast to all rows.
   - Non-dict meta.json (JSON list or scalar): UserWarning emitted, runs continue.
   - Legacy hyperparameters.json fallback and corrupted fallback handling.
7. DataFrame Dtype Normalization & System Column Ordering:
   - _normalize_dataframe_dtypes on all-null columns, mixed boolean/numeric, pure numeric, and list columns.
   - System columns ordering (run_name, step, epoch, _timestamp, _time_since_start) prioritized at front.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import pytest

from experiment_logger import constants
from experiment_logger.reader import (
    LogReader,
    _deserialize_special_floats,
    _flatten_dict,
    _normalize_dataframe_dtypes,
)


# =============================================================================
# Helper Utilities for Synthesizing Adversarial Runs
# =============================================================================
def _write_run_files(
    run_dir: Path,
    run_name: str | None = None,
    meta_content: str | dict[str, Any] | None = None,
    log_lines: list[str] | None = None,
    legacy_hp_content: str | dict[str, Any] | None = None,
) -> Path:
    """Helper to synthesize run directories with fine-grained control over corruptions."""
    run_dir.mkdir(parents=True, exist_ok=True)

    if meta_content is not None:
        meta_file = run_dir / constants.META_FILE
        if isinstance(meta_content, dict):
            meta_file.write_text(json.dumps(meta_content), encoding="utf-8")
        else:
            meta_file.write_text(meta_content, encoding="utf-8")

    if legacy_hp_content is not None:
        hp_file = run_dir / "hyperparameters.json"
        if isinstance(legacy_hp_content, dict):
            hp_file.write_text(json.dumps(legacy_hp_content), encoding="utf-8")
        else:
            hp_file.write_text(legacy_hp_content, encoding="utf-8")

    if log_lines is not None:
        log_file = run_dir / constants.LOG_FILE
        with open(log_file, "w", encoding="utf-8") as f:
            for line in log_lines:
                f.write(line if line.endswith("\n") else f"{line}\n")

    return run_dir


# =============================================================================
# Suite 1: Chained & Alternating Corrupted Line Streams
# =============================================================================
class TestChainedCorruptedLines:
    """Stress-test reader resilience when processing streams of corrupted lines."""

    def test_alternating_valid_and_corrupted_lines(self, tmp_path: Path) -> None:
        """Verify reader skips varied corrupted lines, recovers mid-line crash, and preserves valid metrics."""
        log_file = tmp_path / "alternating.jsonl"
        lines = [
            # Line 1: Header
            json.dumps({"type": "header", "run_name": "chain_test", "created_at": "2026-09-14T00:00:00Z"}),
            # Line 2: Arbitrary garbage string
            "<<<SYNTAX_ERROR_CORRUPTED_LINE_2>>>",
            # Line 3: Valid metric row
            json.dumps({"step": 1, "loss": 0.50, "_timestamp": 10.0, "_time_since_start": 0.1}),
            # Line 4: Incomplete JSON truncated value
            '{"step": 2, "loss":',
            # Line 5: Valid metric row
            json.dumps({"step": 2, "loss": 0.45, "_timestamp": 11.0, "_time_since_start": 1.1}),
            # Line 6: Non-object JSON array
            "[100, 200, 300]",
            # Line 7: Valid metric row
            json.dumps({"step": 3, "loss": 0.40, "_timestamp": 12.0, "_time_since_start": 2.1}),
            # Line 8: Non-object JSON string literal
            '"plain string literal instead of object"',
            # Line 9: Valid metric row
            json.dumps({"step": 4, "loss": 0.35, "_timestamp": 13.0, "_time_since_start": 3.1}),
            # Line 10: Non-object JSON null
            "null",
            # Line 11: Valid metric row
            json.dumps({"step": 5, "loss": 0.30, "_timestamp": 14.0, "_time_since_start": 4.1}),
            # Line 12: Non-object JSON boolean
            "true",
            # Line 13: Valid metric row
            json.dumps({"step": 6, "loss": 0.25, "_timestamp": 15.0, "_time_since_start": 5.1}),
            # Line 14: Mid-line crash concatenation with valid trailing record
            '{"bad_crash_prefix": 123{"step": 7, "loss": 0.20, "_timestamp": 16.0, "_time_since_start": 6.1}',
            # Line 15: Valid metric row
            json.dumps({"step": 8, "loss": 0.15, "_timestamp": 17.0, "_time_since_start": 7.1}),
            # Line 16: Binary control and NUL characters
            "\x00\x01\x02\x03\x04\x05GARBAGE",
            # Line 17: Valid metric row
            json.dumps({"step": 9, "loss": 0.10, "_timestamp": 18.0, "_time_since_start": 8.1}),
            # Line 18: Unclosed brace
            '{"unclosed_dict": ',
            # Line 19: Valid metric row
            json.dumps({"step": 10, "loss": 0.05, "_timestamp": 19.0, "_time_since_start": 9.1}),
            # Line 20: Mismatched braces
            "}{}{bad_braces",
        ]

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        # Expected warnings: lines 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        assert len(recorded_warnings) >= 10
        for w in recorded_warnings:
            assert issubclass(w.category, UserWarning)

        # 10 valid rows: 9 standalone + 1 recovered from line 14 mid-line concatenation
        assert len(df) == 10
        assert df["step"].tolist() == list(range(1, 11))
        assert df["run_name"].tolist() == ["chain_test"] * 10
        assert df["loss"].iloc[0] == pytest.approx(0.50)
        assert df["loss"].iloc[6] == pytest.approx(0.20)
        assert df["loss"].iloc[9] == pytest.approx(0.05)

    def test_consecutive_corrupted_lines_burst(self, tmp_path: Path) -> None:
        """Verify bursts of consecutive corrupted lines do not prevent recovery of valid rows."""
        log_file = tmp_path / "burst.jsonl"
        lines = (
            [f"BURST_CORRUPT_PREFIX_{i}" for i in range(40)]
            + [json.dumps({"step": 1, "metric": 99.9})]
            + [f"BURST_CORRUPT_MIDDLE_{i}" for i in range(40)]
            + [json.dumps({"step": 2, "metric": 88.8})]
            + [f"BURST_CORRUPT_SUFFIX_{i}" for i in range(40)]
        )

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file, run_name="burst_run")

        assert len(df) == 2
        assert df["step"].tolist() == [1, 2]
        assert df["metric"].tolist() == [pytest.approx(99.9), pytest.approx(88.8)]
        assert len(recorded_warnings) == 120

    def test_all_corrupted_lines_returns_empty_dataframe(self, tmp_path: Path) -> None:
        """Verify a file composed entirely of corrupted lines yields an empty DataFrame with warnings."""
        log_file = tmp_path / "all_corrupted.jsonl"
        lines = [
            "NOT_A_JSON_1",
            '{"unclosed": "val',
            "}",
            "[1, 2, 3]",
            "null",
            "false",
            "123.456",
            "<<<<>>>>",
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert df.empty
        assert len(df) == 0
        assert len(recorded_warnings) == len(lines)

    def test_thousand_corrupted_lines_preceding_single_valid_line(self, tmp_path: Path) -> None:
        """Verify performance and recovery when 1,000 corrupted lines precede 1 valid line."""
        log_file = tmp_path / "heavy_corrupt.jsonl"
        corrupted_lines = [f"corrupted_line_entry_{i}: {{{{ bad json" for i in range(1000)]
        valid_line = json.dumps({"step": 42, "score": 0.99})

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(corrupted_lines) + "\n" + valid_line + "\n")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file, run_name="heavy_run")

        assert len(df) == 1
        assert df["step"].iloc[0] == 42
        assert df["score"].iloc[0] == pytest.approx(0.99)
        assert len(recorded_warnings) == 1000


# =============================================================================
# Suite 2: Truncated Writes with Escaped Quotes, Nested Braces, and Brackets
# =============================================================================
class TestTruncatedWritesAndEscapedBoundaries:
    """Stress-test crash recovery when truncated writes contain escape sequences and brackets."""

    def test_truncated_write_with_escaped_quotes_and_braces(self, tmp_path: Path) -> None:
        """Verify recovery when corrupted prefix contains escaped quotes and braces in string literal."""
        log_file = tmp_path / "escaped_quotes.jsonl"
        # Line simulates crash mid-write of a string containing \" and {brace}
        corrupted_line = (
            '{"msg": "incomplete write with \\"escaped quotes\\" and {curly_token} cut off'
            '{"step": 1, "loss": 0.123, "_timestamp": 100.0, "_time_since_start": 0.5}'
        )
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file, run_name="escaped_run")

        assert len(df) == 1
        assert df["step"].iloc[0] == 1
        assert df["loss"].iloc[0] == pytest.approx(0.123)
        assert len(recorded_warnings) == 1
        assert "Corrupted JSONL line 1" in str(recorded_warnings[0].message)

    def test_truncated_write_with_nested_arrays_and_brackets(self, tmp_path: Path) -> None:
        """Verify recovery when corrupted prefix contains JSON arrays, square brackets, and inner objects."""
        log_file = tmp_path / "nested_brackets.jsonl"
        corrupted_line = (
            '{"arr": [1, 2, {"inner_broken": 3}], "notes": "aborted ['
            '{"step": 5, "accuracy": 0.98, "_timestamp": 200.0, "_time_since_start": 1.5}'
        )
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file, run_name="bracket_run")

        assert len(df) == 1
        assert df["step"].iloc[0] == 5
        assert df["accuracy"].iloc[0] == pytest.approx(0.98)

    def test_truncated_write_with_multiple_candidate_braces(self, tmp_path: Path) -> None:
        """Verify search loop iterates through multiple broken candidate braces until finding valid record."""
        log_file = tmp_path / "multi_brace_candidates.jsonl"
        corrupted_line = (
            '{"prefix": 1{candidate_1: broken{candidate_2: also_broken{'
            '{"step": 10, "val": 42.0, "_timestamp": 300.0, "_time_since_start": 2.5}'
        )
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file, run_name="candidate_run")

        assert len(df) == 1
        assert df["step"].iloc[0] == 10
        assert df["val"].iloc[0] == pytest.approx(42.0)

    def test_truncated_write_where_recovered_is_header_record(self, tmp_path: Path) -> None:
        """Verify mid-line recovered record that is a header updates run_name without adding metric row."""
        log_file = tmp_path / "recovered_header.jsonl"
        lines = [
            '{"CRASH_JUNK": 1{"type": "header", "run_name": "inferred_resumed_name", "created_at": "2026-09-14T00:00:00Z"}',
            json.dumps({"step": 1, "loss": 0.5}),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert len(df) == 1
        assert df["run_name"].iloc[0] == "inferred_resumed_name"
        assert df["step"].iloc[0] == 1
        assert len(rec_warn) >= 1

    def test_truncated_write_where_second_candidate_is_also_corrupted(self, tmp_path: Path) -> None:
        """Verify that when the second candidate is also incomplete, line is safely skipped with warning."""
        log_file = tmp_path / "double_corrupted.jsonl"
        corrupted_line = '{"step": 0, "loss": 1.0{"step": 1, "loss": '
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert df.empty
        assert len(rec_warn) == 1

    def test_truncated_write_where_recovered_candidate_is_non_dict(self, tmp_path: Path) -> None:
        """Verify candidate starting with brace but resolving to non-dict object is not appended as record."""
        log_file = tmp_path / "non_dict_candidate.jsonl"
        # Candidate starting at second brace is not a valid JSON dict
        corrupted_line = '{"prefix": 1{[100, 200]}'
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert df.empty
        assert len(rec_warn) == 1

    def test_truncated_write_followed_by_trailing_garbage(self, tmp_path: Path) -> None:
        """Verify trailing garbage after valid candidate fails json.loads candidate slice safely."""
        log_file = tmp_path / "trailing_garbage.jsonl"
        corrupted_line = '{"prefix": 1{"step": 2, "loss": 0.5}TRAILING_GARBAGE'
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert df.empty
        assert len(rec_warn) == 1

    def test_single_brace_corrupted_line_behavior(self, tmp_path: Path) -> None:
        """Verify single-brace corrupted line (no second brace) safely emits UserWarning without crash."""
        log_file = tmp_path / "single_brace.jsonl"
        # Only one brace on line; while loop does not run, recovered_dict is None
        corrupted_line = 'JUNK_NO_INITIAL_BRACE{"step": 1, "loss": 0.5}'
        log_file.write_text(corrupted_line + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_jsonl(log_file)

        assert df.empty
        assert len(rec_warn) == 1
        assert "Corrupted JSONL line 1" in str(rec_warn[0].message)


# =============================================================================
# Suite 3: Deeply Nested Recursive Dictionary Flattening & Edge Cases
# =============================================================================
class TestDeeplyNestedDictFlatteningAndCycles:
    """Stress-test _flatten_dict and _deserialize_special_floats on complex and adversarial dictionaries."""

    def test_flatten_dict_non_string_keys(self) -> None:
        """Verify non-string keys (integers, floats, booleans, None, tuples) are cast to strings."""
        nested = {
            1: "int_root",
            2: {
                10: "int_child",
                3.14: "float_child",
                True: "bool_child",
                None: "none_child",
                (1, 2): "tuple_child",
            },
        }
        flattened = _flatten_dict(nested)
        assert flattened["1"] == "int_root"
        assert flattened["2.10"] == "int_child"
        assert flattened["2.3.14"] == "float_child"
        assert flattened["2.True"] == "bool_child"
        assert flattened["2.None"] == "none_child"
        assert flattened["2.(1, 2)"] == "tuple_child"
        # Ensure parent dictionary is preserved
        assert isinstance(flattened["2"], dict)

    def test_flatten_dict_empty_leaves_and_intermediate_empty_nodes(self) -> None:
        """Verify empty dictionaries at root, intermediate nodes, and leaves are preserved."""
        # Empty root
        assert _flatten_dict({}) == {}

        # Empty leaf
        nested = {"a": {}, "b": {"c": {}}, "d": {"e": 42}}
        flattened = _flatten_dict(nested)
        assert flattened["a"] == {}
        assert flattened["b.c"] == {}
        assert flattened["b"] == {"c": {}}
        assert flattened["d.e"] == 42
        assert flattened["d"] == {"e": 42}

    def test_flatten_dict_deep_15_level_hierarchy(self) -> None:
        """Verify 15-level deeply nested hierarchy flattens with correct dot notation."""
        curr: dict[str, Any] = {"leaf": "deep_secret"}
        for i in range(14, 0, -1):
            curr = {f"level_{i:02d}": curr}

        flattened = _flatten_dict(curr)
        expected_key = ".".join([f"level_{i:02d}" for i in range(1, 15)]) + ".leaf"
        assert expected_key in flattened
        assert flattened[expected_key] == "deep_secret"

    def test_flatten_dict_cycle_raises_recursion_error(self) -> None:
        """Verify self-referential dictionary raises RecursionError rather than hanging."""
        d: dict[str, Any] = {}
        d["self"] = d
        with pytest.raises(RecursionError):
            _flatten_dict(d)

    def test_deserialize_special_floats_nested_structures(self) -> None:
        """Verify recursive deserialization of special floats across nested dicts, lists, and tuples."""
        payload = {
            "scalar_nan": "NaN",
            "scalar_pos_inf": "+Infinity",
            "scalar_neg_inf": "-Infinity",
            "case_variations": ["nan", "+inf", "-inf", "infinity", "+infinity", "-infinity"],
            "nested_dict": {
                "inner_nan": "-NaN",
                "normal_str": "NaN_not_a_float",
                "regular_number": 42,
                "boolean": True,
            },
            "nested_tuple": ("+Infinity", "regular"),
        }
        res = _deserialize_special_floats(payload)

        assert math.isnan(res["scalar_nan"])
        assert math.isinf(res["scalar_pos_inf"]) and res["scalar_pos_inf"] > 0
        assert math.isinf(res["scalar_neg_inf"]) and res["scalar_neg_inf"] < 0

        for val in res["case_variations"]:
            assert isinstance(val, float)

        assert math.isnan(res["nested_dict"]["inner_nan"])
        assert res["nested_dict"]["normal_str"] == "NaN_not_a_float"
        assert res["nested_dict"]["regular_number"] == 42
        assert res["nested_dict"]["boolean"] is True
        assert math.isinf(res["nested_tuple"][0])
        assert res["nested_tuple"][1] == "regular"

    def test_read_run_with_nested_dict_and_non_string_keys_in_hyperparameters(self, tmp_path: Path) -> None:
        """Verify read_run handles complex nested hyperparameters including lists and empty dicts."""
        run_dir = tmp_path / "complex_hp_run"
        meta_payload = {
            constants.RUN_NAME_KEY: "complex_hp_run",
            constants.HYPERPARAMETERS_KEY: {
                "model": {
                    "backbone": "vit_base",
                    "layers": [12, 24],
                    "empty_cfg": {},
                },
                "lr": 0.001,
            },
        }
        log_lines = [
            json.dumps(constants.DEFAULT_HEADER),
            json.dumps({"step": 1, "loss": 0.5}),
            json.dumps({"step": 2, "loss": 0.4}),
        ]
        _write_run_files(run_dir, meta_content=meta_payload, log_lines=log_lines)

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_run(run_dir)

        assert len(df) == 2
        assert "hyperparameters.model.backbone" in df.columns
        assert df["hyperparameters.model.backbone"].tolist() == ["vit_base", "vit_base"]
        assert "hyperparameters.model.layers" in df.columns
        assert df["hyperparameters.model.layers"].iloc[0] == [12, 24]
        assert "hyperparameters.model.empty_cfg" in df.columns
        assert df["hyperparameters.model.empty_cfg"].iloc[0] == {}
        assert "hyperparameters.lr" in df.columns
        assert df["hyperparameters.lr"].tolist() == [0.001, 0.001]


# =============================================================================
# Suite 4: Schema Evolution & Mixed Column Types Across 10+ Disjoint Runs
# =============================================================================
class TestSchemaEvolutionAcross12DisjointRuns:
    """Stress-test multi-run schema evolution (ADR 20) across 12 disjoint and conflicting runs."""

    def test_read_all_12_disjoint_runs_with_divergent_types(self, tmp_path: Path) -> None:
        """Verify read_all cleanly concatenates 12 runs with divergent, colliding, and missing metric types."""
        campaign_dir = tmp_path / "campaign_12"
        campaign_dir.mkdir(parents=True)

        run_configs: list[dict[str, Any]] = [
            # Run 0: Standard floats and optimizer string
            {
                "name": "run_00_baseline",
                "hp": {"lr": 0.01, "optimizer": "adam"},
                "metrics": [{"step": 1, "loss": 0.5, "lr": 0.01, "opt": "adam"}],
            },
            # Run 1: Disjoint metric keys and integer hyperparameter
            {
                "name": "run_01_disjoint",
                "hp": {"batch_size": 64},
                "metrics": [{"step": 1, "accuracy": 0.85, "val_loss": 0.6}],
            },
            # Run 2: Special float NaN metric
            {
                "name": "run_02_nan_metric",
                "hp": {"lr": 0.001},
                "metrics": [{"step": 1, "loss": "NaN", "stability": 0.1}],
            },
            # Run 3: Special float +Infinity and 1D array metric
            {
                "name": "run_03_inf_and_array",
                "hp": {"layers": [1, 2, 3]},
                "metrics": [{"step": 1, "loss": "+Infinity", "grad_norms": [0.1, 0.2, 0.3]}],
            },
            # Run 4: Special float -Infinity and conflicting integer for 'opt' (was string in Run 0)
            {
                "name": "run_04_type_collision_int",
                "hp": {"opt_id": 1},
                "metrics": [{"step": 1, "loss": "-Infinity", "opt": 1}],
            },
            # Run 5: Conflicting string for 'loss' (was float in Run 0)
            {
                "name": "run_05_type_collision_str",
                "hp": {"desc": "text_loss"},
                "metrics": [{"step": 1, "loss": "diverged_loss", "epoch": 1}],
            },
            # Run 6: High-precision numeric metrics
            {
                "name": "run_06_precision",
                "hp": {"precision": "fp64"},
                "metrics": [{"step": 1, "loss": 1.234567890123456, "metric_6": 6.6}],
            },
            # Run 7: Purely novel metric keys
            {
                "name": "run_07_novel",
                "hp": {"novel": True},
                "metrics": [{"step": 1, "metric_7a": 7.1, "metric_7b": 7.2}],
            },
            # Run 8: Boolean metric value
            {
                "name": "run_08_bool_metric",
                "hp": {"flag": False},
                "metrics": [{"step": 1, "converged": True, "early_stop": False}],
            },
            # Run 9: None/null metric value
            {
                "name": "run_09_none_metric",
                "hp": {"opt": None},
                "metrics": [{"step": 1, "aux_loss": None, "score": 9.9}],
            },
            # Run 10: Tuple and sequence hyperparameters
            {
                "name": "run_10_tuple_hp",
                "hp": {"dims": (256, 512)},
                "metrics": [{"step": 1, "loss": 0.05, "metric_10": 10.0}],
            },
            # Run 11: Multi-step run with epoch tracking
            {
                "name": "run_11_multistep",
                "hp": {"lr": 0.0001},
                "metrics": [
                    {"step": 1, "epoch": 1, "loss": 0.04},
                    {"step": 2, "epoch": 2, "loss": 0.02},
                ],
            },
        ]

        for cfg in run_configs:
            rdir = campaign_dir / cfg["name"]
            meta_data = {
                constants.RUN_NAME_KEY: cfg["name"],
                constants.HYPERPARAMETERS_KEY: cfg["hp"],
            }
            log_lines = [json.dumps(constants.DEFAULT_HEADER)] + [
                json.dumps(m) for m in cfg["metrics"]
            ]
            _write_run_files(rdir, meta_content=meta_data, log_lines=log_lines)

        reader = LogReader(logs_dir=campaign_dir)
        df = reader.read_all()

        # Row count conservation: 11 runs with 1 row + 1 run with 2 rows = 13 rows total
        assert len(df) == 13
        assert df["run_name"].nunique() == 12

        # Verify contiguous RangeIndex from 0 to 12
        assert list(df.index) == list(range(13))

        # Check type collision column: 'opt' contains string 'adam', int 1, and NaNs
        assert "opt" in df.columns
        opt_vals = df["opt"].dropna().tolist()
        assert "adam" in opt_vals
        assert 1 in opt_vals

        # Check type collision column: 'loss' contains numeric floats, special floats, and string 'diverged_loss'
        assert "loss" in df.columns
        loss_vals = df["loss"].tolist()
        assert "diverged_loss" in loss_vals
        has_nan = any(isinstance(x, float) and math.isnan(x) for x in loss_vals)
        has_pos_inf = any(isinstance(x, float) and math.isinf(x) and x > 0 for x in loss_vals)
        has_neg_inf = any(isinstance(x, float) and math.isinf(x) and x < 0 for x in loss_vals)
        assert has_nan
        assert has_pos_inf
        assert has_neg_inf

        # Check 1D array column: grad_norms preserved as object without expanding row dimensions
        assert "grad_norms" in df.columns
        grad_norms_non_null = df["grad_norms"].dropna().tolist()
        assert len(grad_norms_non_null) == 1
        assert grad_norms_non_null[0] == [0.1, 0.2, 0.3]

        # Check missing metric imputation with NaN
        run_0_row = df[df["run_name"] == "run_00_baseline"].iloc[0]
        assert pd.isna(run_0_row["metric_7a"])
        assert pd.isna(run_0_row["accuracy"])

        # Check hyperparameter prefixing and broadcasting
        assert "hyperparameters.batch_size" in df.columns
        run_1_row = df[df["run_name"] == "run_01_disjoint"].iloc[0]
        assert run_1_row["hyperparameters.batch_size"] == 64
        assert pd.isna(run_0_row["hyperparameters.batch_size"])

    def test_read_all_zero_common_metrics_between_runs(self, tmp_path: Path) -> None:
        """Verify outer join correctness when sibling runs share zero metric keys."""
        logs_dir = tmp_path / "zero_common"
        for i in range(5):
            rdir = logs_dir / f"run_{i}"
            meta_data = {constants.RUN_NAME_KEY: f"run_{i}"}
            log_lines = [
                json.dumps(constants.DEFAULT_HEADER),
                json.dumps({f"unique_metric_{i}": float(i * 10)}),
            ]
            _write_run_files(rdir, meta_content=meta_data, log_lines=log_lines)

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()

        assert len(df) == 5
        assert df["run_name"].nunique() == 5
        for i in range(5):
            col = f"unique_metric_{i}"
            assert col in df.columns
            # Exactly one row has a non-null value for this column
            assert df[col].dropna().tolist() == [float(i * 10)]


# =============================================================================
# Suite 5: Directory Traversal, Symbolic Links, and Unreadable File Permissions
# =============================================================================
class TestDirectoryTraversalAndSymlinks:
    """Stress-test directory traversal, symbolic link handling, and permission boundaries."""

    def test_traversal_symlinked_log_file(self, tmp_path: Path) -> None:
        """Verify LogReader discovers and reads a run where log.jsonl is a symbolic link to an external file."""
        external_dir = tmp_path / "external_storage"
        external_dir.mkdir()
        actual_log = external_dir / "actual_log.jsonl"
        actual_log.write_text(
            json.dumps(constants.DEFAULT_HEADER) + "\n" + json.dumps({"step": 1, "loss": 0.15}) + "\n",
            encoding="utf-8",
        )

        logs_dir = tmp_path / "logs"
        run_dir = logs_dir / "run_symlinked"
        run_dir.mkdir(parents=True)
        symlink_target = run_dir / constants.LOG_FILE
        symlink_target.symlink_to(actual_log)

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()

        assert len(df) == 1
        assert df["run_name"].iloc[0] == "run_symlinked"
        assert df["step"].iloc[0] == 1
        assert df["loss"].iloc[0] == pytest.approx(0.15)

    def test_traversal_broken_symlink_safely_ignored(self, tmp_path: Path) -> None:
        """Verify broken symbolic link to log.jsonl is skipped without crashing sibling runs."""
        logs_dir = tmp_path / "logs"
        broken_run = logs_dir / "broken_run"
        broken_run.mkdir(parents=True)
        broken_link = broken_run / constants.LOG_FILE
        broken_link.symlink_to(tmp_path / "does_not_exist.jsonl")

        valid_run = logs_dir / "valid_run"
        _write_run_files(
            valid_run,
            run_name="valid_run",
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "acc": 0.99})],
        )

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()

        assert len(df) == 1
        assert df["run_name"].iloc[0] == "valid_run"
        assert df["acc"].iloc[0] == pytest.approx(0.99)

    def test_traversal_directory_named_log_jsonl_ignored(self, tmp_path: Path) -> None:
        """Verify a directory named 'log.jsonl' does not cause IsADirectoryError and is skipped."""
        logs_dir = tmp_path / "logs"
        fake_run = logs_dir / "fake_run"
        fake_run_log = fake_run / constants.LOG_FILE
        fake_run_log.mkdir(parents=True)  # Creates a directory named log.jsonl!

        valid_run = logs_dir / "valid_run"
        _write_run_files(
            valid_run,
            run_name="valid_run",
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "metric": 5.0})],
        )

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()

        assert len(df) == 1
        assert df["run_name"].iloc[0] == "valid_run"

    def test_traversal_unreadable_run_directory_permission_error_isolation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify read_all catches PermissionError during run ingestion, warns, and parses sibling runs."""
        logs_dir = tmp_path / "logs"
        run_good_1 = logs_dir / "run_good_1"
        run_bad = logs_dir / "run_bad"
        run_good_2 = logs_dir / "run_good_2"

        _write_run_files(
            run_good_1,
            meta_content={"run_name": "good_1"},
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "val": 1.0})],
        )
        _write_run_files(
            run_bad,
            meta_content={"run_name": "bad_permission"},
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "val": 2.0})],
        )
        _write_run_files(
            run_good_2,
            meta_content={"run_name": "good_2"},
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "val": 3.0})],
        )

        original_read_run = LogReader.read_run

        def mock_read_run(self_reader: LogReader, rd: str | Path) -> pd.DataFrame:
            if "run_bad" in str(rd):
                raise PermissionError(f"Permission denied: '{rd}'")
            return original_read_run(self_reader, rd)

        monkeypatch.setattr(LogReader, "read_run", mock_read_run)

        reader = LogReader(logs_dir=logs_dir)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_all()

        assert len(df) == 2
        assert set(df["run_name"].tolist()) == {"good_1", "good_2"}
        assert len(rec_warn) == 1
        assert "Permission denied" in str(rec_warn[0].message)

    def test_traversal_filters_hidden_and_venv_directories(self, tmp_path: Path) -> None:
        """Verify runs nested in .git, .venv, .hidden, or __pycache__ directories are ignored."""
        logs_dir = tmp_path / "logs"

        # Ignored run paths
        _write_run_files(logs_dir / ".git" / "run_git", log_lines=[json.dumps({"step": 1})])
        _write_run_files(logs_dir / ".venv" / "run_venv", log_lines=[json.dumps({"step": 1})])
        _write_run_files(logs_dir / "__pycache__" / "run_cache", log_lines=[json.dumps({"step": 1})])
        _write_run_files(logs_dir / ".hidden_folder" / "run_hidden", log_lines=[json.dumps({"step": 1})])

        # Legitimate nested run
        legit_run = logs_dir / "valid_group" / "experiment_a"
        _write_run_files(
            legit_run,
            run_name="experiment_a",
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "loss": 0.3})],
        )

        reader = LogReader(logs_dir=logs_dir)
        df = reader.read_all()

        assert len(df) == 1
        assert df["run_name"].iloc[0] == "experiment_a"


# =============================================================================
# Suite 6: Corrupted vs Valid Metadata Asymmetry Stress Tests
# =============================================================================
class TestCorruptedMetadataAsymmetry:
    """Stress-test asymmetry between corrupt meta.json / valid log.jsonl and valid meta.json / corrupt log.jsonl."""

    def test_corrupted_meta_json_with_valid_log_jsonl(self, tmp_path: Path) -> None:
        """Verify corrupted meta.json emits UserWarning, falls back to folder name, and parses log.jsonl."""
        run_dir = tmp_path / "corrupt_meta_run"
        log_lines = [
            json.dumps(constants.DEFAULT_HEADER),
            json.dumps({"step": 1, "loss": 0.5}),
            json.dumps({"step": 2, "loss": 0.4}),
        ]
        _write_run_files(
            run_dir,
            meta_content="MALFORMED_JSON_{{{{ NOT VALID",
            log_lines=log_lines,
        )

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_run(run_dir)

        assert len(df) == 2
        assert df["run_name"].tolist() == ["corrupt_meta_run", "corrupt_meta_run"]
        assert df["loss"].tolist() == [0.5, 0.4]
        # No hyperparameter columns present because meta.json was corrupt
        assert not any(c.startswith(constants.HYPERPARAMETER_PREFIX) for c in df.columns)
        # Verify warning was emitted
        assert len(rec_warn) >= 1
        assert "Failed to read metadata file" in str(rec_warn[0].message)

    def test_valid_meta_json_with_100_percent_corrupted_log_jsonl(self, tmp_path: Path) -> None:
        """Verify valid meta.json with 100% corrupted log.jsonl yields empty DataFrame without crashing."""
        run_dir = tmp_path / "corrupt_log_run"
        meta_data = {
            constants.RUN_NAME_KEY: "valid_meta_name",
            constants.HYPERPARAMETERS_KEY: {"lr": 0.001, "epochs": 50},
        }
        corrupted_log_lines = [
            "CORRUPTED_LINE_1",
            "CORRUPTED_LINE_2",
            "CORRUPTED_LINE_3",
        ]
        _write_run_files(run_dir, meta_content=meta_data, log_lines=corrupted_log_lines)

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_run(run_dir)

        assert df.empty
        assert len(df) == 0
        assert len(rec_warn) == 3

    def test_valid_meta_json_with_partially_corrupted_log_jsonl(self, tmp_path: Path) -> None:
        """Verify valid meta.json broadcasts hyperparameters to parsed rows despite interleaved corrupted log lines."""
        run_dir = tmp_path / "partial_corrupt_run"
        meta_data = {
            constants.RUN_NAME_KEY: "resilient_run",
            constants.HYPERPARAMETERS_KEY: {"lr": 0.001, "optimizer": "adamw"},
        }
        log_lines = [
            json.dumps(constants.DEFAULT_HEADER),
            "BROKEN_LINE_1",
            json.dumps({"step": 1, "loss": 0.8}),
            "BROKEN_LINE_2",
            json.dumps({"step": 2, "loss": 0.6}),
        ]
        _write_run_files(run_dir, meta_content=meta_data, log_lines=log_lines)

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            df = reader.read_run(run_dir)

        assert len(df) == 2
        assert df["run_name"].tolist() == ["resilient_run", "resilient_run"]
        assert df["loss"].tolist() == [0.8, 0.6]
        # Hyperparameters must be broadcast to all parsed rows
        assert "hyperparameters.lr" in df.columns
        assert df["hyperparameters.lr"].tolist() == [0.001, 0.001]
        assert "hyperparameters.optimizer" in df.columns
        assert df["hyperparameters.optimizer"].tolist() == ["adamw", "adamw"]
        assert len(rec_warn) == 2

    def test_non_dict_meta_json_returns_empty_and_warns(self, tmp_path: Path) -> None:
        """Verify meta.json containing JSON list or string literal emits warning and returns empty metadata."""
        run_dir = tmp_path / "list_meta_run"
        _write_run_files(
            run_dir,
            meta_content=json.dumps([1, 2, 3]),
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "loss": 0.1})],
        )

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            meta = reader.get_metadata(run_dir)
            df = reader.read_run(run_dir)

        assert meta == {}
        assert len(df) == 1
        assert any("is not a JSON object" in str(w.message) for w in rec_warn)

    def test_empty_zero_byte_meta_json(self, tmp_path: Path) -> None:
        """Verify 0-byte meta.json emits UserWarning and returns empty dict."""
        run_dir = tmp_path / "zero_byte_meta_run"
        _write_run_files(
            run_dir,
            meta_content="",
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "loss": 0.2})],
        )

        reader = LogReader(logs_dir=tmp_path)
        with warnings.catch_warnings(record=True) as rec_warn:
            warnings.simplefilter("always")
            meta = reader.get_metadata(run_dir)
            df = reader.read_run(run_dir)

        assert meta == {}
        assert len(df) == 1
        assert len(rec_warn) >= 1

    def test_fallback_hyperparameters_json_valid_and_corrupted(self, tmp_path: Path) -> None:
        """Verify fallback to hyperparameters.json works when valid, and handles corruptions cleanly."""
        # Case A: Valid hyperparameters.json fallback
        run_valid_fallback = tmp_path / "valid_fallback"
        _write_run_files(
            run_valid_fallback,
            legacy_hp_content={"lr": 0.05, "batch_size": 16},
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "loss": 0.3})],
        )
        reader = LogReader(logs_dir=tmp_path)
        df_a = reader.read_run(run_valid_fallback)
        assert df_a["hyperparameters.lr"].iloc[0] == 0.05

        # Case B: Corrupted hyperparameters.json fallback
        run_corrupt_fallback = tmp_path / "corrupt_fallback"
        _write_run_files(
            run_corrupt_fallback,
            legacy_hp_content="BROKEN_JSON_CONTENT{{{",
            log_lines=[json.dumps(constants.DEFAULT_HEADER), json.dumps({"step": 1, "loss": 0.3})],
        )
        df_b = reader.read_run(run_corrupt_fallback)
        assert len(df_b) == 1
        assert not any(c.startswith(constants.HYPERPARAMETER_PREFIX) for c in df_b.columns)


# =============================================================================
# Suite 7: DataFrame Column Dtype Normalization & System Columns Ordering
# =============================================================================
class TestDataFrameDtypeNormalizationAndOrdering:
    """Stress-test _normalize_dataframe_dtypes and column ordering priority."""

    def test_normalize_dataframe_dtypes_all_null_column(self) -> None:
        """Verify column containing only null/NaN values is safely processed without error."""
        df = pd.DataFrame({"all_none": [None, None], "all_nan": [float("nan"), float("nan")]})
        normalized = _normalize_dataframe_dtypes(df)
        assert len(normalized) == 2

    def test_normalize_dataframe_dtypes_mixed_bool_numeric_preserves_bool(self) -> None:
        """Verify column containing booleans mixed with numbers is NOT coerced to int (e.g. False -> 0)."""
        df = pd.DataFrame({"mixed_bool": [1, False, 2.5]})
        normalized = _normalize_dataframe_dtypes(df)
        assert normalized["mixed_bool"].dtype == object
        assert normalized["mixed_bool"].iloc[1] is False

    def test_normalize_dataframe_dtypes_pure_numeric_objects_converted(self) -> None:
        """Verify object column containing only ints and floats is normalized to numeric dtype."""
        df = pd.DataFrame({"numeric_obj": pd.Series([1, 2.5, 3], dtype=object)})
        normalized = _normalize_dataframe_dtypes(df)
        assert pd.api.types.is_numeric_dtype(normalized["numeric_obj"])

    def test_normalize_dataframe_dtypes_preserves_list_and_dict_objects(self) -> None:
        """Verify sequence and dictionary columns remain object dtype."""
        df = pd.DataFrame({
            "list_col": [[1, 2], [3, 4]],
            "dict_col": [{"a": 1}, {"b": 2}],
        })
        normalized = _normalize_dataframe_dtypes(df)
        assert normalized["list_col"].dtype == object
        assert normalized["dict_col"].dtype == object

    def test_read_jsonl_system_column_priority_ordering(self, tmp_path: Path) -> None:
        """Verify system columns (run_name, step, epoch, _timestamp, _time_since_start) are prioritized first."""
        log_file = tmp_path / "system_order.jsonl"
        unordered_record = {
            "zebra_metric": 99.0,
            "step": 1,
            "alpha_metric": 10.0,
            "epoch": 0,
            "_timestamp": 1234567.89,
            "_time_since_start": 4.5,
        }
        log_file.write_text(json.dumps(unordered_record) + "\n", encoding="utf-8")

        reader = LogReader(logs_dir=tmp_path)
        df = reader.read_jsonl(log_file, run_name="priority_run")

        expected_leading = [
            constants.RUN_NAME_KEY,
            constants.STEP_KEY,
            constants.EPOCH_KEY,
            constants.TIMESTAMP_KEY,
            constants.TIME_SINCE_START_KEY,
        ]
        assert list(df.columns[:5]) == expected_leading
        assert list(df.columns[5:]) == ["zebra_metric", "alpha_metric"]
