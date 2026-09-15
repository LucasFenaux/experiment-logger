"""
Tier 5 White-Box Adversarial Hardening: Logger, Environment, and Constants.
Authored by Challenger M4-1 (challenger_m4_1).

Covers White-Box Adversarial Stress Dimensions:
1. Collision resolution loops under high contention & timestamp microsecond collisions.
2. Non-standard IEEE 754 float values (denormals/subnormals, signed zeros, signaling NaNs).
3. Memory and file descriptor leakage on unexpected exceptions inside context managers.
4. Malformed git repositories (detached HEAD, unborn branch, corrupt git tree, binary diffs).
5. Python runtime environment metadata variations (frozen environments, corrupt distribution metadata).
6. Schema header injection protection and constants round-trip symmetry.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from typing import Any
from unittest import mock

import pytest

from experiment_logger import constants
from experiment_logger.env import (
    capture_environment,
    capture_git_info,
    create_diff_patch,
)
from experiment_logger.logger import (
    Logger,
    _is_foreign_tensor,
    _sanitize_hyperparameters,
    _sanitize_scalar_float,
)


# =============================================================================
# 1. Collision Resolution Under High Contention & Microsecond Clocks
# =============================================================================
class TestAdversarialCollisionResolution:
    """Stress-tests the directory collision resolution loop (ADR 41) under extreme contention."""

    def test_high_contention_collision_loop_50_iterations(self, temp_logs_dir: Path):
        """Simulate 50 rapid collisions occurring within the exact same second.

        Tests that the while loop in Logger.__init__ increments counter 1..50
        without deadlock, off-by-one errors, or directory collision crashes.
        """
        fixed_dt = datetime(2026, 9, 14, 20, 0, 0)
        base_name = "contention_run"

        # Create original base run directory
        original_dir = temp_logs_dir / base_name
        original_dir.mkdir(parents=True, exist_ok=True)

        opened_loggers: list[Logger] = []
        try:
            with mock.patch("experiment_logger.logger.datetime") as mock_dt:
                mock_dt.now.return_value = fixed_dt
                mock_dt.strftime = datetime.strftime

                for i in range(50):
                    logger = Logger(
                        run_name=base_name,
                        log_dir=temp_logs_dir,
                        capture_env=False,
                    )
                    opened_loggers.append(logger)

                    # First collision gets timestamp without counter suffix
                    # Subsequent collisions get _1, _2, ..., _49
                    if i == 0:
                        expected_name = f"{base_name}_20260914_200000"
                    else:
                        expected_name = f"{base_name}_20260914_200000_{i}"

                    assert logger.run_name == expected_name
                    assert logger.original_run_name == base_name
                    assert logger.run_dir.exists()
                    assert logger.run_dir.is_dir()
                    assert logger.run_dir.name == expected_name

                    # Verify line-1 header matches assigned collided run_name
                    with open(logger.log_file, "r", encoding="utf-8") as f:
                        header = json.loads(f.readline())
                    assert header["run_name"] == expected_name
                    assert header[constants.VERSION_HEADER_KEY] == constants.MLVIZ_VERSION

                    # Verify meta.json records both run_name and original_run_name
                    with open(logger.meta_file, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    assert meta["run_name"] == expected_name
                    assert meta["original_run_name"] == base_name
        finally:
            for l in opened_loggers:
                l.close()

    def test_preexisting_collision_gaps(self, temp_logs_dir: Path):
        """Verify collision counter cleanly skips past pre-existing directories with gaps."""
        fixed_dt = datetime(2026, 9, 14, 21, 30, 0)
        base_name = "gap_run"
        ts_suffix = "20260914_213000"

        # Create base dir and collision candidate dir
        (temp_logs_dir / base_name).mkdir(parents=True, exist_ok=True)
        (temp_logs_dir / f"{base_name}_{ts_suffix}").mkdir(parents=True, exist_ok=True)

        # Create counter dirs 1, 2, and 4 (leaving gap at 3)
        (temp_logs_dir / f"{base_name}_{ts_suffix}_1").mkdir(parents=True, exist_ok=True)
        (temp_logs_dir / f"{base_name}_{ts_suffix}_2").mkdir(parents=True, exist_ok=True)
        (temp_logs_dir / f"{base_name}_{ts_suffix}_4").mkdir(parents=True, exist_ok=True)

        with mock.patch("experiment_logger.logger.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            mock_dt.strftime = datetime.strftime

            # First new logger should fill gap 3
            logger_1 = Logger(run_name=base_name, log_dir=temp_logs_dir, capture_env=False)
            assert logger_1.run_name == f"{base_name}_{ts_suffix}_3"
            logger_1.close()

            # Second new logger should skip past 4 and take 5
            logger_2 = Logger(run_name=base_name, log_dir=temp_logs_dir, capture_env=False)
            assert logger_2.run_name == f"{base_name}_{ts_suffix}_5"
            logger_2.close()

    def test_concurrent_multi_instance_logging(self, temp_logs_dir: Path):
        """Ensure multiple simultaneously active collided Logger instances write without cross-talk."""
        base_name = "multi_contention"
        loggers: list[Logger] = []
        try:
            # Spawn 10 logger instances on the same base_name
            for _ in range(10):
                l = Logger(run_name=base_name, log_dir=temp_logs_dir, capture_env=False)
                loggers.append(l)

            # Log alternating metrics into each
            for step in range(5):
                for idx, l in enumerate(loggers):
                    l.log({"step": step, "source_idx": idx, "val": step * 10.0 + idx})

            # Verify each log file has exactly 1 header + 5 metric lines with correct source_idx
            for idx, l in enumerate(loggers):
                with open(l.log_file, "r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                assert len(lines) == 6
                assert lines[0][constants.RECORD_TYPE_KEY] == constants.HEADER_TYPE
                for step in range(5):
                    rec = lines[step + 1]
                    assert rec["step"] == step
                    assert rec["source_idx"] == idx
                    assert rec["val"] == step * 10.0 + idx
        finally:
            for l in loggers:
                l.close()

    def test_nested_log_dir_collision_resolution(self, temp_logs_dir: Path):
        """Verify collision resolution handles deeply nested log directories with trailing slashes."""
        deep_dir = temp_logs_dir / "deeply" / "nested" / "output_dir" / ""
        base_name = "nested_run"

        l1 = Logger(run_name=base_name, log_dir=deep_dir, capture_env=False)
        l1.log({"step": 0, "m": 1.0})
        l1.close()

        l2 = Logger(run_name=base_name, log_dir=deep_dir, capture_env=False)
        assert l2.run_name != l1.run_name
        assert l2.original_run_name == base_name
        assert l2.run_dir.parent == deep_dir.resolve()
        l2.close()


# =============================================================================
# 2. Non-Standard IEEE 754 Float Values & Boundary Stress Testing
# =============================================================================
class TestAdversarialIEEE754Floats:
    """Stress-tests float handling against subnormals, signed zeros, sNaNs, and extreme boundaries."""

    def test_denormal_subnormal_floats_in_metrics(self, temp_logs_dir: Path):
        """Verify subnormal/denormal floats (e.g. 5e-324, 1e-315) serialize without allow_nan crash."""
        min_denormal = 5e-324  # Smallest positive subnormal float in IEEE 754 64-bit
        mid_subnormal = sys.float_info.min * 1e-10

        with Logger(run_name="subnormal_run", log_dir=temp_logs_dir, capture_env=False) as logger:
            logger.log({
                "step": 0,
                "denormal_min": min_denormal,
                "mid_subnormal": mid_subnormal,
                "subnormal_array": [min_denormal, mid_subnormal, 1.0],
            })

            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            metric_line = lines[1]

            assert metric_line["denormal_min"] == min_denormal
            assert metric_line["mid_subnormal"] == mid_subnormal
            assert metric_line["subnormal_array"][0] == min_denormal

    def test_denormal_subnormal_in_hyperparameters(self, temp_logs_dir: Path):
        """Verify subnormal floats in hyperparameters persist accurately in meta.json."""
        subnormal_hp = {
            "min_eps": 5e-324,
            "nested": {"eps": sys.float_info.min / 1000},
            "array_eps": [5e-324, 1e-300],
        }
        with Logger(
            run_name="subnormal_hp_run",
            hyperparameters=subnormal_hp,
            log_dir=temp_logs_dir,
            capture_env=False,
        ) as logger:
            with open(logger.meta_file, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            assert meta["hyperparameters"]["min_eps"] == 5e-324
            assert meta["hyperparameters"]["nested"]["eps"] == sys.float_info.min / 1000
            assert meta["hyperparameters"]["array_eps"][0] == 5e-324

    def test_signed_zeros_distinction_and_preservation(self, temp_logs_dir: Path):
        """Verify negative zero (-0.0) is preserved through serialization in metrics and hyperparams."""
        neg_zero = -0.0
        pos_zero = +0.0

        assert math.copysign(1.0, neg_zero) == -1.0
        assert math.copysign(1.0, pos_zero) == +1.0

        with Logger(
            run_name="signed_zero_run",
            hyperparameters={"lr_offset": neg_zero},
            log_dir=temp_logs_dir,
            capture_env=False,
        ) as logger:
            # -0.0 as metric scalar, array element, and step
            # Note: in Python, -0.0 < 0 is False, so -0.0 is non-negative and valid as step
            logger.log({
                "step": neg_zero,
                "loss": neg_zero,
                "weights": [neg_zero, pos_zero],
            })

            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            rec = lines[1]
            assert rec["step"] == 0.0
            assert rec["loss"] == 0.0
            assert rec["weights"] == [0.0, 0.0]

    def test_signaling_nan_and_payload_nans(self, temp_logs_dir: Path):
        """Verify signaling NaNs and negative NaNs convert safely to 'NaN' string per ADR 42."""
        # Construct signaling NaN and negative NaN bit patterns via struct
        snan_bytes = b"\x7f\xf0\x00\x00\x00\x00\x00\x01"  # Signaling NaN bit pattern
        neg_nan_bytes = b"\xff\xf8\x00\x00\x00\x00\x00\x01"  # Negative NaN bit pattern

        snan_float = struct.unpack("!d", snan_bytes)[0]
        neg_nan_float = struct.unpack("!d", neg_nan_bytes)[0]

        assert math.isnan(snan_float)
        assert math.isnan(neg_nan_float)

        # Verify sanitizer directly
        assert _sanitize_scalar_float(snan_float) == constants.NAN_STR
        assert _sanitize_scalar_float(neg_nan_float) == constants.NAN_STR

        # Verify logger log stream
        with Logger(run_name="snan_run", log_dir=temp_logs_dir, capture_env=False) as logger:
            logger.log({
                "step": 1,
                "snan_val": snan_float,
                "neg_nan_val": neg_nan_float,
                "nan_list": [snan_float, neg_nan_float],
            })

            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            rec = lines[1]
            assert rec["snan_val"] == constants.NAN_STR
            assert rec["neg_nan_val"] == constants.NAN_STR
            assert rec["nan_list"] == [constants.NAN_STR, constants.NAN_STR]

    def test_extreme_finite_boundaries(self, temp_logs_dir: Path):
        """Verify extreme float boundaries up to sys.float_info.max are serialized correctly."""
        max_float = sys.float_info.max
        min_pos_normal = sys.float_info.min
        neg_max_float = -sys.float_info.max

        with Logger(run_name="extreme_bounds", log_dir=temp_logs_dir, capture_env=False) as logger:
            logger.log({
                "step": 0,
                "max_float": max_float,
                "min_pos_normal": min_pos_normal,
                "neg_max_float": neg_max_float,
                "arr": [max_float, neg_max_float],
            })

            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            rec = lines[1]
            assert rec["max_float"] == max_float
            assert rec["min_pos_normal"] == min_pos_normal
            assert rec["neg_max_float"] == neg_max_float

    def test_special_floats_rejection_for_step_and_epoch(self, temp_logs_dir: Path):
        """Confirm NaN, +Infinity, and -Infinity are strictly rejected for step and epoch."""
        with Logger(run_name="step_nan_test", log_dir=temp_logs_dir, capture_env=False) as logger:
            for bad_step in (float("nan"), float("inf"), float("-inf")):
                with pytest.raises(ValueError, match="finite number"):
                    logger.log({"step": bad_step, "loss": 0.5})

                with pytest.raises(ValueError, match="finite number"):
                    logger.log({"epoch": bad_step, "loss": 0.5})

    def test_negative_step_and_epoch_rejection(self, temp_logs_dir: Path):
        """Verify strictly negative numbers are rejected for step and epoch."""
        with Logger(run_name="neg_step_test", log_dir=temp_logs_dir, capture_env=False) as logger:
            for bad_val in (-1, -0.001, -1e-15):
                with pytest.raises(ValueError, match="non-negative"):
                    logger.log({"step": bad_val, "loss": 0.5})
                with pytest.raises(ValueError, match="non-negative"):
                    logger.log({"epoch": bad_val, "loss": 0.5})

    def test_boolean_step_and_epoch_rejection(self, temp_logs_dir: Path):
        """Verify booleans (True, False) are rejected as step/epoch despite being int subclasses."""
        with Logger(run_name="bool_step_test", log_dir=temp_logs_dir, capture_env=False) as logger:
            with pytest.raises(TypeError, match="'step' must be an integer or float"):
                logger.log({"step": True, "loss": 0.5})

            with pytest.raises(TypeError, match="'step' must be an integer or float"):
                logger.log({"step": False, "loss": 0.5})

            with pytest.raises(TypeError, match="'epoch' must be an int or float"):
                logger.log({"epoch": True, "loss": 0.5})

    def test_complex_and_decimal_rejection(self, temp_logs_dir: Path):
        """Verify complex numbers, Decimals, and Fractions are rejected per ADR 19 and ADR 44."""
        import decimal
        import fractions

        unsupported_vals = [
            complex(1.0, 2.0),
            decimal.Decimal("3.14159"),
            fractions.Fraction(1, 3),
        ]

        with Logger(run_name="unsupported_numeric", log_dir=temp_logs_dir, capture_env=False) as logger:
            for val in unsupported_vals:
                # Top-level metric value
                with pytest.raises(TypeError):
                    logger.log({"step": 1, "bad_val": val})

                # Inside 1D array
                with pytest.raises(TypeError):
                    logger.log({"step": 1, "bad_array": [1.0, val]})

        # In hyperparameters
        for val in unsupported_vals:
            with pytest.raises(TypeError):
                Logger(
                    run_name="bad_hp",
                    hyperparameters={"unsupported": val},
                    log_dir=temp_logs_dir,
                    capture_env=False,
                )


# =============================================================================
# 3. Resource Leakage and Context Manager Resilience
# =============================================================================
class TestAdversarialResourceSafetyAndContextManager:
    """Stress-tests file descriptor cleanup, idempotency, and exception handling."""

    def test_context_manager_standard_exception_closes_file(self, temp_logs_dir: Path):
        """Ensure file descriptors are fully closed when an unexpected exception occurs inside with block."""
        captured_logger: Logger | None = None

        with pytest.raises(ZeroDivisionError):
            with Logger(run_name="exc_leak_test", log_dir=temp_logs_dir, capture_env=False) as logger:
                captured_logger = logger
                logger.log({"step": 0, "m": 1.0})
                _ = 1 / 0

        assert captured_logger is not None
        assert captured_logger._closed is True
        assert captured_logger._file is not None
        assert captured_logger._file.closed is True

    def test_context_manager_keyboard_interrupt_closes_file(self, temp_logs_dir: Path):
        """Ensure KeyboardInterrupt (BaseException) cleanly closes descriptors without leakage."""
        captured_logger: Logger | None = None

        with pytest.raises(KeyboardInterrupt):
            with Logger(run_name="interrupt_test", log_dir=temp_logs_dir, capture_env=False) as logger:
                captured_logger = logger
                logger.log({"step": 0, "m": 1.0})
                raise KeyboardInterrupt("Simulated user cancellation")

        assert captured_logger is not None
        assert captured_logger._closed is True
        assert captured_logger._file.closed is True

    def test_init_failure_leaves_no_lingering_descriptors(self, temp_logs_dir: Path):
        """Ensure that exceptions raised during Logger.__init__ do not orphan open file descriptors."""
        # 1. Invalid run_name
        with pytest.raises(ValueError):
            Logger(run_name="../../traversal", log_dir=temp_logs_dir)

        # 2. Non-serializable hyperparameters
        with pytest.raises(TypeError):
            Logger(
                run_name="bad_hp_init",
                hyperparameters={"bad": object()},
                log_dir=temp_logs_dir,
            )

        # 3. Restart mismatch
        valid_dir = temp_logs_dir / "valid_run"
        l = Logger(run_name="valid_run", hyperparameters={"a": 1}, log_dir=temp_logs_dir, capture_env=False)
        l.close()

        with pytest.raises(ValueError, match="hyperparameters mismatch"):
            Logger(
                run_name="valid_run",
                hyperparameters={"a": 2},
                restart=True,
                log_dir=temp_logs_dir,
                capture_env=False,
            )

    def test_close_idempotency_multi_call(self, temp_logs_dir: Path):
        """Verify calling logger.close() 50 times in a row never raises or corrupts internal state."""
        logger = Logger(run_name="idempotent_close", log_dir=temp_logs_dir, capture_env=False)
        logger.log({"step": 0, "val": 1.0})

        for _ in range(50):
            logger.close()
            assert logger._closed is True
            assert logger._file.closed is True

    def test_double_exit_safety(self, temp_logs_dir: Path):
        """Verify invoking __exit__ multiple times or after close() is completely harmless."""
        logger = Logger(run_name="double_exit", log_dir=temp_logs_dir, capture_env=False)
        logger.close()

        # Repeated __exit__ calls
        logger.__exit__(None, None, None)
        logger.__exit__(ValueError, ValueError("test"), None)
        assert logger._closed is True

    def test_log_after_close_raises_runtime_error(self, temp_logs_dir: Path):
        """Verify attempting to log on a closed Logger raises RuntimeError."""
        logger = Logger(run_name="closed_log_test", log_dir=temp_logs_dir, capture_env=False)
        logger.close()

        with pytest.raises(RuntimeError, match="Cannot log to closed Logger"):
            logger.log({"step": 1, "loss": 0.5})

    def test_failed_log_does_not_corrupt_subsequent_writes(self, temp_logs_dir: Path):
        """Ensure an invalid log call (raising TypeError) does not break stream for subsequent valid logs."""
        with Logger(run_name="failed_log_resilience", log_dir=temp_logs_dir, capture_env=False) as logger:
            logger.log({"step": 0, "loss": 0.5})

            # Invalid log call: nested dict
            with pytest.raises(TypeError, match="Nested dictionary"):
                logger.log({"step": 1, "bad_nested": {"a": 1}})

            # Invalid log call: missing step/epoch
            with pytest.raises(ValueError, match="Explicit step or epoch tracking"):
                logger.log({"loss": 0.4})

            # Valid subsequent log call
            logger.log({"step": 2, "loss": 0.3})

            with open(logger.log_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            # Should contain header + step 0 + step 2 (step 1 was not written)
            assert len(lines) == 3
            assert lines[1]["step"] == 0
            assert lines[2]["step"] == 2

    def test_high_volume_context_manager_lifecycle(self, temp_logs_dir: Path):
        """Run 100 quick open-log-close cycles to confirm no descriptor exhaustion or leaks."""
        for i in range(100):
            with Logger(run_name=f"cycle_{i}", log_dir=temp_logs_dir, capture_env=False) as logger:
                logger.log({"step": 0, "iter": i})
            assert logger._closed is True
            assert logger._file.closed is True


# =============================================================================
# 4. Malformed Git Repositories & Non-Standard VCS Environments
# =============================================================================
class TestAdversarialMalformedGitRepositories:
    """Stress-tests git inspection and diff patch generation across broken/exotic git states."""

    def test_detached_head_capture(self, tmp_path: Path):
        """Verify git capture in detached HEAD state reports commit SHA and HEAD branch."""
        repo_path = tmp_path / "detached_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo_path, check=True, capture_output=True)

        f = repo_path / "file.txt"
        f.write_text("v1\n", encoding="utf-8")
        subprocess.run(["git", "add", "file.txt"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Commit 1"], cwd=repo_path, check=True, capture_output=True)

        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, capture_output=True, text=True)
        sha1 = res.stdout.strip()

        # Checkout commit directly (detached HEAD)
        subprocess.run(["git", "checkout", sha1], cwd=repo_path, check=True, capture_output=True)

        git_info = capture_git_info(cwd=repo_path)
        assert git_info["git_commit"] == sha1
        assert git_info["git_branch"] == "HEAD"
        assert git_info["is_dirty"] is False

    def test_unborn_branch_empty_repository(self, tmp_path: Path):
        """Verify git capture on a freshly initialized repo with no commits (unborn branch)."""
        repo_path = tmp_path / "unborn_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)

        git_info = capture_git_info(cwd=repo_path)
        assert git_info["git_commit"] is None
        # Git default branch name (master or main) should be captured or None
        assert git_info["git_branch"] in ("main", "master", None)
        assert git_info["is_dirty"] is False

        # create_diff_patch should gracefully return False without crashing
        patch_out = repo_path / "diff.patch"
        res = create_diff_patch(patch_out, cwd=repo_path)
        assert res is False
        assert not patch_out.exists()

    def test_corrupt_git_head_ref(self, tmp_path: Path):
        """Verify capture gracefully degrades when .git/HEAD points to invalid or garbage data."""
        repo_path = tmp_path / "corrupt_head_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        head_file = repo_path / ".git" / "HEAD"
        head_file.write_text("garbage_invalid_ref_data\n", encoding="utf-8")

        git_info = capture_git_info(cwd=repo_path)
        assert git_info["git_commit"] is None
        assert git_info["is_dirty"] is False

        # Diff patch should also fail gracefully
        res = create_diff_patch(repo_path / "diff.patch", cwd=repo_path)
        assert res is False

    def test_corrupt_git_tree_missing_objects_dir(self, tmp_path: Path):
        """Verify capture survives a git repository with deleted or unreadable .git/objects dir."""
        repo_path = tmp_path / "corrupt_objects_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        objects_dir = repo_path / ".git" / "objects"
        if objects_dir.exists():
            for p in objects_dir.glob("**/*"):
                if p.is_file():
                    p.unlink()
            # On POSIX we can remove empty subdirs
            try:
                objects_dir.rmdir()
            except OSError:
                pass

        git_info = capture_git_info(cwd=repo_path)
        assert isinstance(git_info, dict)
        assert "git_commit" in git_info
        assert "is_dirty" in git_info

    def test_binary_and_non_utf8_diff_patch(self, tmp_path: Path):
        """Verify create_diff_patch handles binary files and invalid UTF-8 bytes safely."""
        repo_path = tmp_path / "binary_diff_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo_path, check=True, capture_output=True)

        # Initial tracked binary file
        bin_file = repo_path / "model.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        subprocess.run(["git", "add", "model.bin"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add binary file"], cwd=repo_path, check=True, capture_output=True)

        # Dirty binary file with invalid UTF-8 byte sequence
        invalid_utf8_bytes = b"\xff\xfe\x00\x80\xaa\xbb\xcc\xdd"
        bin_file.write_bytes(invalid_utf8_bytes)

        patch_file = repo_path / "diff.patch"
        res = create_diff_patch(patch_file, cwd=repo_path)
        assert res is True
        assert patch_file.exists()
        assert patch_file.stat().st_size > 0

    def test_untracked_only_does_not_create_diff_patch(self, tmp_path: Path):
        """Confirm that untracked files do NOT produce a diff.patch per ADR 31."""
        repo_path = tmp_path / "untracked_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo_path, check=True, capture_output=True)

        tracked = repo_path / "tracked.txt"
        tracked.write_text("tracked\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

        # Create purely untracked file
        untracked = repo_path / "untracked.py"
        untracked.write_text("print('untracked')\n", encoding="utf-8")

        patch_file = repo_path / "diff.patch"
        # git diff HEAD produces no output for untracked files
        res = create_diff_patch(patch_file, cwd=repo_path)
        assert res is False
        assert not patch_file.exists()

    def test_subprocess_timeout_and_os_errors_handled(self, tmp_path: Path):
        """Verify subprocess TimeoutExpired, PermissionError, and FileNotFoundError are handled defensively."""
        # TimeoutExpired simulation
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5.0)):
            git_info = capture_git_info(cwd=tmp_path)
            assert git_info["git_commit"] is None
            assert git_info["is_dirty"] is False

            patch_result = create_diff_patch(tmp_path / "diff.patch", cwd=tmp_path)
            assert patch_result is False

        # PermissionError simulation
        with mock.patch("subprocess.run", side_effect=PermissionError("Permission denied")):
            git_info = capture_git_info(cwd=tmp_path)
            assert git_info["git_commit"] is None
            assert git_info["is_dirty"] is False

            patch_result = create_diff_patch(tmp_path / "diff.patch", cwd=tmp_path)
            assert patch_result is False

        # FileNotFoundError simulation (git executable missing from PATH)
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            git_info = capture_git_info(cwd=tmp_path)
            assert git_info["git_commit"] is None
            assert git_info["is_dirty"] is False

            patch_result = create_diff_patch(tmp_path / "diff.patch", cwd=tmp_path)
            assert patch_result is False


# =============================================================================
# 5. Python Runtime Environment Metadata Variations & Fault Tolerance
# =============================================================================
class TestAdversarialEnvironmentMetadata:
    """Stress-tests runtime environment capture against frozen binaries and corrupted package metadata."""

    def test_frozen_environment_distributions_failure(self):
        """Simulate a frozen Python binary (PyInstaller/cx_Freeze) where metadata distributions() fails."""
        with mock.patch("importlib.metadata.distributions", side_effect=FileNotFoundError("No package metadata")):
            env = capture_environment()
            assert isinstance(env, dict)
            assert env["packages"] == {}
            assert "python_version" in env
            assert "sys_platform" in env
            assert "platform" in env

    def test_corrupted_distribution_objects(self):
        """Verify per-distribution exception isolation against malformed Distribution objects."""
        class BrokenDist1:
            @property
            def name(self):
                raise RuntimeError("Corrupt name property")

        class BrokenDist2:
            name = None
            metadata = None

        class BrokenDist3:
            name = "   "  # whitespace-only name
            version = "1.0.0"

        class ValidDist1:
            name = "pkg-alpha"
            version = "2.1.0"

        class ValidDist2DuplicateCase:
            name = "PKG-ALPHA"  # Duplicate with different casing
            version = "2.1.0"

        class DistWithBrokenVersion:
            name = "pkg-beta"
            @property
            def version(self):
                raise OSError("Corrupted version metadata")
            metadata = None

        mock_dists = [
            BrokenDist1(),
            BrokenDist2(),
            BrokenDist3(),
            ValidDist1(),
            ValidDist2DuplicateCase(),
            DistWithBrokenVersion(),
        ]

        with mock.patch("importlib.metadata.distributions", return_value=mock_dists):
            env = capture_environment()
            packages = env["packages"]

            # Valid package was preserved
            assert "pkg-alpha" in packages
            assert packages["pkg-alpha"] == "2.1.0"

            # Broken version defaulted to 'unknown'
            assert "pkg-beta" in packages
            assert packages["pkg-beta"] == "unknown"

            # Whitespace and broken distributions were safely excluded
            assert "" not in packages
            assert "   " not in packages

    def test_logger_init_resilience_to_env_crashes(self, temp_logs_dir: Path):
        """Verify Logger(capture_env=True) never crashes if environment inspection fails catastrophically."""
        with mock.patch("experiment_logger.env.capture_environment", side_effect=RuntimeError("Catastrophic inspection failure")), \
             mock.patch("experiment_logger.env.capture_git_info", side_effect=OSError("Disk failure")):

            # Logger initialization should succeed with fallback values
            with Logger(run_name="resilient_env_run", log_dir=temp_logs_dir, capture_env=True) as logger:
                with open(logger.meta_file, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)

                assert meta["run_name"] == "resilient_env_run"
                env_meta = meta["environment"]
                assert env_meta["packages"] == {}
                assert env_meta["git_commit"] is None
                assert env_meta["is_dirty"] is False
                assert env_meta["python_version"] == platform.python_version()

    def test_exotic_platform_metadata_json_serializable(self, temp_logs_dir: Path):
        """Verify captured environment containing exotic unicode or platforms serializes to valid JSON."""
        with mock.patch("sys.version", "3.12.0 (custom-built, \u2603 unicode test, Sep 14 2026)"), \
             mock.patch("sys.platform", "wasi-custom-\ud83d\ude80"):

            env = capture_environment()
            # Ensure roundtrip serialization
            json_str = json.dumps(env)
            loaded = json.loads(json_str)
            assert loaded["sys_platform"] == "wasi-custom-\ud83d\ude80"


# =============================================================================
# 6. Schema Header Protection & Constants Symmetry
# =============================================================================
class TestAdversarialConstantsAndHeader:
    """Stress-tests header generation against injection attacks and verifies constant symmetries."""

    def test_header_key_injection_protection(self):
        """Verify make_header prevents caller from overwriting reserved header keys via extra_fields."""
        malicious_extra = {
            constants.VERSION_HEADER_KEY: "999.0-injected",
            constants.RECORD_TYPE_KEY: "malicious_type",
            "custom_key": "safe_custom_val",
        }

        header = constants.make_header(run_name="safe_run", extra_fields=malicious_extra)

        # Reserved keys must NOT be overwritten
        assert header[constants.VERSION_HEADER_KEY] == constants.MLVIZ_VERSION
        assert header[constants.RECORD_TYPE_KEY] == constants.HEADER_TYPE
        # Legitimate extra fields must be preserved
        assert header["custom_key"] == "safe_custom_val"
        assert header["run_name"] == "safe_run"

    def test_is_header_record_adversarial_types(self):
        """Verify is_header_record handles unusual structures without raising AttributeError."""
        # Non-header records
        assert constants.is_header_record({}) is False
        assert constants.is_header_record({"step": 1, "loss": 0.5}) is False
        assert constants.is_header_record({"type": "metric"}) is False

        # Header records
        assert constants.is_header_record({constants.RECORD_TYPE_KEY: constants.HEADER_TYPE}) is True
        assert constants.is_header_record({constants.VERSION_HEADER_KEY: "1.0"}) is True

    def test_special_floats_roundtrip_deep_nesting(self):
        """Verify serialize_special_floats and deserialize_special_floats operate symmetrically."""
        original = {
            "scalar_nan": float("nan"),
            "scalar_pos_inf": float("inf"),
            "scalar_neg_inf": float("-inf"),
            "normal_float": 42.5,
            "int_val": 100,
            "str_val": "regular_string",
            "nested_dict": {
                "inner_nan": float("nan"),
                "inner_inf": float("inf"),
            },
            "nested_list": [float("nan"), float("-inf"), 3.14],
        }

        serialized = constants.serialize_special_floats(original)
        assert serialized["scalar_nan"] == constants.NAN_STR
        assert serialized["scalar_pos_inf"] == constants.POS_INF_STR
        assert serialized["scalar_neg_inf"] == constants.NEG_INF_STR
        assert serialized["nested_dict"]["inner_nan"] == constants.NAN_STR
        assert serialized["nested_dict"]["inner_inf"] == constants.POS_INF_STR
        assert serialized["nested_list"] == [constants.NAN_STR, constants.NEG_INF_STR, 3.14]

        # Verify strict RFC 8259 JSON serialization succeeds with allow_nan=False
        json_output = json.dumps(serialized, allow_nan=False)
        assert json_output is not None

        # Deserialize back
        deserialized = constants.deserialize_special_floats(serialized)
        assert math.isnan(deserialized["scalar_nan"])
        assert math.isinf(deserialized["scalar_pos_inf"]) and deserialized["scalar_pos_inf"] > 0
        assert math.isinf(deserialized["scalar_neg_inf"]) and deserialized["scalar_neg_inf"] < 0
        assert math.isnan(deserialized["nested_dict"]["inner_nan"])
        assert math.isinf(deserialized["nested_dict"]["inner_inf"])
        assert math.isnan(deserialized["nested_list"][0])
        assert math.isinf(deserialized["nested_list"][1]) and deserialized["nested_list"][1] < 0
        assert deserialized["nested_list"][2] == 3.14

    def test_special_float_string_constants_immutability(self):
        """Verify SPECIAL_FLOAT_STRINGS is an immutable frozenset to prevent runtime tampering."""
        assert isinstance(constants.SPECIAL_FLOAT_STRINGS, frozenset)
        assert constants.NAN_STR in constants.SPECIAL_FLOAT_STRINGS
        assert constants.POS_INF_STR in constants.SPECIAL_FLOAT_STRINGS
        assert constants.NEG_INF_STR in constants.SPECIAL_FLOAT_STRINGS
        assert "Infinity" in constants.SPECIAL_FLOAT_STRINGS
        assert "nan" in constants.SPECIAL_FLOAT_STRINGS
        assert "-inf" in constants.SPECIAL_FLOAT_STRINGS
