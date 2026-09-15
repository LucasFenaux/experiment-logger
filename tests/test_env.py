"""
Milestone 2 Unit Test Suite: Environment & Git State Capture (ADR 31, R2).

Covers:
- Feature 22: Python runtime version capture
- Feature 23: Installed package state capture via importlib.metadata
- Feature 24: Git commit SHA and branch capture
- Feature 25: Uncommitted changes diff.patch generation
- Feature 26: Defensive fallback for non-git environments & missing git binaries
- Feature 27: Clean git tree handling (omission of diff.patch)
- Integration with Logger lifecycle and meta.json persistence
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from experiment_logger import constants
from experiment_logger.env import (
    capture_environment,
    capture_git_info,
    create_diff_patch,
)
from experiment_logger.logger import Logger


# =============================================================================
# 1. Environment & Package Capture Unit Tests (Features 22 & 23, ADR 31)
# =============================================================================
class TestCaptureEnvironment:
    def test_capture_environment_returns_dict(self):
        """Verify capture_environment() returns a standard dictionary."""
        env = capture_environment()
        assert isinstance(env, dict)

    def test_capture_environment_required_keys(self):
        """Verify all mandatory keys exist in the environment payload."""
        env = capture_environment()
        assert "python_version" in env
        assert "packages" in env
        assert "sys_platform" in env
        assert "platform" in env

    def test_capture_environment_python_version_format(self):
        """Verify python_version contains the major and minor interpreter version strings."""
        env = capture_environment()
        pv = env["python_version"]
        assert isinstance(pv, str)
        assert len(pv) > 0
        assert str(sys.version_info.major) in pv
        assert str(sys.version_info.minor) in pv

    def test_capture_environment_sys_platform_matches_system(self):
        """Verify sys_platform exactly matches sys.platform."""
        env = capture_environment()
        assert env["sys_platform"] == sys.platform

    def test_capture_environment_packages_mapping_types(self):
        """Verify packages is a dictionary mapping string names to string versions."""
        env = capture_environment()
        packages = env["packages"]
        assert isinstance(packages, dict)
        for pkg_name, pkg_version in packages.items():
            assert isinstance(pkg_name, str)
            assert isinstance(pkg_version, str)

    def test_capture_environment_packages_contains_installed_distributions(self):
        """Verify standard testing dependencies (pytest) appear in captured packages."""
        env = capture_environment()
        packages = env["packages"]
        lower_keys = {k.lower() for k in packages.keys()}
        assert "pytest" in lower_keys

    def test_capture_environment_packages_are_sorted_deterministically(self):
        """Verify packages dictionary keys are sorted alphabetically."""
        env = capture_environment()
        keys = list(env["packages"].keys())
        sorted_keys = sorted(keys, key=lambda s: s.lower())
        assert keys == sorted_keys

    def test_capture_environment_json_serializable(self):
        """Verify the captured environment payload is fully RFC-8259 JSON serializable."""
        env = capture_environment()
        serialized = json.dumps(env, allow_nan=False)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["python_version"] == env["python_version"]
        assert deserialized["sys_platform"] == env["sys_platform"]

    def test_capture_environment_resilience_to_distributions_exception(self, monkeypatch):
        """Verify capture_environment() returns empty packages dict if importlib.metadata raises."""
        import importlib.metadata

        def mock_distributions():
            raise RuntimeError("Corrupted package metadata database")

        monkeypatch.setattr(importlib.metadata, "distributions", mock_distributions)
        env = capture_environment()
        assert isinstance(env, dict)
        assert "python_version" in env
        assert env["packages"] == {}

    def test_capture_environment_resilience_to_malformed_single_distribution(self, monkeypatch):
        """Verify a single corrupted distribution does not abort environment capture."""
        import importlib.metadata

        class MockGoodDist:
            metadata = {"Name": "ValidPkg"}
            version = "1.0.0"

        class MockBadDist:
            @property
            def metadata(self):
                raise KeyError("Malformed METADATA file")

            version = "0.0.0"

        monkeypatch.setattr(
            importlib.metadata,
            "distributions",
            lambda: [MockGoodDist(), MockBadDist()],
        )
        env = capture_environment()
        assert isinstance(env["packages"], dict)
        assert "ValidPkg" in env["packages"]
        assert env["packages"]["ValidPkg"] == "1.0.0"

    def test_capture_environment_platform_key_present(self):
        """Verify platform field is a non-empty string."""
        env = capture_environment()
        assert isinstance(env.get("platform"), str)
        assert len(env["platform"]) > 0


# =============================================================================
# 2. Git State Capture Unit Tests: Clean & Dirty (Features 24 & 25, ADR 31)
# =============================================================================
class TestCaptureGitInfoCleanAndDirty:
    def test_clean_repo_returns_commit_and_clean_status(self, temp_git_repo: Path):
        """Verify capture_git_info in a clean git repo returns valid commit and is_dirty=False."""
        info = capture_git_info(cwd=temp_git_repo)
        assert isinstance(info, dict)
        assert "git_commit" in info
        assert "git_branch" in info
        assert "is_dirty" in info
        assert info["is_dirty"] is False
        assert info.get("git_dirty") is False

    def test_commit_sha_is_40_hex_characters(self, temp_git_repo: Path):
        """Verify git_commit is a 40-character hexadecimal SHA-1 string."""
        info = capture_git_info(cwd=temp_git_repo)
        sha = info["git_commit"]
        assert isinstance(sha, str)
        assert len(sha) == 40
        assert re.match(r"^[0-9a-f]{40}$", sha) is not None

    def test_branch_name_is_non_empty_string(self, temp_git_repo: Path):
        """Verify git_branch is a non-empty string."""
        info = capture_git_info(cwd=temp_git_repo)
        branch = info["git_branch"]
        assert isinstance(branch, str)
        assert len(branch) > 0

    def test_unstaged_modification_marks_dirty(self, temp_git_repo: Path):
        """Verify modifying a tracked file flags is_dirty=True."""
        tracked_file = temp_git_repo / "initial.txt"
        with open(tracked_file, "a", encoding="utf-8") as f:
            f.write("Uncommitted edit\n")

        info = capture_git_info(cwd=temp_git_repo)
        assert info["is_dirty"] is True
        assert info.get("git_dirty") is True
        assert info["git_commit"] is not None

    def test_staged_modification_marks_dirty(self, temp_git_repo: Path):
        """Verify staging a modified tracked file flags is_dirty=True."""
        tracked_file = temp_git_repo / "initial.txt"
        tracked_file.write_text("Staged content\n", encoding="utf-8")
        subprocess.run(["git", "add", "initial.txt"], cwd=temp_git_repo, check=True, capture_output=True)

        info = capture_git_info(cwd=temp_git_repo)
        assert info["is_dirty"] is True

    def test_staged_new_file_marks_dirty(self, temp_git_repo: Path):
        """Verify staging a newly created file flags is_dirty=True."""
        new_file = temp_git_repo / "feature.py"
        new_file.write_text("x = 42\n", encoding="utf-8")
        subprocess.run(["git", "add", "feature.py"], cwd=temp_git_repo, check=True, capture_output=True)

        info = capture_git_info(cwd=temp_git_repo)
        assert info["is_dirty"] is True

    def test_deleted_tracked_file_marks_dirty(self, temp_git_repo: Path):
        """Verify unlinking a tracked file flags is_dirty=True."""
        tracked_file = temp_git_repo / "initial.txt"
        tracked_file.unlink()

        info = capture_git_info(cwd=temp_git_repo)
        assert info["is_dirty"] is True

    def test_untracked_file_marks_dirty(self, temp_git_repo: Path):
        """Verify adding an untracked file flags is_dirty=True via porcelain status."""
        untracked = temp_git_repo / "scratch.txt"
        untracked.write_text("scratch notes\n", encoding="utf-8")

        info = capture_git_info(cwd=temp_git_repo)
        assert info["is_dirty"] is True

    def test_explicit_path_vs_default_cwd(self, temp_git_repo: Path, monkeypatch):
        """Verify capture_git_info(cwd=None) uses current working directory."""
        monkeypatch.chdir(temp_git_repo)
        info_default = capture_git_info()
        info_explicit = capture_git_info(cwd=temp_git_repo)
        assert info_default["git_commit"] == info_explicit["git_commit"]
        assert info_default["is_dirty"] == info_explicit["is_dirty"]

    def test_supports_string_and_pathlib_cwd(self, temp_git_repo: Path):
        """Verify capture_git_info accepts both str and Path instances for cwd."""
        info_path = capture_git_info(cwd=temp_git_repo)
        info_str = capture_git_info(cwd=str(temp_git_repo))
        assert info_path == info_str


# =============================================================================
# 3. Non-Git Fallback & Resilience Unit Tests (Feature 26)
# =============================================================================
class TestCaptureGitInfoNonGitFallback:
    def test_non_git_directory_returns_null_and_clean(self, non_git_dir: Path):
        """Verify non-git directory returns None for commit/branch and False for is_dirty."""
        info = capture_git_info(cwd=non_git_dir)
        assert info["git_commit"] is None
        assert info["git_branch"] is None
        assert info["is_dirty"] is False
        assert info.get("git_dirty") is False

    def test_empty_repo_without_commits_returns_null_commit(self, tmp_path: Path):
        """Verify a repository with git init but zero commits returns git_commit=None."""
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        subprocess.run(["git", "init"], cwd=empty_repo, check=True, capture_output=True)

        info = capture_git_info(cwd=empty_repo)
        assert info["git_commit"] is None
        assert info["is_dirty"] is False

    def test_missing_git_binary_falls_back_gracefully(self, monkeypatch, temp_git_repo: Path):
        """Verify capture_git_info returns fallback payload if git executable is missing."""
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("[Errno 2] No such file or directory: 'git'")

        monkeypatch.setattr(subprocess, "run", mock_run)
        info = capture_git_info(cwd=temp_git_repo)
        assert info["git_commit"] is None
        assert info["git_branch"] is None
        assert info["is_dirty"] is False

    def test_nonexistent_cwd_falls_back_gracefully(self, tmp_path: Path):
        """Verify passing a nonexistent directory does not crash."""
        ghost_dir = tmp_path / "does_not_exist_abc123"
        info = capture_git_info(cwd=ghost_dir)
        assert info["git_commit"] is None
        assert info["is_dirty"] is False

    def test_subprocess_called_process_error_handled(self, monkeypatch, temp_git_repo: Path):
        """Verify CalledProcessError (exit code 128) is handled without raising."""
        def mock_run(*args, **kwargs):
            raise subprocess.CalledProcessError(128, cmd=["git", "status"])

        monkeypatch.setattr(subprocess, "run", mock_run)
        info = capture_git_info(cwd=temp_git_repo)
        assert info["git_commit"] is None
        assert info["is_dirty"] is False

    def test_subprocess_timeout_handled(self, monkeypatch, temp_git_repo: Path):
        """Verify subprocess.TimeoutExpired is handled without raising."""
        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["git", "status"], timeout=5.0)

        monkeypatch.setattr(subprocess, "run", mock_run)
        info = capture_git_info(cwd=temp_git_repo)
        assert info["git_commit"] is None
        assert info["is_dirty"] is False

    def test_corrupted_git_metadata_handled(self, tmp_path: Path):
        """Verify a directory containing a dummy unreadable .git folder falls back cleanly."""
        corrupt_dir = tmp_path / "corrupt_git"
        corrupt_dir.mkdir()
        fake_git = corrupt_dir / ".git"
        fake_git.write_text("corrupted content not a git dir\n", encoding="utf-8")

        info = capture_git_info(cwd=corrupt_dir)
        assert info["git_commit"] is None
        assert info["is_dirty"] is False


# =============================================================================
# 4. Diff Patch Generation Unit Tests (Features 25 & 27, ADR 31)
# =============================================================================
class TestCreateDiffPatch:
    def test_clean_repo_returns_false_and_no_file(self, temp_git_repo: Path):
        """Verify clean repository returns False and does not create patch file (ADR 31)."""
        patch_path = temp_git_repo / "clean.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is False
        assert not patch_path.exists()

    def test_unstaged_change_writes_patch_and_returns_true(self, temp_git_repo: Path):
        """Verify modifying a tracked file writes non-empty diff.patch and returns True."""
        tracked_file = temp_git_repo / "initial.txt"
        with open(tracked_file, "a", encoding="utf-8") as f:
            f.write("Line added for diff patch test\n")

        patch_path = temp_git_repo / "diff.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        assert patch_path.exists()
        assert patch_path.stat().st_size > 0

    def test_patch_contains_unified_diff_format(self, temp_git_repo: Path):
        """Verify patch content contains unified diff markers."""
        tracked_file = temp_git_repo / "initial.txt"
        with open(tracked_file, "a", encoding="utf-8") as f:
            f.write("Patch format line\n")

        patch_path = temp_git_repo / "diff.patch"
        create_diff_patch(patch_path, cwd=temp_git_repo)

        content = patch_path.read_text(encoding="utf-8")
        assert "diff --git" in content
        assert "initial.txt" in content
        assert "--- a/initial.txt" in content
        assert "+++ b/initial.txt" in content
        assert "+Patch format line" in content

    def test_staged_change_captured_in_patch(self, temp_git_repo: Path):
        """Verify git diff HEAD captures staged changes in diff.patch."""
        new_file = temp_git_repo / "staged_feature.py"
        new_file.write_text("# Staged module\nVALUE = 100\n", encoding="utf-8")
        subprocess.run(["git", "add", "staged_feature.py"], cwd=temp_git_repo, check=True, capture_output=True)

        patch_path = temp_git_repo / "staged.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        content = patch_path.read_text(encoding="utf-8")
        assert "staged_feature.py" in content
        assert "VALUE = 100" in content

    def test_both_staged_and_unstaged_captured_together(self, temp_git_repo: Path):
        """Verify git diff HEAD captures both staged and unstaged changes simultaneously."""
        # 1. Staged change
        file1 = temp_git_repo / "file1.txt"
        file1.write_text("file 1 content\n", encoding="utf-8")
        subprocess.run(["git", "add", "file1.txt"], cwd=temp_git_repo, check=True, capture_output=True)

        # 2. Unstaged change
        tracked = temp_git_repo / "initial.txt"
        with open(tracked, "a", encoding="utf-8") as f:
            f.write("unstaged edit\n")

        patch_path = temp_git_repo / "combined.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        content = patch_path.read_text(encoding="utf-8")
        assert "file1.txt" in content
        assert "initial.txt" in content

    def test_deleted_file_captured_in_patch(self, temp_git_repo: Path):
        """Verify deleted tracked file appears as deletion in diff.patch."""
        tracked = temp_git_repo / "initial.txt"
        tracked.unlink()

        patch_path = temp_git_repo / "deletion.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        content = patch_path.read_text(encoding="utf-8")
        assert "deleted file" in content or "--- a/initial.txt" in content

    def test_non_git_directory_returns_false_no_file(self, non_git_dir: Path):
        """Verify non-git directory returns False and creates no patch file."""
        patch_path = non_git_dir / "patch.diff"
        result = create_diff_patch(patch_path, cwd=non_git_dir)
        assert result is False
        assert not patch_path.exists()

    def test_empty_repo_without_commits_returns_false(self, tmp_path: Path):
        """Verify repository with 0 commits returns False and does not crash."""
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        subprocess.run(["git", "init"], cwd=empty_repo, check=True, capture_output=True)

        patch_path = empty_repo / "patch.diff"
        result = create_diff_patch(patch_path, cwd=empty_repo)
        assert result is False
        assert not patch_path.exists()

    def test_missing_git_binary_returns_false(self, monkeypatch, temp_git_repo: Path):
        """Verify missing git executable returns False and creates no patch file."""
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", mock_run)
        patch_path = temp_git_repo / "patch.diff"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is False
        assert not patch_path.exists()

    def test_binary_file_modification_handled_without_error(self, temp_git_repo: Path):
        """Verify binary file modifications do not trigger UnicodeDecodeError."""
        bin_file = temp_git_repo / "weights.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")
        subprocess.run(["git", "add", "weights.bin"], cwd=temp_git_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add weights"], cwd=temp_git_repo, check=True, capture_output=True)

        # Modify binary file
        bin_file.write_bytes(b"\xff\xfe\x00\x01\x99\x88\x77")
        patch_path = temp_git_repo / "bin.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        assert patch_path.exists()

    def test_creates_parent_directory_if_missing(self, temp_git_repo: Path):
        """Verify create_diff_patch automatically creates missing parent directory paths."""
        tracked = temp_git_repo / "initial.txt"
        with open(tracked, "a", encoding="utf-8") as f:
            f.write("change for nested patch\n")

        patch_path = temp_git_repo / "nested" / "sub_dir" / "diff.patch"
        assert not patch_path.parent.exists()

        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        assert patch_path.exists()
        assert patch_path.parent.is_dir()

    def test_large_diff_handling(self, temp_git_repo: Path):
        """Verify large diff modifications (5,000 lines) are handled efficiently without error."""
        large_file = temp_git_repo / "initial.txt"
        large_file.write_text("data line\n" * 5000, encoding="utf-8")

        patch_path = temp_git_repo / "large.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is True
        assert patch_path.exists()
        assert patch_path.stat().st_size > 5000

    def test_subprocess_timeout_returns_false(self, monkeypatch, temp_git_repo: Path):
        """Verify subprocess timeout returns False and creates no patch file."""
        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=5.0)

        monkeypatch.setattr(subprocess, "run", mock_run)
        patch_path = temp_git_repo / "timeout.patch"
        result = create_diff_patch(patch_path, cwd=temp_git_repo)
        assert result is False
        assert not patch_path.exists()


# =============================================================================
# 5. Logger Integration Unit Tests (ADR 31, ADR 36)
# =============================================================================
class TestLoggerEnvironmentIntegration:
    def test_logger_in_clean_git_repo_saves_meta_omits_patch(self, temp_git_repo: Path, monkeypatch):
        """Verify Logger initialized in a clean git repo records git info and omits diff.patch."""
        monkeypatch.chdir(temp_git_repo)
        logs_dir = temp_git_repo / "logs"

        logger = Logger(run_name="clean_run", log_dir=logs_dir, capture_env=True)
        logger.close()

        meta_file = logs_dir / "clean_run" / constants.META_FILE
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        env_meta = meta["environment"]
        assert env_meta["git_commit"] is not None
        assert len(env_meta["git_commit"]) == 40
        assert env_meta["is_dirty"] is False
        assert "python_version" in env_meta

        diff_file = logs_dir / "clean_run" / constants.DIFF_FILE
        assert not diff_file.exists()

    def test_logger_in_dirty_git_repo_saves_meta_creates_patch(self, temp_git_repo: Path, monkeypatch):
        """Verify Logger in a dirty git repo auto-generates diff.patch and flags is_dirty=True."""
        tracked = temp_git_repo / "initial.txt"
        with open(tracked, "a", encoding="utf-8") as f:
            f.write("dirty change for logger test\n")

        monkeypatch.chdir(temp_git_repo)
        logs_dir = temp_git_repo / "logs"

        logger = Logger(run_name="dirty_run", log_dir=logs_dir, capture_env=True)
        logger.close()

        meta_file = logs_dir / "dirty_run" / constants.META_FILE
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        env_meta = meta["environment"]
        assert env_meta["git_commit"] is not None
        assert env_meta["is_dirty"] is True

        diff_file = logs_dir / "dirty_run" / constants.DIFF_FILE
        assert diff_file.exists()
        assert diff_file.stat().st_size > 0

    def test_logger_in_non_git_dir_saves_clean_meta_omits_patch(self, non_git_dir: Path, monkeypatch):
        """Verify Logger in non-git directory captures Python env, sets git fields null, and logs cleanly."""
        monkeypatch.chdir(non_git_dir)
        logs_dir = non_git_dir / "logs"

        logger = Logger(run_name="non_git_run", log_dir=logs_dir, capture_env=True)
        logger.log({"step": 1, "loss": 0.42})
        logger.close()

        meta_file = logs_dir / "non_git_run" / constants.META_FILE
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        env_meta = meta["environment"]
        assert env_meta["git_commit"] is None
        assert env_meta["is_dirty"] is False
        assert "python_version" in env_meta
        assert "packages" in env_meta

        diff_file = logs_dir / "non_git_run" / constants.DIFF_FILE
        assert not diff_file.exists()

        log_file = logs_dir / "non_git_run" / constants.LOG_FILE
        assert log_file.exists()

    def test_logger_with_capture_env_false_skips_capture(self, temp_git_repo: Path, monkeypatch):
        """Verify Logger(capture_env=False) omits environment metadata and skips diff.patch."""
        tracked = temp_git_repo / "initial.txt"
        with open(tracked, "a", encoding="utf-8") as f:
            f.write("dirty change ignored\n")

        monkeypatch.chdir(temp_git_repo)
        logs_dir = temp_git_repo / "logs"

        logger = Logger(run_name="no_env_run", log_dir=logs_dir, capture_env=False)
        logger.close()

        meta_file = logs_dir / "no_env_run" / constants.META_FILE
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["environment"] == {}

        diff_file = logs_dir / "no_env_run" / constants.DIFF_FILE
        assert not diff_file.exists()

    def test_logger_resumed_run_preserves_existing_environment(self, temp_git_repo: Path, monkeypatch):
        """Verify resumed run (restart=True) preserves initial environment without overwriting diff.patch."""
        monkeypatch.chdir(temp_git_repo)
        logs_dir = temp_git_repo / "logs"

        # Session 1: Clean tree
        logger1 = Logger(run_name="session_run", log_dir=logs_dir, capture_env=True)
        logger1.log({"step": 1, "loss": 0.5})
        logger1.close()

        # Modify repo AFTER session 1
        tracked = temp_git_repo / "initial.txt"
        with open(tracked, "a", encoding="utf-8") as f:
            f.write("late modification\n")

        # Session 2: Resumed run
        logger2 = Logger(run_name="session_run", log_dir=logs_dir, restart=True)
        logger2.log({"step": 2, "loss": 0.4})
        logger2.close()

        diff_file = logs_dir / "session_run" / constants.DIFF_FILE
        # Preserved as clean per Session 1
        assert not diff_file.exists()

    def test_logger_meta_has_both_env_and_git_sections(self, temp_git_repo: Path, monkeypatch):
        """Verify meta.json contains both 'environment' and 'git' sections for contract parity."""
        monkeypatch.chdir(temp_git_repo)
        logs_dir = temp_git_repo / "logs"

        logger = Logger(run_name="parity_run", log_dir=logs_dir, capture_env=True)
        logger.close()

        meta_file = logs_dir / "parity_run" / constants.META_FILE
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        assert "environment" in meta
        assert "git" in meta
        assert meta["git"]["git_commit"] == meta["environment"]["git_commit"]
        assert meta["git"]["is_dirty"] == meta["environment"]["is_dirty"]
