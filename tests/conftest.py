"""
Pytest configuration and shared test fixtures for experiment-logger.

Provides isolated filesystem environments, mock git repositories,
sample metrics/hyperparameters, and corrupted line test factories.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Generator

import pytest


# -----------------------------------------------------------------------------
# 1. Directory & Path Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_logs_dir(tmp_path: Path) -> Path:
    """Provides an isolated logs directory under tmp_path."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


@pytest.fixture
def non_git_dir(tmp_path: Path) -> Path:
    """Provides an isolated directory guaranteed to be outside any Git repository."""
    isolated = tmp_path / "isolated_non_git"
    isolated.mkdir(parents=True, exist_ok=True)
    return isolated


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Fixture that initializes an isolated git repository with an initial commit."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    initial_file = repo_dir / "initial.txt"
    initial_file.write_text("initial repository content\n", encoding="utf-8")

    subprocess.run(["git", "add", "initial.txt"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    return repo_dir


# -----------------------------------------------------------------------------
# 2. Sample Data Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_hyperparameters() -> dict[str, Any]:
    """Provides a realistic set of JSON-serializable experiment hyperparameters."""
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "resnet50",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "epochs": 10,
        "use_amp": True,
        "seed": 42,
    }


@pytest.fixture
def sample_tags() -> list[str]:
    """Provides a standard list of string tags for run classification."""
    return ["baseline", "vision", "seed_42"]


@pytest.fixture
def sample_metrics_step() -> dict[str, Any]:
    """Provides standard step-indexed scalar metrics."""
    return {
        "step": 1,
        "loss": 0.452,
        "accuracy": 0.891,
        "learning_rate": 0.001,
    }


@pytest.fixture
def sample_metrics_epoch() -> dict[str, Any]:
    """Provides epoch-indexed scalar metrics."""
    return {
        "epoch": 1,
        "train_loss": 0.385,
        "val_loss": 0.421,
        "val_accuracy": 0.912,
    }


@pytest.fixture
def sample_1d_array_metrics() -> dict[str, Any]:
    """Provides metrics containing 1D scalar lists per ADR 29."""
    return {
        "step": 1,
        "batch_losses": [0.62, 0.54, 0.48, 0.41],
        "batch_accuracies": [0.72, 0.78, 0.82, 0.86],
    }


@pytest.fixture
def sample_special_floats_metrics() -> dict[str, Any]:
    """Provides metrics containing NaN and +/- Infinity floats per ADR 42."""
    return {
        "step": 1,
        "loss_nan": float("nan"),
        "grad_pos_inf": float("inf"),
        "grad_neg_inf": float("-inf"),
        "array_special": [0.1, float("nan"), float("inf"), float("-inf")],
    }


# -----------------------------------------------------------------------------
# 3. Git Environment Fixture (Isolated Mock Repository)
# -----------------------------------------------------------------------------
class GitRepoController:
    """Controller for interacting with an isolated temporary git repository."""

    def __init__(self, repo_path: Path, head_commit: str) -> None:
        self.path = repo_path
        self.head_commit = head_commit
        self.tracked_file = repo_path / "model.py"

    def make_dirty(self, content: str = "# Modified code for test\n") -> None:
        """Modifies a tracked file to dirty the git working tree."""
        with open(self.tracked_file, "a", encoding="utf-8") as f:
            f.write(content)

    def stage_file(self, filename: str = "new_feature.py", content: str = "x = 1\n") -> None:
        """Creates and stages a new file."""
        target = self.path / filename
        target.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", filename], cwd=self.path, check=True, capture_output=True)

    def add_untracked_file(self, filename: str = "scratch.py", content: str = "temp = True\n") -> None:
        """Creates an untracked file without staging."""
        target = self.path / filename
        target.write_text(content, encoding="utf-8")

    def commit_changes(self, message: str = "Update code") -> str:
        """Commits current changes and updates head_commit."""
        subprocess.run(["git", "add", "."], cwd=self.path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], cwd=self.path, check=True, capture_output=True)
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        )
        self.head_commit = res.stdout.strip()
        return self.head_commit


@pytest.fixture
def mock_git_repo(tmp_path: Path) -> Generator[GitRepoController, None, None]:
    """
    Initializes a real, fully configured temporary Git repository in tmp_path.
    Essential for R2/ADR 31 testing because workspace root is not a git repository.
    """
    repo_path = tmp_path / "mock_git_workspace"
    repo_path.mkdir(parents=True, exist_ok=True)

    # Initialize git repo and local dummy identities
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tester@mlviz.org"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "MLViz Tester"], cwd=repo_path, check=True, capture_output=True)

    # Create initial tracked file and commit
    model_file = repo_path / "model.py"
    model_file.write_text("# Initial model code\nclass ResNet: pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)

    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, capture_output=True, text=True)
    head_sha = res.stdout.strip()

    yield GitRepoController(repo_path, head_sha)


# -----------------------------------------------------------------------------
# 4. Synthetic Run & Corrupted Log Factories
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_run_factory(tmp_path: Path) -> Callable[..., Path]:
    """
    Factory creating complete synthetic run folders with log.jsonl, meta.json,
    and optional diff.patch for standalone LogReader and E2E testing.
    """

    def _create_run(
        run_name: str,
        hyperparameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metric_rows: list[dict[str, Any]] | None = None,
        include_diff_patch: bool = False,
        base_dir: Path | None = None,
    ) -> Path:
        target_dir = (base_dir or (tmp_path / "logs")) / run_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. meta.json
        meta_data: dict[str, Any] = {
            "run_name": run_name,
            "created_at": "2026-09-14T18:00:00Z",
            "hyperparameters": hyperparameters or {"lr": 0.001, "batch_size": 32},
            "tags": tags or ["synthetic"],
            "environment": {
                "python_version": "3.12.11",
                "git_commit": "abcdef1234567890abcdef1234567890abcdef12",
                "git_dirty": include_diff_patch,
                "packages": {"pandas": "2.2.0"},
            },
        }
        (target_dir / "meta.json").write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

        # 2. log.jsonl
        rows = metric_rows or [
            {"step": 1, "loss": 0.5, "accuracy": 0.8, "_timestamp": 1700000000.0, "_time_since_start": 0.1},
            {"step": 2, "loss": 0.4, "accuracy": 0.85, "_timestamp": 1700000001.0, "_time_since_start": 1.1},
        ]
        with open(target_dir / "log.jsonl", "w", encoding="utf-8") as f:
            # Line 1 header per ADR 32
            f.write(json.dumps({"mlviz_version": "1.0", "type": "header", "run_name": run_name}) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")

        # 3. diff.patch if requested
        if include_diff_patch:
            patch_content = (
                "diff --git a/train.py b/train.py\n"
                "--- a/train.py\n"
                "+++ b/train.py\n"
                "@@ -1,1 +1,2 @@\n"
                "+# uncommitted tweak\n"
            )
            (target_dir / "diff.patch").write_text(patch_content, encoding="utf-8")

        return target_dir

    return _create_run


@pytest.fixture
def corrupted_jsonl_factory(tmp_path: Path) -> Callable[..., Path]:
    """
    Factory generating JSONL log files containing various corruption patterns
    to test LogReader resilience per ADR 43.
    """

    def _create_corrupted_file(
        lines: list[str | bytes],
        filename: str = "corrupted_log.jsonl",
    ) -> Path:
        file_path = tmp_path / filename
        with open(file_path, "wb") as f:
            for item in lines:
                if isinstance(item, str):
                    f.write(item.encode("utf-8") + b"\n")
                else:
                    f.write(item + b"\n")
        return file_path

    return _create_corrupted_file
