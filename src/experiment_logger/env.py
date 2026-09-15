"""
ML Research Logging & Plotting Framework — Phase 1 (Lightweight Logger).

Environment and Git reproducibility capture module.
Governed by ADR 31, R2, and PROJECT.md § Interface Contracts.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


def capture_environment() -> dict[str, Any]:
    """Capture Python runtime environment, platform details, and installed packages.

    Dynamically inspects the active Python environment using standard library
    utilities (`sys`, `platform`, and `importlib.metadata`) without subshell overhead
    or external dependencies. Employs a two-tier fault-tolerance architecture to ensure
    corrupted package distributions or inspection errors never crash logger execution.

    Returns:
        dict[str, Any]: Environment state dictionary containing:
            - "python_version" (str): Python runtime compiler and version string (sys.version).
            - "sys_platform" (str): System platform identifier matching sys.platform.
            - "platform" (str): Detailed OS and architecture platform string (platform.platform()).
            - "packages" (dict[str, str]): Sorted mapping of package names to versions.
    """
    packages: dict[str, str] = {}

    # Tier 2: Global environment inspection isolation
    try:
        dists = importlib.metadata.distributions()
        for dist in dists:
            # Tier 1: Per-distribution exception isolation
            try:
                name = None
                try:
                    name = getattr(dist, "name", None)
                except Exception:
                    pass
                if not name:
                    try:
                        meta = getattr(dist, "metadata", None)
                        if meta is not None:
                            name = meta.get("Name") if hasattr(meta, "get") else meta["Name"]
                    except Exception:
                        name = None

                version = None
                try:
                    version = getattr(dist, "version", None)
                except Exception:
                    pass
                if not version:
                    try:
                        meta = getattr(dist, "metadata", None)
                        if meta is not None:
                            version = meta.get("Version") if hasattr(meta, "get") else meta["Version"]
                    except Exception:
                        version = None

                if name:
                    clean_name = str(name).strip()
                    clean_ver = str(version).strip() if version is not None else "unknown"
                    if clean_name and clean_name not in packages:
                        packages[clean_name] = clean_ver
            except Exception:
                # Defensive skip of corrupt individual distribution metadata
                continue
    except Exception:
        # Global inspection failure fallback (e.g. monkeypatched distributions)
        packages = {}

    # Sort packages alphabetically (case-insensitively) for deterministic serialization
    sorted_packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))

    return {
        "python_version": sys.version,
        "sys_platform": sys.platform,
        "platform": platform.platform(),
        "packages": sorted_packages,
    }


def capture_git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """Capture Git repository commit SHA, active branch, and dirty status.

    Parameters
    ----------
    cwd : str | Path | None, optional
        Working directory of the repository to inspect. If None, defaults
        to Path.cwd().

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "git_commit": str | None (40-hex SHA-1 string, or None)
        - "git_branch": str | None (active branch name string, or None)
        - "is_dirty": bool (True if uncommitted changes exist, False otherwise)
        - "git_dirty": bool (alias of is_dirty for backward compatibility)
    """
    fallback_payload: dict[str, Any] = {
        "git_commit": None,
        "git_branch": None,
        "is_dirty": False,
        "git_dirty": False,
    }

    if cwd is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(cwd)
        if not target_dir.exists() or not target_dir.is_dir():
            return fallback_payload

    try:
        # Fast work tree check
        proc_check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc_check.returncode != 0 or proc_check.stdout.strip().lower() != "true":
            return fallback_payload

        # 1. Commit SHA-1 (HEAD)
        git_commit: str | None = None
        proc_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc_commit.returncode == 0:
            commit_sha = proc_commit.stdout.strip()
            if len(commit_sha) == 40 and re.match(r"^[0-9a-f]{40}$", commit_sha):
                git_commit = commit_sha

        # 2. Branch name
        git_branch: str | None = None
        proc_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc_branch.returncode == 0:
            branch_name = proc_branch.stdout.strip()
            if branch_name:
                git_branch = branch_name
        else:
            # Fallback for newly initialized repos without commits
            proc_branch2 = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if proc_branch2.returncode == 0 and proc_branch2.stdout.strip():
                git_branch = proc_branch2.stdout.strip()

        # 3. Dirty status
        is_dirty = False
        proc_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc_status.returncode == 0:
            is_dirty = bool(proc_status.stdout.strip())

        return {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "is_dirty": is_dirty,
            "git_dirty": is_dirty,
        }

    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError, ValueError):
        return fallback_payload


def create_diff_patch(output_path: str | Path, cwd: str | Path | None = None) -> bool:
    """Generate and write a unified diff patch of uncommitted changes against HEAD.

    Executes `git diff HEAD` to capture both staged and unstaged tracked changes.
    Writes patch to `output_path`. Returns False if clean, non-git, or git fails.
    Captures raw bytes to handle binary diffs safely without UnicodeDecodeError.

    Parameters
    ----------
    output_path : str | Path
        Target filesystem path for diff.patch.
    cwd : str | Path | None, optional
        Working directory of the repository. If None, defaults to Path.cwd().

    Returns
    -------
    bool
        True if uncommitted changes were found and written to output_path, False otherwise.
    """
    if cwd is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(cwd)
        if not target_dir.exists() or not target_dir.is_dir():
            return False

    target_path = Path(output_path)

    try:
        # Check if inside git repository
        proc_tree = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc_tree.returncode != 0 or proc_tree.stdout.strip().lower() != "true":
            return False

        # Execute git diff HEAD capturing raw bytes to avoid UnicodeDecodeError on binary files
        proc_diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        if proc_diff.returncode != 0 or not proc_diff.stdout.strip():
            return False

        # Ensure parent directory exists before writing
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(proc_diff.stdout)
        return True

    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError, ValueError):
        return False
