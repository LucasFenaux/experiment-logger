"""
ML Research Logging & Plotting Framework — Phase 1 (Lightweight Logger).

Lightweight, crash-resilient experiment logging and DataFrame reader.
"""

from __future__ import annotations

from experiment_logger import constants

__version__ = "0.1.0"

# Defensive import guards to support incremental milestone implementation
try:
    from experiment_logger.logger import Logger
except ImportError:  # pragma: no cover
    Logger = None  # type: ignore[assignment,misc]

from experiment_logger.reader import LogReader

from experiment_logger.env import (
    capture_environment,
    capture_git_info,
    create_diff_patch,
)

__all__ = [
    "__version__",
    "constants",
    "Logger",
    "LogReader",
    "capture_environment",
    "capture_git_info",
    "create_diff_patch",
]
