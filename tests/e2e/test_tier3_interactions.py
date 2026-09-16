"""Tier 3: Pairwise cross-feature combinatorial E2E tests.

Validates that individual features do not interfere with each other when used in combination:
- Restarted runs with NaN/Inf metrics
- 1D arrays with custom tags and View DSL queries
- Collision resolution alongside restart mode
- Interleaved corrupted lines with special float deserialization
- Multi-run directory discovery with schema evolution and nested hyperparameters
- Git dirty status and environment capture interactions
- Step vs epoch tracking alongside batch array metrics
"""

import json
import math
import os
import shutil
import time
import warnings
from pathlib import Path
import pytest
import pandas as pd

from experiment_logger import Logger, LogReader
from experiment_logger.env import capture_environment, capture_git_info, create_diff_patch


def test_pairwise_01_restart_and_nan_metrics(tmp_path: Path):
    """Pairwise: Restarted run logs NaN and Inf metrics; verifies continuity and deserialization."""
    hp = {"model": "transformer", "lr": 1e-4}
    with Logger(run_name="restart_nan_run", hyperparameters=hp, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5, "grad_norm": 1.2})
        l1.log({"step": 2, "loss": 0.4, "grad_norm": 1.1})

    # Restart session logging NaN and Inf
    with Logger(run_name="restart_nan_run", hyperparameters=hp, log_dir=tmp_path, restart=True) as l2:
        l2.log({"step": 3, "loss": math.nan, "grad_norm": math.inf})
        l2.log({"step": 4, "loss": 0.35, "grad_norm": 0.95})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "restart_nan_run")
    assert len(df) == 4
    assert list(df["step"]) == [1, 2, 3, 4]
    assert math.isnan(df.loc[df["step"] == 3, "loss"].iloc[0])
    assert math.isinf(df.loc[df["step"] == 3, "grad_norm"].iloc[0])
    assert df.loc[df["step"] == 4, "loss"].iloc[0] == 0.35


def test_pairwise_02_1d_arrays_and_custom_tags(tmp_path: Path):
    """Pairwise: 1D arrays logged under runs configured with custom tags."""
    tags = ["reinforcement-learning", "ppo", "cartpole"]
    with Logger(run_name="rl_array_run", tags=tags, log_dir=tmp_path) as logger:
        logger.log({"step": 1, "rewards": [1.0, 1.0, 0.0, 1.0], "mean_reward": 0.75})
        logger.log({"step": 2, "rewards": [1.0, 1.0, 1.0, 1.0], "mean_reward": 1.0})

    reader = LogReader(logs_dir=tmp_path)
    metadata = reader.get_metadata(tmp_path / "rl_array_run")
    assert metadata["tags"] == tags

    df = reader.read_run(tmp_path / "rl_array_run")
    assert len(df) == 2
    assert len(df["rewards"].iloc[0]) == 4
    assert df["mean_reward"].iloc[1] == 1.0


def test_pairwise_03_collision_and_restart_independence(tmp_path: Path):
    """Pairwise: Collision handling (restart=False) and restart mode (restart=True) operate independently."""
    hp = {"lr": 0.01}
    # Initial run A
    with Logger(run_name="exp_branch", hyperparameters=hp, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "acc": 0.7})

    # Colliding run B with restart=False -> creates timestamped dir
    with Logger(run_name="exp_branch", hyperparameters=hp, log_dir=tmp_path, restart=False) as l2:
        l2.log({"step": 1, "acc": 0.75})

    # Resumed run C with restart=True -> appends specifically to run A
    with Logger(run_name="exp_branch", hyperparameters=hp, log_dir=tmp_path, restart=True) as l3:
        l3.log({"step": 2, "acc": 0.8})

    reader = LogReader(logs_dir=tmp_path)
    df_a = reader.read_run(tmp_path / "exp_branch")
    assert len(df_a) == 2
    assert list(df_a["step"]) == [1, 2]
    assert df_a["acc"].iloc[1] == 0.8


def test_pairwise_04_corrupted_lines_and_special_floats(tmp_path: Path):
    """Pairwise: Corrupted lines interleaved with special floats (NaN, +Inf, -Inf)."""
    run_dir = tmp_path / "corrupt_special_run"
    run_dir.mkdir(parents=True)
    log_file = run_dir / "log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write('{"type": "header", "mlviz_version": "1.0"}\n')
        f.write('{"step": 1, "loss": "NaN", "grad": "+Infinity"}\n')
        f.write('MALFORMED_JSON_LINE_HERE\n')
        f.write('{"step": 2, "loss": 0.5, "bound": "-Infinity"}\n')
        f.write('{unclosed json object\n')
        f.write('{"step": 3, "loss": 0.25, "grad": 1.0}\n')

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(run_dir)

    assert len(df) == 3
    assert list(df["step"]) == [1, 2, 3]
    assert math.isnan(df.loc[df["step"] == 1, "loss"].iloc[0])
    assert math.isinf(df.loc[df["step"] == 1, "grad"].iloc[0]) and df.loc[df["step"] == 1, "grad"].iloc[0] > 0
    assert math.isinf(df.loc[df["step"] == 2, "bound"].iloc[0]) and df.loc[df["step"] == 2, "bound"].iloc[0] < 0
    assert df.loc[df["step"] == 3, "loss"].iloc[0] == 0.25


def test_pairwise_05_multi_run_and_schema_evolution(tmp_path: Path):
    """Pairwise: Multi-run directory discovery combined with schema evolution of hyperparameters and metrics."""
    with Logger(run_name="run_v1", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l:
        l.log({"step": 1, "train_loss": 0.8})

    with Logger(run_name="run_v2", hyperparameters={"lr": 0.005, "weight_decay": 1e-4}, log_dir=tmp_path) as l:
        l.log({"step": 1, "train_loss": 0.7, "val_loss": 0.75})

    with Logger(run_name="run_v3", hyperparameters={"lr": 0.001, "optimizer": "adamw", "weight_decay": 1e-5}, log_dir=tmp_path) as l:
        l.log({"step": 1, "train_loss": 0.6, "val_loss": 0.65, "top1_acc": 0.82})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) == 3
    assert set(df["run_name"]) == {"run_v1", "run_v2", "run_v3"}
    assert "hyperparameters.lr" in df.columns
    assert "hyperparameters.weight_decay" in df.columns
    assert "hyperparameters.optimizer" in df.columns
    assert "top1_acc" in df.columns

    # Verify NaNs for missing fields in v1 and v2
    assert pd.isna(df.loc[df["run_name"] == "run_v1", "hyperparameters.weight_decay"].iloc[0])
    assert pd.isna(df.loc[df["run_name"] == "run_v1", "hyperparameters.optimizer"].iloc[0])
    assert pd.isna(df.loc[df["run_name"] == "run_v2", "top1_acc"].iloc[0])


def test_pairwise_06_git_capture_and_uncommitted_diff(temp_git_repo: Path):
    """Pairwise: Git repository state capture with uncommitted modifications generating diff.patch."""
    (temp_git_repo / "initial.txt").write_text("modified for pairwise test\n", encoding="utf-8")
    log_dir = temp_git_repo / "experiments"

    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        with Logger(run_name="git_diff_pair", log_dir=log_dir, capture_env=True) as logger:
            logger.log({"step": 1, "loss": 0.45})
    finally:
        os.chdir(old_cwd)

    run_path = log_dir / "git_diff_pair"
    patch_path = run_path / "diff.patch"
    assert patch_path.exists()
    assert "modified for pairwise test" in patch_path.read_text(encoding="utf-8")

    meta = LogReader(logs_dir=log_dir).get_metadata(run_path)
    git_meta = meta.get("git") or meta
    assert git_meta.get("is_dirty") is True
    assert git_meta.get("git_commit") is not None


def test_pairwise_07_non_git_and_env_capture(tmp_path: Path):
    """Pairwise: Non-git directory captures Python environment without failing git integration."""
    non_git = tmp_path / "non_git_area"
    non_git.mkdir()
    
    old_cwd = os.getcwd()
    os.chdir(non_git)
    try:
        with Logger(run_name="non_git_run", log_dir=non_git, capture_env=True) as logger:
            logger.log({"step": 1, "loss": 0.3})
    finally:
        os.chdir(old_cwd)

    meta = LogReader(logs_dir=non_git).get_metadata(non_git / "non_git_run")
    env_data = meta.get("env") or meta.get("environment")
    assert env_data is not None
    assert "python_version" in env_data
    assert not (non_git / "non_git_run" / "diff.patch").exists()


def test_pairwise_08_1d_array_containing_nans_and_infs(tmp_path: Path):
    """Pairwise: 1D array containing a mixture of finite numbers, NaN, +Infinity, and -Infinity."""
    mixed_arr = [0.1, math.nan, math.inf, -math.inf, 99.9]
    with Logger(run_name="arr_special_pair", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "metrics_vector": mixed_arr})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "arr_special_pair")
    res = list(df["metrics_vector"].iloc[0])
    assert len(res) == 5
    assert res[0] == 0.1
    assert math.isnan(res[1])
    assert math.isinf(res[2]) and res[2] > 0
    assert math.isinf(res[3]) and res[3] < 0
    assert res[4] == 99.9


def test_pairwise_09_context_manager_and_synchronous_flush(tmp_path: Path):
    """Pairwise: Context manager (__enter__/__exit__) combined with immediate synchronous disk flush."""
    log_file = tmp_path / "cm_flush_run" / "log.jsonl"
    with Logger(run_name="cm_flush_run", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "val": "first"})
        assert log_file.exists()
        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 2

        logger.log({"step": 2, "val": "second"})
        assert len(log_file.read_text(encoding="utf-8").splitlines()) == 3

    # Post context manager verification
    assert len(log_file.read_text(encoding="utf-8").splitlines()) == 3


def test_pairwise_10_epoch_tracking_with_1d_batch_metrics(tmp_path: Path):
    """Pairwise: Epoch-based tracking containing 1D batch metrics (e.g. per-batch loss arrays)."""
    with Logger(run_name="epoch_batch_run", log_dir=tmp_path) as logger:
        logger.log({"epoch": 1, "batch_losses": [0.8, 0.7, 0.6], "epoch_loss": 0.7})
        logger.log({"epoch": 2, "batch_losses": [0.55, 0.5, 0.45], "epoch_loss": 0.5})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "epoch_batch_run")
    assert len(df) == 2
    assert "epoch" in df.columns
    assert list(df["epoch"]) == [1, 2]
    assert len(df["batch_losses"].iloc[0]) == 3
    assert df["epoch_loss"].iloc[1] == 0.5


def test_pairwise_11_restart_with_custom_log_dir(tmp_path: Path):
    """Pairwise: Restart mode targeting a custom nested log directory path."""
    custom_dir = tmp_path / "custom" / "experiments" / "logs"
    hp = {"arch": "resnet18"}
    with Logger(run_name="nested_restart", hyperparameters=hp, log_dir=custom_dir) as l1:
        l1.log({"step": 1, "loss": 1.0})

    with Logger(run_name="nested_restart", hyperparameters=hp, log_dir=custom_dir, restart=True) as l2:
        l2.log({"step": 2, "loss": 0.8})

    df = LogReader(logs_dir=custom_dir).read_run(custom_dir / "nested_restart")
    assert len(df) == 2
    assert list(df["step"]) == [1, 2]


def test_pairwise_12_multi_run_query_filtering(tmp_path: Path):
    """Pairwise: Multi-run aggregation queried using pandas DSL filter patterns."""
    models = ["resnet50", "resnet50", "vit_base"]
    for i, model in enumerate(models):
        with Logger(run_name=f"sweep_{i}", hyperparameters={"model": model, "run_id": i}, log_dir=tmp_path) as l:
            l.log({"step": 1, "accuracy": 0.8 + i * 0.05})

    df = LogReader(logs_dir=tmp_path).read_all()
    filtered = df[df["hyperparameters.model"] == "resnet50"]
    assert len(filtered) == 2
    assert set(filtered["run_name"]) == {"sweep_0", "sweep_1"}


def test_pairwise_13_rapid_logging_and_timestamp_monotonicity(tmp_path: Path):
    """Pairwise: Rapid high-frequency logging verifying timestamp monotonicity."""
    with Logger(run_name="rapid_ts_run", log_dir=tmp_path) as logger:
        for i in range(30):
            logger.log({"step": i, "val": i * 2})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "rapid_ts_run")
    assert len(df) == 30
    assert (df["_time_since_start"].diff().dropna() >= 0.0).all()
    assert (df["_timestamp"].diff().dropna() >= 0.0).all()


def test_pairwise_14_truncated_file_and_restart(tmp_path: Path):
    """Pairwise: Run file truncated mid-write, restarted run resumes and LogReader parses all valid lines."""
    with Logger(run_name="trunc_restart", hyperparameters={"lr": 0.01}, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.6})
        l1.log({"step": 2, "loss": 0.5})

    # Simulate sudden power cut / crash during write of step 3
    log_file = tmp_path / "trunc_restart" / "log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"step": 3, "loss":')  # incomplete line

    # Restart session appending step 4
    with Logger(run_name="trunc_restart", hyperparameters={"lr": 0.01}, log_dir=tmp_path, restart=True) as l2:
        l2.log({"step": 4, "loss": 0.3})

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "trunc_restart")
    assert len(df) == 3
    assert list(df["step"]) == [1, 2, 4]


def test_pairwise_15_nested_hyperparameters_and_multi_run(tmp_path: Path):
    """Pairwise: Multiple runs with nested dictionary hyperparameters aggregated via read_all."""
    hp1 = {"optimizer": {"name": "adam", "lr": 1e-3}}
    hp2 = {"optimizer": {"name": "sgd", "lr": 1e-2}}
    with Logger(run_name="nested_hp_1", hyperparameters=hp1, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})
    with Logger(run_name="nested_hp_2", hyperparameters=hp2, log_dir=tmp_path) as l2:
        l2.log({"step": 1, "loss": 0.4})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) == 2


def test_pairwise_16_empty_metrics_and_close_idempotence(tmp_path: Path):
    """Pairwise: Empty metrics run with repeated close() calls parsed cleanly by reader."""
    logger = Logger(run_name="empty_close_run", hyperparameters={"seed": 42}, log_dir=tmp_path)
    logger.close()
    logger.close()

    reader = LogReader(logs_dir=tmp_path)
    df = reader.read_run(tmp_path / "empty_close_run")
    assert len(df) == 0
    meta = reader.get_metadata(tmp_path / "empty_close_run")
    assert meta["hyperparameters"]["seed"] == 42


def test_pairwise_17_step_and_epoch_coexistence(tmp_path: Path):
    """Pairwise: Interleaved logging of step-level metrics and epoch-level summary metrics."""
    with Logger(run_name="coexist_step_epoch", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
        logger.log({"step": 2, "loss": 0.4})
        logger.log({"epoch": 1, "val_loss": 0.45})
        logger.log({"step": 3, "loss": 0.35})
        logger.log({"epoch": 2, "val_loss": 0.38})

    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "coexist_step_epoch")
    assert len(df) == 5
    assert "step" in df.columns
    assert "epoch" in df.columns


def test_pairwise_18_clean_git_tree_and_env_metadata(temp_git_repo: Path):
    """Pairwise: Clean git repository records commit SHA and env, but strictly omits diff.patch."""
    log_dir = temp_git_repo / "clean_logs"
    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        with Logger(run_name="clean_git_pair", log_dir=log_dir, capture_env=True) as logger:
            logger.log({"step": 1, "loss": 0.2})
    finally:
        os.chdir(old_cwd)

    run_dir = log_dir / "clean_git_pair"
    assert not (run_dir / "diff.patch").exists()
    meta = LogReader(logs_dir=log_dir).get_metadata(run_dir)
    git_meta = meta.get("git") or meta
    assert git_meta.get("is_dirty") is False
    assert git_meta.get("git_commit") is not None


def test_pairwise_19_large_arrays_and_multi_run_aggregation(tmp_path: Path):
    """Pairwise: Multi-run aggregation of runs that each log 1D array metrics."""
    for r in range(2):
        with Logger(run_name=f"arr_agg_{r}", hyperparameters={"run": r}, log_dir=tmp_path) as l:
            l.log({"step": 1, "grad_hist": [r * 0.1, r * 0.2, r * 0.3]})

    df = LogReader(logs_dir=tmp_path).read_all()
    assert len(df) == 2
    assert len(df["grad_hist"].iloc[0]) == 3
    assert len(df["grad_hist"].iloc[1]) == 3


def test_pairwise_20_restart_mismatch_leaves_original_untouched(tmp_path: Path):
    """Pairwise: Attempting a restart with mismatched hyperparameters raises ValueError and preserves original run data."""
    hp_orig = {"lr": 0.01, "opt": "sgd"}
    with Logger(run_name="safe_mismatch", hyperparameters=hp_orig, log_dir=tmp_path) as l1:
        l1.log({"step": 1, "loss": 0.5})

    hp_conflict = {"lr": 0.05, "opt": "sgd"}
    with pytest.raises(ValueError):
        Logger(run_name="safe_mismatch", hyperparameters=hp_conflict, log_dir=tmp_path, restart=True)

    # Original files must remain pristine
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "safe_mismatch")
    assert len(df) == 1
    assert df["hyperparameters.lr"].iloc[0] == 0.01


def test_pairwise_21_special_characters_in_tags_and_multi_run(tmp_path: Path):
    """Pairwise: Runs with unicode tags discovered and metadata queried in multi-run scenario."""
    with Logger(run_name="unicode_r1", tags=["α-model", "test/prod"], log_dir=tmp_path) as l1:
        l1.log({"step": 1, "v": 1})
    with Logger(run_name="unicode_r2", tags=["β-model", "dev 2.0"], log_dir=tmp_path) as l2:
        l2.log({"step": 1, "v": 2})

    reader = LogReader(logs_dir=tmp_path)
    meta1 = reader.get_metadata(tmp_path / "unicode_r1")
    meta2 = reader.get_metadata(tmp_path / "unicode_r2")
    assert "α-model" in meta1["tags"]
    assert "β-model" in meta2["tags"]


def test_pairwise_22_read_jsonl_standalone_without_meta_json(tmp_path: Path):
    """Pairwise: Copy log.jsonl to an isolated directory without meta.json; read_jsonl parses metrics successfully."""
    with Logger(run_name="standalone_src", log_dir=tmp_path) as logger:
        logger.log({"step": 1, "score": 95.5})
        logger.log({"step": 2, "score": 98.0})

    src_jsonl = tmp_path / "standalone_src" / "log.jsonl"
    dest_dir = tmp_path / "isolated"
    dest_dir.mkdir()
    dest_file = dest_dir / "copied_log.jsonl"
    shutil.copyfile(src_jsonl, dest_file)

    reader = LogReader(logs_dir=tmp_path)
    df = reader.read_jsonl(dest_file, run_name="copied_run")
    assert len(df) == 2
    assert df["run_name"].iloc[0] == "copied_run"
    assert df["score"].iloc[1] == 98.0
