"""Tier 4: Real-world ML application workflow scenarios.

Implements complex, multi-step ML training lifecycles derived from TEST_INFRA.md:
1. Standard PyTorch Training Simulation (multi-epoch loop, batch loss arrays, learning rate schedules)
2. Crash and Resume Training Workflow (interrupted writes, restart=True continuation, seamless aggregation)
3. Hyperparameter Search Campaign (grid sweep across multiple runs, schema evolution, View DSL group-by queries)
4. Edge Cases Under Unstable Execution (exploding gradients, NaN/Inf serialization, partial write recovery)
5. Git Dirty Reproducibility Pipeline (automatic diff.patch creation, dirty working tree verification, clean fallback)
6. Distributed Rank-0 Simulation (rank-0 logging isolation, synchronization, and reader verification)
"""

import json
import math
import os
import subprocess
import time
import warnings
from pathlib import Path
import pytest
import pandas as pd

from experiment_logger import Logger, LogReader
from experiment_logger.env import capture_environment, capture_git_info, create_diff_patch


# ==============================================================================
# Scenario 1: Standard PyTorch Training Simulation
# ==============================================================================

def test_scenario_01_pytorch_training_simulation(tmp_path: Path):
    """Scenario 1: Simulate a multi-epoch training loop logging step-level, epoch-level, and 1D batch arrays."""
    hyperparameters = {
        "architecture": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "epochs": 3,
    }
    tags = ["computer-vision", "cifar10", "baseline"]

    num_epochs = 3
    steps_per_epoch = 4
    total_steps = num_epochs * steps_per_epoch

    # Execute simulated training loop
    with Logger(
        run_name="pytorch_sim_run",
        hyperparameters=hyperparameters,
        tags=tags,
        log_dir=tmp_path,
        capture_env=True,
    ) as logger:
        global_step = 0
        for epoch in range(1, num_epochs + 1):
            lr = hyperparameters["learning_rate"] * (0.5 ** (epoch - 1))
            for batch_idx in range(steps_per_epoch):
                global_step += 1
                batch_losses = [0.8 / (epoch + 0.1 * i) for i in range(4)]
                step_loss = sum(batch_losses) / len(batch_losses)
                logger.log({
                    "step": global_step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "train_loss": step_loss,
                    "batch_losses": batch_losses,
                    "lr": lr,
                })

            # End of epoch validation
            val_loss = 0.7 / epoch
            val_acc = 0.65 + (epoch * 0.1)
            logger.log({
                "step": global_step,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

    # Verification via LogReader
    reader = LogReader(logs_dir=tmp_path)
    run_dir = tmp_path / "pytorch_sim_run"

    # Verify metadata
    meta = reader.get_metadata(run_dir)
    assert meta["tags"] == tags
    assert meta["hyperparameters"]["architecture"] == "resnet50"
    assert "env" in meta or "environment" in meta

    # Verify DataFrame structure
    df = reader.read_run(run_dir)
    # Total rows = total_steps (12) + val evaluations (3) = 15
    assert len(df) == total_steps + num_epochs

    # Verify View DSL column prefixing
    assert "hyperparameters.architecture" in df.columns
    assert "hyperparameters.learning_rate" in df.columns
    assert (df["hyperparameters.architecture"] == "resnet50").all()

    # Verify 1D batch loss array preservation
    train_rows = df[df["batch_losses"].notna()]
    assert len(train_rows) == total_steps
    assert all(len(arr) == 4 for arr in train_rows["batch_losses"])

    # Verify timestamp injection
    assert "_timestamp" in df.columns
    assert "_time_since_start" in df.columns
    assert (df["_time_since_start"] >= 0.0).all()


# ==============================================================================
# Scenario 2: Crash and Resume Training Workflow
# ==============================================================================

def test_scenario_02_crash_and_resume_training(tmp_path: Path):
    """Scenario 2: Training interrupted mid-run by crash/power failure; resumes seamlessly with restart=True."""
    hp = {"model": "vit_small", "lr": 5e-4, "seed": 42}
    run_name = "crash_resume_exp"

    # Session 1: Train steps 1 to 5
    with Logger(run_name=run_name, hyperparameters=hp, log_dir=tmp_path) as l1:
        for s in range(1, 6):
            l1.log({"step": s, "loss": 1.0 / s, "checkpoint_step": s})

    # Simulate catastrophic crash: partial line appended before exit
    log_file = tmp_path / run_name / "log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"step": 6, "loss": 0.1666, "checkpoi')  # Truncated write

    # Session 2: Resume training from step 6 to 10 using restart=True
    with Logger(run_name=run_name, hyperparameters=hp, log_dir=tmp_path, restart=True) as l2:
        for s in range(6, 11):
            l2.log({"step": s, "loss": 1.0 / s, "checkpoint_step": s})

    # Verify LogReader tolerates truncated line with warning and parses all 10 complete steps
    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(tmp_path / run_name)

    assert len(df) == 10
    assert list(df["step"]) == list(range(1, 11))
    assert df.loc[df["step"] == 1, "loss"].iloc[0] == 1.0
    assert pytest.approx(df.loc[df["step"] == 10, "loss"].iloc[0]) == 0.1


# ==============================================================================
# Scenario 3: Hyperparameter Search Campaign (Grid/Random)
# ==============================================================================

def test_scenario_03_hyperparameter_search_campaign(tmp_path: Path):
    """Scenario 3: Multi-run sweep across hyperparameters, aggregated via read_all with schema evolution."""
    sweep_configs = [
        {"run_name": "sweep_adam_001", "arch": "resnet18", "lr": 0.001, "opt": "adam"},
        {"run_name": "sweep_adam_010", "arch": "resnet18", "lr": 0.010, "opt": "adam"},
        {"run_name": "sweep_sgd_001", "arch": "resnet50", "lr": 0.001, "opt": "sgd"},
        # Run 4 introduces an additional hyperparameter 'dropout' (schema evolution)
        {"run_name": "sweep_sgd_010", "arch": "resnet50", "lr": 0.010, "opt": "sgd", "dropout": 0.2},
    ]

    steps_per_run = 5
    for cfg in sweep_configs:
        name = cfg["run_name"]
        hparams = {k: v for k, v in cfg.items() if k != "run_name"}
        with Logger(run_name=name, hyperparameters=hparams, log_dir=tmp_path) as logger:
            for s in range(1, steps_per_run + 1):
                train_loss = 1.0 / (s * hparams["lr"] * 100 + 1)
                val_acc = 0.5 + (0.05 * s)
                logger.log({"step": s, "train_loss": train_loss, "val_acc": val_acc})

    # Aggregate full campaign
    reader = LogReader(logs_dir=tmp_path)
    df = reader.read_all()

    # Total records: 4 runs * 5 steps = 20
    assert len(df) == 20
    assert len(df["run_name"].unique()) == 4

    # Verify View DSL column prefixing
    assert "hyperparameters.arch" in df.columns
    assert "hyperparameters.lr" in df.columns
    assert "hyperparameters.opt" in df.columns
    assert "hyperparameters.dropout" in df.columns

    # Verify schema evolution: dropout is 0.2 for sweep_sgd_010, and NaN for others
    sgd_010_rows = df[df["run_name"] == "sweep_sgd_010"]
    assert (sgd_010_rows["hyperparameters.dropout"] == 0.2).all()
    other_rows = df[df["run_name"] != "sweep_sgd_010"]
    assert other_rows["hyperparameters.dropout"].isna().all()

    # Perform simulated View DSL query & aggregation
    # Filter by optimizer == "adam" and group by step
    adam_runs = df[df["hyperparameters.opt"] == "adam"]
    assert len(adam_runs) == 10
    mean_by_step = adam_runs.groupby("step")["val_acc"].mean()
    assert len(mean_by_step) == 5


# ==============================================================================
# Scenario 4: Edge Cases Under Unstable Execution
# ==============================================================================

def test_scenario_04_unstable_execution_recovery(tmp_path: Path):
    """Scenario 4: Run experiencing diverging losses, infinities, NaNs, and partial write interruption."""
    run_name = "unstable_run"

    with Logger(run_name=run_name, hyperparameters={"model": "deep_net"}, log_dir=tmp_path) as logger:
        # Step 1: Normal initialization loss
        logger.log({"step": 1, "loss": 0.693, "grad_norm": 1.5})
        # Step 2: Exploding gradients -> loss diverges to +Infinity
        logger.log({"step": 2, "loss": math.inf, "grad_norm": 1e12})
        # Step 3: NaN loss due to 0/0 error
        logger.log({"step": 3, "loss": math.nan, "grad_norm": math.nan})
        # Step 4: Underflow -> -Infinity bound
        logger.log({"step": 4, "loss": -math.inf, "grad_norm": 0.0})
        # Step 5: 1D gradient array containing mixed NaN and Inf
        logger.log({"step": 5, "loss": 0.5, "layer_grads": [0.01, math.nan, math.inf, -math.inf]})

    # Simulate crash right at step 6 write
    log_file = tmp_path / run_name / "log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"step": 6, "loss": "NaN", "unclosed')

    # Step 7 logged in recovered session
    with Logger(run_name=run_name, hyperparameters={"model": "deep_net"}, log_dir=tmp_path, restart=True) as logger:
        logger.log({"step": 7, "loss": 0.25, "grad_norm": 0.8})

    with pytest.warns(UserWarning):
        df = LogReader(logs_dir=tmp_path).read_run(tmp_path / run_name)

    # Steps 1, 2, 3, 4, 5, 7 successfully parsed (6 was skipped)
    assert len(df) == 6
    assert list(df["step"]) == [1, 2, 3, 4, 5, 7]

    # Verify special float representations in DataFrame
    assert math.isinf(df.loc[df["step"] == 2, "loss"].iloc[0]) and df.loc[df["step"] == 2, "loss"].iloc[0] > 0
    assert math.isnan(df.loc[df["step"] == 3, "loss"].iloc[0])
    assert math.isinf(df.loc[df["step"] == 4, "loss"].iloc[0]) and df.loc[df["step"] == 4, "loss"].iloc[0] < 0

    # Verify 1D array special float deserialization
    arr5 = df.loc[df["step"] == 5, "layer_grads"].iloc[0]
    assert arr5[0] == 0.01
    assert math.isnan(arr5[1])
    assert math.isinf(arr5[2]) and arr5[2] > 0
    assert math.isinf(arr5[3]) and arr5[3] < 0

    # Verify recovered step 7
    assert df.loc[df["step"] == 7, "loss"].iloc[0] == 0.25


# ==============================================================================
# Scenario 5: Git Dirty Reproducibility Pipeline
# ==============================================================================

def test_scenario_05_git_dirty_reproducibility_pipeline(temp_git_repo: Path):
    """Scenario 5: Full reproducibility pipeline capturing git commits and dirty working tree patches."""
    # 1. Modify existing tracked file to introduce dirty state
    tracked_file = temp_git_repo / "initial.txt"
    tracked_file.write_text("modified code logic for experimental hypothesis\n", encoding="utf-8")

    log_dir = temp_git_repo / "results"
    old_cwd = os.getcwd()
    os.chdir(temp_git_repo)
    try:
        # Run 1: Executed with dirty working tree
        with Logger(
            run_name="dirty_experiment",
            hyperparameters={"learning_rate": 0.005},
            log_dir=log_dir,
            capture_env=True,
        ) as logger:
            logger.log({"step": 1, "test_metric": 88.5})
    finally:
        os.chdir(old_cwd)

    run_dir = log_dir / "dirty_experiment"
    patch_file = run_dir / "diff.patch"

    # Verify diff.patch exists and contains valid git diff
    assert patch_file.exists()
    patch_content = patch_file.read_text(encoding="utf-8")
    assert "diff --git" in patch_content
    assert "modified code logic for experimental hypothesis" in patch_content

    # Verify meta.json reflects dirty status and commit SHA
    reader = LogReader(logs_dir=log_dir)
    meta = reader.get_metadata(run_dir)
    git_meta = meta.get("git") or meta
    assert git_meta.get("is_dirty") is True
    assert git_meta.get("git_commit") is not None
    assert len(git_meta.get("git_commit")) == 40

    # 2. Revert tracked file to clean state
    subprocess.run(["git", "checkout", "initial.txt"], cwd=temp_git_repo, check=True, capture_output=True)

    # Run 2: Executed in clean state
    os.chdir(temp_git_repo)
    try:
        with Logger(
            run_name="clean_experiment",
            hyperparameters={"learning_rate": 0.005},
            log_dir=log_dir,
            capture_env=True,
        ) as logger:
            logger.log({"step": 1, "test_metric": 91.0})
    finally:
        os.chdir(old_cwd)

    clean_dir = log_dir / "clean_experiment"
    assert not (clean_dir / "diff.patch").exists()

    clean_meta = reader.get_metadata(clean_dir)
    clean_git = clean_meta.get("git") or clean_meta
    assert clean_git.get("is_dirty") is False


# ==============================================================================
# Scenario 6: Distributed Rank-0 Logging Simulation
# ==============================================================================

def test_scenario_06_distributed_rank0_simulation(tmp_path: Path):
    """Scenario 6: Simulate distributed training where only rank-0 instantiates Logger (ADR 15, 30)."""
    world_size = 4
    num_steps = 5

    def simulate_worker(rank: int):
        # In distributed setup (e.g. PyTorch DDP), only rank 0 logs metrics
        if rank == 0:
            with Logger(
                run_name="ddp_rank0_run",
                hyperparameters={"world_size": world_size, "lr": 0.001},
                log_dir=tmp_path,
            ) as logger:
                for step in range(1, num_steps + 1):
                    logger.log({"step": step, "reduced_loss": 0.5 / step})

    for r in range(world_size):
        simulate_worker(r)

    # Verify single clean run directory created and verified
    df = LogReader(logs_dir=tmp_path).read_run(tmp_path / "ddp_rank0_run")
    assert len(df) == num_steps
    assert list(df["step"]) == list(range(1, num_steps + 1))
    assert df["hyperparameters.world_size"].iloc[0] == world_size
