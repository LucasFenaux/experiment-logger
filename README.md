# Experiment Logger

A lightweight, crash-resilient ML research logging and plotting framework. This package provides a robust backend for tracking hyperparameters, capturing environment state, and safely recording metrics iteratively.

## Features

- **Crash-Resilient Storage**: Logs metrics incrementally to append-only JSON Lines (JSONL) files, ensuring no data loss if a training script crashes.
- **Environment & Git Capture**: Automatically logs Python version, installed package states, git commit hashes, and saves uncommitted changes (`diff.patch`) for perfect reproducibility.
- **Data Analysis Ready**: Includes a built-in `LogReader` to instantly parse raw JSONL logs into a pandas DataFrame for seamless integration with visualization tools.
- **Safe Run Resumption**: Supports `restart=True` to gracefully append data to existing runs (e.g., resuming training from a checkpoint) without fragmenting log directories.

## Installation

Install the package directly from source using `pip`:

```bash
# Install the package
pip install -e .

# Install with development/testing dependencies
pip install -e ".[dev]"
```

## Quickstart

### 1. Logging an Experiment

```python
from experiment_logger.logger import Logger

# Initialize the logger (automatically captures environment and git state)
logger = Logger(
    run_name="resnet50_baseline",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "resnet50"
    }
)

# Iteratively log metrics
for step in range(100):
    logger.log({
        "step": step,
        "loss": 0.5 * (0.9 ** step),
        "accuracy": 0.8 + (0.1 * step / 100)
    })
```

### 2. Reading Data into Pandas

```python
from experiment_logger.reader import LogReader

# Load the run data
reader = LogReader(run_name="resnet50_baseline")
df = reader.to_pandas()

print(df.head())
```

## Development & Testing

This project strictly follows Test-Driven Development (TDD). The test suite includes comprehensive unit tests and rigorous E2E adversarial tests.

Run the tests via pytest:

```bash
pytest --cov=src --cov-fail-under=90
```
