"""
run_all.py

Runs the entire log anomaly detection pipeline end to end.

Order of execution:
    HDFS:    hdfs_parser  →  hdfs_preprocessor  →  hdfs_modeling
    Android: android_parser  →  android_preprocessor  →  android_modeling

Each stage is run as a separate Python process. If any stage fails, the
orchestrator stops and reports which script failed so the error can be
debugged in isolation.

Prerequisites:
    conda env create -f environment.yml
    conda activate <env name>

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path


# (script path, human-readable description)
STAGES = [
    ("hadoop_parser.py",          "Parse HDFS raw log"),
    ("hadoop_preprocessing.py",    "Template HDFS events and build traces"),
    ("hadoop_model.py",        "HDFS classification + clustering"),

    ("android_parser.py",       "Parse Android raw log"),
    ("android_preprocessing.py", "Template Android events"),
    ("android_model.py",     "Android anomaly detection"),
]


def run_stage(script: str, description: str, index: int, total: int) -> float:
    """Run one pipeline stage and return its elapsed time in seconds."""
    print(f"\n{'=' * 70}")
    print(f"[{index}/{total}] {description}")
    print(f"  running: {script}")
    print('=' * 70)

    if not Path(script).exists():
        raise FileNotFoundError(f"Script not found: {script}")

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"{script} exited with code {result.returncode}"
        )

    print(f"\n  ✓ completed in {elapsed:.1f}s")
    return elapsed


def format_duration(seconds: float) -> str:
    """Format a duration as 'Xm Ys' or 'Ys'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def main() -> None:
    print("Log Anomaly Detection Pipeline")
    print(f"Running {len(STAGES)} stages\n")

    timings = []
    overall_start = time.time()

    try:
        for i, (script, description) in enumerate(STAGES, start=1):
            elapsed = run_stage(script, description, i, len(STAGES))
            timings.append((script, elapsed))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        print(f"  (ran {len(timings)} of {len(STAGES)} stages)")
        sys.exit(1)

    total = time.time() - overall_start

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print('=' * 70)
    for script, elapsed in timings:
        print(f"  {script:<28} {format_duration(elapsed):>10}")
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL':<28} {format_duration(total):>10}")
    print(f"\nOutputs written to the project directory.")


if __name__ == "__main__":
    main()