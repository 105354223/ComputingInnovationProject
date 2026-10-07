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
    print(f"[{index}/{total}] {description}")
    print(f"  running: {script}")

    if not Path(script).exists():
        raise FileNotFoundError(f"Script not found: {script}")

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"{script} exited with code {result.returncode}"
        )

    print(f"\ncompleted in {elapsed:.1f}s")
    return elapsed


def format_duration(seconds: float) -> str:
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
        print(f"\nPIPELINE FAILED: {e}")
        print(f"  (ran {len(timings)} of {len(STAGES)} stages)")
        sys.exit(1)

    total = time.time() - overall_start

    # ---- Summary ----
    print("PIPELINE COMPLETE")
    for script, elapsed in timings:
        print(f"  {script:<28} {format_duration(elapsed):>10}")
    print(f"  {'TOTAL':<28} {format_duration(total):>10}")
    print(f"\nOutputs written to the project directory.")


if __name__ == "__main__":
    main()