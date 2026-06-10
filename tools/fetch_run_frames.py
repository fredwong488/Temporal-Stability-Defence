"""Fetch single_run_frames.jsonl for a given run from the GPU server via scp."""

import subprocess
from pathlib import Path

REMOTE = "cyw122@gpu35:/homes/cyw122/Developer/year_4/FYP/FYP-experiment-pipeline"


def main() -> None:
    run_name = input("run_name: ").strip()
    if not run_name:
        raise SystemExit("run_name is required")

    remote_path = f"{REMOTE}/results/{run_name}/single_run_frames.jsonl"
    local_path = Path("results") / run_name / "single_run_frames.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(["scp", "-r", remote_path, str(local_path)], check=True)


if __name__ == "__main__":
    main()
