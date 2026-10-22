"""Run the whole collection pipeline: scan -> extract -> build.

    python scripts/collect/run_pipeline.py                  # all engines, all steps
    python scripts/collect/run_pipeline.py godot ogre       # some engines (the dataset is still built from all scanned ones)
    python scripts/collect/run_pipeline.py --steps extract build

Each step can also be run on its own; see scan.py, extract.py and build_dataset.py.
The scan is the slow step and resumes where it stopped.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {"scan": "scan.py", "extract": "extract.py", "build": "build_dataset.py"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="*", help="limit scan and extract to these engines")
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--config")
    args = ap.parse_args()
    for step in STEPS:
        if step not in args.steps:
            continue
        cmd = [sys.executable, str(HERE / STEPS[step])]
        if step != "build":
            cmd += args.engines  # the dataset is always built from every engine that has units
        if args.config:
            cmd += ["--config", args.config]
        print(f"\n##### {step}: {' '.join(cmd[1:])}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"step '{step}' failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
