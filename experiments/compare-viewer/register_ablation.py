#!/usr/bin/env python3
"""Register a completed run in the ablation index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ablation_registry import register_experiment  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-run-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--conf", type=float, required=True)
    p.add_argument("--vote", type=float, required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--vocab", default="construction_vocab.txt")
    p.add_argument("--note", default="")
    args = p.parse_args()

    repo = _HERE.parents[1]
    aid = __import__("ablation_registry", fromlist=["ablation_id"]).ablation_id(
        args.conf, args.vote, args.margin
    )
    manifest = repo / "outputs" / "ablation" / "manifests" / f"{aid}.json"
    if not manifest.is_file():
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                str(_HERE / "build_manifest.py"),
                "--baseline-dir",
                f"outputs/runs/{args.baseline_run_id}",
                "--experiment-dir",
                f"outputs/runs/{args.run_id}",
                "--output",
                str(manifest),
                "--experiment-label",
                f"conf={args.conf} vote={args.vote} margin={args.margin}",
            ],
            cwd=str(repo),
        )

    entry = register_experiment(
        conf=args.conf,
        vote=args.vote,
        margin=args.margin,
        run_id=args.run_id,
        manifest_path=manifest,
        baseline_run_id=args.baseline_run_id,
        vocab=args.vocab,
        note=args.note,
    )
    print(f"Registered {entry['id']} → run {entry['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
