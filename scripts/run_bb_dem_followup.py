#!/usr/bin/env python3
"""Run one explicitly selected Nov 2 follow-up worker, or preview it without weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_bb_hierarchical_dem_cnn import resolve_checkpoint

DEFAULT_MANIFEST = REPO / "scripts/manifests/bb_dem_followup_2026_11_02.json"


def workers(manifest, mode):
    data = json.loads(Path(manifest).read_text())
    if data.get("format") != "bb_dem_hierarchical_sources_v1":
        raise ValueError("Unknown campaign format")
    entries = [e for e in data["experiments"] if e["action"] == mode]
    return [(e, weight) for e in entries
            for weight in (("latest",) if mode == "train" else ("best", "latest"))]


def make_plan(args):
    selected = workers(args.manifest, args.mode)
    if not 0 <= args.worker < len(selected):
        raise ValueError(f"Worker must be in [0, {len(selected) - 1}]")
    entry, weights = selected[args.worker]
    checkpoint, candidates = resolve_checkpoint(entry, args.root, args.source_root or args.root, weights)
    plan = dict(mode=args.mode, worker=args.worker, source=entry, weights=weights,
                checkpoint=str(checkpoint), checkpoint_exists=checkpoint.is_file(),
                searched=[str(p) for p in candidates], output=str(args.output) if args.output else None)
    if not checkpoint.is_file() and (not args.dry_run or args.require_checkpoint):
        raise FileNotFoundError(f"Checkpoint missing: {candidates}. Keep original server resdir_* directories.")
    if args.mode == "evaluate":
        command = [sys.executable, "-u", str(args.root / "scripts/evaluate_bb_hierarchical_dem_cnn.py"),
                   "--root", str(args.root), "--manifest", str(args.manifest),
                   "--task", str(entry["task"]), "--index", str(entry["index"]),
                   "--checkpoint", str(checkpoint), "--weights", weights,
                   "--shots", str(args.shots), "--batch-size", "4", "--edge-chunk-size", "4096",
                   "--seed", str(args.seed), "--confidence", str(args.confidence), "--device", "cuda"]
        if args.output:
            command += ["--output", str(args.output)]
        plan["command"] = command
        return plan
    plan.update(target_epochs=args.target_epochs, restart_learning_rate=args.learning_rate,
                additional_epochs=args.target_epochs - entry["training_epochs"],
                resume_weights="latest model.pt, including AdamW state; original loss unchanged")
    if args.target_epochs <= entry["training_epochs"]:
        raise ValueError("Follow-up target must exceed the manifest's completed epochs")
    if not checkpoint.is_file():
        plan["status"] = "metadata_preview_only; optimizer and weights not checked"
        return plan
    from scripts.resume_bb_dem_cnn import plan_resume, verify_source_stopped

    source = checkpoint.parents[3]
    if not args.dry_run:
        verify_source_stopped(source)
    c = entry["expected_config"]
    resume = plan_resume(args.root, source, c["depth"], c["noise_profile"]["base_error_rate"],
                         args.target_epochs, code=c["code"], learning_rate=args.learning_rate, expected=c)
    if resume["completed_epochs"] != entry["training_epochs"]:
        raise ValueError("Source epochs changed since manifest creation; use an explicit recovery plan")
    if Path(resume["checkpoint"]) != checkpoint.resolve():
        raise ValueError("Resume source differs from the exact manifest source")
    plan["resume"] = resume
    plan["command"] = resume["command"]
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--confidence", type=float, default=.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.count:
        print(len(workers(args.manifest, args.mode)))
        return
    if args.worker is None or (not args.dry_run and args.output is None):
        parser.error("Provide --worker and, for execution, a new --output directory")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or args.shots < 1 or args.seed < 0 or not .5 < args.confidence < 1):
        parser.error("Invalid learning rate, shot count, seed or confidence")
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    if args.source_root:
        args.source_root = args.source_root.resolve()
    if args.output:
        args.output = args.output.resolve()
        if args.output.exists() and not args.dry_run:
            raise FileExistsError(args.output)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    # The evaluator creates its output itself. Training main.py writes outputs/
    # under this isolated worker directory; neither path overwrites its source.
    if args.mode == "train":
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "resume_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        metadata = args.output.parent / "job_metadata.txt"
        if metadata.is_file():
            (args.output / "job_metadata.txt").write_text(metadata.read_text())
    subprocess.run(plan["command"], check=True, cwd=args.output if args.mode == "train" else args.root)
    return plan


if __name__ == "__main__":
    main()
