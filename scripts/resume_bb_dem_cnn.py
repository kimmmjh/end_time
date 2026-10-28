#!/usr/bin/env python3
"""Resume one BB144 DEM sweep point to a total epoch budget, without overwriting its source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

RATES = (.001, .002, .004, .006)


def find_source(root, array, depth):
    task = {8: 5, 12: 7}[depth]
    expected = f"Array job/task: {array}/{task}"
    matches = [p.parent for p in root.glob("resdir_*/job_metadata.txt")
               if expected in p.read_text().splitlines()]
    if len(matches) != 1:
        raise ValueError(f"Expected one source job with '{expected}', found {matches}. "
                         "Use --source-job for an explicit source directory.")
    return matches[0]


def plan_resume(root, source, depth, rate, target):
    matches = []
    for history in source.glob("outputs/*/*/history.json"):
        c = json.loads(history.read_text())["config"]
        if (c.get("architecture") == "bb_dem_cnn" and c.get("code") == "bb144"
                and c.get("depth") == depth and c.get("noise_profile", {}).get("base_error_rate") == rate):
            matches.append(history.with_name("model.pt"))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"Expected one latest model.pt for BB144 D{depth} p={rate}; found {matches}")
    checkpoint_path = matches[0].resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("format") != "bb_dem_cnn_v1":
        raise ValueError("Expected a bb_dem_cnn_v1 checkpoint")
    c, history = saved["config"], saved["history"]
    if not (c["code"] == "bb144" and c["depth"] == depth and c["noise_model"] == "circuit"
            and c["rounds"] == 12 and c["width"] == 64
            and c["noise_profile"]["name"] == "standard"
            and c["noise_profile"]["base_error_rate"] == rate):
        raise ValueError("Checkpoint configuration does not match the requested sweep point")
    completed = saved["epoch"] + 1
    if [r["epoch"] for r in history["train"]] != list(range(completed)):
        raise ValueError("Checkpoint epoch and saved training history disagree")
    plan = dict(source_job=str(source), checkpoint=str(checkpoint_path),
                checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                code=c["code"], depth=depth, p=rate, completed_epochs=completed,
                target_epochs=target, additional_epochs=max(0,target-completed),
                prior_best_epoch=saved["best_epoch"]+1,
                schedule="new cosine over remaining epochs, starting at saved optimizer LR",
                sampler="new independent phase; not bitwise uninterrupted continuation")
    if completed >= target:
        if "final" not in history:
            raise ValueError(f"{checkpoint_path}: {completed} epochs already trained but no final evaluation "
                             "in checkpoint. Needs evaluation-only recovery; no extra epochs will be added.")
        return dict(plan, status="already_complete", command=[])
    phase = history["phases"][-1]
    rates = {float(g["lr"]) for g in saved["optimizer_state_dict"]["param_groups"]}
    if len(rates) != 1 or next(iter(rates)) <= 0:
        raise ValueError("Expected one positive saved optimizer learning rate")
    lr = rates.pop()
    loss = c["loss_weights"]
    values = dict(architecture="bb_dem_cnn", noise_model="circuit", code="bb144",
        bb_circuit_noise_model="standard", rounds=c["rounds"], p=rate, seed=c["seed"],
        load_model=str(checkpoint_path), epochs=target-completed, batch_size=phase["batch_size"],
        batches=phase["batches"], eval_batches=phase["eval_batches"], eval_every=phase["eval_every"],
        final_eval_batches=phase["final_eval_batches"], lr=lr, amp_dtype="none", loss_fn="bb_coset",
        bb_cnn_width=c["width"], bb_cnn_depth=depth, bb_weight_decay=phase["weight_decay"],
        bb_cnn_gradient_clip=phase["gradient_clip"], bb_syndrome_loss_weight=loss["syndrome_weight"],
        bb_logical_loss_weight=loss["logical_weight"], bb_pauli_loss_weight=loss["mechanism_weight"],
        edge_chunk_size=phase["edge_chunk_size"], bb_osd_eval_shots=phase["osd_eval_shots"])
    command = [sys.executable, "-u", str(root / "main.py"),
               *[f"--{k}={v}" for k,v in values.items()], "--save_model"]
    if not phase["gradient_checkpoint"]:
        command.append("--bb_dem_cnn_no_checkpoint")
    return dict(plan, status="resume", resume_learning_rate=lr, command=command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-array", type=int, default=60310800)
    parser.add_argument("--source-job", type=Path)
    parser.add_argument("--depth", type=int, choices=(8,12), default=8)
    parser.add_argument("--index", type=int, choices=range(4), required=True)
    parser.add_argument("--target-epochs", type=int, default=800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.target_epochs < 1:
        parser.error("--target-epochs must be positive")
    source = args.source_job.resolve() if args.source_job else find_source(root,args.source_array,args.depth)
    # An actively changing source is not a resume target. Explicit source jobs
    # (including subsequent resume jobs) are checked by their metadata Job ID.
    metadata = (source / "job_metadata.txt").read_text().splitlines()
    job_id = next((s.removeprefix("Job ID: ") for s in metadata if s.startswith("Job ID: ")), None)
    if not args.dry_run:
        if not job_id or not shutil.which("squeue"):
            raise RuntimeError("Cannot verify source job is stopped; use --dry-run for a local preview")
        active = subprocess.run(["squeue","-h","-j",job_id,"-o","%T"], text=True,capture_output=True)
        if active.returncode:
            # Completed jobs can already have disappeared from slurmctld.
            accounting = subprocess.run(["sacct","-n","-X","-j",job_id,"--format=State","--parsable2"],
                                        text=True,capture_output=True)
            states = [s.strip().rstrip("|") for s in accounting.stdout.splitlines() if s.strip()]
            if accounting.returncode or not states:
                raise RuntimeError(f"Could not verify source job {job_id} is stopped")
        else:
            states = active.stdout.split()
        if any(state not in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY") for state in states):
            raise RuntimeError(f"Source job {job_id} is still active: {states}. Resume after it stops.")
    plan = plan_resume(root,source,args.depth,RATES[args.index],args.target_epochs)
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return
    Path(f"resume_exp_{args.index}.json").write_text(json.dumps(plan,indent=2)+"\n")
    if plan["status"] == "already_complete":
        return
    # Preserve the original invocation for reproducibility; cwd is a new resdir.
    subprocess.run(plan["command"],check=True)


if __name__ == "__main__":
    main()
