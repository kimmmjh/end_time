#!/usr/bin/env python3
"""Resolve a completed depth-2 CNN run and continue its latest checkpoint.

The source can be an original resdir_<job> or its organized results archive.
No matching checkpoint means failure, never an implicit fresh training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys

import torch


def resolve_checkpoint(repo_root, source_job, code, p, seed, source_root=None):
    name = f"resdir_{source_job}"
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        roots = [root if root.name == name else root / name]
    else:
        root = Path(repo_root).resolve()
        archive = root / "results/bb/code_capacity/depolarizing/tanner_cnn"
        roots = [root / name, archive / code / name, archive / "replicates" / name]
    matches = set()
    expected = {"architecture": "bb_tanner_cnn", "code": code, "error_rate": p,
                "seed": seed, "cnn_depth": 2, "cnn_width": 64,
                "noise_model": "capacity", "channel": "depolarizing"}
    for directory in roots:
        for history_path in directory.glob("outputs/*/*/history.json"):
            history = json.loads(history_path.read_text())
            if all(history.get("config", {}).get(key) == value for key, value in expected.items()):
                checkpoint = history_path.with_name("model.pt")
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Latest checkpoint missing: {checkpoint}")
                matches.add(checkpoint.resolve())
    if len(matches) != 1:
        detail = "\n".join(str(path) for path in sorted(matches or roots))
        raise ValueError(f"Expected one source checkpoint, found {len(matches)} for {code} p={p} seed={seed}.\n"
                         f"{detail}\nSet BB_CNN_SOURCE_ROOT to the directory containing {name} "
                         "(or to that result directory itself).")
    return matches.pop()


def build_command(checkpoint, *, repo_root, code, p, seed, target_epochs=400,
                  expected_epochs=100, learning_rate=3e-4):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved.get("config", {})
    expected = {"architecture": "bb_tanner_cnn", "code": code, "error_rate": p,
                "seed": seed, "cnn_depth": 2, "cnn_width": 64,
                "noise_model": "capacity", "channel": "depolarizing"}
    if saved.get("format") != "bb_tanner_cnn_v1" or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint configuration does not match the requested depth-2 experiment.")
    completed = saved["epoch"] + 1
    history = saved["history"]
    if completed != expected_epochs:
        raise ValueError(f"Expected {expected_epochs} completed epochs, found {completed}.")
    if [row["epoch"] for row in history["train"]] != list(range(completed)):
        raise ValueError("Source checkpoint has incomplete training history.")
    if "final" not in history or saved.get("best_model_state_dict") is None:
        raise ValueError("Source run must have completed final evaluation and saved its selected weights.")
    if target_epochs <= completed:
        raise ValueError("Target epochs must exceed the source's completed epochs.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    phase = history["phases"][-1]
    options = {
        "architecture": "bb_tanner_cnn", "noise_model": "capacity", "bb_channel": "depolarizing",
        "code": code, "p": p, "seed": seed, "rounds": 1,
        "measurement_error_rate": 0, "bb_idle_error_rate": 0, "loss_fn": "bb_coset",
        "load_model": str(Path(checkpoint).resolve()), "epochs": target_epochs - completed,
        "batch_size": phase["batch_size"], "batches": phase["batches"],
        "eval_batches": phase["eval_batches"], "eval_every": phase["eval_every"],
        "final_eval_batches": phase["final_eval_batches"], "lr": learning_rate, "amp_dtype": "none",
        "bb_cnn_depth": config["cnn_depth"], "bb_cnn_width": config["cnn_width"],
        "bb_cnn_gradient_clip": config["gradient_clip"], "bb_weight_decay": config["weight_decay"],
        "bb_syndrome_loss_weight": config["syndrome_loss_weight"],
        "bb_logical_loss_weight": config["logical_loss_weight"],
        "bb_pauli_loss_weight": config["pauli_loss_weight"],
    }
    for key in ("x_error_rate", "z_error_rate"):
        if config[key] is not None:
            options[key] = config[key]
    return [sys.executable, "-u", str(Path(repo_root).resolve() / "main.py"),
            *(f"--{key}={value}" for key, value in options.items()), "--save_model", "--bb_cnn_compare_resume"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=os.environ.get("BB_REPO_ROOT", Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-root", type=Path, default=os.environ.get("BB_CNN_SOURCE_ROOT"))
    parser.add_argument("--source-job", type=int, required=True)
    parser.add_argument("--code", choices=("bb72", "bb144"), required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-epochs", type=int, default=400)
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate the source, then print the main.py command.")
    args = parser.parse_args()
    try:
        checkpoint = resolve_checkpoint(args.repo_root, args.source_job, args.code, args.p,
                                        args.seed, args.source_root)
        command = build_command(checkpoint, repo_root=args.repo_root, code=args.code, p=args.p,
                                seed=args.seed, target_epochs=args.target_epochs,
                                expected_epochs=args.expected_epochs, learning_rate=args.lr)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Continuing {checkpoint}: {args.expected_epochs} -> {args.target_epochs} total epochs; "
          f"cosine LR restart at {args.lr:g}", flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        # Replace this process so Slurm signals reach the actual trainer directly.
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
