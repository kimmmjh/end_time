#!/usr/bin/env python3
"""Two-point sample scaling with one cosine schedule and recoverable evaluations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_bb_hierarchical_dem_cnn import write_json

MANIFEST = REPO / "scripts/manifests/bb_dem_sample_scaling.json"


@contextmanager
def worker_lock(directory):
    """Exclude other workers without requiring advisory flock support.

    A hard-killed owner leaves its directory behind. Never infer that a remote
    PID or a missing scheduler response means it is safe to steal that lock.
    """
    path = directory / "worker.lock.d"
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"Worker lock already exists: {path}. Inspect {path / 'owner.json'} "
            "and verify that its Slurm job and worker have stopped before removing "
            "the stale lock directory. No existing lock was removed."
        ) from exc
    try:
        write_json(path / "owner.json", dict(
            job_id=os.environ.get("SLURM_JOB_ID"),
            step_id=os.environ.get("SLURM_STEP_ID"),
            hostname=socket.gethostname(), pid=os.getpid(), created_unix=time.time()))
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def source_path(entry, source_root):
    candidates = [source_root / entry["source_run"] / "model.pt",
                  REPO / entry["imported_run"] / "model.pt"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def validate_checkpoint(saved, campaign, entry, *, initial=False, continuation=None):
    if saved.get("format") != "bb_dem_cnn_v1" or saved["config"] != entry["expected_config"]:
        raise ValueError("Checkpoint configuration differs from the frozen campaign manifest")
    completed = saved["epoch"] + 1
    if [r["epoch"] for r in saved["history"]["train"]] != list(range(completed)):
        raise ValueError("Checkpoint training history is not contiguous")
    if initial and (completed != entry["source_epochs"] or
                    saved["best_epoch"] + 1 != entry["source_best_epoch"]):
        raise ValueError("Source checkpoint changed since the campaign was planned")
    if completed < campaign["start_epoch"] or (continuation is None and completed > campaign["target_epoch"]):
        raise ValueError("Checkpoint epoch is outside this campaign")
    if not saved.get("optimizer_state_dict") or saved.get("best_model_state_dict") is None:
        raise ValueError("Latest checkpoint must contain AdamW and best weights")
    if any(p["batch_size"] != campaign["batch_size"] or p["batches"] != campaign["batches"]
           for p in saved["history"]["phases"]):
        raise ValueError("Sample accounting requires unchanged batch size and batches per epoch")
    if continuation is not None and completed > campaign["target_epoch"]:
        state = saved.get("scheduler_state_dict", {})
        expected_steps = (completed - campaign["target_epoch"]) * campaign["batches"]
        if (saved.get("scheduler_type") != "constant" or state.get("last_epoch") != expected_steps
                or not state.get("base_lrs")
                or any(lr != continuation["learning_rate"] for lr in state["base_lrs"])
                or any(group["lr"] != continuation["learning_rate"]
                       for group in saved["optimizer_state_dict"]["param_groups"])):
            raise ValueError("Checkpoint does not belong to the constant-LR continuation")
    elif completed > campaign["start_epoch"]:
        state = saved.get("scheduler_state_dict", {})
        expected_steps = (completed - campaign["start_epoch"]) * campaign["batches"]
        if (state.get("last_epoch") != expected_steps or state.get("T_max") !=
                (campaign["target_epoch"] - campaign["start_epoch"]) * campaign["batches"]):
            raise ValueError("Checkpoint does not belong to the continuous campaign schedule")
    return completed


def train_segment(checkpoint, output, end_epoch, campaign, entry, stop, *, continuation=None):
    import numpy as np
    import torch
    from models._bb_dem_cnn import BBDemCNN
    from src._bb_circuit_experiment import _graph_fingerprint
    from src._bb_circuit_loss import CircuitDegeneracyAwareLoss
    from src._bb_dem_cnn_experiment import BBDemCNNTrainer
    from src.bb_circuit_data import BBCircuitGenerator
    from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = validate_checkpoint(saved, campaign, entry, continuation=continuation)
    c, phase = saved["config"], saved["history"]["phases"][-1]
    seeds = np.random.SeedSequence([c["seed"], completed, CIRCUIT_SCHEMA_VERSION]).spawn(3)
    torch.manual_seed(int(seeds[2].generate_state(1)[0]))
    noise = c["noise_profile"]
    kwargs = dict(code=c["code"], rounds=c["rounds"], gate_error_rate=noise["base_error_rate"],
                  measurement_error_rate=noise["measurement_error_rate"],
                  idle_error_rate=noise["gate_idle_error_rate"], circuit_noise_model=noise["name"],
                  batch_size=campaign["batch_size"])
    train = BBCircuitGenerator(**kwargs, seed=int(seeds[0].generate_state(1)[0]))
    evaluation = BBCircuitGenerator(**kwargs, seed=int(seeds[1].generate_state(1)[0]), graph=train.graph)
    if (c["circuit_schema_version"] != CIRCUIT_SCHEMA_VERSION or
            asdict(train.noise_profile) != noise or _graph_fingerprint(train.graph) != c["graph_fingerprint"]):
        raise ValueError("Reconstructed circuit differs from source")
    device = torch.device("cuda:0")
    model = BBDemCNN(train.graph, width=c["width"], depth=c["depth"],
                     edge_chunk_size=phase["edge_chunk_size"],
                     gradient_checkpoint=phase["gradient_checkpoint"]).to(device)
    loss = CircuitDegeneracyAwareLoss(check_matrix=train.graph.check_matrix,
                                     observable_matrix=train.graph.observable_matrix,
                                     **c["loss_weights"]).to(device)
    trainer = BBDemCNNTrainer(
        model=model, train_generator=train, eval_generator=evaluation, criterion=loss, device=device,
        output_directory=output, config=c, epochs=end_epoch-completed, batches=campaign["batches"],
        eval_batches=phase["eval_batches"], eval_every=phase["eval_every"],
        final_eval_batches=phase["final_eval_batches"],
        learning_rate=continuation["learning_rate"] if continuation else campaign["learning_rate"],
        weight_decay=phase["weight_decay"], gradient_clip=phase["gradient_clip"],
        osd_eval_shots=0, save_model=True, load_model_path=checkpoint,
        schedule_epochs=campaign["target_epoch"]-campaign["start_epoch"],
        resume_schedule=completed > (campaign["target_epoch"] if continuation else campaign["start_epoch"]),
        scheduler_type="constant" if continuation else "cosine")
    del saved
    trainer.train(stop_requested=stop, final_evaluation=False)
    del trainer, model, loss, train, evaluation
    torch.cuda.empty_cache()


def evaluation_command(checkpoint, weights, output, campaign, seed):
    return [sys.executable, "-u", str(REPO / "scripts/evaluate_bb_hierarchical_dem_cnn.py"),
            "--checkpoint", str(checkpoint), "--weights", weights, "--output", str(output),
            "--shots", str(campaign["shots"]), "--seed", str(seed), "--batch-size", "4",
            "--confidence", str(campaign["confidence"]), "--edge-chunk-size", "4096", "--device", "cuda"]


def run_evaluation(checkpoint, weights, directory, campaign, seed, stop):
    """Reuse only complete evaluations; retain interrupted attempts for inspection."""
    signature = dict(checkpoint_sha256=digest(checkpoint), weights=weights, seed=seed,
                     shots=campaign["shots"], confidence=campaign["confidence"])
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(directory.glob("attempt_*")):
        result_path = attempt / "results.json"
        request_path = directory / (attempt.name + ".json")
        if result_path.is_file() and request_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") == "complete" and json.loads(request_path.read_text()) == signature:
                return dict(output=str(attempt), **result)
    if stop():
        return None
    index = 1
    while (directory / f"attempt_{index:03d}.json").exists():
        index += 1
    output = directory / f"attempt_{index:03d}"
    write_json(output.with_suffix(".json"), signature)
    command = evaluation_command(checkpoint, weights, output, campaign, seed)
    print("Evaluating", weights, checkpoint, "->", output, flush=True)
    with (directory / f"attempt_{index:03d}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if stop():
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return None
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"Evaluation failed; see {log.name}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = json.loads((output / "results.json").read_text())
    if result["status"] != "complete" or result["shots"] != campaign["shots"]:
        raise RuntimeError("Evaluator exited without completing requested shots")
    return dict(output=str(output), **result)


def choose_winner(records):
    # Stable ties prefer the earlier milestone, then best over latest.
    return min(records, key=lambda r: (r["result"]["metrics"]["hierarchical"]["failures"],
                                       r["epoch"], r["weights"] != "best"))


def continuation_recipe(args, campaign):
    return dict(format="bb_dem_continuation_v1", start_epoch=campaign["target_epoch"],
                learning_rate=args.continuation_lr, scheduler="constant",
                evaluation_every=args.continuation_every, batch_size=campaign["batch_size"],
                batches=campaign["batches"], shots=campaign["shots"],
                selection_seed=campaign["selection_seed"], confidence=campaign["confidence"])


def run_continuation(args, campaign, entry, directory, stop):
    """Extend the latest weights after the frozen 2M experiment, under its worker lock.

    Each evaluation pair is committed before advancing the milestone. A job
    interrupted during evaluation retries that pair, not another training block.
    The original checkpoints, selection, and held-out evaluation stay untouched.
    """
    import torch

    extension = directory / "continuous"
    extension.mkdir(exist_ok=True)
    recipe = continuation_recipe(args, campaign)
    recipe_path = extension / "campaign.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("Continuation settings changed; use its original LR and evaluation interval")
    write_json(recipe_path, recipe)
    comparison_path = extension / "comparison.json"
    records = json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    if len(records) % 2:
        raise ValueError("Continuation comparison must contain complete best/latest pairs")
    start = campaign["target_epoch"]
    for i, record in enumerate(records):
        if (record["epoch"] != start + (i // 2 + 1) * recipe["evaluation_every"] or
                record["weights"] != ("best", "latest")[i % 2]):
            raise ValueError("Continuation evaluation history is not contiguous")
    milestone = start + (len(records) // 2 + 1) * recipe["evaluation_every"]
    latest = extension / "training/model.pt"
    baseline = directory / f"checkpoints/epoch_{start}.pt"
    while not stop():
        frozen = extension / f"checkpoints/epoch_{milestone}.pt"
        if not frozen.exists():
            checkpoint = latest if latest.exists() else baseline
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            completed = validate_checkpoint(saved, campaign, entry, continuation=recipe)
            del saved
            if completed < start or completed > milestone:
                raise ValueError("Continuation checkpoint is outside the pending training segment")
            if completed < milestone:
                print(f"Continuing {entry['name']}: {completed} -> {milestone} epochs, "
                      f"constant LR {recipe['learning_rate']:g}", flush=True)
                train_segment(checkpoint, extension / "training", milestone, campaign, entry, stop,
                              continuation=recipe)
                if stop():
                    return False
            atomic_copy(latest, frozen)
        pair = []
        for weights in ("best", "latest"):
            result = run_evaluation(frozen, weights,
                                    extension / f"evaluations/epoch_{milestone}/{weights}",
                                    campaign, campaign["selection_seed"], stop)
            if result is None:
                return False
            pair.append(dict(epoch=milestone,
                             cumulative_samples=milestone*campaign["batch_size"]*campaign["batches"],
                             checkpoint=str(frozen), weights=weights, result=result))
        records.extend(pair)
        write_json(comparison_path, records)
        # Selection uses the recurring selection bank, never the held-out final bank.
        original = json.loads((directory / "comparison.json").read_text())
        write_json(extension / "selected.json", choose_winner(original + records))
        milestone += recipe["evaluation_every"]
    return False


def run(args, campaign, entry):
    import torch
    from scripts.resume_bb_dem_cnn import verify_source_stopped

    directory = args.campaign_root / entry["name"]
    directory.mkdir(parents=True, exist_ok=True)
    with worker_lock(directory):
        recipe = directory / "campaign.json"
        if recipe.exists() and json.loads(recipe.read_text()) != campaign:
            raise ValueError("Campaign settings changed; resume with the original manifest")
        write_json(recipe, campaign)
        baseline = directory / "checkpoints/epoch_1000.pt"
        if not baseline.exists():
            source = source_path(entry, args.source_root)
            verify_source_stopped(source.parents[3])
            saved = torch.load(source, map_location="cpu", weights_only=False)
            validate_checkpoint(saved, campaign, entry, initial=True)
            del saved
            atomic_copy(source, baseline)
            write_json(directory / "source.json", dict(checkpoint=str(source), sha256=digest(baseline)))
        stop = lambda: args.stop_file.is_file() or (args.campaign_root / "STOP_CONTINUOUS").is_file()
        train_output = directory / "training"
        latest = train_output / "model.pt"
        records = []
        for milestone in [campaign["start_epoch"], *campaign["milestones"]]:
            frozen = directory / f"checkpoints/epoch_{milestone}.pt"
            if not frozen.exists():
                checkpoint = latest if latest.exists() else baseline
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                completed = validate_checkpoint(saved, campaign, entry)
                del saved
                if completed > milestone:
                    raise ValueError("Missing immutable checkpoint for an already passed milestone")
                if completed < milestone:
                    if stop():
                        return False
                    train_segment(checkpoint, train_output, milestone, campaign, entry, stop)
                    if stop():
                        return False
                atomic_copy(latest, frozen)
            for weights in ("best", "latest"):
                result = run_evaluation(frozen, weights, directory / f"evaluations/epoch_{milestone}/{weights}",
                                        campaign, campaign["selection_seed"], stop)
                if result is None:
                    return False
                records.append(dict(epoch=milestone, cumulative_samples=milestone*512,
                                    checkpoint=str(frozen), weights=weights, result=result))
                write_json(directory / "comparison.json", records)
        winner = choose_winner(records)
        write_json(directory / "selected.json", winner)
        final = run_evaluation(Path(winner["checkpoint"]), winner["weights"], directory / "final_evaluation",
                               campaign, campaign["final_seed"], stop)
        if final is None:
            return False
        write_json(directory / "completed.json", dict(selected=winner, held_out_evaluation=final))
        if getattr(args, "continuous", False):
            return run_continuation(args, campaign, entry, directory, stop)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--continuous", action="store_true",
                        help="Finish the frozen campaign, then train until a stop request")
    parser.add_argument("--continuation-lr", type=float, default=1e-5)
    parser.add_argument("--continuation-every", type=int, default=1000)
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.continuation_lr) or args.continuation_lr <= 0 or args.continuation_every < 1:
        parser.error("Continuation LR must be finite and positive; interval must be positive")
    for key in ("source_root", "campaign_root", "stop_file"):
        setattr(args, key, getattr(args, key).resolve())
    campaign = json.loads(args.manifest.read_text())
    if (campaign["format"] != "bb_dem_sample_scaling_v1" or campaign["start_epoch"] != 1000 or
            campaign["target_epoch"] != 3907 or campaign["milestones"] != [1953, 2930, 3907] or
            campaign["batch_size"] != 4 or campaign["batches"] != 128 or
            campaign["selection_seed"] == campaign["final_seed"]):
        raise ValueError("Invalid sample-scaling recipe")
    entry = campaign["experiments"][args.worker]
    source = source_path(entry, args.source_root)
    baseline = args.campaign_root / entry["name"] / "checkpoints/epoch_1000.pt"
    print(json.dumps(dict(experiment=entry["name"], source=str(source), source_exists=source.is_file(),
                          campaign_root=str(args.campaign_root), total_samples=3907*512,
                          additional_epochs=2907, milestones=campaign["milestones"],
                          learning_rate=campaign["learning_rate"], evaluation_shots=campaign["shots"],
                          continuous=args.continuous,
                          continuation=continuation_recipe(args, campaign) if args.continuous else None,
                          evaluation="best/latest at baseline and milestones; independent final bank"), indent=2), flush=True)
    extension_recipe = args.campaign_root / entry["name"] / "continuous/campaign.json"
    if args.continuous and extension_recipe.exists():
        if json.loads(extension_recipe.read_text()) != continuation_recipe(args, campaign):
            raise ValueError("Continuation settings changed; use its original LR and evaluation interval")
    if args.require_checkpoint:
        import torch
        checkpoint = baseline if baseline.exists() else source
        validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False),
                            campaign, entry, initial=True)
    if args.dry_run:
        return 0
    return 0 if run(args, campaign, entry) else 75


if __name__ == "__main__":
    sys.exit(main())
