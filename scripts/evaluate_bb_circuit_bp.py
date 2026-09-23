#!/usr/bin/env python3
"""Evaluate ordinary circuit-level BB min-sum BP on one saved circuit shot bank.

No training, Relay, learned updates, or OSD. Multiple iteration caps share one
trajectory and stop each shot at its first syndrome-valid correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models._equivariant_neural_bp2 import EquivariantNeuralBP2
from src._bb_circuit_metrics import score_corrections
from src._bb_circuit_trainer import BP_EVALUATION_POLICY
from src.bb_circuit_data import BBCircuitGenerator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("bb72", "bb144"), required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--rounds", type=int, help="Noisy cycles; default: code distance.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iteration-caps", type=int, nargs="+", default=[12, 1000])
    parser.add_argument("--normalisation", type=float, default=0.625)
    parser.add_argument("--message-clip", type=float, default=30.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=16, help="Batches per report.")
    parser.add_argument("--out", type=Path, required=True, help="New result directory.")
    args = parser.parse_args(argv)
    if not 0 < args.p < 0.5:
        parser.error("require 0 < p < 0.5")
    for name in ("shots", "batch_size", "threads", "progress_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.rounds is not None and args.rounds < 1:
        parser.error("rounds must be positive")
    if args.seed < 0 or any(cap < 1 for cap in args.iteration_caps):
        parser.error("seed must be non-negative and iteration caps must be positive")
    if not 0 < args.normalisation <= 1:
        parser.error("normalisation must lie in (0, 1]")
    if not math.isfinite(args.message_clip) or args.message_clip <= 0:
        parser.error("message-clip must be finite and positive")
    args.iteration_caps = sorted(set(args.iteration_caps))
    return args


def summarize(success, converged, iterations, reference):
    shots = success.size
    failures = int((~success).sum())
    ler = failures / shots
    z = 1.959963984540054
    denominator = 1 + z * z / shots
    center = (ler + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(ler * (1 - ler) / shots + z * z / (4 * shots**2)) / denominator
    difference = success.astype(float) - reference.astype(float)
    return {
        "shots": shots, "failures": failures, "logical_error_rate": ler,
        "logical_error_ci95_low": max(0., center - half),
        "logical_error_ci95_high": min(1., center + half),
        "syndrome_convergence": float(converged.mean()),
        "flagged_failures": int((~converged).sum()),
        "unflagged_failures": int((converged & ~success).sum()),
        "mean_bp_iterations": float(iterations.mean()),
        "max_used_iterations": int(iterations.max()),
        "paired_accuracy_gain": float(difference.mean()),
        "paired_gain_ci95_half_width": (
            z * float(difference.std(ddof=1)) / math.sqrt(shots) if shots > 1 else None
        ),
        "rescued": int((success & ~reference).sum()),
        "harmed": int((~success & reference).sum()),
    }


def write_report(directory, report):
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "results.json")
    if report["decoders"]:
        temporary = directory / "summary.csv.tmp"
        rows = [dict(code=report["config"]["code"], p=report["config"]["p"],
                     seed=report["config"]["seed"], status=report["status"],
                     decoder=name, **values) for name, values in report["decoders"].items()]
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(directory / "summary.csv")


@torch.inference_mode()
def run(args):
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Refuse to mix a new bank with a previous evaluation's outputs.
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"Building {args.code}, p={args.p:g}, device={device}, caps={args.iteration_caps}",
          flush=True)
    generator = BBCircuitGenerator(
        args.code, rounds=args.rounds, gate_error_rate=args.p,
        measurement_error_rate=args.p, idle_error_rate=0,
        circuit_noise_model="legacy", seed=args.seed, batch_size=args.batch_size,
    )
    graph = generator.graph
    # Sample once on CPU: changing only decode batch size leaves this bank intact.
    batch = generator.sample_circuit(args.shots)
    detectors = batch.detectors.numpy().astype(np.uint8)
    observables = batch.observables.numpy().astype(np.uint8)
    config = {
        "code": args.code, "p": args.p, "seed": args.seed,
        "noise_model": "circuit", "circuit_noise_model": "legacy",
        "measurement_error_rate": args.p, "idle_error_rate": 0,
        "rounds": generator.rounds, "detector_frames": graph.detector_frames,
        "circuit_schema_version": graph.circuit_schema_version,
        "num_detectors": graph.num_detectors, "num_mechanisms": graph.num_mechanisms,
        "num_observables": graph.num_observables, "dem_fingerprint": graph.dem_fingerprint,
        "decoder": "normalized_min_sum", "schedule": "parallel",
        "bp_evaluation_policy": BP_EVALUATION_POLICY,
        "iteration_caps": args.iteration_caps, "normalisation": args.normalisation,
        "message_clip": args.message_clip, "batch_size": args.batch_size,
        "device": str(device), "threads": args.threads,
        "neural": False, "relay": False, "osd": False,
        "metric": "Block failure over the full experiment: invalid syndrome OR any logical observable mismatch",
    }
    np.savez_compressed(args.out / "shots.npz", detectors=detectors, observables=observables,
                        metadata_json=json.dumps(config))
    report = {
        "status": "running", "config": config, "shots_requested": args.shots,
        "shots_completed": 0, "decoders": {}, "shot_bank": "shots.npz",
        "shot_content_sha256": hashlib.sha256(detectors.tobytes() + observables.tobytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "stim")},
        "python": platform.python_version(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("scripts/evaluate_bb_circuit_bp.py",
                                       "models/_equivariant_neural_bp2.py",
                                       "src/_bb_circuit_metrics.py", "src/bb_circuit_data.py",
                                       "src/bb_stim_utils.py", "src/bb_dem.py", "src/bb_code.py")},
    }
    write_report(args.out, report)
    model = EquivariantNeuralBP2(
        graph, iterations=min(args.iteration_caps), hidden_dim=4, orbit_embedding_dim=0,
        sharing="global", normalisation=args.normalisation, message_clip=args.message_clip,
        gradient_checkpoint=False,
    ).to(device).eval()
    model.requires_grad_(False)
    outcomes = {cap: {"success": np.zeros(args.shots, dtype=bool),
                      "converged": np.zeros(args.shots, dtype=bool),
                      "iterations": np.zeros(args.shots, dtype=np.int64)}
                for cap in args.iteration_caps}
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, args.shots, args.batch_size), start=1):
        end = min(start + args.batch_size, args.shots)
        results = model.decode_bp_budgets(
            batch.detectors[start:end].to(device), budgets=tuple(args.iteration_caps), neural=False,
        )
        for cap, result in results.items():
            score = score_corrections(
                result.correction.cpu().numpy(), detectors=detectors[start:end],
                observables=observables[start:end], check_matrix=graph.check_matrix,
                observable_matrix=graph.observable_matrix,
            )
            if not np.array_equal(score.syndrome_converged, result.converged.cpu().numpy()):
                raise RuntimeError("BP convergence flag disagrees with independent syndrome scoring.")
            outcomes[cap]["success"][start:end] = score.success
            outcomes[cap]["converged"][start:end] = score.syndrome_converged
            outcomes[cap]["iterations"][start:end] = result.iterations.cpu().numpy()
        if batch_index % args.progress_every == 0 or end == args.shots:
            report["shots_completed"] = end
            # Shared execution plus scoring time, not a per-cap latency measurement.
            report["evaluation_wall_seconds"] = time.perf_counter() - started
            for cap, values in outcomes.items():
                report["decoders"][f"bp_{cap}"] = {
                    "max_iterations": cap, "early_stopping": True,
                    "paired_reference": f"bp_{min(args.iteration_caps)}",
                    **summarize(**{key: value[:end] for key, value in values.items()},
                                reference=outcomes[min(args.iteration_caps)]["success"][:end]),
                }
            write_report(args.out, report)
            summary = ", ".join(f"{name}: LER={row['logical_error_rate']:.6g}, "
                                f"mean iterations={row['mean_bp_iterations']:.2f}"
                                for name, row in report["decoders"].items())
            print(f"{end}/{args.shots} shots | {summary}", flush=True)
    np.savez_compressed(args.out / "outcomes.npz",
                        **{f"bp_{cap}_{key}": value for cap, values in outcomes.items()
                           for key, value in values.items()})
    report["outcomes"] = "outcomes.npz"
    report["status"] = "complete"
    write_report(args.out, report)
    print(f"Saved {args.out / 'results.json'}", flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
