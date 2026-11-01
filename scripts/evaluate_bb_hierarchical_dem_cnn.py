#!/usr/bin/env python3
"""Evaluate saved DEM CNN weights with hierarchical residual decoding; never train."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_MANIFEST = REPO / "scripts/manifests/bb_dem_hierarchical_800_epochs.json"


def source_entry(manifest, task, index):
    data = json.loads(Path(manifest).read_text())
    if data.get("format") != "bb_dem_hierarchical_sources_v1":
        raise ValueError("Unrecognized source manifest format")
    matches = [e for e in data["experiments"] if e["task"] == task and e["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest experiment for task={task}, index={index}")
    return matches[0]


def resolve_checkpoint(entry, root, source_root, weights):
    """Prefer original server layout, then imported results; never choose by mtime."""
    directories = [Path(source_root) / entry["source_run"], Path(root) / entry["imported_run"]]
    names = ("best_model.pt", "model.pt") if weights == "best" else ("model.pt",)
    candidates = [directory / name for directory in directories for name in names]
    return next((p for p in candidates if p.is_file()), candidates[0]), candidates


def load_weights(path, weights, expected=None):
    import torch

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("Checkpoint changed while reading; evaluate a stopped run or a stable copy")
    if saved.get("format") != "bb_dem_cnn_v1":
        raise ValueError("Expected a bb_dem_cnn_v1 checkpoint")
    config = saved["config"]
    if config.get("architecture") != "bb_dem_cnn" or config.get("noise_model") != "circuit":
        raise ValueError("Expected circuit-level DEM CNN configuration")
    if expected:
        mismatch = [key for key, value in expected.items() if config.get(key) != value]
        if mismatch:
            raise ValueError("Checkpoint disagrees with source manifest: " + ", ".join(mismatch))
    if weights == "best":
        epoch = saved["best_epoch"]
        state = saved.get("best_model_state_dict")
        if state is None and saved["epoch"] == epoch:
            state = saved["model_state_dict"]
        if state is None or epoch < 0:
            raise ValueError("No validation-selected best weights in checkpoint")
    else:
        state, epoch = saved["model_state_dict"], saved["epoch"]
    return config, state, dict(checkpoint=str(path.resolve()), sha256=digest.hexdigest(),
                               weights=weights, selected_epoch=epoch + 1,
                               checkpoint_epoch=saved["epoch"] + 1)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def aggregate(success, converged, observable_match):
    n = len(success)
    failures = int((~success).sum())
    p, z = failures / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return dict(shots=n, failures=failures, block_failure_rate=p,
                block_failure_wilson95=[max(0., center - half), min(1., center + half)],
                syndrome_failure_rate=float((~converged).mean()),
                unflagged_failure_rate=float((converged & ~success).mean()),
                observable_mismatch_rate=float((~observable_match).mean()))


def paired(before, after):
    difference = after.astype(float) - before.astype(float)
    return dict(shots=len(before), accuracy_gain=float(difference.mean()),
                paired_se=float(difference.std(ddof=1) / math.sqrt(len(before))) if len(before) > 1 else None,
                rescued=int((~before & after).sum()), harmed=int((before & ~after).sum()))


def evaluate(args, checkpoint, expected=None):
    import numpy as np
    import stim
    import torch
    from models._bb_hierarchical_dem_cnn import BBHierarchicalDemCNN
    from src._bb_circuit_experiment import _graph_fingerprint
    from src._bb_circuit_metrics import score_corrections
    from src._direct_osd import DirectOSD0
    from src.bb_circuit_data import BBCircuitGenerator
    from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION

    config, state, provenance = load_weights(checkpoint, args.weights, expected)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device=cpu for a local smoke test")
    if config["circuit_schema_version"] != CIRCUIT_SCHEMA_VERSION:
        raise ValueError("Circuit schema differs from checkpoint")
    noise = config["noise_profile"]
    # A new stream, shared across depths at equal code/rounds/noise/seed.
    seed_material = json.dumps([args.seed, config["code"], config["rounds"], noise], sort_keys=True)
    sampler_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "little")
    generator = BBCircuitGenerator(
        config["code"], rounds=config["rounds"], gate_error_rate=noise["base_error_rate"],
        measurement_error_rate=noise["measurement_error_rate"],
        idle_error_rate=noise["gate_idle_error_rate"], circuit_noise_model=noise["name"],
        batch_size=args.batch_size, seed=sampler_seed)
    graph = generator.graph
    if asdict(generator.noise_profile) != noise or _graph_fingerprint(graph) != config["graph_fingerprint"]:
        raise ValueError("Reconstructed noise profile/graph does not match checkpoint")
    model = BBHierarchicalDemCNN(graph, width=config["width"], depth=config["depth"],
                                 edge_chunk_size=args.edge_chunk_size, gradient_checkpoint=False,
                                 confidence_threshold=args.confidence).to(device).eval()
    model.load_state_dict(state, strict=True)
    del state
    # Construct once so allocations are separately reported from decoding time.
    started = time.perf_counter()
    backend = DirectOSD0(graph.check_matrix)
    setup_seconds = time.perf_counter() - started
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "shots").mkdir()
    metadata = dict(format="bb_hierarchical_dem_eval_v1", source=provenance, config=config,
                    requested_shots=args.shots, batch_size=args.batch_size,
                    confidence_threshold=args.confidence, active_candidates_only=True,
                    backend="direct_hard_centered_osd0_v1", compare_direct_osd=args.compare_direct_osd,
                    seed=args.seed, sampler_seed=sampler_seed, sampling="fresh_exact_circuit",
                    device=str(device), torch_version=torch.__version__, stim_version=stim.__version__,
                    edge_chunk_size=args.edge_chunk_size, save_corrections=args.save_corrections,
                    backend_setup_seconds=setup_seconds,
                    timing_note="Batch totals, not single-shot latency. Hierarchical time includes CNN, gating, CPU transfers and OSD; excludes sampling/scoring/file IO. Direct OSD time is incremental after shared CNN.")
    write_json(output / "metadata.json", metadata)
    names = ["raw", "local", "hierarchical"] + (["direct_osd"] if args.compare_direct_osd else [])
    flags = {name: [[], [], []] for name in names}
    hierarchy_seconds = direct_seconds = 0.
    global_calls = residual_weight = local_flips = count = 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        while count < args.shots:
            batch = generator.sample_circuit(min(args.batch_size, args.shots - count), device=device)
            sync()
            started = time.perf_counter()
            result = model.decode(batch.detectors, backend=backend)
            sync()
            hierarchy_seconds += time.perf_counter() - started
            detectors = batch.detectors.cpu().numpy().astype(np.uint8)
            observables = batch.observables.cpu().numpy().astype(np.uint8)
            corrections = dict(raw=model.hard_decision(result.local.log_odds).cpu().numpy(),
                               local=result.local.local_correction.cpu().numpy(),
                               hierarchical=result.correction.cpu().numpy())
            if args.compare_direct_osd:
                started = time.perf_counter()
                fixed = corrections["raw"].copy()
                invalid = np.any((graph.check_matrix @ fixed.T).T % 2 != detectors, axis=1)
                if invalid.any():
                    probabilities = torch.sigmoid(-result.local.log_odds.double()).cpu().numpy()
                    fixed[invalid] = backend.decode_batch(detectors[invalid], probabilities[invalid])
                direct_seconds += time.perf_counter() - started
                corrections["direct_osd"] = fixed
            bank = dict(detectors=detectors, observables=observables,
                        global_used=result.global_used.cpu().numpy(),
                        residual_weight=result.local.residual_detectors.sum(dim=1).cpu().numpy(),
                        local_flip_count=corrections["local"].sum(axis=1))
            for name, correction in corrections.items():
                outcome = score_corrections(correction, detectors=detectors, observables=observables,
                                            check_matrix=graph.check_matrix, observable_matrix=graph.observable_matrix)
                predicted = (graph.observable_matrix @ correction.T).T % 2
                match = np.all(predicted == observables, axis=1)
                if name in ("hierarchical", "direct_osd") and not outcome.syndrome_converged.all():
                    raise RuntimeError(f"{name} returned an inconsistent correction")
                for dest, value in zip(flags[name], (outcome.success, outcome.syndrome_converged, match)):
                    dest.append(value)
                bank.update({f"{name}_success": outcome.success,
                             f"{name}_converged": outcome.syndrome_converged,
                             f"{name}_predicted_observables": predicted})
                if args.save_corrections:
                    bank[f"{name}_correction"] = correction
            # Save each completed batch so interrupted runs retain paired observations.
            np.savez_compressed(output / "shots" / f"batch_{count:08d}.npz", **bank)
            count += len(detectors)
            global_calls += int(bank["global_used"].sum())
            residual_weight += int(bank["residual_weight"].sum())
            local_flips += int(bank["local_flip_count"].sum())
            joined = {name: [np.concatenate(v) for v in values] for name, values in flags.items()}
            summary = dict(status="complete" if count == args.shots else "running", shots=count,
                           metrics={name: aggregate(*values) for name, values in joined.items()},
                           hierarchical_vs_raw=paired(joined["raw"][0], joined["hierarchical"][0]),
                           global_call_fraction=global_calls / count,
                           mean_residual_weight=residual_weight / count, mean_local_flips=local_flips / count,
                           hierarchical_seconds=hierarchy_seconds, direct_osd_extra_seconds=direct_seconds)
            if args.compare_direct_osd:
                summary["hierarchical_vs_direct_osd"] = paired(joined["direct_osd"][0], joined["hierarchical"][0])
            write_json(output / "results.json", summary)
            if count == args.shots or count % max(args.batch_size, 64) == 0:
                print(json.dumps(summary), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--source-root", type=Path, help="Parent of original resdir_* directories; defaults to --root")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", type=int)
    parser.add_argument("--index", type=int, choices=range(4))
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint, including resumed runs")
    parser.add_argument("--weights", choices=("best", "latest"), default="best")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--edge-chunk-size", type=int, default=4096)
    parser.add_argument("--confidence", type=float, default=.9)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare-direct-osd", action="store_true")
    parser.add_argument("--save-corrections", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true", help="Fail preview if weights are absent")
    args = parser.parse_args(argv)
    if min(args.shots, args.batch_size, args.edge_chunk_size) < 1 or args.seed < 0:
        parser.error("Shot/batch/chunk counts must be positive; seed must be nonnegative")
    if not .5 < args.confidence < 1:
        parser.error("Confidence must be strictly between 0.5 and 1")
    if (args.task is None) != (args.index is None):
        parser.error("--task and --index must be provided together")
    if args.checkpoint is None and args.task is None:
        parser.error("Provide --checkpoint or --task and --index")
    if not args.dry_run and args.output is None:
        parser.error("--output is required for evaluation")
    return args


def main(argv=None):
    args = parse_args(argv)
    entry = source_entry(args.manifest, args.task, args.index) if args.task is not None else None
    if args.checkpoint:
        path, candidates = args.checkpoint, [args.checkpoint]
    else:
        path, candidates = resolve_checkpoint(entry, args.root, args.source_root or args.root, args.weights)
    plan = dict(checkpoint=str(path), checkpoint_exists=path.is_file(),
                searched=[str(p) for p in candidates], weights=args.weights, source=entry,
                shots=args.shots, confidence=args.confidence, output=str(args.output) if args.output else None)
    print(json.dumps(plan, indent=2), flush=True)
    if not path.is_file() and (not args.dry_run or args.require_checkpoint):
        raise FileNotFoundError("DEM weights missing. Metadata/history alone cannot run inference. "
                                "Keep server checkpoints in the original resdir_* layout, set --source-root, "
                                "or provide --checkpoint. Searched: " + ", ".join(map(str, candidates)))
    if args.dry_run:
        return plan
    return evaluate(args, path, entry["expected_config"] if entry else None)


if __name__ == "__main__":
    main()
