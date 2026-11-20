#!/usr/bin/env python3
"""Compare identical CNN residuals with reference CPU, packed CPU and packed CUDA OSD.

No training, checkpoint writes to the campaign, or automatic backend promotion.
The small default shot count is for correctness/timing, not a precise LER claim.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_bb_hierarchical_dem_cnn import aggregate, load_weights, write_json
from scripts.run_bb_dem_sample_scaling import MANIFEST, atomic_copy, digest, worker_lock


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=REPO / "recovered_60828644_rhhDfV")
    parser.add_argument("--worker", type=int, choices=(0,1), default=1)
    parser.add_argument("--checkpoint", type=Path, help="Optional immutable checkpoint instead of campaign latest")
    parser.add_argument("--weights", choices=("best", "latest"), default="latest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--confidence", type=float, default=.9)
    parser.add_argument("--edge-chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CPU is only a local algorithm smoke test, never a GPU benchmark")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.shots, args.batch_size, args.repeats, args.edge_chunk_size) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    if not .5 <= args.confidence < 1:
        parser.error("Confidence must be in [0.5,1)")
    return args


def source_checkpoint(args, entry):
    if args.checkpoint is not None:
        return args.checkpoint.resolve()
    experiment = args.campaign_root.resolve() / entry["name"]
    extension = experiment / "continuous/training/model.pt"
    return extension if extension.is_file() else experiment / "training/model.pt"


def snapshot_checkpoint(args, entry):
    destination = args.output / "checkpoint.pt"
    if args.checkpoint is not None:
        source = source_checkpoint(args, entry)
        before = digest(source)
        atomic_copy(source, destination)
        if digest(source) != before or digest(destination) != before:
            raise RuntimeError("Explicit checkpoint changed while copying; use immutable weights")
    else:
        experiment = args.campaign_root.resolve() / entry["name"]
        if not experiment.is_dir():
            raise FileNotFoundError(experiment)
        # Same per-model lock as training. Never remove/steal a running owner's lock.
        with worker_lock(experiment):
            source = source_checkpoint(args, entry)
            before = digest(source)
            atomic_copy(source, destination)
            if digest(destination) != before:
                raise RuntimeError("Checkpoint copy failed integrity verification")
    return destination, dict(original_path=str(source), sha256=before)


def device_self_check(device):
    """Exercise ties, saturation, dependent rows and invalid syndromes on this device."""
    import numpy as np
    from src._direct_osd import DirectOSD0
    from src._packed_osd import PackedDirectOSD0

    rng = np.random.default_rng(61)
    cases = 0
    for rows, columns in ((3,4), (7,19), (11,29)):
        h = rng.integers(0, 2, (rows, columns), dtype=np.uint8)
        h[-1] = h[0] ^ h[1]
        faults = rng.integers(0, 2, (8, columns), dtype=np.uint8)
        s = (h @ faults.T).T % 2
        q = rng.choice([0., 1., .5, .1, .9, .499999, .500001], size=(8,columns))
        ref, packed = DirectOSD0(h), PackedDirectOSD0(h, device=device)
        np.testing.assert_array_equal(ref.decode_batch(s,q), packed.decode_batch(s,q))
        cases += len(s)
    packed = PackedDirectOSD0(np.array([[1,0],[1,0]], dtype=np.uint8), device=device)
    try:
        packed.decode([1,0], [.5,.5])
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent syndrome was accepted")
    return cases


def benchmark(args, checkpoint, original, expected=None):
    import numpy as np
    import stim
    import torch
    from models._bb_hierarchical_dem_cnn import BBHierarchicalDemCNN
    from src._bb_circuit_experiment import _graph_fingerprint
    from src._bb_circuit_metrics import score_corrections
    from src._direct_osd import DirectOSD0
    from src._packed_osd import PackedDirectOSD0
    from src.bb_circuit_data import BBCircuitGenerator
    from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION

    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; benchmark will not fall back to CPU")
    sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    tested = device_self_check(args.device)
    config, state, provenance = load_weights(checkpoint, args.weights, expected)
    if config["circuit_schema_version"] != CIRCUIT_SCHEMA_VERSION:
        raise ValueError("Circuit schema differs from checkpoint")
    noise = config["noise_profile"]
    material = json.dumps([args.seed, config["code"], config["rounds"], noise], sort_keys=True)
    sampler_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "little")
    generator = BBCircuitGenerator(config["code"], rounds=config["rounds"],
        gate_error_rate=noise["base_error_rate"], measurement_error_rate=noise["measurement_error_rate"],
        idle_error_rate=noise["gate_idle_error_rate"], circuit_noise_model=noise["name"],
        batch_size=args.batch_size, seed=sampler_seed)
    graph = generator.graph
    if asdict(generator.noise_profile) != noise or _graph_fingerprint(graph) != config["graph_fingerprint"]:
        raise ValueError("Reconstructed circuit differs from checkpoint")
    model = BBHierarchicalDemCNN(graph, width=config["width"], depth=config["depth"],
        confidence_threshold=args.confidence, edge_chunk_size=args.edge_chunk_size,
        gradient_checkpoint=False).to(args.device).eval()
    model.load_state_dict(state, strict=True)
    del state
    candidate_name = "packed_cuda" if args.device == "cuda" else "packed_cpu_smoke"
    backends, setup = {}, {}
    factories = {"reference_cpu": lambda: DirectOSD0(graph.check_matrix),
                 "packed_cpu": lambda: PackedDirectOSD0(graph.check_matrix, device="cpu")}
    if args.device == "cuda":
        factories[candidate_name] = lambda: PackedDirectOSD0(graph.check_matrix, device="cuda")
    else:
        candidate_name = "packed_cpu"
    for name, factory in factories.items():
        sync()
        start = time.perf_counter()
        backends[name] = factory()
        sync()
        setup[name] = time.perf_counter()-start
        print(f"Setup {name}: {setup[name]:.3f}s", flush=True)
    metadata = dict(format="bb_dem_osd_benchmark_v1", config=config, source=provenance,
        original_checkpoint=original, confidence=args.confidence, active_candidates_only=True,
        comparison="gt" if args.confidence == .5 else "ge", requested_shots=args.shots,
        batch_size=args.batch_size, repeats=args.repeats, sampler_seed=sampler_seed,
        seed=args.seed, sampling="fresh_exact_circuit_shared_across_backends",
        torch_version=torch.__version__, numpy_version=np.__version__, stim_version=stim.__version__,
        torch_threads=torch.get_num_threads(), cuda_version=torch.version.cuda, device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        device_self_check_shots=tested, setup_seconds=setup,
        candidate="Torch bit-packed row elimination; NumPy sort and sparse syndrome checks on CPU",
        timing="Synchronized wall time including backend host work/transfers. Setup/warmup excluded. "
               "Pipeline estimate adds shared CNN/local/transfer time; excludes sampling/scoring/file IO.",
        purpose="Correctness and throughput experiment; not a precise LER estimate or backend promotion")
    write_json(args.output / "metadata.json", metadata)
    totals = {name:0. for name in backends}
    flags = {name:[[],[],[]] for name in backends}
    correction_mismatches = {name:0 for name in backends}
    logical_mismatches = {name:0 for name in backends}
    common_seconds = 0.
    calls = count = 0
    warmed = False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (args.output / "timings.csv").open("w", newline="") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=["batch_start", "repeat", "backend", "osd_shots", "seconds"])
        writer.writeheader()
        while count < args.shots:
            batch = generator.sample_circuit(min(args.batch_size, args.shots-count), device=args.device)
            sync()
            start = time.perf_counter()
            local = model.preprocess(batch.detectors)
            active = local.residual_detectors.bool().any(dim=1)
            s = local.residual_detectors[active].cpu().numpy()
            q = torch.sigmoid(-local.residual_log_odds[active].double()).cpu().numpy()
            a = local.local_correction.cpu().numpy()
            used = active.cpu().numpy()
            sync()
            common_seconds += time.perf_counter()-start
            if len(s) and not warmed:
                warm = [decoder.decode_batch(s[:1], q[:1]) for decoder in backends.values()]
                for other in warm[1:]:
                    np.testing.assert_array_equal(warm[0], other)
                sync()
                warmed = True
            times = {name:[] for name in backends}
            corrections = {}
            names = list(backends)
            for repeat in range(args.repeats):
                offset = (count//args.batch_size + repeat) % len(names)
                for name in names[offset:]+names[:offset]:
                    sync()
                    start = time.perf_counter()
                    residual = (backends[name].decode_batch(s,q) if len(s)
                                else np.empty((0, graph.num_mechanisms), dtype=np.uint8))
                    sync()
                    seconds = time.perf_counter()-start
                    times[name].append(seconds)
                    correction = a.copy()
                    correction[used] ^= residual
                    if name in corrections:
                        np.testing.assert_array_equal(correction, corrections[name])
                    corrections[name] = correction
                    writer.writerow(dict(batch_start=count, repeat=repeat, backend=name,
                                         osd_shots=len(s), seconds=seconds))
            detectors = batch.detectors.cpu().numpy().astype(np.uint8)
            observables = batch.observables.cpu().numpy().astype(np.uint8)
            reference = corrections["reference_cpu"]
            reference_logical = (graph.observable_matrix @ reference.T).T % 2
            for name, correction in corrections.items():
                totals[name] += float(np.median(times[name]))
                outcome = score_corrections(correction, detectors=detectors, observables=observables,
                    check_matrix=graph.check_matrix, observable_matrix=graph.observable_matrix)
                predicted = (graph.observable_matrix @ correction.T).T % 2
                for dest, value in zip(flags[name], (outcome.success, outcome.syndrome_converged,
                                                    np.all(predicted == observables, axis=1))):
                    dest.append(value)
                correction_mismatches[name] += int(np.any(correction != reference, axis=1).sum())
                logical_mismatches[name] += int(np.any(predicted != reference_logical, axis=1).sum())
                if not outcome.syndrome_converged.all() or correction_mismatches[name]:
                    np.savez_compressed(args.output / "mismatch.npz", detectors=detectors,
                        observables=observables, reference=reference, candidate=correction,
                        residual_syndrome=s, residual_probabilities=q, active=used)
                    raise RuntimeError(f"{name} differs from reference; saved mismatch.npz")
            count += len(detectors)
            calls += len(s)
            result = dict(status="complete" if count == args.shots else "running", shots=count,
                osd_shots=calls, confidence=args.confidence, common_cnn_local_transfer_seconds=common_seconds,
                backend_seconds=totals, pipeline_seconds_estimate={k:common_seconds+v for k,v in totals.items()},
                backend_speedup_vs_reference={k:totals["reference_cpu"]/v for k,v in totals.items()},
                correction_mismatch_shots=correction_mismatches, logical_mismatch_shots=logical_mismatches,
                metrics={k:aggregate(*(np.concatenate(v) for v in values)) for k,values in flags.items()},
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                cuda_tested=args.device == "cuda", candidate=candidate_name,
                real_shot_osd_exercised=calls > 0,
                acceptance="Do not change production backend automatically; review correctness and timing")
            write_json(args.output / "results.json", result)
            stream.flush()
            print(f"Shots {count}/{args.shots}; OSD calls {calls}; seconds {totals}", flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    entry = json.loads(MANIFEST.read_text())["experiments"][args.worker]
    plan = dict(experiment=entry["name"], checkpoint=str(source_checkpoint(args, entry)),
                output=str(args.output), shots=args.shots, confidence=args.confidence,
                device=args.device, training=False, automatic_resubmission=False)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return plan
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint, original = snapshot_checkpoint(args, entry)
        result = benchmark(args, checkpoint, original,
                           entry["expected_config"] if args.checkpoint is None else None)
        if digest(checkpoint) != original["sha256"]:
            raise RuntimeError("Benchmark modified its checkpoint snapshot")
        return result
    except Exception as exc:
        write_json(args.output / "failure.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
