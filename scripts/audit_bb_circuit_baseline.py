#!/usr/bin/env python3
"""Audit the archived circuit BP/OSD pipeline on a shared, saved shot bank.

This is a diagnostic, not a reproduction of a paper's memory circuit. It
preserves the training decoder and adds reference decoders only in this script.
The archived rows explicitly replay the old clipping order; the corrected rows
use the current model. Their only difference is clipping versus subtraction order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from ldpc import BpOsdDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models._equivariant_neural_bp2 import EquivariantNeuralBP2
from src._bb_circuit_metrics import OsdPostprocessor, score_corrections
from src.bb_circuit_data import BBCircuitGenerator


def archived_variable_update(self, check_messages):
    """Replay the pre-2026-09-17 update for the historical comparison only."""
    aggregate = torch.zeros(
        (check_messages.shape[0], self.num_mechanisms),
        dtype=check_messages.dtype,
        device=check_messages.device,
    ).index_add(1, self.edge_mechanism, check_messages)
    posterior = (self.prior_log_odds.unsqueeze(0) + aggregate).clamp(
        -self.message_clip, self.message_clip
    )
    outgoing = posterior[:, self.edge_mechanism] - check_messages
    return posterior, outgoing.clamp(
        -self.message_clip, self.message_clip
    )


def numerical_probes():
    graph = SimpleNamespace(
        edge_detector=np.array([0, 1]),
        edge_mechanism=np.array([0, 0]),
        edge_orbit=np.array([0, 0]),
        num_detectors=2,
        num_mechanisms=1,
        num_observables=0,
        num_orbits=1,
        prior_log_odds=np.array([4.0]),
    )
    model = EquivariantNeuralBP2(graph, gradient_checkpoint=False)
    incoming = torch.tensor([[18.0, 18.0]])
    _, archived = archived_variable_update(model, incoming)
    _, expected = model._variable_update(incoming)
    assert archived.tolist() == [[12.0, 12.0]]
    assert expected.tolist() == [[22.0, 22.0]]

    check = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    supplied = np.array([[0.5, 1.0, 2.0]])
    processor = OsdPostprocessor(
        check, priors=np.array([0.1, 0.2, 0.3]), method="OSD_0", order=0
    )
    correction = processor.decode_batch(
        np.array([[1, 0]], dtype=np.uint8), posterior=supplied
    )
    actual = processor._decoder.log_prob_ratios.copy()
    assert not np.allclose(actual, supplied[0])
    assert processor._decoder.converge  # The new BP succeeds and bypasses OSD.
    return {
        "clipping": {
            "prior_llr": 4.0,
            "incoming_check_messages": incoming[0].tolist(),
            "clip": 30.0,
            "archived_variable_messages": archived[0].tolist(),
            "extrinsic_before_clip": expected[0].tolist(),
        },
        "posterior_reseeding": {
            "check_matrix": check.tolist(),
            "syndrome": [1, 0],
            "supplied_posterior": supplied[0].tolist(),
            "backend_posterior_after_new_bp": actual.tolist(),
            "backend_bp_converged": bool(processor._decoder.converge),
            "correction": correction[0].tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("bb72", "bb144"), default="bb72")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1 or args.batch_size < 1 or not 0 < args.p < 0.5:
        parser.error("shots and batch-size must be positive; require 0 < p < 0.5")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    probes = numerical_probes()
    generator = BBCircuitGenerator(
        args.code, gate_error_rate=args.p, measurement_error_rate=args.p,
        idle_error_rate=0, circuit_noise_model="legacy", seed=args.seed,
    )
    graph = generator.graph
    batch = generator.sample_circuit(args.shots)
    syndrome = batch.detectors.numpy().astype(np.uint8)
    observables = batch.observables.numpy().astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shot_path = args.out.with_suffix(".shots.npz")
    np.savez_compressed(shot_path, detectors=syndrome, observables=observables)
    report = {
        "purpose": "Implementation audit on legacy circuit, not paper reproduction",
        "code": args.code, "p": args.p, "shots": args.shots, "seed": args.seed,
        "rounds": generator.rounds, "noise_model": "legacy", "idle_error_rate": 0,
        "metric": "Any of 24 logical correlation-sheet mismatches OR invalid syndrome",
        "graph": {
            "detectors": graph.num_detectors, "mechanisms": graph.num_mechanisms,
            "observables": graph.num_observables, "dem_fingerprint": graph.dem_fingerprint,
        },
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "torch", "stim", "ldpc"
        )},
        "python": platform.python_version(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "models/_equivariant_neural_bp2.py", "src/_bb_circuit_metrics.py",
                "src/bb_stim_utils.py", "src/bb_dem.py", "src/bb_circuit_data.py",
                "scripts/audit_bb_circuit_baseline.py",
            )
        },
        "shot_bank": shot_path.name,
        "shot_content_sha256": hashlib.sha256(
            syndrome.tobytes() + observables.tobytes()
        ).hexdigest(),
        "numerical_probes": probes, "decoders": {},
    }
    success_vectors = {}

    def record(name, correction, elapsed, **settings):
        outcomes = score_corrections(
            correction, detectors=syndrome, observables=observables,
            check_matrix=graph.check_matrix, observable_matrix=graph.observable_matrix,
        )
        predicted = np.asarray(graph.observable_matrix @ correction.T).T % 2
        row = {
            "failures": int((~outcomes.success).sum()),
            "block_ler": outcomes.logical_error_rate,
            "flagged_failures": int((~outcomes.syndrome_converged).sum()),
            "observable_only_failures_first_12": int(np.any(
                predicted[:, :12] != observables[:, :12], axis=1
            ).sum()),
            "observable_only_failures_last_12": int(np.any(
                predicted[:, 12:] != observables[:, 12:], axis=1
            ).sum()),
            "seconds": elapsed, **settings,
        }
        success_vectors[name] = outcomes.success
        report["decoders"][name] = row
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"decoder": name, **row}), flush=True)

    model = EquivariantNeuralBP2(graph, iterations=12, gradient_checkpoint=False).eval()
    corrected_variable_update = model._variable_update
    for ablate_clipping in (False, True):
        model._variable_update = (
            corrected_variable_update if ablate_clipping
            else MethodType(archived_variable_update, model)
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            posterior = np.concatenate([
                model(batch.detectors[i:i + args.batch_size], neural=False).numpy()
                for i in range(0, args.shots, args.batch_size)
            ])
        bp_seconds = time.perf_counter() - t0
        name = "extrinsic_before_clip" if ablate_clipping else "archived"
        record(name + "_bp12_raw", (posterior < 0).astype(np.uint8), bp_seconds,
               max_iter=12, min_sum_scale=0.625, early_stopping=False)
        processor = OsdPostprocessor(
            graph.check_matrix, priors=graph.priors, method="OSD_0", order=0
        )
        t0 = time.perf_counter()
        correction = processor.decode_batch(syndrome, posterior=posterior)
        record(name + "_bp12_reseed_bp1_osd0", correction,
               bp_seconds + time.perf_counter() - t0,
               max_iter=12, min_sum_scale=0.625, reseeded_bp_iterations=1, osd_order=0)

    for iterations, scale, order in ((12, 0.625, 0), (1000, 1.0, 0), (1000, 1.0, 3)):
        t0 = time.perf_counter()
        decoder = BpOsdDecoder(
            graph.check_matrix, error_channel=list(graph.priors), max_iter=iterations,
            bp_method="ms", ms_scaling_factor=scale, schedule="parallel",
            osd_method="OSD_CS" if order else "OSD_0", osd_order=order,
        )
        correction = np.empty((args.shots, graph.num_mechanisms), dtype=np.uint8)
        bp_successes = 0
        for i, shot in enumerate(syndrome):
            correction[i] = decoder.decode(shot)
            bp_successes += int(decoder.converge)
            if (i + 1) % 128 == 0:
                print(f"Reference BP{iterations}/scale{scale}/OSD{order}: {i + 1}/{args.shots}", flush=True)
        record(f"ldpc_bp{iterations}_scale{scale}_osd{order}", correction,
               time.perf_counter() - t0, max_iter=iterations, min_sum_scale=scale,
               osd_order=order, early_stopping=True, bp_converged_shots=bp_successes)

    archived = success_vectors["archived_bp12_reseed_bp1_osd0"]
    report["paired_vs_archived_pipeline"] = {
        name: {
            "reference_succeeds_archived_fails": int((success & ~archived).sum()),
            "archived_succeeds_reference_fails": int((archived & ~success).sum()),
        }
        for name, success in success_vectors.items()
    }
    np.savez_compressed(args.out.with_suffix(".outcomes.npz"), **success_vectors)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
