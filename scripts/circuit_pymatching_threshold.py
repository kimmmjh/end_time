#!/usr/bin/env python3
"""Benchmark PyMatching on the repo's Stim circuit-level toric-code noise."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
from panqec.codes import Toric2DCode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stim_utils import generate_toric_memory_circuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PyMatching threshold scan on the same Stim circuit-level "
            "toric-code noise used by the neural decoder."
        )
    )
    parser.add_argument("--L", type=int, nargs="+", required=True)
    parser.add_argument("--p", type=float, nargs="+", required=True)
    parser.add_argument(
        "--measurement_error_rate",
        type=float,
        default=None,
        help="Measurement flip probability q. Defaults to q=p at each point.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "Number of detector frames, including the final perfect closing "
            "frame. Defaults to rounds=L."
        ),
    )
    parser.add_argument("--shots", type=int, default=262_144)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2_048,
        help="Sampling/decoding chunk size; does not change the result.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--enable_correlations",
        action="store_true",
        help=(
            "Use PyMatching's correlated-matching pass. The matching graph is "
            "then built with correlation information from decomposed DEM errors."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("circuit_pymatching_threshold.csv"),
    )
    args = parser.parse_args()

    if any(L < 2 for L in args.L):
        parser.error("Every --L value must be at least 2.")
    if any(not 0.0 <= p < 0.5 for p in args.p):
        parser.error("Every --p value must be in [0, 0.5).")
    if args.measurement_error_rate is not None and not (
        0.0 <= args.measurement_error_rate < 0.5
    ):
        parser.error("--measurement_error_rate must be in [0, 0.5).")
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be positive.")
    if args.shots < 1 or args.batch_size < 1:
        parser.error("--shots and --batch_size must be positive.")
    return args


def point_seed(*, seed: int, L: int, rounds: int, p: float, q: float) -> int:
    """Derive a deterministic Stim seed that is distinct for each scan point."""
    sequence = np.random.SeedSequence(
        [seed, L, rounds, int(round(p * 1e9)), int(round(q * 1e9))]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def build_circuit_matching(
    *,
    L: int,
    rounds: int,
    p: float,
    q: float,
    enable_correlations: bool = False,
):
    """Construct the exact sampled circuit and its DEM-derived matcher."""
    circuit = generate_toric_memory_circuit(
        Toric2DCode(L),
        rounds=rounds,
        gate_error_rate=p,
        measurement_error_rate=q,
    )
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(
        detector_error_model,
        enable_correlations=enable_correlations,
    )
    return circuit, matching


def decode_observables(
    matching: pymatching.Matching,
    detectors: np.ndarray,
    *,
    num_observables: int,
    enable_correlations: bool = False,
) -> np.ndarray:
    """Decode detector shots and normalize the prediction to all observables."""
    predictions = np.asarray(
        matching.decode_batch(
            detectors,
            enable_correlations=enable_correlations,
        ),
        dtype=np.uint8,
    )
    if predictions.ndim == 1:
        predictions = predictions.reshape(detectors.shape[0], -1)
    if predictions.shape[1] > num_observables:
        raise RuntimeError(
            "PyMatching returned more observable columns than the circuit has: "
            f"{predictions.shape[1]} > {num_observables}."
        )
    if predictions.shape[1] == num_observables:
        return predictions

    # A noiseless DEM contains no fault edges, so PyMatching can return zero
    # fault-id columns even though the circuit declares logical observables.
    padded = np.zeros((detectors.shape[0], num_observables), dtype=np.uint8)
    padded[:, : predictions.shape[1]] = predictions
    return padded


def benchmark_point(
    *,
    L: int,
    rounds: int,
    p: float,
    q: float,
    shots: int,
    batch_size: int,
    seed: int,
    enable_correlations: bool = False,
) -> dict[str, int | float | str | bool]:
    """Sample and decode one circuit-level threshold point."""
    circuit, matching = build_circuit_matching(
        L=L,
        rounds=rounds,
        p=p,
        q=q,
        enable_correlations=enable_correlations,
    )
    sampler = circuit.compile_detector_sampler(
        seed=point_seed(seed=seed, L=L, rounds=rounds, p=p, q=q)
    )
    num_observables = circuit.num_observables
    failures = 0
    processed = 0
    start = time.perf_counter()

    while processed < shots:
        current_batch = min(batch_size, shots - processed)
        detectors, actual = sampler.sample(
            shots=current_batch,
            separate_observables=True,
        )
        predicted = decode_observables(
            matching,
            detectors,
            num_observables=num_observables,
            enable_correlations=enable_correlations,
        )
        if actual.shape != (current_batch, num_observables):
            raise RuntimeError(
                "Unexpected Stim observable shape: "
                f"{actual.shape}, expected {(current_batch, num_observables)}."
            )
        failures += int(np.count_nonzero(np.any(predicted != actual, axis=1)))
        processed += current_batch

    elapsed = time.perf_counter() - start
    failure = failures / shots
    return {
        "decoder": "pymatching",
        "noise_model": "circuit",
        "matching_correlations": enable_correlations,
        "L": L,
        "rounds": rounds,
        "p": p,
        "q": q,
        "shots": shots,
        "eval_samples": shots,
        "failures": failures,
        "accuracy": 1.0 - failure,
        "failure": failure,
        "standard_error": math.sqrt(failure * (1.0 - failure) / shots),
        "seed": seed,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    print("Executed Command: python " + shlex.join(sys.argv), flush=True)
    rows = []
    for L in args.L:
        rounds = L if args.rounds is None else args.rounds
        for p in args.p:
            q = p if args.measurement_error_rate is None else args.measurement_error_rate
            row = benchmark_point(
                L=L,
                rounds=rounds,
                p=p,
                q=q,
                shots=args.shots,
                batch_size=args.batch_size,
                seed=args.seed,
                enable_correlations=args.enable_correlations,
            )
            rows.append(row)
            print(
                f"PyMatching circuit L={L} rounds={rounds} p={p:g} q={q:g} | "
                f"correlations={args.enable_correlations} | "
                f"failures={row['failures']}/{args.shots} | "
                f"failure={row['failure']:.8g} | "
                f"accuracy={row['accuracy']:.8g} | "
                f"time={row['elapsed_seconds']:.2f}s",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
