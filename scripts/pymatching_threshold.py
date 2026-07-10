#!/usr/bin/env python3
"""Benchmark PyMatching on the repo's phenomenological toric-code noise."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymatching
from panqec.codes import Toric2DCode
from scipy.sparse import csc_matrix, csr_matrix, vstack


@dataclass
class MatchingSector:
    """One independently decoded CSS error sector."""

    matching: pymatching.Matching
    check_matrix: csc_matrix
    faults_matrix: csc_matrix
    logical_rows: np.ndarray
    component: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PyMatching threshold scan using exactly the phenomenological "
            "noise convention used by the neural decoder."
        )
    )
    parser.add_argument("--L", type=int, nargs="+", required=True)
    parser.add_argument("--p", type=float, nargs="+", required=True)
    parser.add_argument(
        "--measurement_error_rate",
        type=float,
        default=None,
        help="Measurement error rate q. Defaults to q=p for every scan point.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of syndrome rounds. Defaults to rounds=L.",
    )
    parser.add_argument("--shots", type=int, default=262_144)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2_048,
        help="Sampling/decoding chunk size; does not change the reported result.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output", type=Path, default=Path("pymatching_threshold.csv")
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


def probability_weight(probability: float) -> float:
    """Convert a Bernoulli probability to an MWPM log-likelihood weight."""
    if probability == 0.0:
        # PyMatching stores integer-scaled weights with a finite upper bound.
        return 1e6
    return math.log((1.0 - probability) / probability)


def model_ordered_matrices(
    code: Toric2DCode,
) -> tuple[csr_matrix, csr_matrix]:
    """Return checks/logicals in the same spatial order as DataGenerator."""
    logicals = vstack(
        (csr_matrix(code.logicals_x), csr_matrix(code.logicals_z))
    ).tocsr()
    block_size = code.size[0] ** len(code.size)
    original = np.asarray(code.stabilizer_matrix.todense())
    reordered = np.zeros_like(original)
    rows, columns = original.shape
    for row_block in range(rows // block_size):
        for column_block in range(columns // block_size):
            row_slice = slice(row_block * block_size, (row_block + 1) * block_size)
            column_slice = slice(
                column_block * block_size, (column_block + 1) * block_size
            )
            reordered[row_slice, column_slice] = original[
                row_slice, column_slice
            ].T
    return csr_matrix(reordered), logicals


def build_matching_sectors(
    *, L: int, rounds: int, p: float, q: float
) -> tuple[list[MatchingSector], int]:
    """Build the two CSS matchers in the generator's detector ordering."""
    code = Toric2DCode(L)
    stabilizers, logicals = model_ordered_matrices(code)
    checks_per_sector = stabilizers.shape[0] // 2

    # A depolarizing Pauli has an X component with probability 2p/3 and a Z
    # component with the same probability. Standard CSS MWPM decodes these
    # components independently and therefore does not exploit Y correlations.
    space_weight = probability_weight(2.0 * p / 3.0)
    time_weight = probability_weight(q)
    sectors: list[MatchingSector] = []

    for sector_index in range(2):
        start = sector_index * checks_per_sector
        stop = start + checks_per_sector
        sector_rows = stabilizers[start:stop]
        active_columns = np.flatnonzero(
            np.asarray(sector_rows.getnnz(axis=0)).ravel()
        )
        if active_columns.size != code.n:
            raise RuntimeError(
                f"Expected {code.n} physical edges in sector {sector_index}, "
                f"found {active_columns.size}."
            )

        check_matrix = sector_rows[:, active_columns].tocsc().astype(np.uint8)
        faults = logicals[:, active_columns]
        logical_rows = np.flatnonzero(np.asarray(faults.getnnz(axis=1)).ravel())
        faults_matrix = faults[logical_rows].tocsc().astype(np.uint8)
        if logical_rows.size != 2:
            raise RuntimeError(
                "Each toric-code CSS sector should affect two logical bits; "
                f"sector {sector_index} affects {logical_rows.tolist()}."
            )

        if np.all(active_columns < code.n):
            component = "z"
        elif np.all(active_columns >= code.n):
            component = "x"
        else:
            raise RuntimeError("A CSS sector unexpectedly mixes X and Z columns.")

        matching = pymatching.Matching.from_check_matrix(
            check_matrix,
            weights=space_weight,
            repetitions=rounds,
            timelike_weights=time_weight,
            faults_matrix=faults_matrix,
        )
        sectors.append(
            MatchingSector(
                matching=matching,
                check_matrix=check_matrix,
                faults_matrix=faults_matrix,
                logical_rows=logical_rows,
                component=component,
            )
        )

    return sectors, code.n


def parity_product(matrix: csc_matrix, states: np.ndarray) -> np.ndarray:
    """Return matrix @ states.T over GF(2), with shots on axis zero."""
    return (np.asarray(matrix @ states.T).T.astype(np.uint8, copy=False) & 1)


def sample_batch(
    *,
    rng: np.random.Generator,
    batch_size: int,
    num_qubits: int,
    rounds: int,
    p: float,
    q: float,
    sectors: list[MatchingSector],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample detection events and the four true logical bits."""
    checks_per_sector = sectors[0].check_matrix.shape[0]
    x_state = np.zeros((batch_size, num_qubits), dtype=np.uint8)
    z_state = np.zeros_like(x_state)
    previous = [
        np.zeros((batch_size, checks_per_sector), dtype=np.uint8)
        for _ in sectors
    ]
    detectors = [
        np.empty((batch_size, rounds * checks_per_sector), dtype=np.uint8)
        for _ in sectors
    ]

    for time_index in range(rounds):
        pauli_draw = rng.random((batch_size, num_qubits), dtype=np.float32)
        # Equal X/Y/Z probabilities p/3, represented without allocating a
        # string-valued Pauli tensor.
        x_state ^= pauli_draw < (2.0 * p / 3.0)
        z_state ^= (pauli_draw >= (p / 3.0)) & (pauli_draw < p)

        for sector_index, sector in enumerate(sectors):
            state = z_state if sector.component == "z" else x_state
            syndrome = parity_product(sector.check_matrix, state)
            if time_index < rounds - 1 and q > 0.0:
                syndrome ^= rng.random(
                    syndrome.shape, dtype=np.float32
                ) < q
            detection_events = syndrome ^ previous[sector_index]
            offset = time_index * checks_per_sector
            detectors[sector_index][
                :, offset : offset + checks_per_sector
            ] = detection_events
            previous[sector_index] = syndrome

    logical_bits = np.zeros((batch_size, 4), dtype=np.uint8)
    for sector in sectors:
        state = z_state if sector.component == "z" else x_state
        logical_bits[:, sector.logical_rows] = parity_product(
            sector.faults_matrix, state
        )

    return detectors, logical_bits


def benchmark_point(
    *,
    L: int,
    rounds: int,
    p: float,
    q: float,
    shots: int,
    batch_size: int,
    seed: int,
) -> dict[str, int | float | str]:
    sectors, num_qubits = build_matching_sectors(
        L=L, rounds=rounds, p=p, q=q
    )
    point_seed = np.random.SeedSequence(
        [seed, L, rounds, int(round(p * 1e9)), int(round(q * 1e9))]
    )
    rng = np.random.default_rng(point_seed)
    failures = 0
    processed = 0
    start = time.perf_counter()

    while processed < shots:
        current_batch = min(batch_size, shots - processed)
        detectors, actual = sample_batch(
            rng=rng,
            batch_size=current_batch,
            num_qubits=num_qubits,
            rounds=rounds,
            p=p,
            q=q,
            sectors=sectors,
        )
        predicted = np.zeros_like(actual)
        for sector, sector_detectors in zip(sectors, detectors):
            predicted[:, sector.logical_rows] = sector.matching.decode_batch(
                sector_detectors
            )
        failures += int(np.count_nonzero(np.any(predicted != actual, axis=1)))
        processed += current_batch

    elapsed = time.perf_counter() - start
    failure = failures / shots
    return {
        "decoder": "pymatching",
        "noise_model": "phenomenological",
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
            )
            rows.append(row)
            print(
                f"PyMatching L={L} rounds={rounds} p={p:g} q={q:g} | "
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
