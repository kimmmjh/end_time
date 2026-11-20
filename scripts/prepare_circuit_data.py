#!/usr/bin/env python3
"""Generate an offline circuit-level toric-code dataset with Stim."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from panqec.codes import Toric2DCode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import CircuitLevelDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare detector and logical-label arrays from a Stim circuit."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--measurement_error_rate", type=float, default=0.005)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = args.L if args.rounds is None else args.rounds
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("--samples and --batch_size must be positive.")

    code = Toric2DCode(args.L)
    generator = CircuitLevelDataGenerator(
        code=code,
        error_rate=args.p,
        measurement_error_rate=args.measurement_error_rate,
        batch_size=min(args.batch_size, args.samples),
        rounds=rounds,
        categorical_classification=False,
        verbose=False,
        seed=args.seed,
    )

    syndromes = np.empty(
        (args.samples, 2, rounds, args.L**2),
        dtype=np.uint8,
    )
    logical_bits = np.empty((args.samples, 2 * code.k), dtype=np.uint8)
    device = torch.device("cpu")

    start = 0
    while start < args.samples:
        stop = min(start + args.batch_size, args.samples)
        generator.batch_size = stop - start
        batch_syndromes, batch_logicals = generator.generate_batch(device)
        syndromes[start:stop] = batch_syndromes.numpy().astype(np.uint8)
        logical_bits[start:stop] = batch_logicals.numpy().astype(np.uint8)
        start = stop

    class_weights = 2 ** np.arange(2 * code.k - 1, -1, -1)
    classes = logical_bits @ class_weights
    metadata = {
        "noise_model": "circuit",
        "L": args.L,
        "rounds": rounds,
        "gate_error_rate": args.p,
        "measurement_error_rate": args.measurement_error_rate,
        "samples": args.samples,
        "seed": args.seed,
        "syndrome_shape": list(syndromes.shape),
        "logical_bit_order": [
            "logical_x_0",
            "logical_x_1",
            "logical_z_0",
            "logical_z_1",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        syndromes=syndromes,
        logical_bits=logical_bits,
        classes=classes.astype(np.uint8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"Saved {args.samples} samples to {args.output}")
    print(f"syndromes={syndromes.shape}, logical_bits={logical_bits.shape}")


if __name__ == "__main__":
    main()
