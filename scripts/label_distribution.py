#!/usr/bin/env python3
"""Sample generated labels and print their class distribution."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from panqec.codes import Toric2DCode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._data_generator import CapacityDataGenerator, PhenomenologicalDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check logical-label distribution from the data generator."
    )
    parser.add_argument("--L", type=int, nargs="+", default=[9])
    parser.add_argument("--p", type=float, default=0.01, help="Physical error rate.")
    parser.add_argument(
        "--measurement_error_rate",
        type=float,
        default=0.01,
        help="Phenomenological measurement error rate.",
    )
    parser.add_argument(
        "--noise_model",
        choices=["phenomenological", "capacity"],
        default="phenomenological",
    )
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def make_generator(args: argparse.Namespace, code: Toric2DCode, batch_size: int):
    if args.noise_model == "capacity":
        return CapacityDataGenerator(
            code=code,
            error_rate=args.p,
            batch_size=batch_size,
            verbose=False,
        )

    return PhenomenologicalDataGenerator(
        code=code,
        error_rate=args.p,
        batch_size=batch_size,
        verbose=False,
        measurement_error_rate=args.measurement_error_rate,
    )


def sample_counts(args: argparse.Namespace, L: int) -> np.ndarray:
    code = Toric2DCode(L)
    num_classes = 2 ** (2 * len(code.size))
    counts = np.zeros(num_classes, dtype=np.int64)
    remaining = args.samples
    device = torch.device("cpu")

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        generator = make_generator(args, code, batch_size)
        _, labels = generator.generate_batch(device)
        counts += np.bincount(labels.cpu().numpy(), minlength=num_classes)
        remaining -= batch_size

    return counts


def print_counts(args: argparse.Namespace, L: int, counts: np.ndarray) -> None:
    total = counts.sum()
    probabilities = counts / total
    majority_class = int(np.argmax(counts))
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    bit_width = int(math.log2(len(counts)))

    print(
        f"L={L} p={args.p} q={args.measurement_error_rate} "
        f"noise_model={args.noise_model} samples={total}"
    )
    print("class bits count probability")
    for class_id, (count, probability) in enumerate(zip(counts, probabilities)):
        bits = format(class_id, f"0{bit_width}b")
        print(f"{class_id:>5} {bits} {count:>7} {probability:.6f}")

    print(
        f"majority_class={majority_class} "
        f"majority_baseline={probabilities[majority_class]:.6f} "
        f"entropy_bits={entropy:.4f} max_entropy_bits={math.log2(len(counts)):.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    for index, L in enumerate(args.L):
        if index:
            print()
        counts = sample_counts(args, L)
        print_counts(args, L, counts)


if __name__ == "__main__":
    main()
