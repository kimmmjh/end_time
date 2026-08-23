#!/usr/bin/env python3
"""CPU classical baselines for the built-in BB code-capacity experiments.

The script samples one depolarizing error bank per ``(code, p)`` point and
decodes the exact same shots with three CSS-separated binary decoders:

* BP+OSD-0;
* BP+OSD-CS, order 7; and
* BP+LSD-0.

The X and Z components of the depolarizing channel each have marginal error
rate ``2*p/3``.  They are decoded separately, so these baselines intentionally
discard the on-qubit X/Z correlation carried by a Y error.  Success is still
scored on the joint Pauli residual: it must reproduce the measured syndrome
and have trivial commutation with the complete logical basis.

This module deliberately loads ``src/bb_code.py`` without importing the
``src`` package.  Consequently the command only requires NumPy, SciPy, and
``ldpc``; it does not import Torch, PanQEC, Stim, or the neural training stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"bb-ldpc-matplotlib-{os.getpid()}"),
)

try:
    from ldpc import BpLsdDecoder, BpOsdDecoder
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment.
    raise SystemExit(
        "The 'ldpc' package is required. Install it with `python -m pip install ldpc`."
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bposd_0", "bposd_cs7", "bplsd_0")
CSV_FIELDS = (
    "code",
    "n",
    "k",
    "d",
    "channel",
    "p",
    "binary_component_error_rate",
    "method",
    "sector_strategy",
    "uses_xz_correlation",
    "samples",
    "logical_accuracy",
    "logical_error_rate",
    "logical_error_count",
    "logical_error_se",
    "logical_error_ci95_low",
    "logical_error_ci95_high",
    "syndrome_convergence",
    "flagged_failure_rate",
    "flagged_failure_count",
    "unflagged_logical_failure_rate",
    "unflagged_logical_failure_count",
    "paired_reference",
    "paired_accuracy_gain",
    "paired_gain_se",
    "paired_ci95_low",
    "paired_ci95_high",
    "rescued",
    "harmed",
    "decoder_build_seconds",
    "decode_wall_seconds",
    "throughput_shots_per_second",
    "latency_mean_us",
    "latency_median_us",
    "latency_p95_us",
    "latency_p99_us",
    "sampling_seconds",
    "bp_iterations",
    "bp_method",
    "ms_scaling_factor",
    "schedule",
    "omp_threads",
    "seed",
    "point_seed",
    "sample_sha256",
    "ldpc_version",
)

BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]


def _load_bb_code_spec() -> type[Any]:
    """Load ``BBCodeSpec`` without executing ``src/__init__.py``."""

    path = REPOSITORY_ROOT / "src" / "bb_code.py"
    module_name = "_standalone_bb_code_spec"
    if module_name in sys.modules:
        return sys.modules[module_name].BBCodeSpec

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load BB code definition from {path}.")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules while executing.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.BBCodeSpec


def _ldpc_version() -> str:
    try:
        from importlib.metadata import version

        return version("ldpc")
    except Exception:  # pragma: no cover - metadata is expected in real installs.
        return "unknown"


def _point_seed(master_seed: int, code_name: str, p: float) -> int:
    """Return an order-independent deterministic seed for one sweep point."""

    code_id = {"bb72": 72, "bb144": 144}[code_name]
    bits = struct.unpack("<Q", struct.pack("<d", float(p)))[0]
    sequence = np.random.SeedSequence(
        [master_seed, code_id, bits & 0xFFFFFFFF, bits >> 32]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def sample_depolarizing_errors(
    *, shots: int, n: int, p: float, seed: int
) -> tuple[UInt8Array, UInt8Array, str]:
    """Sample binary X/Z components and return a reproducibility checksum."""

    rng = np.random.default_rng(seed)
    uniform = rng.random((shots, n), dtype=np.float32)
    boundary_i = 1.0 - p
    boundary_x = boundary_i + p / 3.0
    boundary_y = boundary_x + p / 3.0

    x_error = ((uniform >= boundary_i) & (uniform < boundary_y)).astype(np.uint8)
    z_error = (uniform >= boundary_x).astype(np.uint8)
    del uniform

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x_error).tobytes())
    digest.update(np.ascontiguousarray(z_error).tobytes())
    return x_error, z_error, digest.hexdigest()


def syndromes_for_errors(
    x_error: UInt8Array,
    z_error: UInt8Array,
    *,
    hx: UInt8Array,
    hz: UInt8Array,
) -> tuple[UInt8Array, UInt8Array]:
    """Return X-check and Z-check syndrome sectors."""

    syndrome_x = (z_error @ hx.T).astype(np.uint8) & 1
    syndrome_z = (x_error @ hz.T).astype(np.uint8) & 1
    return syndrome_x, syndrome_z


def score_css_corrections(
    *,
    x_error: UInt8Array,
    z_error: UInt8Array,
    correction_x: UInt8Array,
    correction_z: UInt8Array,
    syndrome_x: UInt8Array,
    syndrome_z: UInt8Array,
    hx: UInt8Array,
    hz: UInt8Array,
    logicals_x: UInt8Array,
    logicals_z: UInt8Array,
) -> dict[str, BoolArray]:
    """Score corrections exactly as the neural BB hard metrics do."""

    expected_shape = x_error.shape
    for name, value in (
        ("z_error", z_error),
        ("correction_x", correction_x),
        ("correction_z", correction_z),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}."
            )

    predicted_x = (correction_z @ hx.T).astype(np.uint8) & 1
    predicted_z = (correction_x @ hz.T).astype(np.uint8) & 1
    syndrome_converged = np.all(predicted_x == syndrome_x, axis=1) & np.all(
        predicted_z == syndrome_z, axis=1
    )

    residual_x = np.bitwise_xor(x_error, correction_x)
    residual_z = np.bitwise_xor(z_error, correction_z)
    logical_x_bits = (residual_x @ logicals_z.T).astype(np.uint8) & 1
    logical_z_bits = (residual_z @ logicals_x.T).astype(np.uint8) & 1
    logical_trivial = ~(
        np.any(logical_x_bits, axis=1) | np.any(logical_z_bits, axis=1)
    )

    success = syndrome_converged & logical_trivial
    flagged = ~syndrome_converged
    unflagged = syndrome_converged & ~logical_trivial
    return {
        "success": success,
        "syndrome_converged": syndrome_converged,
        "flagged_failure": flagged,
        "unflagged_logical_failure": unflagged,
    }


def _wilson_interval(proportion: float, samples: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1.0 + z * z / samples
    center = (proportion + z * z / (2.0 * samples)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_comparison(
    success: BoolArray, reference: BoolArray
) -> dict[str, float | int]:
    difference = success.astype(np.float64) - reference.astype(np.float64)
    samples = int(difference.size)
    gain = float(difference.mean())
    standard_error = (
        float(difference.std(ddof=1) / math.sqrt(samples)) if samples > 1 else 0.0
    )
    rescued = int(np.count_nonzero(success & ~reference))
    harmed = int(np.count_nonzero(~success & reference))
    return {
        "gain": gain,
        "se": standard_error,
        "ci_low": gain - 1.96 * standard_error,
        "ci_high": gain + 1.96 * standard_error,
        "rescued": rescued,
        "harmed": harmed,
    }


def _make_decoder_pair(
    method: str,
    *,
    hx: UInt8Array,
    hz: UInt8Array,
    binary_error_rate: float,
    args: argparse.Namespace,
) -> tuple[Any, Any, float, dict[str, Any]]:
    is_lsd = method == "bplsd_0"
    max_iter = args.lsd_max_iter if is_lsd else args.osd_max_iter
    # ldpc interprets max_iter=0 as the block length.  Resolve it here so the
    # CSV records the actual, code-dependent literature default.
    if max_iter == 0:
        max_iter = int(hx.shape[1])
    scaling = (
        args.lsd_ms_scaling_factor if is_lsd else args.osd_ms_scaling_factor
    )
    common: dict[str, Any] = {
        "error_rate": binary_error_rate,
        "max_iter": max_iter,
        "bp_method": args.bp_method,
        "ms_scaling_factor": scaling,
        "schedule": args.schedule,
        "omp_thread_count": args.omp_threads,
        "random_schedule_seed": 1,
        "input_vector_type": "syndrome",
    }
    if method == "bposd_0":
        constructor: Callable[..., Any] = BpOsdDecoder
        method_args = {"osd_method": "OSD_0", "osd_order": 0}
    elif method == "bposd_cs7":
        constructor = BpOsdDecoder
        method_args = {"osd_method": "OSD_CS", "osd_order": 7}
    elif method == "bplsd_0":
        constructor = BpLsdDecoder
        method_args = {
            "lsd_method": "LSD_0",
            "lsd_order": 0,
            "bits_per_step": 1,
            "always_run_lsd": False,
        }
    else:  # pragma: no cover - argparse and caller validate this.
        raise ValueError(f"Unknown method {method!r}.")

    started = time.perf_counter()
    decoder_z = constructor(hx, **common, **method_args)
    decoder_x = constructor(hz, **common, **method_args)
    config = {
        "bp_iterations": max_iter,
        "bp_method": args.bp_method,
        "ms_scaling_factor": scaling,
    }
    return decoder_x, decoder_z, time.perf_counter() - started, config


def _decode_bank(
    decoder_x: Any,
    decoder_z: Any,
    *,
    syndrome_x: UInt8Array,
    syndrome_z: UInt8Array,
    n: int,
    warmup_shots: int,
    progress_every: int,
    method: str,
) -> tuple[UInt8Array, UInt8Array, NDArray[np.int64], float]:
    shots = syndrome_x.shape[0]
    correction_x = np.empty((shots, n), dtype=np.uint8)
    correction_z = np.empty((shots, n), dtype=np.uint8)

    for index in range(min(warmup_shots, shots)):
        decoder_z.decode(syndrome_x[index])
        decoder_x.decode(syndrome_z[index])

    latency_ns = np.empty(shots, dtype=np.int64)
    wall_started = time.perf_counter()
    for index in range(shots):
        shot_started = time.perf_counter_ns()
        correction_z[index] = decoder_z.decode(syndrome_x[index])
        correction_x[index] = decoder_x.decode(syndrome_z[index])
        latency_ns[index] = time.perf_counter_ns() - shot_started
        if progress_every and (index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - wall_started
            rate = (index + 1) / elapsed if elapsed else float("inf")
            print(
                f"  {method}: {index + 1}/{shots} shots "
                f"({rate:.1f} shots/s)",
                flush=True,
            )
    return correction_x, correction_z, latency_ns, time.perf_counter() - wall_started


def _point_key(code: str, p: float) -> str:
    p_text = format(p, ".12g").replace("-", "m").replace(".", "p")
    return f"{code}_p{p_text}"


def _build_row(
    *,
    code: Any,
    p: float,
    method: str,
    outcomes: dict[str, BoolArray],
    paired: dict[str, float | int],
    latency_ns: NDArray[np.int64],
    build_seconds: float,
    decode_wall_seconds: float,
    sampling_seconds: float,
    args: argparse.Namespace,
    decoder_config: dict[str, Any],
    point_seed: int,
    sample_sha256: str,
) -> dict[str, Any]:
    samples = int(outcomes["success"].size)
    success_count = int(np.count_nonzero(outcomes["success"]))
    accuracy = success_count / samples
    error_rate = 1.0 - accuracy
    error_count = samples - success_count
    error_se = math.sqrt(max(error_rate * (1.0 - error_rate), 0.0) / samples)
    ci_low, ci_high = _wilson_interval(error_rate, samples)
    flagged_count = int(np.count_nonzero(outcomes["flagged_failure"]))
    unflagged_count = int(
        np.count_nonzero(outcomes["unflagged_logical_failure"])
    )
    latency_us = latency_ns.astype(np.float64) / 1_000.0

    return {
        "code": code.name,
        "n": code.n,
        "k": code.k,
        "d": code.d,
        "channel": "depolarizing",
        "p": p,
        "binary_component_error_rate": 2.0 * p / 3.0,
        "method": method,
        "sector_strategy": "css_separated",
        "uses_xz_correlation": False,
        "samples": samples,
        "logical_accuracy": accuracy,
        "logical_error_rate": error_rate,
        "logical_error_count": error_count,
        "logical_error_se": error_se,
        "logical_error_ci95_low": ci_low,
        "logical_error_ci95_high": ci_high,
        "syndrome_convergence": float(outcomes["syndrome_converged"].mean()),
        "flagged_failure_rate": flagged_count / samples,
        "flagged_failure_count": flagged_count,
        "unflagged_logical_failure_rate": unflagged_count / samples,
        "unflagged_logical_failure_count": unflagged_count,
        "paired_reference": "bposd_0",
        "paired_accuracy_gain": paired["gain"],
        "paired_gain_se": paired["se"],
        "paired_ci95_low": paired["ci_low"],
        "paired_ci95_high": paired["ci_high"],
        "rescued": paired["rescued"],
        "harmed": paired["harmed"],
        "decoder_build_seconds": build_seconds,
        "decode_wall_seconds": decode_wall_seconds,
        "throughput_shots_per_second": samples / decode_wall_seconds,
        "latency_mean_us": float(latency_us.mean()),
        "latency_median_us": float(np.median(latency_us)),
        "latency_p95_us": float(np.percentile(latency_us, 95)),
        "latency_p99_us": float(np.percentile(latency_us, 99)),
        "sampling_seconds": sampling_seconds,
        "bp_iterations": decoder_config["bp_iterations"],
        "bp_method": decoder_config["bp_method"],
        "ms_scaling_factor": decoder_config["ms_scaling_factor"],
        "schedule": args.schedule,
        "omp_threads": args.omp_threads,
        "seed": args.seed,
        "point_seed": point_seed,
        "sample_sha256": sample_sha256,
        "ldpc_version": _ldpc_version(),
    }


def run_point(
    code: Any,
    p: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, NDArray[Any]]]:
    point_seed = _point_seed(args.seed, code.name, p)
    sampling_started = time.perf_counter()
    x_error, z_error, sample_sha256 = sample_depolarizing_errors(
        shots=args.shots, n=code.n, p=p, seed=point_seed
    )
    hx = np.asarray(code.hx, dtype=np.uint8)
    hz = np.asarray(code.hz, dtype=np.uint8)
    syndrome_x, syndrome_z = syndromes_for_errors(
        x_error, z_error, hx=hx, hz=hz
    )
    sampling_seconds = time.perf_counter() - sampling_started
    print(
        f"{code.name} [[{code.n},{code.k},{code.d}]] p={p:g} | "
        f"shots={args.shots} | sample={sample_sha256[:12]}",
        flush=True,
    )

    stored: dict[str, dict[str, Any]] = {}
    npz_arrays: dict[str, NDArray[Any]] = {}
    point_key = _point_key(code.name, p)
    if args.save_test_bank is not None:
        npz_arrays[f"{point_key}__x_error"] = x_error
        npz_arrays[f"{point_key}__z_error"] = z_error
        npz_arrays[f"{point_key}__syndrome_x"] = syndrome_x
        npz_arrays[f"{point_key}__syndrome_z"] = syndrome_z
    for method in METHODS:
        decoder_x, decoder_z, build_seconds, decoder_config = _make_decoder_pair(
            method,
            hx=hx,
            hz=hz,
            binary_error_rate=2.0 * p / 3.0,
            args=args,
        )
        correction_x, correction_z, latency_ns, decode_wall_seconds = _decode_bank(
            decoder_x,
            decoder_z,
            syndrome_x=syndrome_x,
            syndrome_z=syndrome_z,
            n=code.n,
            warmup_shots=args.warmup_shots,
            progress_every=args.progress_every,
            method=method,
        )
        outcomes = score_css_corrections(
            x_error=x_error,
            z_error=z_error,
            correction_x=correction_x,
            correction_z=correction_z,
            syndrome_x=syndrome_x,
            syndrome_z=syndrome_z,
            hx=hx,
            hz=hz,
            logicals_x=np.asarray(code.logicals_x, dtype=np.uint8),
            logicals_z=np.asarray(code.logicals_z, dtype=np.uint8),
        )
        stored[method] = {
            "outcomes": outcomes,
            "latency_ns": latency_ns,
            "build_seconds": build_seconds,
            "decode_wall_seconds": decode_wall_seconds,
        }
        if args.save_test_bank is not None:
            for outcome_name, values in outcomes.items():
                npz_arrays[f"{point_key}__{method}__{outcome_name}"] = values
            npz_arrays[f"{point_key}__{method}__latency_ns"] = latency_ns
        stored[method]["decoder_config"] = decoder_config

    reference = stored["bposd_0"]["outcomes"]["success"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        values = stored[method]
        paired = _paired_comparison(values["outcomes"]["success"], reference)
        row = _build_row(
            code=code,
            p=p,
            method=method,
            outcomes=values["outcomes"],
            paired=paired,
            latency_ns=values["latency_ns"],
            build_seconds=values["build_seconds"],
            decode_wall_seconds=values["decode_wall_seconds"],
            sampling_seconds=sampling_seconds,
            args=args,
            point_seed=point_seed,
            sample_sha256=sample_sha256,
            decoder_config=values["decoder_config"],
        )
        rows.append(row)
        print(
            f"  {method}: LER={row['logical_error_rate']:.8f} "
            f"flagged={row['flagged_failure_rate']:.8f} "
            f"latency_p50={row['latency_median_us']:.1f} us",
            flush=True,
        )
    return rows, npz_arrays


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_npz(
    path: Path, arrays: dict[str, NDArray[Any]], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSS-separated BP+OSD/LSD baselines for BB code capacity."
    )
    parser.add_argument(
        "--code",
        nargs="+",
        choices=("bb72", "bb144"),
        default=("bb72", "bb144"),
    )
    parser.add_argument("--p", nargs="+", required=True, type=float)
    parser.add_argument("--shots", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--osd_max_iter",
        type=int,
        default=0,
        help="BP iterations before OSD; 0 uses n (default).",
    )
    parser.add_argument(
        "--lsd_max_iter",
        type=int,
        default=0,
        help="BP iterations before LSD; 0 uses n (default).",
    )
    parser.add_argument(
        "--bp_method",
        choices=("product_sum", "minimum_sum"),
        default="minimum_sum",
    )
    parser.add_argument(
        "--osd_ms_scaling_factor",
        type=float,
        default=0.625,
        help="Min-sum scale for BP+OSD (default: 0.625).",
    )
    parser.add_argument(
        "--lsd_ms_scaling_factor",
        type=float,
        default=0.625,
        help="Min-sum scale for BP+LSD (default: 0.625).",
    )
    parser.add_argument(
        "--schedule", choices=("parallel", "serial"), default="parallel"
    )
    parser.add_argument("--omp_threads", type=int, default=1)
    parser.add_argument("--warmup_shots", type=int, default=8)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1_000,
        help="Print progress every N shots; use 0 to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bb_classical_baselines.csv"),
    )
    parser.add_argument(
        "--save_test_bank",
        "--outcomes_npz",
        dest="save_test_bank",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing the sampled X/Z errors, syndromes, "
            "per-method outcomes, and latency."
        ),
    )
    args = parser.parse_args(argv)

    if args.shots < 1:
        parser.error("--shots must be positive.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.osd_max_iter < 0:
        parser.error("--osd_max_iter must be non-negative (0 means n).")
    if args.lsd_max_iter < 0:
        parser.error("--lsd_max_iter must be non-negative (0 means n).")
    if args.osd_ms_scaling_factor < 0.0:
        parser.error("--osd_ms_scaling_factor must be non-negative.")
    if args.lsd_ms_scaling_factor <= 0.0:
        parser.error("--lsd_ms_scaling_factor must be positive.")
    if args.omp_threads < 1:
        parser.error("--omp_threads must be positive.")
    if args.warmup_shots < 0:
        parser.error("--warmup_shots must be non-negative.")
    if args.progress_every < 0:
        parser.error("--progress_every must be non-negative.")
    if len(set(args.code)) != len(args.code):
        parser.error("--code contains a duplicate value.")
    if len(set(args.p)) != len(args.p):
        parser.error("--p contains a duplicate value.")
    if any(not 0.0 < p < 0.75 for p in args.p):
        parser.error("Every --p value must lie in (0, 0.75).")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    BBCodeSpec = _load_bb_code_spec()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, NDArray[Any]] = {}

    for code_name in args.code:
        code = BBCodeSpec.from_name(code_name)
        for p in args.p:
            point_rows, point_arrays = run_point(code, p, args)
            rows.extend(point_rows)
            arrays.update(point_arrays)

    _write_csv(args.output, rows)
    print(f"Wrote {args.output}")
    if args.save_test_bank is not None:
        metadata = {
            "format_version": 1,
            "methods": METHODS,
            "channel": "depolarizing",
            "sector_strategy": "css_separated",
            "uses_xz_correlation": False,
            "codes": list(args.code),
            "p": list(args.p),
            "shots_per_point": args.shots,
            "seed": args.seed,
            "bp_method": args.bp_method,
            "osd_max_iter": args.osd_max_iter,
            "lsd_max_iter": args.lsd_max_iter,
            "osd_ms_scaling_factor": args.osd_ms_scaling_factor,
            "lsd_ms_scaling_factor": args.lsd_ms_scaling_factor,
            "schedule": args.schedule,
            "omp_threads": args.omp_threads,
            "ldpc_version": _ldpc_version(),
        }
        _write_npz(args.save_test_bank, arrays, metadata)
        print(f"Wrote {args.save_test_bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
