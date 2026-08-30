#!/usr/bin/env python3
"""Paired BB Neural-BP versus classical-decoder evaluation.

This script re-evaluates archived selected-best Neural-BP checkpoints on the
exact error banks saved by ``bb_classical_baselines.py``.  It never resamples
errors or reruns a classical decoder.  Instead, it validates the stored bank
and classical CSV, runs the neural checkpoint on the stored syndromes, and
writes shot-paired accuracy gains, confidence intervals, rescues, and harms.

The built-in paths reproduce the BB72/BB144 code-capacity depolarizing audit
at p = 0.04, 0.06, 0.08, and 0.10.  Paths can be overridden with repeated
``CODE=PATH`` options when the archived artifacts are relocated.
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
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"bb-paired-matplotlib-{os.getpid()}"),
)

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised by CLI environment.
    raise SystemExit(
        "PyTorch is required to evaluate the archived neural checkpoints."
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIRS = {
    "bb72": Path(
        "results/bb/code_capacity/depolarizing/orbit/bb72/resdir_57181711"
    ),
    "bb144": Path(
        "results/bb/code_capacity/depolarizing/orbit/bb144/resdir_57181713"
    ),
}
DEFAULT_BANK_DIRS = {
    "bb72": Path(
        "results/bb/code_capacity/depolarizing/classical/css_separated/"
        "bb72/resdir_57452848"
    ),
    "bb144": Path(
        "results/bb/code_capacity/depolarizing/classical/css_separated/"
        "bb144/resdir_57452850"
    ),
}
DEFAULT_P = (0.04, 0.06, 0.08, 0.10)
CLASSICAL_METHOD = "bposd_cs7"
CSV_FIELDS = (
    "code",
    "n",
    "k",
    "d",
    "channel",
    "p",
    "samples",
    "neural_method",
    "neural_sector_strategy",
    "neural_uses_xz_correlation",
    "classical_method",
    "classical_sector_strategy",
    "classical_uses_xz_correlation",
    "vanilla_method",
    "vanilla_sector_strategy",
    "vanilla_uses_xz_correlation",
    "neural_logical_accuracy",
    "neural_logical_error_rate",
    "neural_logical_error_count",
    "neural_logical_error_se",
    "neural_logical_error_ci95_low",
    "neural_logical_error_ci95_high",
    "neural_syndrome_convergence",
    "neural_flagged_failure_rate",
    "neural_flagged_failure_count",
    "neural_unflagged_logical_failure_rate",
    "neural_unflagged_logical_failure_count",
    "classical_logical_accuracy",
    "classical_logical_error_rate",
    "classical_logical_error_count",
    "classical_logical_error_se",
    "classical_logical_error_ci95_low",
    "classical_logical_error_ci95_high",
    "classical_syndrome_convergence",
    "classical_flagged_failure_rate",
    "classical_flagged_failure_count",
    "classical_unflagged_logical_failure_rate",
    "classical_unflagged_logical_failure_count",
    "vanilla_logical_accuracy",
    "vanilla_logical_error_rate",
    "vanilla_logical_error_count",
    "vanilla_logical_error_se",
    "vanilla_logical_error_ci95_low",
    "vanilla_logical_error_ci95_high",
    "vanilla_syndrome_convergence",
    "vanilla_flagged_failure_rate",
    "vanilla_flagged_failure_count",
    "vanilla_unflagged_logical_failure_rate",
    "vanilla_unflagged_logical_failure_count",
    "paired_reference",
    "paired_accuracy_gain",
    "paired_gain_se",
    "paired_ci95_low",
    "paired_ci95_high",
    "paired_ci_method",
    "rescued",
    "harmed",
    "discordant",
    "neural_vs_vanilla_paired_accuracy_gain",
    "neural_vs_vanilla_paired_gain_se",
    "neural_vs_vanilla_paired_ci95_low",
    "neural_vs_vanilla_paired_ci95_high",
    "neural_vs_vanilla_rescued",
    "neural_vs_vanilla_harmed",
    "neural_vs_vanilla_discordant",
    "vanilla_vs_classical_paired_accuracy_gain",
    "vanilla_vs_classical_paired_gain_se",
    "vanilla_vs_classical_paired_ci95_low",
    "vanilla_vs_classical_paired_ci95_high",
    "vanilla_vs_classical_rescued",
    "vanilla_vs_classical_harmed",
    "vanilla_vs_classical_discordant",
    "relative_ler_reduction",
    "classical_to_neural_ler_ratio",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_state",
    "checkpoint_format_version",
    "checkpoint_epoch",
    "checkpoint_best_epoch",
    "checkpoint_best_validation_accuracy",
    "neural_bp_iterations",
    "neural_residual_hidden_dim",
    "neural_parameter_sharing",
    "neural_residual_scale",
    "neural_max_relaxation_delta",
    "test_bank_path",
    "test_bank_sha256",
    "sample_sha256",
    "classical_csv_path",
    "classical_ldpc_version",
    "classical_bp_method",
    "classical_bp_iterations",
    "classical_ms_scaling_factor",
    "classical_schedule",
    "evaluation_device",
    "evaluation_batch_size",
    "torch_version",
    "numpy_version",
    "analysis_script_path",
    "analysis_script_sha256",
)

BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]


def _load_class(path: Path, module_name: str, class_name: str) -> type[Any]:
    """Load one self-contained module without executing package __init__.py."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {class_name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during execution.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


BBCodeSpec = _load_class(
    REPOSITORY_ROOT / "src" / "bb_code.py",
    "_paired_bb_code_spec",
    "BBCodeSpec",
)
EquivariantNeuralBP4 = _load_class(
    REPOSITORY_ROOT / "models" / "_equivariant_neural_bp.py",
    "_paired_equivariant_neural_bp",
    "EquivariantNeuralBP4",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_sha256(x_error: UInt8Array, z_error: UInt8Array) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x_error).tobytes())
    digest.update(np.ascontiguousarray(z_error).tobytes())
    return digest.hexdigest()


def _graph_fingerprint(code: Any) -> str:
    digest = hashlib.sha256()
    digest.update(code.name.encode("utf-8"))
    digest.update(np.asarray(code.hx, dtype=np.uint8).tobytes())
    digest.update(np.asarray(code.hz, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _point_key(code: str, p: float) -> str:
    p_text = format(p, ".12g").replace("-", "m").replace(".", "p")
    return f"{code}_p{p_text}"


def _same_probability(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only flag.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return value


def _parse_code_paths(
    values: Sequence[str] | None,
    defaults: Mapping[str, Path],
    *,
    option: str,
) -> dict[str, Path]:
    result = {code: REPOSITORY_ROOT / path for code, path in defaults.items()}
    if values is None:
        return result
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects CODE=PATH, got {value!r}.")
        code, raw_path = value.split("=", 1)
        code = code.strip().lower()
        if code not in defaults:
            raise ValueError(f"{option} has unknown code {code!r}.")
        path = Path(raw_path).expanduser()
        result[code] = path if path.is_absolute() else REPOSITORY_ROOT / path
    return result


def _checkpoint_index(
    roots: Mapping[str, Path], codes: Sequence[str]
) -> dict[tuple[str, float], tuple[Path, dict[str, Any]]]:
    index: dict[tuple[str, float], tuple[Path, dict[str, Any]]] = {}
    for expected_code in codes:
        root = roots[expected_code]
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        paths = sorted(root.glob("**/best_model.pt"))
        if not paths:
            raise FileNotFoundError(f"No best_model.pt below {root}")
        for path in paths:
            checkpoint = _torch_load(path)
            config = checkpoint.get("experiment_config")
            if not isinstance(config, dict):
                raise ValueError(f"Checkpoint has no experiment_config: {path}")
            code = str(config.get("code", "")).lower()
            if code != expected_code:
                raise ValueError(
                    f"Checkpoint code mismatch below {root}: {path} says {code!r}."
                )
            if config.get("architecture") != "bb_neural_bp":
                raise ValueError(f"Unexpected checkpoint architecture: {path}")
            if config.get("noise_model") != "capacity":
                raise ValueError(f"Checkpoint is not code-capacity: {path}")
            if config.get("channel") != "depolarizing":
                raise ValueError(f"Checkpoint is not depolarizing: {path}")
            p = float(config["error_rate"])
            key = (code, p)
            if key in index:
                raise ValueError(
                    f"Duplicate selected-best checkpoints for {code}, p={p}: "
                    f"{index[key][0]} and {path}"
                )
            index[key] = (path, config)
    return index


def _bank_index(
    roots: Mapping[str, Path], codes: Sequence[str]
) -> dict[tuple[str, float], tuple[Path, dict[str, Any]]]:
    index: dict[tuple[str, float], tuple[Path, dict[str, Any]]] = {}
    for expected_code in codes:
        root = roots[expected_code]
        if not root.is_dir():
            raise FileNotFoundError(f"Test-bank directory not found: {root}")
        paths = sorted(root.glob("*_test_bank.npz"))
        if not paths:
            raise FileNotFoundError(f"No test-bank NPZ below {root}")
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if "metadata_json" not in archive:
                    raise ValueError(f"Test bank has no metadata_json: {path}")
                metadata = json.loads(str(archive["metadata_json"].item()))
            codes_in_bank = metadata.get("codes")
            probabilities = metadata.get("p")
            if codes_in_bank != [expected_code] or not isinstance(
                probabilities, list
            ) or len(probabilities) != 1:
                raise ValueError(f"Expected one {expected_code} point in {path}.")
            if metadata.get("channel") != "depolarizing":
                raise ValueError(f"Test bank is not depolarizing: {path}")
            p = float(probabilities[0])
            key = (expected_code, p)
            if key in index:
                raise ValueError(f"Duplicate test banks for {expected_code}, p={p}.")
            index[key] = (path, metadata)
    return index


def _find_key(
    index: Mapping[tuple[str, float], Any], code: str, p: float, kind: str
) -> tuple[str, float]:
    candidates = [
        key for key in index if key[0] == code and _same_probability(key[1], p)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {kind} for {code}, p={p:g}; found {len(candidates)}."
        )
    return candidates[0]


def _require_binary(name: str, value: NDArray[Any]) -> UInt8Array:
    array = np.asarray(value, dtype=np.uint8)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} contains non-binary entries.")
    return array


def _validate_outcome_partition(
    *,
    success: BoolArray,
    syndrome_converged: BoolArray,
    flagged_failure: BoolArray,
    unflagged_logical_failure: BoolArray,
    label: str,
) -> None:
    shape = success.shape
    if any(
        value.shape != shape
        for value in (
            syndrome_converged,
            flagged_failure,
            unflagged_logical_failure,
        )
    ):
        raise ValueError(f"{label} outcome vectors have inconsistent shapes.")
    if not np.array_equal(flagged_failure, ~syndrome_converged):
        raise ValueError(f"{label} flagged outcomes contradict convergence.")
    if np.any(flagged_failure & unflagged_logical_failure):
        raise ValueError(f"{label} failure categories overlap.")
    if not np.array_equal(~success, flagged_failure | unflagged_logical_failure):
        raise ValueError(f"{label} failure categories do not partition failures.")


def _load_bank(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    code: Any,
    p: float,
) -> dict[str, NDArray[Any]]:
    prefix = _point_key(code.name, p)
    required = (
        "x_error",
        "z_error",
        "syndrome_x",
        "syndrome_z",
        f"{CLASSICAL_METHOD}__success",
        f"{CLASSICAL_METHOD}__syndrome_converged",
        f"{CLASSICAL_METHOD}__flagged_failure",
        f"{CLASSICAL_METHOD}__unflagged_logical_failure",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if f"{prefix}__{name}" not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        result = {
            name: np.asarray(archive[f"{prefix}__{name}"]).copy()
            for name in required
        }

    x_error = _require_binary("x_error", result["x_error"])
    z_error = _require_binary("z_error", result["z_error"])
    syndrome_x = _require_binary("syndrome_x", result["syndrome_x"])
    syndrome_z = _require_binary("syndrome_z", result["syndrome_z"])
    shots = int(metadata["shots_per_point"])
    if x_error.shape != (shots, code.n) or z_error.shape != x_error.shape:
        raise ValueError(f"Unexpected error-array shapes in {path}.")
    if syndrome_x.shape != (shots, code.num_x_checks):
        raise ValueError(f"Unexpected X-syndrome shape in {path}.")
    if syndrome_z.shape != (shots, code.num_z_checks):
        raise ValueError(f"Unexpected Z-syndrome shape in {path}.")

    recomputed_x = (z_error @ np.asarray(code.hx, dtype=np.uint8).T) & 1
    recomputed_z = (x_error @ np.asarray(code.hz, dtype=np.uint8).T) & 1
    if not np.array_equal(syndrome_x, recomputed_x) or not np.array_equal(
        syndrome_z, recomputed_z
    ):
        raise ValueError(f"Stored syndromes do not match stored errors in {path}.")

    result["x_error"] = x_error
    result["z_error"] = z_error
    result["syndrome_x"] = syndrome_x
    result["syndrome_z"] = syndrome_z
    for name in required[4:]:
        result[name] = np.asarray(result[name], dtype=np.bool_)
        if result[name].shape != (shots,):
            raise ValueError(f"Unexpected outcome shape for {name} in {path}.")
    _validate_outcome_partition(
        success=result[f"{CLASSICAL_METHOD}__success"],
        syndrome_converged=result[
            f"{CLASSICAL_METHOD}__syndrome_converged"
        ],
        flagged_failure=result[f"{CLASSICAL_METHOD}__flagged_failure"],
        unflagged_logical_failure=result[
            f"{CLASSICAL_METHOD}__unflagged_logical_failure"
        ],
        label=CLASSICAL_METHOD,
    )
    return result


def _load_classical_csv_row(
    path: Path,
    *,
    code: str,
    p: float,
    samples: int,
    sample_sha256: str,
    success: BoolArray,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Classical CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == CLASSICAL_METHOD
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {CLASSICAL_METHOD} row in {path}.")
    row = matches[0]
    accuracy = float(success.mean())
    checks = (
        row["code"] == code,
        _same_probability(float(row["p"]), p),
        int(row["samples"]) == samples,
        row["sample_sha256"] == sample_sha256,
        math.isclose(
            float(row["logical_accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-15
        ),
    )
    if not all(checks):
        raise ValueError(f"Classical CSV does not match the NPZ outcomes: {path}")
    return row


def _edge_orbits(code: Any, sharing: str) -> NDArray[np.int64]:
    if sharing == "orbit":
        return np.asarray(code.edge_orbit, dtype=np.int64)
    if sharing == "global":
        return np.zeros(code.num_edges, dtype=np.int64)
    if sharing == "edge":
        return np.arange(code.num_edges, dtype=np.int64)
    raise ValueError(f"Unknown checkpoint parameter sharing {sharing!r}.")


def _validate_checkpoint_config(config: Mapping[str, Any], code: Any, p: float) -> None:
    expected = {
        "architecture": "bb_neural_bp",
        "code": code.name,
        "n": code.n,
        "k": code.k,
        "d": code.d,
        "noise_model": "capacity",
        "channel": "depolarizing",
    }
    mismatches = [
        f"{key}: checkpoint={config.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if not _same_probability(float(config.get("error_rate", -1.0)), p):
        mismatches.append(
            f"error_rate: checkpoint={config.get('error_rate')!r}, expected={p!r}"
        )
    fingerprint = _graph_fingerprint(code)
    if config.get("graph_fingerprint") != fingerprint:
        mismatches.append("graph_fingerprint does not match the current BB graph")
    if mismatches:
        raise ValueError("Incompatible checkpoint:\n  " + "\n  ".join(mismatches))


def _make_model(
    checkpoint: Mapping[str, Any], code: Any, p: float, device: torch.device
) -> tuple[Any, Mapping[str, Any], str]:
    config = checkpoint.get("experiment_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no experiment_config.")
    _validate_checkpoint_config(config, code, p)
    state = checkpoint.get("best_model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(
            "Selected-best evaluation requires best_model_state_dict; refusing "
            "to substitute a latest-epoch state."
        )
    model = EquivariantNeuralBP4(
        code.hx,
        code.hz,
        edge_orbits=_edge_orbits(code, str(config["bp_parameter_sharing"])),
        iterations=int(config["bp_iterations"]),
        residual_hidden_dim=int(config["bp_residual_hidden_dim"]),
        residual_scale=float(config["bp_residual_scale"]),
        max_relaxation_delta=float(config["bp_max_relaxation_delta"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config, "best_model_state_dict"


def _score_corrections(
    correction: UInt8Array,
    *,
    x_error: UInt8Array,
    z_error: UInt8Array,
    syndrome_x: UInt8Array,
    syndrome_z: UInt8Array,
    code: Any,
) -> dict[str, BoolArray]:
    correction_x = np.isin(correction, (1, 2)).astype(np.uint8)
    correction_z = np.isin(correction, (2, 3)).astype(np.uint8)
    predicted_x = (correction_z @ np.asarray(code.hx, dtype=np.uint8).T) & 1
    predicted_z = (correction_x @ np.asarray(code.hz, dtype=np.uint8).T) & 1
    converged = np.all(predicted_x == syndrome_x, axis=1) & np.all(
        predicted_z == syndrome_z, axis=1
    )
    residual_x = np.bitwise_xor(x_error, correction_x)
    residual_z = np.bitwise_xor(z_error, correction_z)
    logical_x = (
        residual_x @ np.asarray(code.logicals_z, dtype=np.uint8).T
    ) & 1
    logical_z = (
        residual_z @ np.asarray(code.logicals_x, dtype=np.uint8).T
    ) & 1
    logical_trivial = ~(np.any(logical_x, axis=1) | np.any(logical_z, axis=1))
    success = converged & logical_trivial
    return {
        "success": success,
        "syndrome_converged": converged,
        "flagged_failure": ~converged,
        "unflagged_logical_failure": converged & ~logical_trivial,
    }


def _evaluate_neural_and_vanilla(
    model: Any,
    *,
    bank: Mapping[str, NDArray[Any]],
    code: Any,
    p: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, BoolArray], dict[str, BoolArray]]:
    shots = int(bank["x_error"].shape[0])
    names = (
        "success",
        "syndrome_converged",
        "flagged_failure",
        "unflagged_logical_failure",
    )
    parts: dict[str, dict[str, list[BoolArray]]] = {
        mode: {name: [] for name in names} for mode in ("neural", "vanilla")
    }
    with torch.inference_mode():
        for start in range(0, shots, batch_size):
            stop = min(start + batch_size, shots)
            syndrome = np.concatenate(
                (bank["syndrome_x"][start:stop], bank["syndrome_z"][start:stop]),
                axis=1,
            )
            syndrome_tensor = torch.as_tensor(
                syndrome, dtype=torch.float32, device=device
            )
            for mode, neural_enabled in (("neural", True), ("vanilla", False)):
                logits = model(syndrome_tensor, p=p, neural=neural_enabled)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError(
                        f"Non-finite {mode} logits for {code.name}, p={p:g}."
                    )
                correction = (
                    logits.argmax(dim=-1).to(dtype=torch.uint8).cpu().numpy()
                )
                scored = _score_corrections(
                    correction,
                    x_error=bank["x_error"][start:stop],
                    z_error=bank["z_error"][start:stop],
                    syndrome_x=bank["syndrome_x"][start:stop],
                    syndrome_z=bank["syndrome_z"][start:stop],
                    code=code,
                )
                for name, values in scored.items():
                    parts[mode][name].append(values)
    outcomes = {
        mode: {name: np.concatenate(values) for name, values in by_name.items()}
        for mode, by_name in parts.items()
    }
    for mode, values in outcomes.items():
        _validate_outcome_partition(label=mode, **values)
    return outcomes["neural"], outcomes["vanilla"]


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


def _summarize(outcomes: Mapping[str, BoolArray]) -> dict[str, float | int]:
    success = outcomes["success"]
    samples = int(success.size)
    success_count = int(np.count_nonzero(success))
    error_count = samples - success_count
    error_rate = error_count / samples
    error_se = math.sqrt(max(error_rate * (1.0 - error_rate), 0.0) / samples)
    ci_low, ci_high = _wilson_interval(error_rate, samples)
    flagged_count = int(np.count_nonzero(outcomes["flagged_failure"]))
    unflagged_count = int(
        np.count_nonzero(outcomes["unflagged_logical_failure"])
    )
    return {
        "samples": samples,
        "logical_accuracy": success_count / samples,
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
    }


def _paired_comparison(
    neural_success: BoolArray, classical_success: BoolArray
) -> dict[str, float | int | str]:
    if neural_success.shape != classical_success.shape:
        raise ValueError("Paired outcome vectors have different shapes.")
    difference = (
        neural_success.astype(np.float64) - classical_success.astype(np.float64)
    )
    samples = int(difference.size)
    gain = float(difference.mean())
    standard_error = (
        float(difference.std(ddof=1) / math.sqrt(samples)) if samples > 1 else 0.0
    )
    rescued = int(np.count_nonzero(neural_success & ~classical_success))
    harmed = int(np.count_nonzero(~neural_success & classical_success))
    if not math.isclose(
        gain, (rescued - harmed) / samples, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("Paired gain does not match rescue/harm counts.")
    return {
        "gain": gain,
        "se": standard_error,
        "ci_low": gain - 1.96 * standard_error,
        "ci_high": gain + 1.96 * standard_error,
        "ci_method": "normal_wald_on_paired_shot_differences",
        "rescued": rescued,
        "harmed": harmed,
        "discordant": rescued + harmed,
    }


def _prefixed_summary(
    prefix: str, summary: Mapping[str, float | int]
) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in summary.items() if key != "samples"}


def _build_row(
    *,
    code: Any,
    p: float,
    neural: Mapping[str, BoolArray],
    vanilla: Mapping[str, BoolArray],
    classical: Mapping[str, BoolArray],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_state: str,
    bank_path: Path,
    sample_sha256: str,
    classical_csv_path: Path,
    classical_csv_row: Mapping[str, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    neural_summary = _summarize(neural)
    vanilla_summary = _summarize(vanilla)
    classical_summary = _summarize(classical)
    paired = _paired_comparison(neural["success"], classical["success"])
    neural_vs_vanilla = _paired_comparison(
        neural["success"], vanilla["success"]
    )
    vanilla_vs_classical = _paired_comparison(
        vanilla["success"], classical["success"]
    )
    classical_ler = float(classical_summary["logical_error_rate"])
    neural_ler = float(neural_summary["logical_error_rate"])
    return {
        "code": code.name,
        "n": code.n,
        "k": code.k,
        "d": code.d,
        "channel": "depolarizing",
        "p": p,
        "samples": neural_summary["samples"],
        "neural_method": "selected_best_neural_bp4",
        "neural_sector_strategy": "joint_pauli",
        "neural_uses_xz_correlation": True,
        "classical_method": CLASSICAL_METHOD,
        "classical_sector_strategy": classical_csv_row["sector_strategy"],
        "classical_uses_xz_correlation": classical_csv_row[
            "uses_xz_correlation"
        ],
        "vanilla_method": "vanilla_bp4",
        "vanilla_sector_strategy": "joint_pauli",
        "vanilla_uses_xz_correlation": True,
        **_prefixed_summary("neural", neural_summary),
        **_prefixed_summary("classical", classical_summary),
        **_prefixed_summary("vanilla", vanilla_summary),
        "paired_reference": CLASSICAL_METHOD,
        "paired_accuracy_gain": paired["gain"],
        "paired_gain_se": paired["se"],
        "paired_ci95_low": paired["ci_low"],
        "paired_ci95_high": paired["ci_high"],
        "paired_ci_method": paired["ci_method"],
        "rescued": paired["rescued"],
        "harmed": paired["harmed"],
        "discordant": paired["discordant"],
        "neural_vs_vanilla_paired_accuracy_gain": neural_vs_vanilla["gain"],
        "neural_vs_vanilla_paired_gain_se": neural_vs_vanilla["se"],
        "neural_vs_vanilla_paired_ci95_low": neural_vs_vanilla["ci_low"],
        "neural_vs_vanilla_paired_ci95_high": neural_vs_vanilla["ci_high"],
        "neural_vs_vanilla_rescued": neural_vs_vanilla["rescued"],
        "neural_vs_vanilla_harmed": neural_vs_vanilla["harmed"],
        "neural_vs_vanilla_discordant": neural_vs_vanilla["discordant"],
        "vanilla_vs_classical_paired_accuracy_gain": vanilla_vs_classical[
            "gain"
        ],
        "vanilla_vs_classical_paired_gain_se": vanilla_vs_classical["se"],
        "vanilla_vs_classical_paired_ci95_low": vanilla_vs_classical["ci_low"],
        "vanilla_vs_classical_paired_ci95_high": vanilla_vs_classical[
            "ci_high"
        ],
        "vanilla_vs_classical_rescued": vanilla_vs_classical["rescued"],
        "vanilla_vs_classical_harmed": vanilla_vs_classical["harmed"],
        "vanilla_vs_classical_discordant": vanilla_vs_classical["discordant"],
        "relative_ler_reduction": (
            (classical_ler - neural_ler) / classical_ler
            if classical_ler > 0.0
            else float("nan")
        ),
        "classical_to_neural_ler_ratio": (
            classical_ler / neural_ler if neural_ler > 0.0 else float("inf")
        ),
        "checkpoint_path": _display_path(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_state": checkpoint_state,
        "checkpoint_format_version": checkpoint.get("format_version", ""),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "checkpoint_best_epoch": checkpoint.get("best_epoch", ""),
        "checkpoint_best_validation_accuracy": checkpoint.get("best_accuracy", ""),
        "neural_bp_iterations": config["bp_iterations"],
        "neural_residual_hidden_dim": config["bp_residual_hidden_dim"],
        "neural_parameter_sharing": config["bp_parameter_sharing"],
        "neural_residual_scale": config["bp_residual_scale"],
        "neural_max_relaxation_delta": config["bp_max_relaxation_delta"],
        "test_bank_path": _display_path(bank_path),
        "test_bank_sha256": _sha256_file(bank_path),
        "sample_sha256": sample_sha256,
        "classical_csv_path": _display_path(classical_csv_path),
        "classical_ldpc_version": classical_csv_row["ldpc_version"],
        "classical_bp_method": classical_csv_row["bp_method"],
        "classical_bp_iterations": classical_csv_row["bp_iterations"],
        "classical_ms_scaling_factor": classical_csv_row["ms_scaling_factor"],
        "classical_schedule": classical_csv_row["schedule"],
        "evaluation_device": str(device),
        "evaluation_batch_size": batch_size,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "analysis_script_path": _display_path(Path(__file__)),
        "analysis_script_sha256": _sha256_file(Path(__file__)),
    }


def _classical_outcomes(bank: Mapping[str, NDArray[Any]]) -> dict[str, BoolArray]:
    return {
        name: np.asarray(bank[f"{CLASSICAL_METHOD}__{name}"], dtype=np.bool_)
        for name in (
            "success",
            "syndrome_converged",
            "flagged_failure",
            "unflagged_logical_failure",
        )
    }


def _classical_csv_for_bank(bank_path: Path) -> Path:
    suffix = "_test_bank.npz"
    if not bank_path.name.endswith(suffix):
        raise ValueError(f"Unexpected test-bank filename: {bank_path}")
    return bank_path.with_name(bank_path.name[: -len(suffix)] + "_classical.csv")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {name}")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS device requested but unavailable.")
    return device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code",
        nargs="+",
        choices=("bb72", "bb144"),
        default=("bb72", "bb144"),
    )
    parser.add_argument("--p", nargs="+", type=float, default=DEFAULT_P)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu; use auto only when device variance is acceptable).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="CPU intra-op threads; 0 leaves the Torch default unchanged.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        metavar="CODE=PATH",
        help="Override one archived selected-best checkpoint root.",
    )
    parser.add_argument(
        "--bank-dir",
        action="append",
        metavar="CODE=PATH",
        help="Override one classical test-bank root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/bb_neural_vs_classical_paired.csv"),
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.torch_threads < 0:
        parser.error("--torch-threads must be non-negative.")
    if len(set(args.code)) != len(args.code):
        parser.error("--code contains a duplicate.")
    if len(set(args.p)) != len(args.p):
        parser.error("--p contains a duplicate.")
    if any(not 0.0 < p < 0.75 for p in args.p):
        parser.error("Every --p must lie in (0, 0.75).")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checkpoint_roots = _parse_code_paths(
            args.checkpoint_dir,
            DEFAULT_CHECKPOINT_DIRS,
            option="--checkpoint-dir",
        )
        bank_roots = _parse_code_paths(
            args.bank_dir, DEFAULT_BANK_DIRS, option="--bank-dir"
        )
        device = _resolve_device(args.device)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)

    checkpoint_index = _checkpoint_index(checkpoint_roots, args.code)
    bank_index = _bank_index(bank_roots, args.code)
    rows: list[dict[str, Any]] = []
    for code_name in args.code:
        code = BBCodeSpec.from_name(code_name)
        for p in args.p:
            checkpoint_key = _find_key(
                checkpoint_index, code_name, p, "selected-best checkpoint"
            )
            bank_key = _find_key(bank_index, code_name, p, "classical test bank")
            checkpoint_path, discovered_config = checkpoint_index[checkpoint_key]
            bank_path, metadata = bank_index[bank_key]
            started = time.perf_counter()
            bank = _load_bank(bank_path, metadata=metadata, code=code, p=p)
            sample_sha256 = _sample_sha256(bank["x_error"], bank["z_error"])
            classical = _classical_outcomes(bank)
            classical_csv_path = _classical_csv_for_bank(bank_path)
            classical_csv_row = _load_classical_csv_row(
                classical_csv_path,
                code=code_name,
                p=p,
                samples=int(bank["x_error"].shape[0]),
                sample_sha256=sample_sha256,
                success=classical["success"],
            )

            checkpoint = _torch_load(checkpoint_path)
            config = checkpoint.get("experiment_config")
            if config != discovered_config:
                raise ValueError(f"Checkpoint config changed while running: {checkpoint_path}")
            model, config, checkpoint_state = _make_model(
                checkpoint, code, p, device
            )
            neural, vanilla = _evaluate_neural_and_vanilla(
                model,
                bank=bank,
                code=code,
                p=p,
                batch_size=args.batch_size,
                device=device,
            )
            row = _build_row(
                code=code,
                p=p,
                neural=neural,
                vanilla=vanilla,
                classical=classical,
                checkpoint=checkpoint,
                config=config,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
                bank_path=bank_path,
                sample_sha256=sample_sha256,
                classical_csv_path=classical_csv_path,
                classical_csv_row=classical_csv_row,
                device=device,
                batch_size=args.batch_size,
            )
            rows.append(row)
            elapsed = time.perf_counter() - started
            print(
                f"{code_name} p={p:g}: neural LER="
                f"{row['neural_logical_error_rate']:.8f}, "
                f"vanilla BP4 LER={row['vanilla_logical_error_rate']:.8f}, "
                f"{CLASSICAL_METHOD} LER="
                f"{row['classical_logical_error_rate']:.8f}, paired gain="
                f"{row['paired_accuracy_gain']:.8f}, rescued/harmed="
                f"{row['rescued']}/{row['harmed']} ({elapsed:.1f}s)",
                flush=True,
            )
            del model, checkpoint, neural, vanilla, classical, bank

    expected_rows = len(args.code) * len(args.p)
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(rows)}.")
    rows.sort(key=lambda row: (int(row["n"]), float(row["p"])))
    output = args.output.expanduser()
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    _write_csv(output, rows)
    print(f"Wrote {len(rows)} paired rows to {_display_path(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
