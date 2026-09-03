#!/usr/bin/env python3
"""Extract and plot completed circuit-level BB Neural-BP experiments.

The authoritative result is ``history.json`` in each timestamped output
directory.  Top-level Slurm exit codes are recorded as provenance, but they do
not replace the per-run history and selected-best final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import shlex
import statistics
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY / "results" / "bb" / "circuit"
ANALYSIS_ROOT = REPOSITORY / "results" / "analysis"
PLOT_ROOT = REPOSITORY / "results" / "plots"
CSV_PATH = ANALYSIS_ROOT / "bb_circuit_campaign_2026_08.csv"
REPORT_PATH = ANALYSIS_ROOT / "bb_circuit_campaign_2026_08.md"
PLOT_PATH = PLOT_ROOT / "bb_circuit_campaign_2026_08.png"
ABLATION_PLOT_PATH = PLOT_ROOT / "bb_circuit_campaign_2026_08_ablations.png"
PARTIAL_CSV_PATH = ANALYSIS_ROOT / "bb_circuit_campaign_2026_08_partial.csv"
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class CircuitResult:
    job_id: str
    experiment_index: int | None
    purpose: str
    variant: str
    scientific_status: str
    launcher_exit_code: int | None
    code: str
    n: int
    k: int
    d: int
    circuit_schema_version: int
    circuit_noise_model: str
    gate_error_rate: float
    measurement_error_rate: float
    idle_error_rate: float
    rounds: int
    detector_frames: int
    num_detectors: int
    num_mechanisms: int
    num_edges: int
    num_orbits: int
    iterations: int
    hidden_dim: int
    orbit_embedding_dim: int
    sharing: str
    normalisation: float
    residual_scale: float
    relaxation_delta: float
    deep_supervision_weight: float
    syndrome_loss_weight: float
    logical_loss_weight: float
    mechanism_loss_weight: float
    learning_rate: float
    trainable_parameters: int
    seed: int
    requested_epochs: int
    completed_epochs: int
    batch_size: int | None
    batches_per_epoch: int | None
    training_shots_per_epoch: int | None
    eval_batches: int | None
    final_eval_batches: int | None
    best_epoch: int
    selection_metric: str
    first_train_loss: float
    last_train_loss: float
    minimum_train_loss: float
    minimum_train_loss_epoch: int
    maximum_validation_neural_accuracy: float
    maximum_validation_neural_accuracy_epoch: int
    mean_epoch_seconds: float | None
    wall_minutes: float | None
    final_shots: int
    neural_accuracy: float
    neural_logical_error_rate: float
    vanilla_accuracy: float
    vanilla_logical_error_rate: float
    neural_syndrome_convergence: float
    neural_flagged_failure_rate: float
    neural_unflagged_logical_failure_rate: float
    raw_paired_gain: float
    raw_paired_gain_ci95_halfwidth: float
    raw_paired_ci95_low: float
    raw_paired_ci95_high: float
    raw_rescued: int
    raw_harmed: int
    osd_method: str
    osd_order: int
    osd_shots: int
    neural_osd_accuracy: float | None
    neural_osd_logical_error_rate: float | None
    vanilla_osd_accuracy: float | None
    vanilla_osd_logical_error_rate: float | None
    osd_paired_gain: float | None
    osd_paired_gain_ci95_halfwidth: float | None
    osd_paired_ci95_low: float | None
    osd_paired_ci95_high: float | None
    osd_relative_ler_reduction: float | None
    source_resdir: str
    output_directory: str
    latest_checkpoint: str
    selected_checkpoint: str


@dataclass(frozen=True)
class PartialCircuitResult:
    """A scientifically incomplete run retained for provenance and resume."""

    job_id: str
    experiment_index: int | None
    purpose: str
    variant: str
    code: str
    circuit_noise_model: str
    gate_error_rate: float
    measurement_error_rate: float
    idle_error_rate: float
    sharing: str
    iterations: int
    seed: int
    requested_epochs: int
    completed_epochs: int
    remaining_epochs: int
    best_epoch: int
    validation_evaluations: int
    best_validation_osd_gain: float | None
    has_latest_checkpoint: bool
    has_selected_checkpoint: bool
    stop_reason: str
    source_resdir: str
    output_directory: str


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY.resolve()))


def option_map(command: str) -> dict[str, str | bool]:
    tokens = shlex.split(command.strip())
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            key, value = token[2:].split("=", 1)
            options[key] = value
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token[2:]] = tokens[index + 1]
            index += 2
        else:
            options[token[2:]] = True
            index += 1
    return options


def scalar_option(
    options: dict[str, str | bool], name: str, converter: type[int] | type[float]
) -> int | float | None:
    value = options.get(name)
    if value is None or value is True:
        return None
    return converter(value)


def enclosing_resdir(path: Path) -> Path:
    for parent in path.parents:
        if re.fullmatch(r"resdir_\d+", parent.name):
            return parent
    raise ValueError(f"No enclosing resdir_<id> for {path}")


def command_matches(options: dict[str, str | bool], config: dict[str, Any]) -> bool:
    profile = str(config.get("circuit_noise_model", "legacy"))
    command_profile = str(options.get("bb_circuit_noise_model", "legacy"))
    if command_profile != profile:
        return False
    string_keys = {
        "code": "code",
        "bp_parameter_sharing": "bp_parameter_sharing",
    }
    integer_keys = {
        "seed": "seed",
        "rounds": "rounds",
        "bp_iterations": "bp_iterations",
        "bp_residual_hidden_dim": "bp_residual_hidden_dim",
        "bp_orbit_embedding_dim": "bp_orbit_embedding_dim",
    }
    float_keys = {
        "p": "gate_error_rate",
        "measurement_error_rate": "measurement_error_rate",
        "bb_idle_error_rate": "idle_error_rate",
        "bp_normalisation": "bp_normalisation",
        "bp_residual_scale": "bp_residual_scale",
        "bp_max_relaxation_delta": "bp_max_relaxation_delta",
        "bp_deep_supervision_weight": "bp_deep_supervision_weight",
        "bb_syndrome_loss_weight": "bb_syndrome_loss_weight",
        "bb_logical_loss_weight": "bb_logical_loss_weight",
        "bb_pauli_loss_weight": "bb_pauli_loss_weight",
    }
    for option, key in string_keys.items():
        if str(options.get(option)) != str(config[key]):
            return False
    for option, key in integer_keys.items():
        value = scalar_option(options, option, int)
        if value is None or int(value) != int(config[key]):
            return False
    for option, key in float_keys.items():
        value = scalar_option(options, option, float)
        # Paper profiles derive q and idle from p, so their runner deliberately
        # omits these two override flags.
        if (
            value is None
            and profile != "legacy"
            and option
            in {
                "measurement_error_rate",
                "bb_idle_error_rate",
            }
        ):
            continue
        if value is None or not math.isclose(float(value), float(config[key])):
            return False
    return True


def find_command(
    resdir: Path, config: dict[str, Any]
) -> tuple[int | None, dict[str, str | bool]]:
    matches: list[tuple[int, dict[str, str | bool]]] = []
    for path in sorted(resdir.glob("command_exp_*.txt")):
        match = re.fullmatch(r"command_exp_(\d+)\.txt", path.name)
        if match is None:
            continue
        options = option_map(path.read_text(encoding="utf-8"))
        if command_matches(options, config):
            matches.append((int(match.group(1)), options))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple commands match {config['code']} seed={config['seed']}"
        )
    return None, {}


def read_optional_integer(path: Path) -> int | None:
    if not path.is_file():
        return None
    return int(path.read_text(encoding="utf-8").strip())


def classify_run(config: dict[str, Any], resdir: Path) -> tuple[str, str]:
    """Classify a run by scientific purpose, independently of Slurm job id."""

    profile = str(config.get("circuit_noise_model", "legacy"))
    p = float(config["gate_error_rate"])
    q = float(config["measurement_error_rate"])
    idle = float(config["idle_error_rate"])
    sharing = str(config["bp_parameter_sharing"])
    iterations = int(config["bp_iterations"])
    residual = float(config["bp_residual_scale"])
    relaxation = float(config["bp_max_relaxation_delta"])
    deep = float(config["bp_deep_supervision_weight"])
    mechanism = float(config["bb_pauli_loss_weight"])

    if profile != "legacy":
        return "noise_profile", profile
    if not math.isclose(q, p) or not math.isclose(idle, 0.0):
        if math.isclose(idle, p) and math.isclose(q, p):
            variant = "idle=p"
        elif math.isclose(idle, 0.0) and math.isclose(q, 2.0 * p):
            variant = "q=2p"
        else:
            variant = f"q={q:g},idle={idle:g}"
        return "noise_balance", variant
    if sharing != "orbit":
        return "sharing", sharing
    if iterations != 12:
        return "iterations", f"T={iterations}"
    if math.isclose(residual, 0.0) and not math.isclose(relaxation, 0.0):
        return "mechanism", "relaxation_only"
    if math.isclose(relaxation, 0.0) and not math.isclose(residual, 0.0):
        return "mechanism", "residual_only"
    if math.isclose(residual, 0.0) and math.isclose(relaxation, 0.0):
        return "mechanism", "vanilla_only"
    if math.isclose(mechanism, 0.0):
        return "loss_auxiliary", "no_mechanism_bce"
    if math.isclose(deep, 0.0):
        return "loss_auxiliary", "no_deep_supervision"
    if "replicates" in resdir.parts:
        return "replicate", "seed_replicate"
    return "baseline", "reference"


def parse_epoch_times(log_path: Path) -> list[float]:
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [
        float(value)
        for value in re.findall(r"^\[Train\].*\| Time: ([0-9.]+)s$", text, re.M)
    ]


def wall_minutes(
    output_directory: Path, resdir: Path, experiment_index: int | None
) -> float | None:
    if experiment_index is None:
        return None
    start_match = re.match(r"(\d{2}-\d{2}-\d{2}-\d{6})_", output_directory.name)
    finish_path = resdir / f"finished_exp_{experiment_index}.txt"
    if start_match is None or not finish_path.is_file():
        return None
    started = datetime.datetime.strptime(
        f"{output_directory.parent.name} {start_match.group(1)}",
        "%Y-%m-%d %H-%M-%S-%f",
    )
    finish_match = re.search(
        r"^Finished: (.+)$", finish_path.read_text(encoding="utf-8"), re.M
    )
    if finish_match is None:
        return None
    finished = datetime.datetime.fromisoformat(finish_match.group(1)).replace(
        tzinfo=None
    )
    return (finished - started).total_seconds() / 60.0


def require_final_float(final: dict[str, Any], key: str) -> float:
    value = final.get(key)
    if value is None:
        raise ValueError(f"Completed circuit result is missing final.{key}")
    return float(value)


def parse_history(path: Path) -> CircuitResult:
    history = json.loads(path.read_text(encoding="utf-8"))
    config = history.get("config")
    final = history.get("final")
    train = history.get("train", [])
    evaluations = history.get("eval", [])
    if (
        not isinstance(config, dict)
        or config.get("architecture") != "bb_neural_bp_circuit"
    ):
        raise ValueError(f"Not a circuit-level BB Neural-BP history: {path}")
    if not isinstance(final, dict):
        raise ValueError(
            f"Circuit result has no selected-best final evaluation: {path}"
        )
    if not train or not evaluations:
        raise ValueError(f"Circuit result has incomplete train/eval history: {path}")
    if any(
        not math.isfinite(float(row[key]))
        for row in train
        for key in ("total", "syndrome", "logical", "mechanism")
    ):
        raise ValueError(f"Circuit training history contains a non-finite loss: {path}")

    resdir = enclosing_resdir(path)
    purpose, variant = classify_run(config, resdir)
    job_match = re.fullmatch(r"resdir_(\d+)", resdir.name)
    assert job_match is not None
    job_id = job_match.group(1)
    experiment_index, options = find_command(resdir, config)
    launcher_exit = (
        read_optional_integer(resdir / f"exit_code_exp_{experiment_index}.txt")
        if experiment_index is not None
        else None
    )

    requested_epochs = sum(
        int(phase.get("epochs", 0)) for phase in history.get("phases", [])
    )
    completed_epochs = len(train)
    status = "complete"
    if requested_epochs != completed_epochs or launcher_exit not in (None, 0):
        status = "partial"

    batch_size = scalar_option(options, "batch_size", int)
    batches_per_epoch = scalar_option(options, "batches", int)
    training_shots = (
        int(batch_size) * int(batches_per_epoch)
        if batch_size is not None and batches_per_epoch is not None
        else None
    )
    eval_batches = scalar_option(options, "eval_batches", int)
    final_eval_batches = scalar_option(options, "final_eval_batches", int)
    epoch_times = parse_epoch_times(path.with_name("training_log.txt"))
    minimum_loss_row = min(train, key=lambda row: float(row["total"]))
    maximum_validation_row = max(
        evaluations, key=lambda row: float(row["neural_accuracy"])
    )

    shots = int(final["shots"])
    neural_accuracy = require_final_float(final, "neural_accuracy")
    vanilla_accuracy = require_final_float(final, "vanilla_accuracy")
    raw_gain = require_final_float(final, "paired_gain")
    raw_halfwidth = require_final_float(final, "paired_gain_error")
    osd_shots = int(final.get("osd_shots", 0))
    if osd_shots > 0:
        neural_osd_accuracy = require_final_float(final, "neural_osd_accuracy")
        vanilla_osd_accuracy = require_final_float(final, "vanilla_osd_accuracy")
        osd_gain = require_final_float(final, "osd_paired_gain")
        osd_halfwidth = require_final_float(final, "osd_paired_gain_error")
        vanilla_osd_ler = 1.0 - vanilla_osd_accuracy
        neural_osd_ler = 1.0 - neural_osd_accuracy
    else:
        neural_osd_accuracy = None
        vanilla_osd_accuracy = None
        osd_gain = None
        osd_halfwidth = None
        vanilla_osd_ler = None
        neural_osd_ler = None
    relative_reduction = (
        (vanilla_osd_ler - neural_osd_ler) / vanilla_osd_ler
        if vanilla_osd_ler is not None
        and neural_osd_ler is not None
        and vanilla_osd_ler > 0.0
        else None
    )

    latest_checkpoint = path.with_name("model.pt")
    selected_checkpoint = path.with_name("best_model.pt")
    for checkpoint in (latest_checkpoint, selected_checkpoint):
        if not checkpoint.is_file():
            raise ValueError(f"Missing checkpoint: {checkpoint}")

    return CircuitResult(
        job_id=job_id,
        experiment_index=experiment_index,
        purpose=purpose,
        variant=variant,
        scientific_status=status,
        launcher_exit_code=launcher_exit,
        code=str(config["code"]),
        n=int(config["n"]),
        k=int(config["k"]),
        d=int(config["d"]),
        circuit_schema_version=int(config["circuit_schema_version"]),
        circuit_noise_model=str(config.get("circuit_noise_model", "legacy")),
        gate_error_rate=float(config["gate_error_rate"]),
        measurement_error_rate=float(config["measurement_error_rate"]),
        idle_error_rate=float(config["idle_error_rate"]),
        rounds=int(config["rounds"]),
        detector_frames=int(config["detector_frames"]),
        num_detectors=int(config["num_detectors"]),
        num_mechanisms=int(config["num_mechanisms"]),
        num_edges=int(config["num_edges"]),
        num_orbits=int(config["num_orbits"]),
        iterations=int(config["bp_iterations"]),
        hidden_dim=int(config["bp_residual_hidden_dim"]),
        orbit_embedding_dim=int(config["bp_orbit_embedding_dim"]),
        sharing=str(config["bp_parameter_sharing"]),
        normalisation=float(config["bp_normalisation"]),
        residual_scale=float(config["bp_residual_scale"]),
        relaxation_delta=float(config["bp_max_relaxation_delta"]),
        deep_supervision_weight=float(config["bp_deep_supervision_weight"]),
        syndrome_loss_weight=float(config["bb_syndrome_loss_weight"]),
        logical_loss_weight=float(config["bb_logical_loss_weight"]),
        mechanism_loss_weight=float(config["bb_pauli_loss_weight"]),
        learning_rate=float(config["learning_rate"]),
        trainable_parameters=int(config["trainable_parameters"]),
        seed=int(config["seed"]),
        requested_epochs=requested_epochs,
        completed_epochs=completed_epochs,
        batch_size=int(batch_size) if batch_size is not None else None,
        batches_per_epoch=(
            int(batches_per_epoch) if batches_per_epoch is not None else None
        ),
        training_shots_per_epoch=training_shots,
        eval_batches=int(eval_batches) if eval_batches is not None else None,
        final_eval_batches=(
            int(final_eval_batches) if final_eval_batches is not None else None
        ),
        best_epoch=int(history["best_epoch"]),
        selection_metric=str(history["best_selection_metric"]),
        first_train_loss=float(train[0]["total"]),
        last_train_loss=float(train[-1]["total"]),
        minimum_train_loss=float(minimum_loss_row["total"]),
        minimum_train_loss_epoch=int(minimum_loss_row["epoch"]),
        maximum_validation_neural_accuracy=float(
            maximum_validation_row["neural_accuracy"]
        ),
        maximum_validation_neural_accuracy_epoch=int(maximum_validation_row["epoch"]),
        mean_epoch_seconds=(
            sum(epoch_times) / len(epoch_times) if epoch_times else None
        ),
        wall_minutes=wall_minutes(path.parent, resdir, experiment_index),
        final_shots=shots,
        neural_accuracy=neural_accuracy,
        neural_logical_error_rate=1.0 - neural_accuracy,
        vanilla_accuracy=vanilla_accuracy,
        vanilla_logical_error_rate=1.0 - vanilla_accuracy,
        neural_syndrome_convergence=require_final_float(final, "neural_converged"),
        neural_flagged_failure_rate=require_final_float(final, "neural_flagged"),
        neural_unflagged_logical_failure_rate=require_final_float(
            final, "neural_unflagged"
        ),
        raw_paired_gain=raw_gain,
        raw_paired_gain_ci95_halfwidth=raw_halfwidth,
        raw_paired_ci95_low=raw_gain - raw_halfwidth,
        raw_paired_ci95_high=raw_gain + raw_halfwidth,
        raw_rescued=int(final["rescued"]),
        raw_harmed=int(final["harmed"]),
        osd_method=str(config["bb_osd_method"]),
        osd_order=int(config["bb_osd_order"]),
        osd_shots=osd_shots,
        neural_osd_accuracy=neural_osd_accuracy,
        neural_osd_logical_error_rate=neural_osd_ler,
        vanilla_osd_accuracy=vanilla_osd_accuracy,
        vanilla_osd_logical_error_rate=vanilla_osd_ler,
        osd_paired_gain=osd_gain,
        osd_paired_gain_ci95_halfwidth=osd_halfwidth,
        osd_paired_ci95_low=(
            osd_gain - osd_halfwidth
            if osd_gain is not None and osd_halfwidth is not None
            else None
        ),
        osd_paired_ci95_high=(
            osd_gain + osd_halfwidth
            if osd_gain is not None and osd_halfwidth is not None
            else None
        ),
        osd_relative_ler_reduction=relative_reduction,
        source_resdir=relative(resdir),
        output_directory=relative(path.parent),
        latest_checkpoint=relative(latest_checkpoint),
        selected_checkpoint=relative(selected_checkpoint),
    )


def parse_partial_history(path: Path, history: dict[str, Any]) -> PartialCircuitResult:
    config = history["config"]
    train = history.get("train", [])
    evaluations = history.get("eval", [])
    resdir = enclosing_resdir(path)
    purpose, variant = classify_run(config, resdir)
    job_match = re.fullmatch(r"resdir_(\d+)", resdir.name)
    assert job_match is not None
    experiment_index, _ = find_command(resdir, config)
    requested_epochs = sum(
        int(phase.get("epochs", 0)) for phase in history.get("phases", [])
    )
    completed_epochs = len(train)
    validation_gains = [
        float(row["osd_paired_gain"])
        for row in evaluations
        if row.get("osd_paired_gain") is not None
    ]
    stop_reason = (
        "Slurm time limit"
        if (resdir / "interrupted.txt").is_file()
        else "missing selected-best final evaluation"
    )
    return PartialCircuitResult(
        job_id=job_match.group(1),
        experiment_index=experiment_index,
        purpose=purpose,
        variant=variant,
        code=str(config["code"]),
        circuit_noise_model=str(config.get("circuit_noise_model", "legacy")),
        gate_error_rate=float(config["gate_error_rate"]),
        measurement_error_rate=float(config["measurement_error_rate"]),
        idle_error_rate=float(config["idle_error_rate"]),
        sharing=str(config["bp_parameter_sharing"]),
        iterations=int(config["bp_iterations"]),
        seed=int(config["seed"]),
        requested_epochs=requested_epochs,
        completed_epochs=completed_epochs,
        remaining_epochs=max(0, requested_epochs - completed_epochs),
        best_epoch=int(history.get("best_epoch", -1)),
        validation_evaluations=len(evaluations),
        best_validation_osd_gain=max(validation_gains) if validation_gains else None,
        has_latest_checkpoint=path.with_name("model.pt").is_file(),
        has_selected_checkpoint=path.with_name("best_model.pt").is_file(),
        stop_reason=stop_reason,
        source_resdir=relative(resdir),
        output_directory=relative(path.parent),
    )


def collect_results() -> tuple[list[CircuitResult], list[PartialCircuitResult]]:
    histories = sorted(RESULTS_ROOT.rglob("history.json"))
    rows: list[CircuitResult] = []
    partial_rows: list[PartialCircuitResult] = []
    for history in histories:
        try:
            data = json.loads(history.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {history}: {error}") from error
        if data.get("config", {}).get("architecture") != "bb_neural_bp_circuit":
            continue
        requested_epochs = sum(
            int(phase.get("epochs", 0)) for phase in data.get("phases", [])
        )
        complete = (
            isinstance(data.get("final"), dict)
            and bool(data.get("train"))
            and bool(data.get("eval"))
            and len(data.get("train", [])) == requested_epochs
        )
        if complete:
            rows.append(parse_history(history))
        else:
            partial_rows.append(parse_partial_history(history, data))
    if not rows and not partial_rows:
        raise ValueError(f"No circuit-level BB histories found under {RESULTS_ROOT}")
    complete_sorted = sorted(
        rows,
        key=lambda row: (
            row.n,
            row.circuit_noise_model,
            row.gate_error_rate,
            row.measurement_error_rate,
            row.idle_error_rate,
            row.seed,
        ),
    )

    partial_sorted = sorted(
        partial_rows,
        key=lambda row: (
            int(row.code[2:]),
            row.circuit_noise_model,
            row.gate_error_rate,
            row.measurement_error_rate,
            row.idle_error_rate,
            row.seed,
        ),
    )
    return complete_sorted, partial_sorted


def is_default_sweep(row: CircuitResult) -> bool:
    return (
        row.scientific_status == "complete"
        and row.circuit_schema_version == 2
        and row.circuit_noise_model == "legacy"
        and math.isclose(row.measurement_error_rate, row.gate_error_rate)
        and math.isclose(row.idle_error_rate, 0.0)
        and row.sharing == "orbit"
        and row.iterations == 12
        and row.hidden_dim == 32
        and row.orbit_embedding_dim == 8
        and math.isclose(row.normalisation, 0.625)
        and math.isclose(row.residual_scale, 2.0)
        and math.isclose(row.relaxation_delta, 0.5)
        and math.isclose(row.deep_supervision_weight, 0.2)
        and math.isclose(row.syndrome_loss_weight, 1.0)
        and math.isclose(row.logical_loss_weight, 1.0)
        and math.isclose(row.mechanism_loss_weight, 0.1)
        and row.osd_shots > 0
        and row.osd_method == "OSD_0"
        and row.osd_order == 0
    )


def is_primary_sweep(row: CircuitResult) -> bool:
    """Return true only for the canonical seed-A reference curve."""

    return is_default_sweep(row) and row.purpose == "baseline"


def write_csv(rows: list[CircuitResult]) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    names = list(asdict(rows[0]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_partial_csv(rows: list[PartialCircuitResult]) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    names = (
        list(asdict(rows[0]))
        if rows
        else list(PartialCircuitResult.__dataclass_fields__)
    )
    with PARTIAL_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def percent(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def write_report(
    rows: list[CircuitResult], partial_rows: list[PartialCircuitResult]
) -> None:
    primary = sorted(
        (row for row in rows if is_primary_sweep(row)),
        key=lambda row: (row.n, row.gate_error_rate),
    )
    if not primary:
        raise ValueError("No complete primary circuit sweep is available for report.")

    def gain_pp(row: CircuitResult) -> float:
        assert row.osd_paired_gain is not None
        return 100.0 * row.osd_paired_gain

    def gain_ci_pp(row: CircuitResult) -> float:
        assert row.osd_paired_gain_ci95_halfwidth is not None
        return 100.0 * row.osd_paired_gain_ci95_halfwidth

    def add_result_table(
        lines: list[str],
        title: str,
        table_rows: list[CircuitResult],
        label,
    ) -> None:
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Code | p | Variant | Neural+OSD LER | BP+OSD LER | Accuracy gain (paired 95% half-width) |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in table_rows:
            lines.append(
                "| "
                f"{row.code.upper()} | {row.gate_error_rate:.3f} | {label(row)} | "
                f"{percent(float(row.neural_osd_logical_error_rate))} | "
                f"{percent(float(row.vanilla_osd_logical_error_rate))} | "
                f"{gain_pp(row):+.3f} ± {gain_ci_pp(row):.3f} pp |"
            )

    reference_by_key = {(row.code, row.gate_error_rate): row for row in primary}
    replicate_rows = [
        row
        for row in rows
        if is_default_sweep(row)
        and row.purpose in {"baseline", "replicate"}
        and math.isclose(row.gate_error_rate, 0.004)
    ]
    replicate_groups: dict[str, list[CircuitResult]] = defaultdict(list)
    for row in replicate_rows:
        replicate_groups[row.code].append(row)

    sharing_rows: list[CircuitResult] = []
    for row in rows:
        if row.purpose != "sharing":
            continue
        reference = reference_by_key.get((row.code, row.gate_error_rate))
        if reference is not None and reference not in sharing_rows:
            sharing_rows.append(reference)
        sharing_rows.append(row)
    sharing_rows.sort(
        key=lambda row: (
            row.n,
            row.gate_error_rate,
            0 if row.purpose == "baseline" else 1,
        )
    )

    p004_reference = [
        row for row in primary if math.isclose(row.gate_error_rate, 0.004)
    ]
    iteration_rows = sorted(
        p004_reference + [row for row in rows if row.purpose == "iterations"],
        key=lambda row: (row.n, row.iterations),
    )
    mechanism_rows = sorted(
        p004_reference + [row for row in rows if row.purpose == "mechanism"],
        key=lambda row: (row.n, 0 if row.purpose == "baseline" else 1, row.variant),
    )
    loss_rows = sorted(
        p004_reference + [row for row in rows if row.purpose == "loss_auxiliary"],
        key=lambda row: (row.n, 0 if row.purpose == "baseline" else 1, row.variant),
    )
    p003_reference = [
        row for row in primary if math.isclose(row.gate_error_rate, 0.003)
    ]
    noise_rows = sorted(
        p003_reference + [row for row in rows if row.purpose == "noise_balance"],
        key=lambda row: (
            row.n,
            row.measurement_error_rate,
            row.idle_error_rate,
        ),
    )

    lines = [
        "# BB circuit-level Neural BP campaign (August–September 2026)",
        "",
        "This report is generated from the archived `history.json` files under ",
        "`results/bb/circuit/`. A run is counted as complete only when all 100 ",
        "training epochs and the fresh selected-best final evaluation are present. ",
        "Incomplete time-limited runs are retained for resume but are not used as ",
        "final performance points.",
        "",
        "## Inventory",
        "",
        f"- Histories: **{len(rows) + len(partial_rows)}** = **{len(rows)} complete** + **{len(partial_rows)} partial**",
        f"- Complete selected-best evaluations: **{sum(row.osd_shots == 4096 for row in rows)}/{len(rows)}** use 4,096 exact Stim shots",
        "- Primary reference: circuit schema v2, `q=p`, idle error 0, orbit sharing, T=12, h=32, embedding=8, normalized min-sum scale 0.625",
        "- Comparison: Neural-BP posterior + OSD-0 versus vanilla-BP posterior + the same OSD-0 post-processor",
        "",
        "## Primary reference sweep",
        "",
        "| Code | p | Best epoch | Neural+OSD LER | BP+OSD LER | Accuracy gain (paired 95% half-width) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            "| "
            f"{row.code.upper()} | {row.gate_error_rate:.3f} | {row.best_epoch} | "
            f"{percent(float(row.neural_osd_logical_error_rate))} | "
            f"{percent(float(row.vanilla_osd_logical_error_rate))} | "
            f"{gain_pp(row):+.3f} ± {gain_ci_pp(row):.3f} pp |"
        )

    lines.extend(
        [
            "",
            "## p=0.004 training-seed replication",
            "",
            "The reference seed and two new replicate seeds are pooled below. The ± value is the sample standard deviation across trained models, not a confidence interval.",
            "",
            "| Code | Seeds | Mean Neural+OSD accuracy | Mean BP+OSD accuracy | Mean gain |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for code in sorted(replicate_groups, key=lambda value: int(value[2:])):
        group = replicate_groups[code]
        neural = [float(row.neural_osd_accuracy) for row in group]
        vanilla = [float(row.vanilla_osd_accuracy) for row in group]
        gains = [float(row.osd_paired_gain) for row in group]
        lines.append(
            f"| {code.upper()} | {len(group)} | "
            f"{percent(statistics.mean(neural))} ± {100.0 * statistics.stdev(neural):.3f} pp | "
            f"{percent(statistics.mean(vanilla))} ± {100.0 * statistics.stdev(vanilla):.3f} pp | "
            f"{100.0 * statistics.mean(gains):+.3f} ± {100.0 * statistics.stdev(gains):.3f} pp |"
        )

    lines.extend(["", "## Controlled ablations"])
    add_result_table(
        lines,
        "Parameter sharing",
        sharing_rows,
        lambda row: f"{row.sharing} ({row.trainable_parameters:,} params)",
    )
    add_result_table(
        lines,
        "BP iterations at p=0.004",
        iteration_rows,
        lambda row: f"T={row.iterations}",
    )
    add_result_table(
        lines,
        "Learned update components at p=0.004",
        mechanism_rows,
        lambda row: "full" if row.purpose == "baseline" else row.variant,
    )
    add_result_table(
        lines,
        "Auxiliary losses at p=0.004",
        loss_rows,
        lambda row: "full" if row.purpose == "baseline" else row.variant,
    )
    add_result_table(
        lines,
        "Noise-balance controls at p=0.003",
        noise_rows,
        lambda row: "q=p, idle=0" if row.purpose == "baseline" else row.variant,
    )

    lines.extend(
        [
            "",
            "## Partial runs (not final results)",
            "",
            "| Job/exp | Code | p | Variant | Epochs | Best validation OSD gain | Checkpoints | Stop reason |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in partial_rows:
        validation = (
            f"{100.0 * row.best_validation_osd_gain:+.3f} pp"
            if row.best_validation_osd_gain is not None
            else "n/a"
        )
        checkpoints = (
            f"latest={'yes' if row.has_latest_checkpoint else 'no'}, "
            f"best={'yes' if row.has_selected_checkpoint else 'no'}"
        )
        lines.append(
            f"| {row.job_id}/{row.experiment_index} | {row.code.upper()} | "
            f"{row.gate_error_rate:.3f} | {row.variant} | "
            f"{row.completed_epochs}/{row.requested_epochs} | {validation} | "
            f"{checkpoints} | {row.stop_reason} |"
        )

    significant = [row for row in primary if float(row.osd_paired_ci95_low) > 0.0]
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "- The primary paired interval is strictly positive at "
        + ", ".join(
            f"{row.code.upper()} p={row.gate_error_rate:.3f}" for row in significant
        )
        + ". Low-p saturation points remain under-resolved with 4,096 shots."
    )
    for code in sorted(replicate_groups, key=lambda value: int(value[2:])):
        group = replicate_groups[code]
        gains = [100.0 * float(row.osd_paired_gain) for row in group]
        neural_lers = [
            100.0 * float(row.neural_osd_logical_error_rate) for row in group
        ]
        lines.append(
            f"- At p=0.004, {code.upper()} across three seeds has mean Neural+OSD LER "
            f"**{statistics.mean(neural_lers):.3f}%** and mean accuracy gain "
            f"**{statistics.mean(gains):+.3f} pp** (seed SD {statistics.stdev(gains):.3f} pp)."
        )
    lines.extend(
        [
            "- Global sharing is competitive but not consistently better than orbit sharing. It uses far fewer trainable parameters (618 versus 13,299), so it remains a useful efficiency baseline.",
            "- T=24 provides no benefit here and hurts BB72. T=6 is strongest for the single BB144 p=0.004 seed, while T=12 is strongest for the corresponding BB72 seed; this needs replication before choosing T globally.",
            "- The learned residual is the essential update: relaxation-only is approximately tied with vanilla OSD. Relaxation helps BB72 beyond residual-only, while BB144 residual-only is similar to the full update in this seed.",
            "- Removing mechanism BCE or deep supervision changes the two code sizes in opposite ways: the full loss wins on BB72, while the reduced losses win on BB144. Treat this as a seed-sensitive result, not a final loss choice.",
            "- All completed noise-balance controls retain a positive paired gain. Idle noise especially degrades the BB144 vanilla posterior at this point, but the claim is based on one seed and 4,096 shots.",
            "- The four high-p BB144 reference runs and BB144 global p=0.005 did not finish within 24 hours. Their validation numbers are resume diagnostics only; no selected-best final evaluation exists.",
            "- This campaign is not a threshold estimate. Completing BB144 high-p points, increasing the fixed final test bank, adding training seeds, and comparing stronger OSD-CS/LSD settings are still required.",
            "",
            f"Complete rows: [`{CSV_PATH.name}`]({CSV_PATH.name})",
            f"Partial rows: [`{PARTIAL_CSV_PATH.name}`]({PARTIAL_CSV_PATH.name})",
            f"Primary plot: [`{PLOT_PATH.name}`](../plots/{PLOT_PATH.name})",
            f"Ablation plot: [`{ABLATION_PLOT_PATH.name}`](../plots/{ABLATION_PLOT_PATH.name})",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def wilson_interval(successes: int, shots: int) -> tuple[float, float]:
    proportion = successes / shots
    denominator = 1.0 + Z_95**2 / shots
    center = (proportion + Z_95**2 / (2.0 * shots)) / denominator
    radius = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / shots + Z_95**2 / (4.0 * shots**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def plot_results(rows: list[CircuitResult], dpi: int) -> None:
    cache = Path(tempfile.gettempdir()) / "theend_bb_circuit_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install requirements.txt or pass --no-plot."
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    sweep = [row for row in rows if is_primary_sweep(row)]
    if not sweep:
        raise ValueError("No complete default circuit sweep is available to plot.")
    by_code: dict[str, list[CircuitResult]] = defaultdict(list)
    for row in sweep:
        by_code[row.code].append(row)
    for code_rows in by_code.values():
        code_rows.sort(key=lambda row: row.gate_error_rate)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10.2,
            "legend.frameon": False,
        }
    )
    figure, (ler_axis, gain_axis) = plt.subplots(
        1, 2, figsize=(12.2, 4.8), constrained_layout=True
    )
    colors = {"bb72": "#2563eb", "bb144": "#dc2626"}
    fallback_colors = ["#0f766e", "#7c3aed", "#d97706"]

    zero_points: list[tuple[float, float, str]] = []
    for code_index, (code, code_rows) in enumerate(sorted(by_code.items())):
        color = colors.get(code, fallback_colors[code_index % len(fallback_colors)])
        xs = [row.gate_error_rate for row in code_rows]
        for decoder, neural, linestyle, marker in (
            ("Neural posterior + OSD-0", True, "-", "o"),
            ("Vanilla BP posterior + OSD-0", False, "--", "s"),
        ):
            ys: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for row in code_rows:
                ler = (
                    row.neural_osd_logical_error_rate
                    if neural
                    else row.vanilla_osd_logical_error_rate
                )
                failures = round(ler * row.osd_shots)
                low, high = wilson_interval(failures, row.osd_shots)
                shown = ler if failures else 0.5 / row.osd_shots
                ys.append(shown)
                lower_errors.append(
                    0.0 if failures == 0 else max(0.0, shown - max(low, 1e-12))
                )
                upper_errors.append(max(0.0, high - shown))
                if failures == 0:
                    zero_points.append((row.gate_error_rate, shown, color))
            ler_axis.errorbar(
                xs,
                ys,
                yerr=[lower_errors, upper_errors],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                capsize=2.5,
                label=f"{code.upper()} {decoder}",
            )

        gains = [100.0 * row.osd_paired_gain for row in code_rows]
        halfwidths = [100.0 * row.osd_paired_gain_ci95_halfwidth for row in code_rows]
        gain_axis.errorbar(
            xs,
            gains,
            yerr=halfwidths,
            color=color,
            marker="o",
            linewidth=1.8,
            capsize=3.0,
            label=code.upper(),
        )

    for x, y, color in zero_points:
        ler_axis.scatter([x], [y], marker="v", s=42, color=color, zorder=5)
    ler_axis.set_yscale("log")
    ler_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ler_axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{100.0 * value:.0f}%"
                if value >= 0.1
                else (
                    f"{100.0 * value:.1f}%"
                    if value >= 0.001
                    else f"{100.0 * value:.2f}%"
                )
            )
        )
    )
    ler_axis.set_xlabel("Circuit gate/readout error rate, p=q")
    ler_axis.set_ylabel("Block logical error rate")
    ler_axis.set_title("Selected-best OSD-0 logical error rate")
    ler_axis.grid(True, which="major", color="#d7dce2", linewidth=0.7)
    ler_axis.grid(True, which="minor", color="#edf0f3", linewidth=0.5)
    ler_axis.legend(fontsize=8.3)
    ler_axis.text(
        0.02,
        0.03,
        "▼: zero failures; displayed at 0.5/N",
        transform=ler_axis.transAxes,
        fontsize=8.1,
        color="#475569",
    )

    gain_axis.axhline(0.0, color="#64748b", linewidth=1.0)
    gain_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    gain_axis.set_xlabel("Circuit gate/readout error rate, p=q")
    gain_axis.set_ylabel("Accuracy gain (percentage points)")
    gain_axis.set_title("Neural-posterior OSD-0 minus BP-posterior OSD-0")
    gain_axis.grid(True, color="#d7dce2", linewidth=0.7)
    gain_axis.legend(fontsize=8.5)

    figure.suptitle(
        "BB circuit-level Neural BP2: paired selected-best evaluation\n"
        "Circuit schema v2, idle error=0; error bars are paired 95% intervals",
        fontsize=12.3,
        fontweight="semibold",
    )
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ablations(rows: list[CircuitResult], dpi: int) -> None:
    cache = Path(tempfile.gettempdir()) / "theend_bb_circuit_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install requirements.txt or pass --no-plot."
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if is_primary_sweep(row)]
    reference = {(row.code, row.gate_error_rate): row for row in primary}

    def with_matching_reference(candidates: list[CircuitResult]) -> list[CircuitResult]:
        selected: list[CircuitResult] = []
        for row in candidates:
            baseline = reference.get((row.code, row.gate_error_rate))
            if baseline is not None and baseline not in selected:
                selected.append(baseline)
            if row not in selected:
                selected.append(row)
        return selected

    sharing = with_matching_reference([row for row in rows if row.purpose == "sharing"])
    p004 = [row for row in primary if math.isclose(row.gate_error_rate, 0.004)]
    iterations = p004 + [row for row in rows if row.purpose == "iterations"]
    mechanisms = p004 + [row for row in rows if row.purpose == "mechanism"]
    losses = p004 + [row for row in rows if row.purpose == "loss_auxiliary"]
    p003 = [row for row in primary if math.isclose(row.gate_error_rate, 0.003)]
    noise = p003 + [row for row in rows if row.purpose == "noise_balance"]
    replicates = [
        row
        for row in rows
        if is_default_sweep(row)
        and row.purpose in {"baseline", "replicate"}
        and math.isclose(row.gate_error_rate, 0.004)
    ]

    panels = [
        (
            "Sharing",
            sorted(
                sharing,
                key=lambda row: (
                    row.n,
                    row.gate_error_rate,
                    0 if row.purpose == "baseline" else 1,
                ),
            ),
            lambda row: (
                f"{row.code.upper()}\np={row.gate_error_rate:.3f}\n{row.sharing}"
            ),
        ),
        (
            "Iterations (p=0.004)",
            sorted(iterations, key=lambda row: (row.n, row.iterations)),
            lambda row: f"{row.code.upper()}\nT={row.iterations}",
        ),
        (
            "Update components (p=0.004)",
            sorted(
                mechanisms,
                key=lambda row: (
                    row.n,
                    0 if row.purpose == "baseline" else 1,
                    row.variant,
                ),
            ),
            lambda row: (
                f"{row.code.upper()}\n{('full' if row.purpose == 'baseline' else row.variant.replace('_only', ' only'))}"
            ),
        ),
        (
            "Auxiliary losses (p=0.004)",
            sorted(
                losses,
                key=lambda row: (
                    row.n,
                    0 if row.purpose == "baseline" else 1,
                    row.variant,
                ),
            ),
            lambda row: (
                f"{row.code.upper()}\n{('full' if row.purpose == 'baseline' else {'no_deep_supervision': 'no deep', 'no_mechanism_bce': 'no mech BCE'}[row.variant])}"
            ),
        ),
        (
            "Noise balance (p=0.003)",
            sorted(
                noise,
                key=lambda row: (
                    row.n,
                    row.measurement_error_rate,
                    row.idle_error_rate,
                ),
            ),
            lambda row: (
                f"{row.code.upper()}\n{'q=p,idle=0' if row.purpose == 'baseline' else row.variant}"
            ),
        ),
        (
            "Training seeds (p=0.004)",
            sorted(replicates, key=lambda row: (row.n, row.seed)),
            lambda row: f"{row.code.upper()}\nseed {str(row.seed)[-4:]}",
        ),
    ]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9.2,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), constrained_layout=True)
    colors = {"bb72": "#2563eb", "bb144": "#dc2626"}
    for axis, (title, panel_rows, label) in zip(axes.flat, panels):
        xs = list(range(len(panel_rows)))
        gains = [100.0 * float(row.osd_paired_gain) for row in panel_rows]
        errors = [
            100.0 * float(row.osd_paired_gain_ci95_halfwidth) for row in panel_rows
        ]
        axis.bar(
            xs,
            gains,
            yerr=errors,
            capsize=3,
            color=[colors.get(row.code, "#64748b") for row in panel_rows],
            alpha=0.88,
        )
        axis.axhline(0.0, color="#475569", linewidth=0.9)
        axis.set_xticks(xs, [label(row) for row in panel_rows], fontsize=7.4)
        axis.set_title(title, fontweight="semibold")
        axis.set_ylabel("Accuracy gain (pp)")
        axis.grid(True, axis="y", color="#d7dce2", linewidth=0.7)

    figure.suptitle(
        "BB circuit-level Neural BP2 ablations\n"
        "Neural-posterior OSD-0 minus vanilla-BP-posterior OSD-0; bars show paired 95% intervals",
        fontsize=12.3,
        fontweight="semibold",
    )
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ABLATION_PLOT_PATH, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize archived circuit-level BB Neural-BP runs."
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--no-plot", action="store_true", help="Write CSV/report without matplotlib."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows, partial_rows = collect_results()
    write_csv(rows)
    write_partial_csv(partial_rows)
    write_report(rows, partial_rows)
    if not args.no_plot:
        plot_results(rows, args.dpi)
        plot_ablations(rows, args.dpi)
    print(f"Parsed {len(rows) + len(partial_rows)} circuit-level BB run(s).")
    print(f"Complete: {len(rows)}; partial: {len(partial_rows)}")
    print(f"CSV: {CSV_PATH}")
    print(f"Partial CSV: {PARTIAL_CSV_PATH}")
    print(f"Report: {REPORT_PATH}")
    if not args.no_plot:
        print(f"Plot: {PLOT_PATH}")
        print(f"Ablation plot: {ABLATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
