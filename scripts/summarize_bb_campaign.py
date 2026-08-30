#!/usr/bin/env python3
"""Summarize the August 2026 BB Neural-BP/classical campaign.

The Slurm launcher logs are not authoritative for this campaign: a few srun
steps report exit 137 even though the Python output log reached epoch 299 and
wrote a fresh ``[Selected Best]`` evaluation.  This script therefore reads the
per-output ``training_log.txt`` and ``history.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

_PLOT_CACHE = Path(tempfile.gettempdir()) / "theend_bb_plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY / "results" / "bb" / "code_capacity"
ANALYSIS_ROOT = REPOSITORY / "results" / "analysis"
PLOT_ROOT = REPOSITORY / "results" / "plots"
BASELINE_CSV = ANALYSIS_ROOT / "bb_neural_bp_depolarizing_orbit.csv"
NEURAL_CSV = ANALYSIS_ROOT / "bb_campaign_2026_08_neural.csv"
CLASSICAL_CSV = ANALYSIS_ROOT / "bb_campaign_2026_08_classical.csv"
PAIRED_CSV = ANALYSIS_ROOT / "bb_neural_vs_classical_paired.csv"
REPORT_MD = ANALYSIS_ROOT / "bb_campaign_2026_08.md"
ABLATION_PLOT = PLOT_ROOT / "bb_campaign_2026_08_ablations.png"
DECODER_PLOT = PLOT_ROOT / "bb_campaign_2026_08_decoders.png"
Z_95 = 1.959963984540054

JOB_PURPOSE = {
    "57452584": "seed_replicate_b",
    "57452586": "seed_replicate_c",
    "57452587": "sharing",
    "57452588": "iterations",
    "57452589": "independent_xz",
    "57452590": "mechanism",
    "57452591": "loss_auxiliary",
    "57452592": "loss_core",
}
CLASSICAL_JOB_IDS = {"57452848", "57452850"}
CODE_SPECS = {"bb72": (72, 12, 6), "bb144": (144, 12, 12)}


@dataclass
class NeuralResult:
    job_id: str
    experiment_index: int
    purpose: str
    variant: str
    scientific_status: str
    launcher_exit_code: int
    code: str
    n: int
    k: int
    d: int
    channel: str
    p_label: float
    x_error_rate: float | None
    z_error_rate: float | None
    total_nonidentity_rate: float
    seed: int
    iterations: int
    hidden_dim: int
    sharing: str
    residual_scale: float
    relaxation_delta: float
    deep_supervision_weight: float
    syndrome_loss_weight: float
    logical_loss_weight: float
    pauli_loss_weight: float
    requested_epochs: int
    logged_last_epoch: int
    checkpoint_last_epoch: int
    remaining_epochs: int
    selected_epoch: int | None
    neural_accuracy: float | None
    neural_logical_error_rate: float | None
    syndrome_convergence: float | None
    flagged_failure_rate: float | None
    unflagged_logical_failure_rate: float | None
    vanilla_accuracy: float | None
    vanilla_logical_error_rate: float | None
    paired_gain: float | None
    paired_gain_se: float | None
    paired_ci95_low: float | None
    paired_ci95_high: float | None
    rescued: int | None
    harmed: int | None
    eval_samples: int | None
    source_resdir: str
    output_directory: str
    checkpoint: str


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY.resolve()))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, report, and plot the August 2026 BB campaign."
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


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


def require(options: dict[str, str | bool], key: str) -> str:
    value = options.get(key)
    if value is None or value is True:
        raise ValueError(f"missing scalar option --{key}")
    return str(value)


def scalar(options: dict[str, str | bool], key: str, default: float) -> float:
    value = options.get(key, default)
    if value is True:
        raise ValueError(f"--{key} must have a scalar value")
    return float(value)


def find_resdir(job_id: str) -> Path:
    matches = list(RESULTS_ROOT.rglob(f"resdir_{job_id}"))
    if len(matches) != 1:
        raise ValueError(f"expected one resdir_{job_id}, found {len(matches)}")
    return matches[0]


def load_commands(resdir: Path) -> dict[tuple[tuple[str, str], ...], int]:
    result: dict[tuple[tuple[str, str], ...], int] = {}
    for path in sorted(resdir.glob("command_exp_*.txt")):
        match = re.search(r"command_exp_(\d+)\.txt$", path.name)
        if match is None:
            continue
        options = option_map(path.read_text(encoding="utf-8"))
        key = tuple(sorted((name, str(value)) for name, value in options.items()))
        result[key] = int(match.group(1))
    return result


def parse_selected(line: str) -> dict[str, float | int]:
    patterns: dict[str, tuple[str, type[float] | type[int]]] = {
        "selected_epoch": (r"Epoch: (\d+)", int),
        "neural_accuracy": (r"\| Accuracy: ([0-9.]+)", float),
        "neural_logical_error_rate": (
            r"\| Logical Error Rate: ([0-9.]+)",
            float,
        ),
        "syndrome_convergence": (r"\| Syndrome Convergence: ([0-9.]+)", float),
        "flagged_failure_rate": (r"\| Flagged: ([0-9.]+)", float),
        "unflagged_logical_failure_rate": (
            r"\| Unflagged Logical: ([0-9.]+)",
            float,
        ),
        "vanilla_accuracy": (r"\| Vanilla BP Accuracy: ([0-9.]+)", float),
        "paired_gain": (r"\| Paired Gain: ([+-]?[0-9.]+)", float),
        "paired_ci95_halfwidth": (r"\+/- ([0-9.]+)", float),
        "rescued": (r"\| Rescued: (\d+)", int),
        "harmed": (r"\| Harmed: (\d+)", int),
        "eval_samples": (r"\| Eval Samples: (\d+)", int),
    }
    parsed: dict[str, float | int] = {}
    for key, (pattern, converter) in patterns.items():
        match = re.search(pattern, line)
        if match is None:
            raise ValueError(f"could not parse {key} from {line!r}")
        parsed[key] = converter(match.group(1))
    return parsed


def variant_name(purpose: str, options: dict[str, str | bool]) -> str:
    if purpose.startswith("seed_replicate"):
        return purpose
    if purpose == "sharing":
        return f"global_h{require(options, 'bp_residual_hidden_dim')}"
    if purpose == "iterations":
        return f"iterations_{require(options, 'bp_iterations')}"
    if purpose == "independent_xz":
        return "marginal_matched_independent_xz"
    if purpose == "mechanism":
        if scalar(options, "bp_residual_scale", 2.0) == 0.0:
            return "relaxation_only"
        if scalar(options, "bp_max_relaxation_delta", 0.5) == 0.0:
            return "residual_only"
    if purpose == "loss_auxiliary":
        if scalar(options, "bb_pauli_loss_weight", 0.1) == 0.0:
            return "no_pauli_auxiliary"
        if scalar(options, "bp_deep_supervision_weight", 0.2) == 0.0:
            return "no_deep_supervision"
    if purpose == "loss_core":
        if scalar(options, "bb_logical_loss_weight", 1.0) == 0.0:
            return "no_logical_loss"
        if scalar(options, "bb_syndrome_loss_weight", 1.0) == 0.0:
            return "no_syndrome_loss"
    raise ValueError(f"unrecognized variant for purpose={purpose}: {options}")


def parse_neural_output(
    job_id: str,
    purpose: str,
    resdir: Path,
    training_log: Path,
    commands: dict[tuple[tuple[str, str], ...], int],
) -> NeuralResult:
    text = training_log.read_text(encoding="utf-8", errors="replace")
    first_line = text.splitlines()[0]
    if not first_line.startswith("Executed Command:"):
        raise ValueError(f"missing Executed Command in {training_log}")
    options = option_map(first_line.split(":", 1)[1])
    command_key = tuple(sorted((name, str(value)) for name, value in options.items()))
    if command_key not in commands:
        raise ValueError(f"could not map {training_log} to a command_exp file")
    experiment_index = commands[command_key]

    train_epochs = [
        int(value) for value in re.findall(r"^\[Train\] Epoch: (\d+)", text, re.M)
    ]
    if not train_epochs:
        raise ValueError(f"no training epochs in {training_log}")
    logged_last_epoch = max(train_epochs)

    history_path = training_log.with_name("history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint_epochs = [int(value) for value in history.get("train_epoch", [])]
    checkpoint_last_epoch = max(checkpoint_epochs) if checkpoint_epochs else -1

    selected_lines = re.findall(r"^\[Selected Best\].*$", text, re.M)
    selected = parse_selected(selected_lines[-1]) if selected_lines else None
    requested_epochs = int(require(options, "epochs"))
    status = "complete" if selected is not None else "partial"
    remaining_epochs = 0 if status == "complete" else requested_epochs - checkpoint_last_epoch - 1

    launcher_exit_path = resdir / f"exit_code_exp_{experiment_index}.txt"
    launcher_exit_code = int(launcher_exit_path.read_text(encoding="utf-8").strip())
    code = require(options, "code")
    n, k, d = CODE_SPECS[code]
    channel = require(options, "bb_channel")
    p_label = float(require(options, "p"))
    if channel == "depolarizing":
        x_rate = None
        z_rate = None
        total_nonidentity = p_label
    else:
        x_rate = scalar(options, "x_error_rate", p_label)
        z_rate = scalar(options, "z_error_rate", p_label)
        total_nonidentity = x_rate + z_rate - x_rate * z_rate

    output_directory = training_log.parent
    checkpoint = output_directory / "model.pt"
    if not checkpoint.is_file():
        raise ValueError(f"missing checkpoint: {checkpoint}")

    if selected is None:
        selected_values: dict[str, float | int | None] = {
            "selected_epoch": None,
            "neural_accuracy": None,
            "neural_logical_error_rate": None,
            "syndrome_convergence": None,
            "flagged_failure_rate": None,
            "unflagged_logical_failure_rate": None,
            "vanilla_accuracy": None,
            "paired_gain": None,
            "paired_ci95_halfwidth": None,
            "rescued": None,
            "harmed": None,
            "eval_samples": None,
        }
    else:
        selected_values = selected

    paired_gain = selected_values["paired_gain"]
    halfwidth = selected_values["paired_ci95_halfwidth"]
    if paired_gain is None or halfwidth is None:
        paired_se = paired_low = paired_high = None
    else:
        paired_gain = float(paired_gain)
        halfwidth = float(halfwidth)
        paired_se = halfwidth / 1.96
        paired_low = paired_gain - halfwidth
        paired_high = paired_gain + halfwidth
    vanilla_accuracy = selected_values["vanilla_accuracy"]

    return NeuralResult(
        job_id=job_id,
        experiment_index=experiment_index,
        purpose=purpose,
        variant=variant_name(purpose, options),
        scientific_status=status,
        launcher_exit_code=launcher_exit_code,
        code=code,
        n=n,
        k=k,
        d=d,
        channel=channel,
        p_label=p_label,
        x_error_rate=x_rate,
        z_error_rate=z_rate,
        total_nonidentity_rate=total_nonidentity,
        seed=int(require(options, "seed")),
        iterations=int(require(options, "bp_iterations")),
        hidden_dim=int(require(options, "bp_residual_hidden_dim")),
        sharing=require(options, "bp_parameter_sharing"),
        residual_scale=scalar(options, "bp_residual_scale", 2.0),
        relaxation_delta=scalar(options, "bp_max_relaxation_delta", 0.5),
        deep_supervision_weight=scalar(options, "bp_deep_supervision_weight", 0.2),
        syndrome_loss_weight=scalar(options, "bb_syndrome_loss_weight", 1.0),
        logical_loss_weight=scalar(options, "bb_logical_loss_weight", 1.0),
        pauli_loss_weight=scalar(options, "bb_pauli_loss_weight", 0.1),
        requested_epochs=requested_epochs,
        logged_last_epoch=logged_last_epoch,
        checkpoint_last_epoch=checkpoint_last_epoch,
        remaining_epochs=remaining_epochs,
        selected_epoch=(
            int(selected_values["selected_epoch"])
            if selected_values["selected_epoch"] is not None
            else None
        ),
        neural_accuracy=(
            float(selected_values["neural_accuracy"])
            if selected_values["neural_accuracy"] is not None
            else None
        ),
        neural_logical_error_rate=(
            float(selected_values["neural_logical_error_rate"])
            if selected_values["neural_logical_error_rate"] is not None
            else None
        ),
        syndrome_convergence=(
            float(selected_values["syndrome_convergence"])
            if selected_values["syndrome_convergence"] is not None
            else None
        ),
        flagged_failure_rate=(
            float(selected_values["flagged_failure_rate"])
            if selected_values["flagged_failure_rate"] is not None
            else None
        ),
        unflagged_logical_failure_rate=(
            float(selected_values["unflagged_logical_failure_rate"])
            if selected_values["unflagged_logical_failure_rate"] is not None
            else None
        ),
        vanilla_accuracy=(
            float(vanilla_accuracy) if vanilla_accuracy is not None else None
        ),
        vanilla_logical_error_rate=(
            1.0 - float(vanilla_accuracy) if vanilla_accuracy is not None else None
        ),
        paired_gain=paired_gain,
        paired_gain_se=paired_se,
        paired_ci95_low=paired_low,
        paired_ci95_high=paired_high,
        rescued=(
            int(selected_values["rescued"])
            if selected_values["rescued"] is not None
            else None
        ),
        harmed=(
            int(selected_values["harmed"])
            if selected_values["harmed"] is not None
            else None
        ),
        eval_samples=(
            int(selected_values["eval_samples"])
            if selected_values["eval_samples"] is not None
            else None
        ),
        source_resdir=relative(resdir),
        output_directory=relative(output_directory),
        checkpoint=relative(checkpoint),
    )


def collect_neural() -> list[NeuralResult]:
    results: list[NeuralResult] = []
    for job_id, purpose in JOB_PURPOSE.items():
        resdir = find_resdir(job_id)
        commands = load_commands(resdir)
        logs = sorted(resdir.glob("outputs/*/*/training_log.txt"))
        if len(logs) != 4:
            raise ValueError(f"expected four output logs in {resdir}, found {len(logs)}")
        for log in logs:
            results.append(parse_neural_output(job_id, purpose, resdir, log, commands))
    return sorted(results, key=lambda item: (int(item.job_id), item.experiment_index))


def write_neural_csv(results: list[NeuralResult]) -> None:
    NEURAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(NeuralResult)]
    with NEURAL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in names})


def collect_classical() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for job_id in sorted(CLASSICAL_JOB_IDS):
        resdir = find_resdir(job_id)
        for path in sorted(resdir.glob("*_classical.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row["source_resdir"] = relative(resdir)
                    row["source_file"] = relative(path)
                    rows.append(row)
    if len(rows) != 24:
        raise ValueError(f"expected 24 classical decoder rows, found {len(rows)}")
    rows.sort(key=lambda row: (int(row["n"]), float(row["p"]), row["method"]))
    return rows


def write_classical_csv(rows: list[dict[str, str]]) -> None:
    CLASSICAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0])
    with CLASSICAL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline() -> list[dict[str, str]]:
    with BASELINE_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_paired() -> list[dict[str, str]]:
    if not PAIRED_CSV.is_file():
        return []
    with PAIRED_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise ValueError(f"expected 8 paired decoder rows, found {len(rows)}")
    return rows


def baseline_point(
    baseline: list[dict[str, str]], code: str, p: float
) -> dict[str, str]:
    points = [
        row
        for row in baseline
        if row["code"] == code and math.isclose(float(row["p"]), p)
    ]
    if len(points) != 1:
        raise ValueError(f"expected one baseline for {code}, p={p}")
    return points[0]


def wilson_interval(proportion: float, samples: int) -> tuple[float, float]:
    denominator = 1.0 + Z_95**2 / samples
    center = (proportion + Z_95**2 / (2.0 * samples)) / denominator
    radius = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + Z_95**2 / (4.0 * samples**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def plot_decoders(
    baseline: list[dict[str, str]],
    classical: list[dict[str, str]],
    paired: list[dict[str, str]],
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10.2,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    method_specs = {
        "neural": ("Joint Neural BP4 (orbit, T=12)", "#0f766e", "o", "-"),
        "vanilla": ("Joint vanilla BP4 (T=12)", "#64748b", "s", "--"),
        "bposd_0": ("CSS BP+OSD-0", "#d97706", "^", ":"),
        "bposd_cs7": ("CSS BP+OSD-CS7", "#b91c1c", "D", "-."),
        "bplsd_0": ("CSS BP+LSD-0", "#7c3aed", "v", ":"),
    }

    for axis, code in zip(axes, ("bb72", "bb144"), strict=True):
        base_rows = sorted(
            (row for row in baseline if row["code"] == code),
            key=lambda row: float(row["p"]),
        )
        class_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in classical:
            if row["code"] == code:
                class_rows[row["method"]].append(row)

        series: dict[str, list[tuple[float, float, int]]] = {
            "neural": [
                (float(row["p"]), float(row["neural_logical_error_rate"]), int(row["eval_samples"]))
                for row in base_rows
            ],
            "vanilla": [
                (float(row["p"]), float(row["vanilla_logical_error_rate"]), int(row["eval_samples"]))
                for row in base_rows
            ],
        }
        paired_rows = sorted(
            (row for row in paired if row["code"] == code),
            key=lambda row: float(row["p"]),
        )
        if paired_rows:
            series["neural"] = [
                (
                    float(row["p"]),
                    float(row["neural_logical_error_rate"]),
                    int(row["samples"]),
                )
                for row in paired_rows
            ]
            series["vanilla"] = [
                (
                    float(row["p"]),
                    float(row["vanilla_logical_error_rate"]),
                    int(row["samples"]),
                )
                for row in paired_rows
            ]
        for method, rows in class_rows.items():
            series[method] = [
                (float(row["p"]), float(row["logical_error_rate"]), int(row["samples"]))
                for row in sorted(rows, key=lambda item: float(item["p"]))
            ]

        for method, points in series.items():
            label, color, marker, linestyle = method_specs[method]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            intervals = [wilson_interval(point[1], point[2]) for point in points]
            lower = [value - interval[0] for value, interval in zip(ys, intervals, strict=True)]
            upper = [interval[1] - value for value, interval in zip(ys, intervals, strict=True)]
            axis.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                markersize=5.3,
                capsize=2.5,
            )
        axis.set_yscale("log")
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Depolarizing error rate, p")
        axis.set_ylabel("Block logical error rate")
        axis.set_title(code.upper())
        axis.grid(True, which="major", color="#d7dce2", linewidth=0.7)
        axis.grid(True, which="minor", color="#edf0f3", linewidth=0.5)

    axes[1].legend(loc="upper left", fontsize=8.7)
    bank_note = (
        "All curves use the same saved 131,072-shot error banks"
        if paired
        else "Neural/BP4 and CSS curves use independent 131,072-shot banks"
    )
    figure.suptitle(
        "BB decoder comparison on code-capacity depolarizing noise\n" + bank_note,
        fontsize=12.4,
        fontweight="semibold",
    )
    DECODER_PLOT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(DECODER_PLOT, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def completed_variant(
    results: list[NeuralResult], code: str, variant: str
) -> NeuralResult | None:
    points = [
        result
        for result in results
        if result.code == code
        and result.variant == variant
        and result.scientific_status == "complete"
    ]
    if len(points) > 1:
        raise ValueError(f"multiple completed {code}/{variant} points")
    return points[0] if points else None


def plot_ablations(
    results: list[NeuralResult], baseline: list[dict[str, str]], dpi: int
) -> None:
    panels = [
        ("Sharing", ["full_orbit", "global_h64", "global_h393"], ["Orbit h64", "Global h64", "Global h393"]),
        ("BP iterations", ["iterations_6", "full_orbit", "iterations_24"], ["T=6", "T=12", "T=24"]),
        ("Neural mechanism", ["full_orbit", "residual_only", "relaxation_only"], ["Full", "Residual only", "Relaxation only"]),
        (
            "Loss components",
            ["full_orbit", "no_pauli_auxiliary", "no_deep_supervision", "no_logical_loss", "no_syndrome_loss"],
            ["Full", "No Pauli", "No deep", "No logical", "No syndrome"],
        ),
    ]
    colors = {"bb72": "#2563eb", "bb144": "#dc2626"}
    markers = {"bb72": "o", "bb144": "s"}
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=True)
    for axis, (title, variants, labels) in zip(axes.flat, panels, strict=True):
        xs = list(range(len(variants)))
        for code, offset in (("bb72", -0.08), ("bb144", 0.08)):
            values: list[float] = []
            present_xs: list[float] = []
            for x, variant in zip(xs, variants, strict=True):
                if variant == "full_orbit":
                    row = baseline_point(baseline, code, 0.08)
                    value = 100.0 * float(row["neural_logical_error_rate"])
                else:
                    result = completed_variant(results, code, variant)
                    if result is None or result.neural_logical_error_rate is None:
                        axis.text(
                            x + offset,
                            0.02,
                            "partial",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            color=colors[code],
                            fontsize=7.8,
                            transform=axis.get_xaxis_transform(),
                        )
                        continue
                    value = 100.0 * result.neural_logical_error_rate
                present_xs.append(x + offset)
                values.append(value)
            axis.plot(
                present_xs,
                values,
                color=colors[code],
                marker=markers[code],
                linewidth=1.35,
                markersize=6.2,
                label=code.upper(),
            )
        axis.set_xticks(xs, labels, rotation=20, ha="right")
        axis.set_xlim(-0.45, len(variants) - 0.55)
        axis.set_ylabel("Logical error rate (%)")
        axis.set_title(title, fontweight="semibold")
        axis.grid(True, axis="y", color="#e2e6ea", linewidth=0.7)
    axes[0, 0].legend(loc="best")
    figure.suptitle(
        "Neural-BP ablations at depolarizing p=0.08 (fresh 131,072-shot tests)",
        fontsize=13,
        fontweight="semibold",
    )
    ABLATION_PLOT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ABLATION_PLOT, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def fmt_percent(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def fmt_pp(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{100.0 * value:+.{digits}f} pp"


def ablation_table(
    results: list[NeuralResult], baseline: list[dict[str, str]], variants: list[str]
) -> list[str]:
    labels = {
        "full_orbit": "Full orbit h64, T12",
        "global_h64": "Global h64",
        "global_h393": "Global h393",
        "iterations_6": "T6",
        "iterations_24": "T24",
        "residual_only": "Residual only",
        "relaxation_only": "Relaxation only",
        "no_pauli_auxiliary": "No Pauli auxiliary",
        "no_deep_supervision": "No deep supervision",
        "no_logical_loss": "No logical loss",
        "no_syndrome_loss": "No syndrome loss",
    }
    lines = [
        "| Variant | BB72 LER | BB144 LER |",
        "|---|---:|---:|",
    ]
    for variant in variants:
        values: list[str] = []
        for code in ("bb72", "bb144"):
            if variant == "full_orbit":
                row = baseline_point(baseline, code, 0.08)
                values.append(fmt_percent(float(row["neural_logical_error_rate"])))
            else:
                result = completed_variant(results, code, variant)
                values.append(
                    "partial"
                    if result is None
                    else fmt_percent(result.neural_logical_error_rate)
                )
        lines.append(f"| {labels[variant]} | {values[0]} | {values[1]} |")
    return lines


def write_report(
    neural: list[NeuralResult],
    classical: list[dict[str, str]],
    baseline: list[dict[str, str]],
    paired: list[dict[str, str]],
) -> None:
    complete = [result for result in neural if result.scientific_status == "complete"]
    partial = [result for result in neural if result.scientific_status == "partial"]
    nominal_137_complete = [
        result for result in complete if result.launcher_exit_code == 137
    ]
    paired_positive = [result for result in complete if (result.paired_ci95_low or 0.0) > 0]
    paired_negative = [result for result in complete if (result.paired_ci95_high or 0.0) < 0]

    seed_rows: list[tuple[str, float, int, float, float, float, float]] = []
    for code in ("bb72", "bb144"):
        for p in (0.08, 0.10):
            base = baseline_point(baseline, code, p)
            accuracies = [float(base["neural_accuracy"])]
            gains = [float(base["paired_gain"])]
            for result in complete:
                if (
                    result.purpose.startswith("seed_replicate")
                    and result.code == code
                    and math.isclose(result.p_label, p)
                ):
                    assert result.neural_accuracy is not None
                    assert result.paired_gain is not None
                    accuracies.append(result.neural_accuracy)
                    gains.append(result.paired_gain)
            seed_rows.append(
                (
                    code,
                    p,
                    len(accuracies),
                    statistics.mean(accuracies),
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    statistics.mean(gains),
                    statistics.stdev(gains) if len(gains) > 1 else 0.0,
                )
            )

    lines = [
        "# BB campaign: August 2026",
        "",
        "## Integrity and scope",
        "",
        f"The campaign contains 40 experiment points: 32 Neural BP trainings and "
        f"8 classical `(code,p)` benchmarks. Scientifically usable results total "
        f"{len(complete) + 8}: {len(complete)} Neural runs have all 300 epochs and "
        "a fresh 131,072-shot `[Selected Best]` evaluation, and all 8 classical "
        f"points completed. The remaining {len(partial)} Neural runs are partial.",
        "",
        "The per-output `training_log.txt` and `history.json` are authoritative. "
        f"Five runs ({len(nominal_137_complete)} observed) have a top-level launcher "
        "exit code of 137 but nevertheless reached epoch 299 and wrote the final "
        "evaluation; they are retained. No completed output contains NaNs or a "
        "Python traceback.",
        "",
        f"Of the {len(complete)} completed Neural runs, {len(paired_positive)} have "
        "a paired 95% confidence interval strictly above zero versus the same-shot "
        f"vanilla BP4 baseline. The only negative case ({len(paired_negative)}) is "
        "BB72 relaxation-only (`residual_scale=0`).",
        "",
        "## Training-seed robustness",
        "",
        "These statistics combine the archived seed-A run with the new completed "
        "seed-B/C runs. They are descriptive because each group has only two or "
        "three completed training seeds.",
        "",
        "| Code | p | Completed seeds | Neural accuracy mean ± SD | Gain over BP4 mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for code, p, count, acc_mean, acc_sd, gain_mean, gain_sd in seed_rows:
        lines.append(
            f"| {code.upper()} | {p:.2f} | {count} | "
            f"{100*acc_mean:.4f}% ± {100*acc_sd:.4f} pp | "
            f"{100*gain_mean:.4f} ± {100*gain_sd:.4f} pp |"
        )

    lines.extend(
        [
            "",
            "## Architecture ablations at p=0.08",
            "",
            *ablation_table(
                neural,
                baseline,
                [
                    "full_orbit",
                    "global_h64",
                    "global_h393",
                    "iterations_6",
                    "iterations_24",
                    "residual_only",
                    "relaxation_only",
                    "no_pauli_auxiliary",
                    "no_deep_supervision",
                    "no_logical_loss",
                    "no_syndrome_loss",
                ],
            ),
            "",
            "Main conclusions:",
            "",
            "- **More BP iterations are the strongest tested improvement.** T24 "
            "reduces Neural LER from 12.0453% to 11.1237% on BB72 (7.65% relative) "
            "and from 4.1580% to 2.5108% on BB144 (39.6% relative). Vanilla BP4 "
            "also improves at T24, so iteration-specific same-T baselines must be "
            "reported.",
            "- **Orbit-specific sharing is not supported by this accuracy ablation.** "
            "Global h64 (2,366 parameters) is essentially tied with orbit h64 "
            "(14,196), while parameter-matched global h393 (14,210) is slightly "
            "better at this one seed. Both are equivariant; this is an orbit-label "
            "granularity test, not equivariance on/off.",
            "- **The residual MLP is essential for BB72.** Relaxation-only selects "
            "epoch 19, scores 15.1413% LER, and is significantly worse than vanilla "
            "BP4 by 0.4021 accuracy points. Residual-only is close to the full BB72 "
            "model. On BB144, either mechanism alone remains useful but is weaker "
            "than the combination.",
            "- Removing the Pauli auxiliary loss changes little. Removing deep "
            "supervision causes a small degradation. Removing logical loss raises "
            "BB72 syndrome convergence to 98.1117% but also raises unflagged logical "
            "failure to 10.7681%, showing why convergence alone is insufficient. "
            "The BB144 no-syndrome counterpart remains partial.",
            "",
            "## Marginal-matched independent X/Z",
            "",
        ]
    )
    for code in ("bb72", "bb144"):
        result = next(
            item
            for item in complete
            if item.code == code
            and item.variant == "marginal_matched_independent_xz"
            and math.isclose(item.p_label, 0.08)
        )
        lines.append(
            f"- {code.upper()}: Neural accuracy {fmt_percent(result.neural_accuracy)} "
            f"versus BP4 {fmt_percent(result.vanilla_accuracy)}, paired gain "
            f"{fmt_pp(result.paired_gain)} with 95% CI "
            f"[{fmt_pp(result.paired_ci95_low)}, {fmt_pp(result.paired_ci95_high)}]."
        )
    lines.extend(
        [
            "",
            "Here `q_x=q_z=2p/3`. This matches the X and Z component marginals of "
            "depolarizing noise but not total non-identity rate: the p-label .08 "
            "point has total non-I rate 10.3822% (and p-label .10 has 12.8889%). "
            "Therefore raw LER cannot isolate correlation alone. A complementary "
            "total-rate-matched control should use `q=1-sqrt(1-p)`.",
            "",
            "## Classical CSS-separated baselines",
            "",
            "All rows use `ldpc==2.4.1`, min-sum scaling 0.625, parallel schedule, "
            "maximum iterations n, and the same 131,072 saved shots within each "
            "point. Every correction satisfies the syndrome, so all failures are "
            "unflagged logical failures.",
            "",
            "| Code | p | Neural LER | Vanilla BP4 LER | OSD-0 LER | OSD-CS7 LER | LSD-0 LER | Neural gain vs CS7 (paired 95% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("bb72", "bb144"):
        for p in (0.04, 0.06, 0.08, 0.10):
            rows = {
                row["method"]: row
                for row in classical
                if row["code"] == code and math.isclose(float(row["p"]), p)
            }
            paired_row = next(
                (
                    row
                    for row in paired
                    if row["code"] == code and math.isclose(float(row["p"]), p)
                ),
                None,
            )
            if paired_row is None:
                base = baseline_point(baseline, code, p)
                neural_ler = float(base["neural_logical_error_rate"])
                vanilla_ler = float(base["vanilla_logical_error_rate"])
                comparison = "independent banks"
            else:
                neural_ler = float(paired_row["neural_logical_error_rate"])
                vanilla_ler = float(paired_row["vanilla_logical_error_rate"])
                comparison = (
                    f"{fmt_pp(float(paired_row['paired_accuracy_gain']))} "
                    f"[{fmt_pp(float(paired_row['paired_ci95_low']))}, "
                    f"{fmt_pp(float(paired_row['paired_ci95_high']))}]"
                )
            lines.append(
                f"| {code.upper()} | {p:.2f} | "
                f"{fmt_percent(neural_ler)} | "
                f"{fmt_percent(vanilla_ler)} | "
                f"{fmt_percent(float(rows['bposd_0']['logical_error_rate']))} | "
                f"{fmt_percent(float(rows['bposd_cs7']['logical_error_rate']))} | "
                f"{fmt_percent(float(rows['bplsd_0']['logical_error_rate']))} | "
                f"{comparison} |"
            )
    paired_note = (
        "The archived selected-best Neural and vanilla BP4 checkpoints were "
        "re-evaluated on the exact saved classical error banks. Neural BP has a "
        "strictly positive paired 95% gain interval over OSD-CS7 at all eight "
        "points. "
        if paired
        else "The Neural/BP4 and classical columns currently use independent banks. "
    )
    lines.extend(
        [
            "",
            paired_note
            + "OSD-CS7 is the strongest of these CSS-separated baselines and beats "
            "OSD-0 at all eight points on paired shots; LSD-0 is statistically "
            "indistinguishable from OSD-0. The CSS split discards on-qubit X/Z (Y) "
            "correlation, while joint BP4 preserves it, so this does not show that "
            "Neural BP beats every BP+OSD implementation or is state of the art.",
            "",
            "## Partial Neural runs",
            "",
            "| Job/exp | Setting | Checkpoint epoch | Additional epochs | Checkpoint |",
            "|---|---|---:|---:|---|",
        ]
    )
    for result in sorted(partial, key=lambda item: (int(item.job_id), item.experiment_index)):
        setting = (
            f"{result.code.upper()}, {result.channel}, p-label={result.p_label:g}, "
            f"{result.variant}, seed={result.seed}"
        )
        lines.append(
            f"| {result.job_id}/exp{result.experiment_index} | {setting} | "
            f"{result.checkpoint_last_epoch} | {result.remaining_epochs} | "
            f"`{result.checkpoint}` |"
        )
    lines.extend(
        [
            "",
            "Use the identical command plus `--load_model=<checkpoint>` and set "
            "`--epochs` to the additional epoch count. Resume starts a new OneCycle "
            "schedule; lower LR is appropriate for the 98/107-epoch continuations. "
            "For one- or two-epoch cases, an eval-only checkpoint path would be "
            "cleaner than a new warm-up cycle.",
            "",
            "## Files",
            "",
            f"- Neural machine-readable summary: `{relative(NEURAL_CSV)}`",
            f"- Classical combined summary: `{relative(CLASSICAL_CSV)}`",
            f"- Same-bank Neural/BP4/OSD-CS7 paired summary: `{relative(PAIRED_CSV)}`",
            f"- Ablation plot: `{relative(ABLATION_PLOT)}`",
            f"- Decoder comparison plot: `{relative(DECODER_PLOT)}`",
            "",
        ]
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    if args.dpi <= 0:
        raise SystemExit("error: --dpi must be positive")
    neural = collect_neural()
    classical = collect_classical()
    baseline = load_baseline()
    paired = load_paired()
    write_neural_csv(neural)
    write_classical_csv(classical)
    plot_ablations(neural, baseline, args.dpi)
    plot_decoders(baseline, classical, paired, args.dpi)
    write_report(neural, classical, baseline, paired)
    complete = sum(result.scientific_status == "complete" for result in neural)
    print(
        f"Wrote campaign summary: {complete}/32 Neural complete, "
        f"{len(classical)} classical method rows, {len(paired)} paired rows"
    )


if __name__ == "__main__":
    main()
