#!/usr/bin/env python3
"""Summarize circuit-level BB Neural-BP runs evaluated without OSD.

This campaign is deliberately kept separate from the OSD-selected circuit
campaign.  A raw decode is successful only when the hard BP decision satisfies
the detector syndrome and has the correct logical observable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import summarize_bb_circuit_campaign as common


REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY / "results" / "bb" / "circuit" / "no_osd"
ANALYSIS_ROOT = REPOSITORY / "results" / "analysis"
PLOT_ROOT = REPOSITORY / "results" / "plots"
CSV_PATH = ANALYSIS_ROOT / "bb_circuit_no_osd_2026_09.csv"
PARTIAL_CSV_PATH = ANALYSIS_ROOT / "bb_circuit_no_osd_2026_09_partial.csv"
REPORT_PATH = ANALYSIS_ROOT / "bb_circuit_no_osd_2026_09.md"
PLOT_PATH = PLOT_ROOT / "bb_circuit_no_osd_2026_09.png"
COMBINED_PLOT_PATH = PLOT_ROOT / "bb_circuit_raw_vs_osd_2026_09.png"
Z_95 = 1.959963984540054

RAW_FIELDS = [
    "job_id",
    "experiment_index",
    "purpose",
    "variant",
    "scientific_status",
    "launcher_exit_code",
    "code",
    "n",
    "k",
    "d",
    "circuit_schema_version",
    "circuit_noise_model",
    "gate_error_rate",
    "measurement_error_rate",
    "idle_error_rate",
    "rounds",
    "detector_frames",
    "num_detectors",
    "num_mechanisms",
    "num_edges",
    "num_orbits",
    "iterations",
    "hidden_dim",
    "orbit_embedding_dim",
    "sharing",
    "normalisation",
    "residual_scale",
    "relaxation_delta",
    "deep_supervision_weight",
    "syndrome_loss_weight",
    "logical_loss_weight",
    "mechanism_loss_weight",
    "learning_rate",
    "trainable_parameters",
    "seed",
    "requested_epochs",
    "completed_epochs",
    "batch_size",
    "batches_per_epoch",
    "training_shots_per_epoch",
    "eval_batches",
    "final_eval_batches",
    "best_epoch",
    "selection_metric",
    "first_train_loss",
    "last_train_loss",
    "minimum_train_loss",
    "minimum_train_loss_epoch",
    "maximum_validation_neural_accuracy",
    "maximum_validation_neural_accuracy_epoch",
    "mean_epoch_seconds",
    "wall_minutes",
    "final_shots",
    "neural_accuracy",
    "neural_logical_error_rate",
    "vanilla_accuracy",
    "vanilla_logical_error_rate",
    "neural_syndrome_convergence",
    "vanilla_syndrome_convergence",
    "neural_flagged_failure_rate",
    "neural_unflagged_logical_failure_rate",
    "raw_paired_gain",
    "raw_paired_gain_ci95_halfwidth",
    "raw_paired_ci95_low",
    "raw_paired_ci95_high",
    "raw_rescued",
    "raw_harmed",
    "osd_shots",
    "source_resdir",
    "output_directory",
    "latest_checkpoint",
    "selected_checkpoint",
]

PARTIAL_FIELDS = [
    "job_id",
    "experiment_index",
    "code",
    "gate_error_rate",
    "measurement_error_rate",
    "idle_error_rate",
    "seed",
    "requested_epochs",
    "completed_epochs",
    "remaining_epochs",
    "validation_evaluations",
    "best_epoch",
    "best_validation_raw_gain",
    "has_latest_checkpoint",
    "has_selected_checkpoint",
    "stop_reason",
    "source_resdir",
    "output_directory",
]


def is_raw_config(config: dict[str, Any]) -> bool:
    return (
        config.get("architecture") == "bb_neural_bp_circuit"
        and config.get("checkpoint_selection_metric") == "neural_paired_gain"
    )


def requested_epochs(history: dict[str, Any]) -> int:
    return sum(int(phase.get("epochs", 0)) for phase in history.get("phases", []))


def is_complete(history: dict[str, Any]) -> bool:
    train = history.get("train", [])
    return (
        isinstance(history.get("final"), dict)
        and bool(train)
        and bool(history.get("eval"))
        and len(train) == requested_epochs(history)
    )


def parse_complete(path: Path, history: dict[str, Any]) -> dict[str, Any]:
    final = history["final"]
    if int(final.get("osd_shots", -1)) != 0:
        raise ValueError(f"Raw campaign unexpectedly contains OSD shots: {path}")
    parsed = asdict(common.parse_history(path))
    parsed["vanilla_syndrome_convergence"] = float(final["vanilla_converged"])
    return {field: parsed[field] for field in RAW_FIELDS}


def parse_partial(path: Path, history: dict[str, Any]) -> dict[str, Any]:
    config = history["config"]
    train = history.get("train", [])
    evaluations = history.get("eval", [])
    resdir = common.enclosing_resdir(path)
    experiment_index, _ = common.find_command(resdir, config)
    requested = requested_epochs(history)
    gains = [
        float(row["paired_gain"])
        for row in evaluations
        if row.get("paired_gain") is not None
    ]
    job_id = resdir.name.removeprefix("resdir_")
    stop_reason = (
        "Slurm time limit"
        if (resdir / "interrupted.txt").is_file()
        else "missing selected-best final evaluation"
    )
    values = {
        "job_id": job_id,
        "experiment_index": experiment_index,
        "code": str(config["code"]),
        "gate_error_rate": float(config["gate_error_rate"]),
        "measurement_error_rate": float(config["measurement_error_rate"]),
        "idle_error_rate": float(config["idle_error_rate"]),
        "seed": int(config["seed"]),
        "requested_epochs": requested,
        "completed_epochs": len(train),
        "remaining_epochs": max(0, requested - len(train)),
        "validation_evaluations": len(evaluations),
        "best_epoch": int(history.get("best_epoch", -1)),
        "best_validation_raw_gain": max(gains) if gains else None,
        "has_latest_checkpoint": path.with_name("model.pt").is_file(),
        "has_selected_checkpoint": path.with_name("best_model.pt").is_file(),
        "stop_reason": stop_reason,
        "source_resdir": common.relative(resdir),
        "output_directory": common.relative(path.parent),
    }
    return {field: values[field] for field in PARTIAL_FIELDS}


def collect_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS_ROOT.rglob("history.json")):
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {path}: {error}") from error
        config = history.get("config", {})
        if not isinstance(config, dict) or not is_raw_config(config):
            continue
        if is_complete(history):
            complete_rows.append(parse_complete(path, history))
        else:
            partial_rows.append(parse_partial(path, history))
    if not complete_rows and not partial_rows:
        raise ValueError(f"No no-OSD circuit-level BB histories under {RESULTS_ROOT}")

    def key(row: dict[str, Any]) -> tuple[int, float, int]:
        return (
            int(str(row["code"])[2:]),
            float(row["gate_error_rate"]),
            int(row["seed"]),
        )

    return sorted(complete_rows, key=key), sorted(partial_rows, key=key)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def exact_paired_sign_p_value(rescued: int, harmed: int) -> float:
    """Two-sided exact sign/McNemar p-value over discordant paired shots."""

    discordant = rescued + harmed
    if discordant == 0:
        return 1.0
    tail = (
        sum(math.comb(discordant, index) for index in range(min(rescued, harmed) + 1))
        / 2**discordant
    )
    return min(1.0, 2.0 * tail)


def write_report(
    rows: list[dict[str, Any]], partial_rows: list[dict[str, Any]]
) -> None:
    total_shots = sum(int(row["final_shots"]) for row in rows)
    unflagged_failures = sum(
        round(
            float(row["neural_unflagged_logical_failure_rate"])
            * int(row["final_shots"])
        )
        for row in rows
    )
    significant = [row for row in rows if float(row["raw_paired_ci95_low"]) > 0.0]
    exact_significant = [
        row
        for row in rows
        if exact_paired_sign_p_value(int(row["raw_rescued"]), int(row["raw_harmed"]))
        < 0.05
    ]
    lines = [
        "# BB circuit-level Neural BP without OSD (September 2026)",
        "",
        "This report is generated from the raw-decoder campaign under "
        "`results/bb/circuit/no_osd/`. It is kept separate from the OSD-selected "
        "campaign: every validation and final point here has `osd_shots=0`, and "
        "the selected checkpoint maximizes paired Neural-BP minus vanilla-BP raw "
        "accuracy.",
        "",
        "## Inventory and configuration",
        "",
        f"- Histories: **{len(rows) + len(partial_rows)}** = **{len(rows)} complete** + **{len(partial_rows)} partial**",
        f"- Fresh selected-best evaluation: **{total_shots:,} shots** total ({int(rows[0]['final_shots']):,} per point)",
        "- Codes: BB72 `[[72,12,6]]`, 6 rounds; BB144 `[[144,12,12]]`, 12 rounds",
        "- Noise: circuit-schema-v2 legacy profile, gate/readout `q=p`, idle error 0",
        "- Decoder: orbit-shared Neural normalized min-sum BP2, T=12; no OSD or other repair",
        "- Training: 100 epochs, learning rate 3e-4; one training seed per `(code,p)` point",
        "",
        "A raw decode counts as correct only if the hard correction both satisfies "
        "the detector syndrome and lies in the correct logical sector.",
        "",
        "## Selected-best final results",
        "",
        "| Code | p | Best epoch | Neural raw success | Vanilla raw success | Neural convergence | Flagged | Unflagged logical failure | Rescued / harmed | Paired gain (95% half-width) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{str(row['code']).upper()} | {float(row['gate_error_rate']):.3f} | "
            f"{int(row['best_epoch'])} | {percent(float(row['neural_accuracy']))} | "
            f"{percent(float(row['vanilla_accuracy']))} | "
            f"{percent(float(row['neural_syndrome_convergence']))} | "
            f"{percent(float(row['neural_flagged_failure_rate']))} | "
            f"{percent(float(row['neural_unflagged_logical_failure_rate']))} | "
            f"{int(row['raw_rescued'])} / {int(row['raw_harmed'])} | "
            f"{100.0 * float(row['raw_paired_gain']):+.3f} ± "
            f"{100.0 * float(row['raw_paired_gain_ci95_halfwidth']):.3f} pp |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The stored normal-approximation paired interval is strictly positive at **{len(significant)}/{len(rows)}** points, while a two-sided exact paired sign/McNemar test is significant at **{len(exact_significant)}/{len(rows)}**. In particular, BB144 p=0.004 has only 4 rescued versus 0 harmed shots (exact p=0.125), so its tiny positive normal interval is under-resolved.",
            f"- Across all **{total_shots:,}** final shots, only **{unflagged_failures}** Neural-BP failure occurred after syndrome convergence. Raw success is therefore almost identical to syndrome convergence; the dominant failure is an invalid hard correction, not a wrong logical sector after convergence.",
            "- BB144 effectively collapses for `p>=0.002`: raw success is 0.488%, 0%, and 0.098%. BB72 also has 93.408% and 88.403% flagged failures at p=0.003 and 0.004.",
            "- The BB72 curve is strongly non-monotone (`p=0.003` is much worse than `p=0.004`). With only one training seed per point, this indicates optimization or seed sensitivity and prevents a threshold interpretation.",
            "- Decreasing training losses do not guarantee syndrome-valid hard decisions. The next architectural target should be syndrome-constrained output/repair or a feasibility-focused objective; simply adding higher-p points or more final shots will not fix this bottleneck.",
            "- Do not compare these checkpoints directly with the older OSD-selected checkpoints as an ablation: the selection metric and code commit differ. Use the within-run paired Neural-versus-vanilla result shown here.",
            "",
            "## Scope",
            "",
            "This is **not a threshold curve** and does not show that OSD can yet be removed. It is a diagnostic of the raw learned-BP stage. OSD in the older campaign was providing an essential global feasibility repair, especially for BB144.",
        ]
    )

    if partial_rows:
        lines.extend(
            [
                "",
                "## Partial runs (excluded from final results)",
                "",
                "| Job/exp | Code | p | Epochs | Best validation raw gain | Stop reason |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in partial_rows:
            best = row["best_validation_raw_gain"]
            shown = "n/a" if best is None else f"{100.0 * float(best):+.3f} pp"
            lines.append(
                f"| {row['job_id']}/{row['experiment_index']} | "
                f"{str(row['code']).upper()} | "
                f"{float(row['gate_error_rate']):.3f} | "
                f"{row['completed_epochs']}/{row['requested_epochs']} | "
                f"{shown} | {row['stop_reason']} |"
            )

    lines.extend(
        [
            "",
            f"Complete rows: [`{CSV_PATH.name}`]({CSV_PATH.name})",
            f"Partial rows: [`{PARTIAL_CSV_PATH.name}`]({PARTIAL_CSV_PATH.name})",
            f"Combined raw/OSD plot: [`{COMBINED_PLOT_PATH.name}`](../plots/{COMBINED_PLOT_PATH.name})",
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


def plot_results(rows: list[dict[str, Any]], dpi: int) -> None:
    cache = Path(tempfile.gettempdir()) / "theend_bb_circuit_raw_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib; install requirements.txt or omit "
            "--standalone-plot."
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_code.setdefault(str(row["code"]), []).append(row)
    for code_rows in by_code.values():
        code_rows.sort(key=lambda row: float(row["gate_error_rate"]))

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10.2,
            "legend.frameon": False,
        }
    )
    figure, (success_axis, gain_axis) = plt.subplots(
        1, 2, figsize=(12.2, 4.8), constrained_layout=True
    )
    colors = {"bb72": "#2563eb", "bb144": "#dc2626"}
    fallback = ["#0f766e", "#7c3aed", "#d97706"]

    ordered_codes = sorted(by_code.items(), key=lambda item: int(item[0][2:]))
    for code_index, (code, code_rows) in enumerate(ordered_codes):
        color = colors.get(code, fallback[code_index % len(fallback)])
        xs = [float(row["gate_error_rate"]) for row in code_rows]
        for label, key, linestyle, marker in (
            ("Neural BP2", "neural_accuracy", "-", "o"),
            ("Vanilla BP2", "vanilla_accuracy", "--", "s"),
        ):
            ys = [float(row[key]) for row in code_rows]
            lower: list[float] = []
            upper: list[float] = []
            for value, row in zip(ys, code_rows):
                shots = int(row["final_shots"])
                low, high = wilson_interval(round(value * shots), shots)
                lower.append(value - low)
                upper.append(high - value)
            success_axis.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                capsize=2.5,
                label=f"{code.upper()} {label}",
            )

        gains = [100.0 * float(row["raw_paired_gain"]) for row in code_rows]
        errors = [
            100.0 * float(row["raw_paired_gain_ci95_halfwidth"]) for row in code_rows
        ]
        gain_axis.errorbar(
            xs,
            gains,
            yerr=errors,
            color=color,
            marker="o",
            linewidth=1.8,
            capsize=3.0,
            label=code.upper(),
        )

    p_ticks = sorted({float(row["gate_error_rate"]) for row in rows})
    success_axis.set_xticks(p_ticks)
    success_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    success_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    success_axis.set_xlabel("Circuit gate/readout error rate, p=q")
    success_axis.set_ylabel("Raw block success rate")
    success_axis.set_title("Hard decode, without OSD")
    success_axis.set_ylim(bottom=-0.02)
    success_axis.grid(True, color="#d7dce2", linewidth=0.7)
    success_axis.legend(fontsize=8.5)

    gain_axis.axhline(0.0, color="#64748b", linewidth=1.0)
    gain_axis.set_xticks(p_ticks)
    gain_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    gain_axis.set_xlabel("Circuit gate/readout error rate, p=q")
    gain_axis.set_ylabel("Accuracy gain (percentage points)")
    gain_axis.set_title("Neural BP2 minus vanilla BP2")
    gain_axis.grid(True, color="#d7dce2", linewidth=0.7)
    gain_axis.legend(fontsize=8.5)

    figure.suptitle(
        "BB circuit-level Neural BP2: raw no-OSD diagnostic\n"
        "Selected-best, 4,096 paired shots per point; bars are 95% intervals",
        fontsize=12.3,
        fontweight="semibold",
    )
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standalone-plot",
        action="store_true",
        help="Also recreate the retired raw-only diagnostic plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    arguments = parser.parse_args()

    rows, partial_rows = collect_results()
    write_csv(CSV_PATH, rows, RAW_FIELDS)
    write_csv(PARTIAL_CSV_PATH, partial_rows, PARTIAL_FIELDS)
    write_report(rows, partial_rows)
    if arguments.standalone_plot:
        plot_results(rows, arguments.dpi)

    print(f"Wrote {CSV_PATH.relative_to(REPOSITORY)} ({len(rows)} complete rows)")
    print(
        f"Wrote {PARTIAL_CSV_PATH.relative_to(REPOSITORY)} "
        f"({len(partial_rows)} partial rows)"
    )
    print(f"Wrote {REPORT_PATH.relative_to(REPOSITORY)}")
    if arguments.standalone_plot:
        print(f"Wrote {PLOT_PATH.relative_to(REPOSITORY)}")


if __name__ == "__main__":
    main()
