#!/usr/bin/env python3
"""Compare raw and OSD-assisted circuit-level BB Neural-BP campaigns."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPOSITORY / "results" / "analysis"
PLOT_ROOT = REPOSITORY / "results" / "plots"
DEFAULT_RAW_CSV = ANALYSIS_ROOT / "bb_circuit_no_osd_2026_09.csv"
DEFAULT_OSD_CSV = ANALYSIS_ROOT / "bb_circuit_campaign_2026_08.csv"
DEFAULT_MERGED_CSV = ANALYSIS_ROOT / "bb_circuit_raw_vs_osd_2026_09.csv"
DEFAULT_PLOT = PLOT_ROOT / "bb_circuit_raw_vs_osd_2026_09.png"
Z_95 = 1.959963984540054

MERGED_FIELDS = [
    "code",
    "n",
    "k",
    "d",
    "gate_error_rate",
    "measurement_error_rate",
    "idle_error_rate",
    "final_shots",
    "raw_job_id",
    "raw_best_epoch",
    "raw_selection_metric",
    "raw_neural_accuracy",
    "raw_neural_logical_error_rate",
    "raw_vanilla_accuracy",
    "raw_vanilla_logical_error_rate",
    "raw_neural_syndrome_convergence",
    "raw_neural_flagged_failure_rate",
    "raw_neural_unflagged_logical_failure_rate",
    "raw_neural_vs_vanilla_gain",
    "raw_gain_ci95_halfwidth",
    "raw_source_resdir",
    "osd_job_id",
    "osd_best_epoch",
    "osd_selection_metric",
    "osd_checkpoint_raw_neural_accuracy",
    "osd_checkpoint_raw_neural_logical_error_rate",
    "osd_shots",
    "neural_osd_accuracy",
    "neural_osd_logical_error_rate",
    "vanilla_osd_accuracy",
    "vanilla_osd_logical_error_rate",
    "neural_vs_vanilla_osd_gain",
    "osd_gain_ci95_halfwidth",
    "same_checkpoint_neural_osd_lift",
    "cross_campaign_neural_accuracy_gap",
    "osd_source_resdir",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def primary_raw_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, float], dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["scientific_status"] == "complete"
        and row["purpose"] == "baseline"
        and row["variant"] == "reference"
        and row["selection_metric"] == "neural_paired_gain"
        and int(row["osd_shots"]) == 0
    ]
    return unique_index(selected, "raw")


def primary_osd_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, float], dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["scientific_status"] == "complete"
        and row["purpose"] == "baseline"
        and row["variant"] == "reference"
        and row["selection_metric"] == "neural_osd_paired_gain"
        and int(row["osd_shots"]) > 0
        and row["osd_method"] == "OSD_0"
    ]
    return unique_index(selected, "OSD")


def unique_index(
    rows: list[dict[str, str]], label: str
) -> dict[tuple[str, float], dict[str, str]]:
    result: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        key = (row["code"], float(row["gate_error_rate"]))
        if key in result:
            raise ValueError(f"Duplicate {label} primary point: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"No complete primary {label} rows found")
    return result


def close(left: str, right: str) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def merge_rows(
    raw: dict[tuple[str, float], dict[str, str]],
    osd: dict[tuple[str, float], dict[str, str]],
) -> list[dict[str, Any]]:
    common_keys = sorted(
        raw.keys() & osd.keys(), key=lambda item: (int(item[0][2:]), item[1])
    )
    if not common_keys:
        raise ValueError("Raw and OSD campaigns have no common (code,p) points")

    merged: list[dict[str, Any]] = []
    comparable_fields = (
        "n",
        "k",
        "d",
        "circuit_schema_version",
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
        "normalisation",
        "residual_scale",
        "relaxation_delta",
        "deep_supervision_weight",
        "syndrome_loss_weight",
        "logical_loss_weight",
        "mechanism_loss_weight",
        "learning_rate",
        "seed",
        "requested_epochs",
        "final_shots",
    )
    for key in common_keys:
        raw_row = raw[key]
        osd_row = osd[key]
        for field in comparable_fields:
            if not close(raw_row[field], osd_row[field]):
                raise ValueError(f"Campaign mismatch for {key}: {field}")
        for field in ("code", "circuit_noise_model", "sharing"):
            if raw_row[field] != osd_row[field]:
                raise ValueError(f"Campaign mismatch for {key}: {field}")
        if not close(raw_row["vanilla_accuracy"], osd_row["vanilla_accuracy"]):
            raise ValueError(f"Final vanilla raw accuracy differs for {key}")

        raw_neural = float(raw_row["neural_accuracy"])
        osd_checkpoint_raw_neural = float(osd_row["neural_accuracy"])
        neural_osd = float(osd_row["neural_osd_accuracy"])
        merged.append(
            {
                "code": raw_row["code"],
                "n": int(raw_row["n"]),
                "k": int(raw_row["k"]),
                "d": int(raw_row["d"]),
                "gate_error_rate": float(raw_row["gate_error_rate"]),
                "measurement_error_rate": float(raw_row["measurement_error_rate"]),
                "idle_error_rate": float(raw_row["idle_error_rate"]),
                "final_shots": int(raw_row["final_shots"]),
                "raw_job_id": raw_row["job_id"],
                "raw_best_epoch": int(raw_row["best_epoch"]),
                "raw_selection_metric": raw_row["selection_metric"],
                "raw_neural_accuracy": raw_neural,
                "raw_neural_logical_error_rate": 1.0 - raw_neural,
                "raw_vanilla_accuracy": float(raw_row["vanilla_accuracy"]),
                "raw_vanilla_logical_error_rate": float(
                    raw_row["vanilla_logical_error_rate"]
                ),
                "raw_neural_syndrome_convergence": float(
                    raw_row["neural_syndrome_convergence"]
                ),
                "raw_neural_flagged_failure_rate": float(
                    raw_row["neural_flagged_failure_rate"]
                ),
                "raw_neural_unflagged_logical_failure_rate": float(
                    raw_row["neural_unflagged_logical_failure_rate"]
                ),
                "raw_neural_vs_vanilla_gain": float(raw_row["raw_paired_gain"]),
                "raw_gain_ci95_halfwidth": float(
                    raw_row["raw_paired_gain_ci95_halfwidth"]
                ),
                "raw_source_resdir": raw_row["source_resdir"],
                "osd_job_id": osd_row["job_id"],
                "osd_best_epoch": int(osd_row["best_epoch"]),
                "osd_selection_metric": osd_row["selection_metric"],
                "osd_checkpoint_raw_neural_accuracy": osd_checkpoint_raw_neural,
                "osd_checkpoint_raw_neural_logical_error_rate": (
                    1.0 - osd_checkpoint_raw_neural
                ),
                "osd_shots": int(osd_row["osd_shots"]),
                "neural_osd_accuracy": neural_osd,
                "neural_osd_logical_error_rate": 1.0 - neural_osd,
                "vanilla_osd_accuracy": float(osd_row["vanilla_osd_accuracy"]),
                "vanilla_osd_logical_error_rate": float(
                    osd_row["vanilla_osd_logical_error_rate"]
                ),
                "neural_vs_vanilla_osd_gain": float(osd_row["osd_paired_gain"]),
                "osd_gain_ci95_halfwidth": float(
                    osd_row["osd_paired_gain_ci95_halfwidth"]
                ),
                "same_checkpoint_neural_osd_lift": (
                    neural_osd - osd_checkpoint_raw_neural
                ),
                "cross_campaign_neural_accuracy_gap": neural_osd - raw_neural,
                "osd_source_resdir": osd_row["source_resdir"],
            }
        )
    return merged


def write_merged_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def failure_series(
    rows: list[dict[str, Any]], accuracy_key: str, shots_key: str
) -> tuple[list[float], list[float], list[float], list[bool]]:
    shown: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    zero_failures: list[bool] = []
    for row in rows:
        shots = int(row[shots_key])
        failures = round((1.0 - float(row[accuracy_key])) * shots)
        rate = failures / shots
        low, high = wilson_interval(failures, shots)
        display = rate if failures else 0.5 / shots
        shown.append(display)
        lower_errors.append(0.0 if failures == 0 else max(0.0, display - low))
        upper_errors.append(max(0.0, high - display))
        zero_failures.append(failures == 0)
    return shown, lower_errors, upper_errors, zero_failures


def plot_rows(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    cache = Path(tempfile.gettempdir()) / "theend_bb_raw_osd_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_code.setdefault(str(row["code"]), []).append(row)
    for code_rows in by_code.values():
        code_rows.sort(key=lambda row: float(row["gate_error_rate"]))
    ordered_codes = sorted(by_code, key=lambda code: int(code[2:]))
    if len(ordered_codes) != 2:
        raise ValueError(f"Expected two code families, got {ordered_codes}")

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10.0,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    p_ticks = sorted({float(row["gate_error_rate"]) for row in rows})
    methods = (
        (
            "Raw Neural BP2 (raw-selected)",
            "raw_neural_accuracy",
            "final_shots",
            "#2563eb",
            "-",
            "o",
        ),
        (
            "Raw vanilla BP2",
            "raw_vanilla_accuracy",
            "final_shots",
            "#64748b",
            "--",
            "s",
        ),
        (
            "Neural BP2 + OSD-0 (OSD-selected)",
            "neural_osd_accuracy",
            "osd_shots",
            "#7c3aed",
            "-",
            "D",
        ),
        (
            "Vanilla BP2 + OSD-0",
            "vanilla_osd_accuracy",
            "osd_shots",
            "#d97706",
            "--",
            "^",
        ),
    )
    for axis, code in zip(axes[0], ordered_codes):
        code_rows = by_code[code]
        xs = [float(row["gate_error_rate"]) for row in code_rows]
        for label, accuracy_key, shots_key, color, linestyle, marker in methods:
            ys, lower, upper, zero = failure_series(code_rows, accuracy_key, shots_key)
            axis.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                capsize=2.5,
                label=label,
            )
            for x_value, y_value, is_zero in zip(xs, ys, zero):
                if is_zero:
                    axis.scatter(
                        [x_value], [y_value], marker="v", s=42, color=color, zorder=5
                    )
        axis.set_yscale("log")
        axis.set_xticks(p_ticks)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        axis.yaxis.set_major_formatter(
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
        axis.set_xlabel("Circuit gate/readout error rate, p=q")
        axis.set_ylabel("Total decoding failure rate")
        axis.set_title(code.upper())
        axis.grid(True, which="major", color="#d7dce2", linewidth=0.7)
        axis.grid(True, which="minor", color="#edf0f3", linewidth=0.5)

    axes[0, 0].text(
        0.98,
        0.035,
        "▼: zero failures displayed at 0.5/N",
        transform=axes[0, 0].transAxes,
        fontsize=8.0,
        color="#475569",
        ha="right",
    )

    colors = {"bb72": "#2563eb", "bb144": "#dc2626"}
    gain_panels = (
        (
            axes[1, 0],
            "raw_neural_vs_vanilla_gain",
            "raw_gain_ci95_halfwidth",
            "Raw Neural BP2 minus raw vanilla BP2",
        ),
        (
            axes[1, 1],
            "neural_vs_vanilla_osd_gain",
            "osd_gain_ci95_halfwidth",
            "Neural+OSD-0 minus vanilla+OSD-0",
        ),
    )
    for axis, gain_key, error_key, title in gain_panels:
        axis.axhline(0.0, color="#64748b", linewidth=1.0)
        for code in ordered_codes:
            code_rows = by_code[code]
            xs = [float(row["gate_error_rate"]) for row in code_rows]
            gains = [100.0 * float(row[gain_key]) for row in code_rows]
            errors = [100.0 * float(row[error_key]) for row in code_rows]
            axis.errorbar(
                xs,
                gains,
                yerr=errors,
                color=colors.get(code, "#0f766e"),
                marker="o",
                linewidth=1.8,
                capsize=3.0,
                label=code.upper(),
            )
        axis.set_xticks(p_ticks)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        axis.set_xlabel("Circuit gate/readout error rate, p=q")
        axis.set_ylabel("Paired accuracy gain (percentage points)")
        axis.set_title(title)
        axis.grid(True, color="#d7dce2", linewidth=0.7)
        axis.legend(fontsize=8.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        fontsize=9.0,
    )
    figure.suptitle(
        "BB circuit-level Neural BP2: raw versus OSD-0\n"
        "Common p-grid, 4,096 shots/point; raw and OSD pipelines use separately selected checkpoints",
        y=0.985,
        fontsize=13.0,
        fontweight="semibold",
    )
    figure.subplots_adjust(top=0.80, bottom=0.08, hspace=0.34, wspace=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--osd-csv", type=Path, default=DEFAULT_OSD_CSV)
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--dpi", type=int, default=180)
    arguments = parser.parse_args()

    raw = primary_raw_rows(read_rows(arguments.raw_csv))
    osd = primary_osd_rows(read_rows(arguments.osd_csv))
    rows = merge_rows(raw, osd)
    write_merged_csv(rows, arguments.merged_csv)
    plot_rows(rows, arguments.output, arguments.dpi)
    print(f"Compared {len(rows)} common raw/OSD point(s)")
    print(f"CSV: {arguments.merged_csv}")
    print(f"Plot: {arguments.output}")


if __name__ == "__main__":
    main()
