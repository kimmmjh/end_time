#!/usr/bin/env python3
"""Plot circuit Relay BP, Neural Relay BP, Neural BP, and Neural BP+OSD.

All four curves use the original idle-free X/Z circuit experiment. Corrected
library BP/OSD results use a different task and remain in library_bp_osd.
The script name and merged raw/OSD CSV are retained for existing workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPOSITORY / "results" / "analysis"
PLOT_ROOT = REPOSITORY / "results/plots/bb/circuit/neural_bp"
DEFAULT_RAW_CSV = ANALYSIS_ROOT / "bb_circuit_no_osd_2026_09.csv"
DEFAULT_OSD_CSV = ANALYSIS_ROOT / "bb_circuit_campaign_2026_08.csv"
DEFAULT_MERGED_CSV = ANALYSIS_ROOT / "bb_circuit_raw_vs_osd_2026_09.csv"
DEFAULT_RELAY_CSV = ANALYSIS_ROOT / "bb_neural_relay_bb72_bb144_2026_09_15.csv"
DEFAULT_PLOT = PLOT_ROOT / "overview.png"
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


def primary_relay_rows(
    rows: list[dict[str, Any]],
    raw: dict[tuple[str, float], dict[str, Any]],
    osd: dict[tuple[str, float], dict[str, Any]],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Check saved final rates and shared DEMs against original histories."""
    normalized = []
    matched = {"legacy_raw": set(), "legacy_osd": set()}
    for row in rows:
        row = dict(row, gate_error_rate=row["p"])
        key = row["code"], float(row["p"])
        history = json.loads((REPOSITORY / row["source_history"]).read_text())
        config, final = history["config"], history["final"]
        if (config["code"] != key[0]
                or not close(config["gate_error_rate"], key[1])
                or config["graph_fingerprint"] != row["graph_fingerprint"]
                or config["idle_error_rate"] != 0
                or config["measurement_error_rate"] != config["gate_error_rate"]
                or config["circuit_schema_version"] != 2
                or config["circuit_noise_model"] != "legacy"
                or int(row["shots"]) != final["shots"]
                or final["osd_shots"] != 0
                or tuple(int(row[k]) for k in ("relay_legs", "iterations_per_leg", "solutions")) != (4, 12, 2)):
            raise ValueError(f"Relay configuration mismatch: {key}")
        for prefix in ("neural", "vanilla"):
            accuracy = float(row[f"{prefix}_accuracy"])
            if (not close(accuracy, final[f"{prefix}_accuracy"])
                    or not close(1 - accuracy, row[f"{prefix}_ler"])
                    or not close((1 - accuracy) * final["shots"], row[f"{prefix}_failures"])):
                raise ValueError(f"Relay final result mismatch: {key}, {prefix}")
        for tag, reference, accuracy_key, shots_key in (
            ("legacy_raw", raw, "neural_accuracy", "final_shots"),
            ("legacy_osd", osd, "neural_osd_accuracy", "osd_shots"),
        ):
            if key not in reference:
                continue
            source = reference[key]
            old = json.loads((REPOSITORY / row[f"{tag}_source"]).read_text())
            # Earlier schema-v2 histories predate the named noise-model field.
            # Their original circuit is also checked by the exact DEM fingerprint.
            old["config"].setdefault("circuit_noise_model", "legacy")
            for field in ("code", "graph_fingerprint", "circuit_schema_version", "circuit_noise_model",
                          "gate_error_rate", "measurement_error_rate", "idle_error_rate", "rounds", "seed"):
                if old["config"][field] != config[field]:
                    raise ValueError(f"Circuit mismatch: {key}, {tag}, {field}")
            if (not close(source[accuracy_key], old["final"][accuracy_key])
                    or not close(1 - float(source[accuracy_key]), row[f"{tag}_neural_ler"])
                    or int(source[shots_key]) != old["final"]["shots" if tag == "legacy_raw" else "osd_shots"]):
                raise ValueError(f"Neural reference mismatch: {key}, {tag}")
            matched[tag].add(key)
        normalized.append(row)
    if matched["legacy_raw"] != raw.keys() or matched["legacy_osd"] != osd.keys():
        raise ValueError("Missing Relay circuit references for a Neural BP point")
    return unique_index(normalized, "Relay")


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


def plot_rows(
    raw: dict[tuple[str, float], dict[str, Any]],
    osd: dict[tuple[str, float], dict[str, Any]],
    path: Path,
    dpi: int,
    relay: dict[tuple[str, float], dict[str, Any]] | None = None,
) -> None:
    """Plot all four methods on their own measured p-grids."""
    if relay is None:
        relay = primary_relay_rows(read_rows(DEFAULT_RELAY_CSV), raw, osd)
    cache = Path(tempfile.gettempdir()) / "theend_bb_raw_osd_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    ordered_codes = sorted({code for code, _ in raw.keys() | osd.keys()}, key=lambda code: int(code[2:]))
    if ordered_codes != ["bb72", "bb144"]:
        raise ValueError(f"Expected BB72 and BB144, got {ordered_codes}")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "legend.frameon": False})
    figure, neural_axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)
    methods = (
        (relay, "Relay BP", "vanilla_accuracy", "shots", "#64748b", "s", "--"),
        (relay, "Neural Relay BP", "neural_accuracy", "shots", "#059669", "^", "-"),
        (raw, "Neural BP", "neural_accuracy", "final_shots", "#2563eb", "o", "-"),
        (osd, "Neural BP + OSD-0", "neural_osd_accuracy", "osd_shots", "#9333ea", "D", "-"),
    )
    for column, code in enumerate(ordered_codes):
        axis = neural_axes[column]
        ticks = set()
        for source, label, accuracy_key, shots_key, color, marker, linestyle in methods:
            code_rows = [row for (family, p), row in sorted(source.items()) if family == code]
            if not code_rows:
                continue
            xs = [float(row["gate_error_rate"]) for row in code_rows]
            ticks.update(xs)
            ys, lower, upper, zero = failure_series(code_rows, accuracy_key, shots_key)
            axis.errorbar(xs, ys, yerr=[lower, upper], color=color,
                          linestyle=linestyle,
                          marker=marker, linewidth=1.8, markersize=5, capsize=2.5, label=label)
            for x_value, y_value, is_zero in zip(xs, ys, zero):
                if is_zero:
                    axis.scatter(x_value, y_value, marker="v", s=42, color=color, zorder=5)
        axis.set(yscale="log", ylim=(1e-5, 1.25),
                 xlabel="Gate / readout error probability p (idle error = 0)",
                 ylabel="X/Z block decoding failure (%)",
                 title=f"{code.upper()} | {int(code_rows[0]['rounds'])} noisy rounds | 4,096 shots/point")
        axis.set_xticks(sorted(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
        axis.grid(alpha=.2)
        axis.legend(loc="lower right", fontsize=9)
    figure.suptitle("BB circuit decoding | Relay BP and Neural BP\n"
                   "Same circuit / noise / XZ block metric | idle error = 0",
                   fontsize=13, fontweight="semibold", y=.985)
    figure.text(.5, .105, "Relay methods: up to 4 x 12 BP iterations (2-solution search). Neural BP: 12 iterations. "
                "Curves include completed final evaluations only.", ha="center", fontsize=9, color="#475569")
    figure.text(.5, .067, "Error bars: 95% Wilson intervals. Downward triangles: zero failures, displayed at 0.5/N. "
                "Model selection and training budgets differ.",
                ha="center", fontsize=9, color="#475569")
    figure.text(.5, .03, "Neural BP + OSD uses the original posterior-seeded wrapper and an OSD-selected checkpoint; "
                "raw vs OSD is not a same-checkpoint ablation.", ha="center", fontsize=9, color="#475569")
    figure.subplots_adjust(left=.075, right=.985, top=.80, bottom=.24, wspace=.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--osd-csv", type=Path, default=DEFAULT_OSD_CSV)
    parser.add_argument("--relay-csv", type=Path, default=DEFAULT_RELAY_CSV)
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--dpi", type=int, default=180)
    arguments = parser.parse_args()

    raw = primary_raw_rows(read_rows(arguments.raw_csv))
    osd = primary_osd_rows(read_rows(arguments.osd_csv))
    rows = merge_rows(raw, osd)
    relay = primary_relay_rows(read_rows(arguments.relay_csv), raw, osd)
    write_merged_csv(rows, arguments.merged_csv)
    plot_rows(raw, osd, arguments.output, arguments.dpi, relay)
    print(f"Compared {len(rows)} common raw/OSD point(s)")
    print(f"Plotted {len(relay)} points each for Relay / Neural Relay, {len(raw)} Neural BP and {len(osd)} Neural+OSD points")
    print(f"CSV: {arguments.merged_csv}")
    print(f"Plot: {arguments.output}")


if __name__ == "__main__":
    main()
