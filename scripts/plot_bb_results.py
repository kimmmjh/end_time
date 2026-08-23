#!/usr/bin/env python3
"""Plot BB neural-BP results against the vanilla BP4 baseline."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


DEFAULT_INPUT = Path("results/analysis/bb_neural_bp_depolarizing_orbit.csv")
DEFAULT_OUTPUT = Path("results/plots/bb_neural_bp_vs_vanilla_bp.png")
REQUIRED_COLUMNS = {
    "code",
    "p",
    "neural_accuracy",
    "vanilla_accuracy",
    "paired_gain",
    "paired_ci95_low",
    "paired_ci95_high",
    "eval_samples",
}
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Result:
    code: str
    p: float
    neural_accuracy: float
    vanilla_accuracy: float
    paired_gain: float
    paired_ci95_low: float
    paired_ci95_high: float
    eval_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot BB neural-BP and vanilla-BP4 logical error rates, together "
            "with the paired neural accuracy gain."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Summary CSV to read (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG path to write (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output resolution in dots per inch (default: 240).",
    )
    return parser.parse_args()


def parse_float(row: dict[str, str], column: str, line_number: int) -> float:
    value = row.get(column, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: {column!r} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"line {line_number}: {column!r} must be finite, got {value!r}"
        )
    return parsed


def load_results(path: Path) -> list[Result]:
    if not path.is_file():
        raise ValueError(f"input CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "input CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        results: list[Result] = []
        seen: set[tuple[str, float]] = set()
        for line_number, row in enumerate(reader, start=2):
            code = row.get("code", "").strip()
            if not code:
                raise ValueError(f"line {line_number}: 'code' must not be empty")

            p = parse_float(row, "p", line_number)
            neural_accuracy = parse_float(row, "neural_accuracy", line_number)
            vanilla_accuracy = parse_float(row, "vanilla_accuracy", line_number)
            paired_gain = parse_float(row, "paired_gain", line_number)
            paired_ci95_low = parse_float(row, "paired_ci95_low", line_number)
            paired_ci95_high = parse_float(row, "paired_ci95_high", line_number)
            samples_float = parse_float(row, "eval_samples", line_number)

            if not 0.0 <= p <= 1.0:
                raise ValueError(f"line {line_number}: p must be in [0, 1]")
            for name, accuracy in (
                ("neural_accuracy", neural_accuracy),
                ("vanilla_accuracy", vanilla_accuracy),
            ):
                if not 0.0 <= accuracy <= 1.0:
                    raise ValueError(f"line {line_number}: {name} must be in [0, 1]")
            if not samples_float.is_integer() or samples_float <= 0:
                raise ValueError(
                    f"line {line_number}: eval_samples must be a positive integer"
                )
            if not paired_ci95_low <= paired_gain <= paired_ci95_high:
                raise ValueError(
                    f"line {line_number}: paired_gain must lie inside its 95% CI"
                )

            key = (code, p)
            if key in seen:
                raise ValueError(
                    f"line {line_number}: duplicate result for code={code!r}, p={p:g}"
                )
            seen.add(key)
            results.append(
                Result(
                    code=code,
                    p=p,
                    neural_accuracy=neural_accuracy,
                    vanilla_accuracy=vanilla_accuracy,
                    paired_gain=paired_gain,
                    paired_ci95_low=paired_ci95_low,
                    paired_ci95_high=paired_ci95_high,
                    eval_samples=int(samples_float),
                )
            )

    if not results:
        raise ValueError(f"input CSV has no data rows: {path}")
    return results


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def wilson_interval(proportion: float, samples: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson binomial interval."""
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


def group_by_code(results: Iterable[Result]) -> dict[str, list[Result]]:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        grouped.setdefault(result.code, []).append(result)
    return {
        code: sorted(points, key=lambda point: point.p)
        for code, points in sorted(grouped.items(), key=lambda item: natural_key(item[0]))
    }


def plot(results: list[Result], output: Path, dpi: int) -> None:
    grouped = group_by_code(results)
    palette = plt.get_cmap("tab10")
    colors = {code: palette(index) for index, code in enumerate(grouped)}

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "font.size": 10.5,
            "legend.frameon": False,
        }
    )
    figure, (error_axis, gain_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": (1.08, 1.0)},
        constrained_layout=True,
    )

    methods = (
        ("Neural BP", "neural_accuracy", "-", "o"),
        ("Vanilla BP4", "vanilla_accuracy", "--", "s"),
    )
    for code, points in grouped.items():
        xs = [point.p for point in points]
        for _, accuracy_field, linestyle, marker in methods:
            failures = [1.0 - getattr(point, accuracy_field) for point in points]
            intervals = [
                wilson_interval(failure, point.eval_samples)
                for failure, point in zip(failures, points, strict=True)
            ]
            lower_errors = [
                failure - interval[0]
                for failure, interval in zip(failures, intervals, strict=True)
            ]
            upper_errors = [
                interval[1] - failure
                for failure, interval in zip(failures, intervals, strict=True)
            ]
            error_axis.errorbar(
                xs,
                failures,
                yerr=[lower_errors, upper_errors],
                color=colors[code],
                linestyle=linestyle,
                linewidth=1.8,
                marker=marker,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                capsize=3,
                capthick=1.1,
                zorder=3,
            )

        gains = [100.0 * point.paired_gain for point in points]
        lower_gain_errors = [
            100.0 * (point.paired_gain - point.paired_ci95_low) for point in points
        ]
        upper_gain_errors = [
            100.0 * (point.paired_ci95_high - point.paired_gain) for point in points
        ]
        gain_axis.errorbar(
            xs,
            gains,
            yerr=[lower_gain_errors, upper_gain_errors],
            color=colors[code],
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3,
            capthick=1.1,
            label=code.upper(),
            zorder=3,
        )

    error_axis.set_yscale("log")
    error_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    error_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    error_axis.set_xlabel("Physical depolarizing error rate, p")
    error_axis.set_ylabel("Logical error rate")
    error_axis.set_title("Decoder performance")
    error_axis.grid(True, which="major", color="#d8dce2", linewidth=0.75)
    error_axis.grid(True, which="minor", color="#edf0f3", linewidth=0.55)

    code_handles = [
        Line2D([0], [0], color=colors[code], linewidth=2.2, label=code.upper())
        for code in grouped
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#4b5563",
            linestyle=linestyle,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            label=label,
        )
        for label, _, linestyle, marker in methods
    ]
    code_legend = error_axis.legend(
        handles=code_handles,
        title="Code",
        loc="upper left",
    )
    error_axis.add_artist(code_legend)
    error_axis.legend(handles=method_handles, title="Decoder", loc="lower right")

    gain_axis.axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.2, zorder=1)
    gain_axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    gain_axis.set_xlabel("Physical depolarizing error rate, p")
    gain_axis.set_ylabel("Neural BP accuracy gain (percentage points)")
    gain_axis.set_title("Paired improvement over vanilla BP4")
    gain_axis.grid(True, color="#e0e3e7", linewidth=0.7)
    gain_axis.legend(title="Code", loc="upper left")

    figure.suptitle(
        "BB neural belief propagation under code-capacity depolarizing noise",
        fontsize=13,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("error: --dpi must be positive")
    try:
        results = load_results(args.input)
        plot(results, args.output, args.dpi)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Wrote {args.output} from {len(results)} rows in {args.input}")


if __name__ == "__main__":
    main()
