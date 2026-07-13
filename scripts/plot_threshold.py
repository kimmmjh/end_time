#!/usr/bin/env python3
"""Plot logical error rate versus physical error rate from training logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

COMMAND_RE = re.compile(r"Executed Command: (.*)")
METRIC_RE = re.compile(
    r"\[Epoch (\d+)\] Loss: ([0-9.eE+-]+) \| Accuracy: ([0-9.eE+-]+) "
    r"\(±([0-9.eE+-]+)\)(?: \| Eval Samples: (\d+))?"
)


@dataclass
class Record:
    source: Path
    epoch: int
    loss: float
    accuracy: float
    eval_samples: int | None
    L: int
    p: float
    q: float
    channels: str
    depths: str
    decoder: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot threshold curves from resdir/log files."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Result dirs or log files.")
    parser.add_argument("--out", type=Path, default=Path("threshold_plot.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--metric",
        choices=["final", "best"],
        default="final",
        help="Use final epoch accuracy or best accuracy in each log.",
    )
    parser.add_argument(
        "--group",
        choices=["L", "L_arch", "L_decoder", "L_arch_decoder"],
        default="L_arch_decoder",
        help=(
            "Curve grouping. The default keeps neural architectures and "
            "PyMatching results separate."
        ),
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis instead of log scale.",
    )
    parser.add_argument(
        "--title",
        default="Threshold Plot",
        help="Plot title.",
    )
    return parser.parse_args()


def flag(command: str, name: str, default: str | None = None) -> str | None:
    match = re.search(rf"--{re.escape(name)}(?:=|\s+)([^\s]+)", command)
    return match.group(1) if match else default


def multi_flag(command: str, name: str, next_name: str) -> str:
    match = re.search(
        rf"--{re.escape(name)}\s+(.+?)\s+--{re.escape(next_name)}", command
    )
    if not match:
        return "unknown"
    return "-".join(match.group(1).split())


def discover_inputs(paths: list[Path]) -> list[Path]:
    inputs: set[Path] = set()
    for path in paths:
        if path.is_file():
            inputs.add(path)
            continue

        found = list(path.rglob("log_exp_*.txt"))
        if found:
            inputs.update(found)
        else:
            inputs.update(path.rglob("training_log.txt"))
        inputs.update(path.rglob("pymatching*.csv"))

    return sorted(inputs)


def parse_log(path: Path, metric: str) -> Record | None:
    text = path.read_text(errors="replace")
    command_match = COMMAND_RE.search(text)
    if command_match is None:
        return None

    metrics = [
        (
            int(epoch),
            float(loss),
            float(accuracy),
            int(eval_samples) if eval_samples else None,
        )
        for epoch, loss, accuracy, _std, eval_samples in METRIC_RE.findall(text)
    ]
    if not metrics:
        return None

    epoch, loss, accuracy, eval_samples = (
        metrics[-1] if metric == "final" else max(metrics, key=lambda item: item[2])
    )

    command = command_match.group(1)
    L = int(flag(command, "L", "0"))
    p = float(flag(command, "p", "nan"))
    q = float(flag(command, "measurement_error_rate", str(p)))
    channels = multi_flag(command, "channels", "depths")
    depths = multi_flag(command, "depths", "lr")
    decoder = flag(command, "decoder")
    if decoder is None:
        decoder = "pymatching" if "pymatching_threshold.py" in command else "neural"

    return Record(
        source=path,
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        eval_samples=eval_samples,
        L=L,
        p=p,
        q=q,
        channels=channels,
        depths=depths,
        decoder=decoder,
    )


def parse_csv(path: Path) -> list[Record]:
    records = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"p", "accuracy"}.issubset(fields) or not (
            "L" in fields or "label" in fields
        ):
            return records
        for row in reader:
            p = float(row["p"])
            samples_text = row.get("eval_samples") or row.get("shots")
            label = row.get("label") or ""
            if row.get("L"):
                L = int(row["L"])
            else:
                lattice_match = re.search(r"\bL=(\d+)", label)
                if lattice_match is None:
                    continue
                L = int(lattice_match.group(1))
            decoder = row.get("decoder") or ""
            if not decoder:
                decoder = "pymatching" if label.startswith("PyMatching ") else "neural"
            records.append(
                Record(
                    source=path,
                    epoch=int(row.get("epoch") or 0),
                    loss=float(row.get("loss") or "nan"),
                    accuracy=float(row["accuracy"]),
                    eval_samples=int(samples_text) if samples_text else None,
                    L=L,
                    p=p,
                    q=float(row.get("q") or p),
                    channels=row.get("channels") or "n/a",
                    depths=row.get("depths") or "n/a",
                    decoder=decoder.lower(),
                )
            )
    return records


def parse_input(path: Path, metric: str) -> list[Record]:
    if path.suffix.lower() == ".csv":
        return parse_csv(path)
    record = parse_log(path, metric)
    return [] if record is None else [record]


def decoder_name(decoder: str) -> str:
    names = {"neural": "NN", "pymatching": "PyMatching"}
    return names.get(decoder.lower(), decoder)


def label_for(record: Record, group: str) -> str:
    include_arch = (
        group in {"L_arch", "L_arch_decoder"}
        and record.decoder.lower() != "pymatching"
    )
    if include_arch:
        label = f"L={record.L} ch={record.channels} d={record.depths}"
    else:
        label = f"L={record.L}"
    if group in {"L_decoder", "L_arch_decoder"}:
        return f"{decoder_name(record.decoder)} {label}"
    return label


def aggregate(records: list[Record], group: str):
    buckets = defaultdict(list)
    for record in records:
        buckets[(label_for(record, group), record.p)].append(record)

    curves = defaultdict(list)
    rows = []
    for (label, p), items in buckets.items():
        sampled_items = [item for item in items if item.eval_samples]
        if len(sampled_items) == len(items):
            total = sum(item.eval_samples for item in sampled_items)
            correct = sum(
                round(item.accuracy * item.eval_samples) for item in sampled_items
            )
            accuracy = correct / total
            eval_samples = total
        else:
            accuracy = sum(item.accuracy for item in items) / len(items)
            eval_samples = None

        failure = 1.0 - accuracy
        if eval_samples:
            # One-standard-deviation Wilson interval. Unlike a symmetric
            # normal error bar, this remains non-negative for zero/rare
            # failures and therefore behaves properly on a logarithmic axis.
            half_count = 0.5 / eval_samples
            failure_for_plot = max(failure, half_count)
            denominator = 1.0 + 1.0 / eval_samples
            interval_center = (failure + 0.5 / eval_samples) / denominator
            interval_half_width = (
                math.sqrt(
                    max(failure * (1.0 - failure), 0.0) / eval_samples
                    + 0.25 / eval_samples**2
                )
                / denominator
            )
            interval_low = max(
                interval_center - interval_half_width,
                half_count,
            )
            interval_high = max(
                interval_center + interval_half_width,
                failure_for_plot,
            )
            yerr_low = max(failure_for_plot - interval_low, 0.0)
            yerr_high = max(interval_high - failure_for_plot, 0.0)
        else:
            failure_for_plot = max(failure, 1e-12)
            yerr_low = 0.0
            yerr_high = 0.0

        point = {
            "label": label,
            "L": items[0].L if all(item.L == items[0].L for item in items) else None,
            "decoder": (
                items[0].decoder
                if all(item.decoder == items[0].decoder for item in items)
                else "mixed"
            ),
            "channels": (
                items[0].channels
                if all(item.channels == items[0].channels for item in items)
                else "mixed"
            ),
            "depths": (
                items[0].depths
                if all(item.depths == items[0].depths for item in items)
                else "mixed"
            ),
            "p": p,
            "failure": failure,
            "failure_for_plot": failure_for_plot,
            "accuracy": accuracy,
            "eval_samples": eval_samples,
            "num_runs": len(items),
            "yerr_low": yerr_low,
            "yerr_high": yerr_high,
        }
        curves[label].append(point)
        rows.append(point)

    for points in curves.values():
        points.sort(key=lambda item: item["p"])

    return curves, sorted(rows, key=lambda item: (item["label"], item["p"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "decoder",
                "L",
                "channels",
                "depths",
                "p",
                "failure",
                "accuracy",
                "eval_samples",
                "num_runs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "decoder": row["decoder"],
                    "L": row["L"],
                    "channels": row["channels"],
                    "depths": row["depths"],
                    "p": row["p"],
                    "failure": row["failure"],
                    "accuracy": row["accuracy"],
                    "eval_samples": row["eval_samples"],
                    "num_runs": row["num_runs"],
                }
            )


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args.paths)
    records = [
        record
        for path in inputs
        for record in parse_input(path, args.metric)
    ]
    if not records:
        raise SystemExit("No parseable training logs or PyMatching CSV files found.")

    curves, rows = aggregate(records, args.group)

    plt.figure(figsize=(8, 5))
    lattice_sizes = sorted(
        {point["L"] for points in curves.values() for point in points if point["L"]}
    )
    color_map = plt.get_cmap("tab10")
    colors = {
        L: color_map(index % color_map.N) for index, L in enumerate(lattice_sizes)
    }
    neural_architectures = sorted(
        {
            (point["channels"], point["depths"])
            for points in curves.values()
            for point in points
            if point["decoder"].lower() == "neural"
        }
    )
    neural_styles = [
        ("-", "o"),
        ("-.", "^"),
        (":", "D"),
    ]
    architecture_styles = {
        architecture: neural_styles[index % len(neural_styles)]
        for index, architecture in enumerate(neural_architectures)
    }
    for label, points in curves.items():
        xs = [point["p"] for point in points]
        ys = [point["failure_for_plot"] for point in points]
        yerr = [
            [point["yerr_low"] for point in points],
            [point["yerr_high"] for point in points],
        ]
        decoder = points[0]["decoder"].lower()
        is_pymatching = decoder == "pymatching"
        linestyle, marker = (
            ("--", "s")
            if is_pymatching
            else architecture_styles[(points[0]["channels"], points[0]["depths"])]
        )
        plt.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=colors.get(points[0]["L"]),
            linestyle=linestyle,
            marker=marker,
            capsize=3,
            label=label,
        )

    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate (1 - accuracy)")
    plt.title(args.title)
    if not args.linear:
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    if args.csv:
        write_csv(args.csv, rows)

    print(f"Wrote {args.out}")
    if args.csv:
        print(f"Wrote {args.csv}")
    for row in rows:
        print(
            f"{row['label']} p={row['p']:.6g} "
            f"failure={row['failure']:.6g} accuracy={row['accuracy']:.6g} "
            f"samples={row['eval_samples']} runs={row['num_runs']}"
        )


if __name__ == "__main__":
    main()
