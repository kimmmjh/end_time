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
        choices=["L", "L_arch"],
        default="L",
        help="Group curves only by L, or split by L plus architecture.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis instead of log scale.",
    )
    parser.add_argument(
        "--title",
        default="Threshold Scan",
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


def discover_logs(paths: list[Path]) -> list[Path]:
    logs: set[Path] = set()
    for path in paths:
        if path.is_file():
            logs.add(path)
            continue

        found = list(path.rglob("log_exp_*.txt"))
        if found:
            logs.update(found)
        else:
            logs.update(path.rglob("training_log.txt"))

    return sorted(logs)


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
    )


def label_for(record: Record, group: str) -> str:
    if group == "L_arch":
        return f"L={record.L} ch={record.channels} d={record.depths}"
    return f"L={record.L}"


def aggregate(records: list[Record], group: str):
    buckets = defaultdict(list)
    for record in records:
        buckets[(label_for(record, group), record.p)].append(record)

    curves = defaultdict(list)
    rows = []
    for (label, p), items in buckets.items():
        samples = [item.eval_samples for item in items if item.eval_samples]
        if samples:
            total = sum(samples)
            correct = sum(round(item.accuracy * item.eval_samples) for item in items)
            accuracy = correct / total
            eval_samples = total
        else:
            accuracy = sum(item.accuracy for item in items) / len(items)
            eval_samples = None

        failure = 1.0 - accuracy
        if eval_samples:
            failure_for_plot = max(failure, 0.5 / eval_samples)
            yerr = math.sqrt(max(failure * (1.0 - failure), 0.0) / eval_samples)
        else:
            failure_for_plot = max(failure, 1e-12)
            yerr = 0.0

        point = {
            "label": label,
            "p": p,
            "failure": failure,
            "failure_for_plot": failure_for_plot,
            "accuracy": accuracy,
            "eval_samples": eval_samples,
            "num_runs": len(items),
            "yerr": yerr,
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
                    "p": row["p"],
                    "failure": row["failure"],
                    "accuracy": row["accuracy"],
                    "eval_samples": row["eval_samples"],
                    "num_runs": row["num_runs"],
                }
            )


def main() -> None:
    args = parse_args()
    logs = discover_logs(args.paths)
    records = [record for path in logs if (record := parse_log(path, args.metric))]
    if not records:
        raise SystemExit("No parseable training logs found.")

    curves, rows = aggregate(records, args.group)

    plt.figure(figsize=(8, 5))
    for label, points in curves.items():
        xs = [point["p"] for point in points]
        ys = [point["failure_for_plot"] for point in points]
        yerr = [point["yerr"] for point in points]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label)

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
