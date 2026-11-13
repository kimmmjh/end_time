#!/usr/bin/env python3
"""Plot audited milestone evaluations; no inference, training, or result selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/analysis/bb_dem_2026_11_11_evaluations.csv"
OUTPUT = ROOT / "results/plots/bb/circuit/standard/dem_cnn/sample_scaling_milestones"


def main():
    rows = list(csv.DictReader(SOURCE.open()))
    for row in rows:
        result = json.loads((ROOT / row["result_path"]).read_text())
        metric = result["metrics"]["hierarchical"]
        assert result["status"] == "complete" and result["shots"] == int(row["shots"]) == 8192
        assert metric["failures"] == int(row["hierarchical_failures"])
        assert metric["block_failure_rate"] == float(row["hierarchical_LER"])
        assert metric["syndrome_failure_rate"] == 0
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.78, wspace=.12)
    fig.suptitle("Hierarchical LER by milestone (p = 0.002)", y=.98, fontsize=16, weight="bold")
    colors = {"best": "#2563a5", "latest": "#d47916"}
    milestones = [1000, 1953, 2930, 3907]
    for ax, name, title in zip(axes, ["bb72_d8_p0.002", "bb144_d6_p0.002"],
                               ["BB72  |  CNN depth 8", "BB144  |  CNN depth 6"]):
        selected = [r for r in rows if r["experiment"] == name]
        for choice in ("best", "latest"):
            data = sorted((r for r in selected if r["milestone"] != "final" and r["choice"] == choice),
                          key=lambda r: int(r["milestone"].split("_")[1]))
            x = [int(r["milestone"].split("_")[1]) for r in data]
            y = [100*float(r["hierarchical_LER"]) for r in data]
            errors = [[p-100*float(r["ci95_low"]) for p,r in zip(y,data)],
                      [100*float(r["ci95_high"])-p for p,r in zip(y,data)]]
            ax.errorbar(x, y, yerr=errors, color=colors[choice],
                        marker="o" if choice == "best" else "s", markersize=7,
                        markerfacecolor=colors[choice] if choice == "best" else "none",
                        linestyle="-" if choice == "best" else "--", linewidth=1.8,
                        elinewidth=1, capsize=3, alpha=.95, zorder=3)
            if name.startswith("bb72"):
                for xx, yy in zip(x,y):
                    above = (choice == "best") != (xx == 1953)
                    ax.annotate(f"{yy:.3f}", (xx,yy), xytext=(0, 12 if above else -21),
                                textcoords="offset points", ha="center", color=colors[choice], fontsize=9)
            elif choice == "best":
                for xx, yy in zip(x,y):
                    ax.annotate(f"{yy:.3f}%", (xx,yy), xytext=(0, 23), textcoords="offset points",
                                ha="center", color=colors[choice], fontsize=10)
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=12)
        ax.set_xlim(760, 4150)
        ax.set_ylim(0, 4.05)
        ax.set_xticks(milestones, [f"{e:,}" for e in milestones])
        ax.set_xlabel("Milestone (epoch)", labelpad=10)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        if name.startswith("bb144"):
            ax.text(.96, .93, "Trained through epoch 3,484",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#475569")
    axes[0].set_ylabel("Hierarchical LER (%)")
    fig.legend(handles=[Line2D([],[],color=colors['best'],marker='o',label='Best'),
                        Line2D([],[],color=colors['latest'],marker='s',markerfacecolor='none',linestyle='--',
                               label='Latest')], loc="upper center", bbox_to_anchor=(.5,.92),
               ncol=2, frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT.with_suffix(suffix), dpi=200)
    plt.close(fig)
    print(OUTPUT.with_suffix('.png').relative_to(ROOT))


if __name__ == "__main__":
    main()
