#!/usr/bin/env python3
"""Keep one combined DEM CNN figure, one selected point per configuration and decoder."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_dem_november3 as prior

PLOTS = prior.PLOTS
KEEP = {"overview.png"}


def emit_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_selection(original, training):
    current = dict(original)
    for row, history in training:
        k = prior.key(history["config"])
        if row["status"] == "complete" and len(history["train"]) > len(current[k]["train"]):
            current[k] = history
    rows = []
    for k, history in sorted(current.items()):
        final = history["final"]
        assert final["epoch"] == history["best_epoch"]
        failures, shots = final["raw"]["failures"], final["shots"]
        low, high = prior.hier.wilson(failures, shots)
        rows.append(dict(code=k[0], depth=k[1], p=k[2], epochs=len(history["train"]),
                         selected_epoch=history["best_epoch"]+1, shots=shots, failures=failures,
                         rate=failures/shots, low=low, high=high,
                         selection="raw_validation_best_from_latest_completed_training"))
    return rows


def hierarchical_selection(evaluations):
    candidates = []
    # The original 24 complete evaluations are retained as candidates, not pooled.
    with (prior.OUT / "bb_dem_2026_11_02_hierarchical_summary.csv").open() as stream:
        for r in csv.DictReader(stream):
            candidates.append(dict(code=r["code"], depth=int(r["depth"]), p=float(r["p"]),
                selected_epoch=int(r["selected_epoch"]), shots=int(r["shots"]),
                failures=int(r["hierarchical_failures"]), weights="best", job=int(r["job"]),
                checkpoint_sha256=r["source_checkpoint_sha256"], training_epochs=800))
    for r, _ in evaluations:
        if r["status"] == "complete":
            candidates.append(dict(code=r["code"], depth=r["depth"], p=r["p"],
                selected_epoch=r["selected_epoch"], shots=r["shots"],
                failures=r["hierarchical_failures"], weights=r["weights"], job=r["job"],
                checkpoint_sha256=r["checkpoint_sha256"], training_epochs=800))
    points = sorted({(r["code"], r["depth"], r["p"]) for r in candidates})
    rows = []
    for k in points:
        options = [r for r in candidates if (r["code"], r["depth"], r["p"]) == k]
        assert {r["shots"] for r in options} == {8192}
        # Earliest selected epoch and best weight label break equal-score ties.
        chosen = dict(min(options, key=lambda r:(r["failures"], r["selected_epoch"],
                                                r["weights"] != "best", r["job"])))
        low, high = prior.hier.wilson(chosen["failures"], chosen["shots"])
        chosen.update(rate=chosen["failures"]/chosen["shots"], low=low, high=high,
                      selection="minimum_measured_hierarchical_LER_on_complete_evaluation_bank",
                      independent_selected_model_test=False)
        rows.append(chosen)
    return rows


def figure(raw, hierarchical_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 2, figsize=(13,9.5), layout="constrained")
    panels = [(axes[i,j], code, j == 1, hierarchical_rows if j == 1 else raw)
              for i, code in enumerate(("bb72", "bb144")) for j in range(2)]
    for ax, code, hierarchical, rows in panels:
        for depth, color in prior.base.COLORS.items():
            by_rate = {r["p"]:r for r in rows if r["code"] == code and r["depth"] == depth}
            if not by_rate:
                continue
            rates = [.001,.002,.004,.006]
            # NaN leaves gaps for missing evaluations and zero-failure log points.
            y = [100*by_rate[p]["rate"] if p in by_rate and
                 (not hierarchical or by_rate[p]["failures"] > 0) else np.nan for p in rates]
            ax.plot([100*p for p in rates], y, color=color, label=f"D{depth}")
            for p, r in sorted(by_rate.items()):
                value = 100*r["rate"]
                if hierarchical and r["failures"] == 0:
                    ax.scatter(100*p,100*r["high"],marker="v",facecolors="none",edgecolors=color)
                else:
                    ax.errorbar(100*p,value,yerr=[[max(0,value-100*r['low'])],
                                                [max(0,100*r['high']-value)]],
                                fmt="o",color=color,capsize=3)
        ax.set(title=code.upper()+ (" — Hierarchical" if hierarchical else " — Raw CNN"),
               xlabel="Per-location error rate p (%)",
               ylabel="Hierarchical logical block failure (%)" if hierarchical else "Raw block failure (%)",
               xticks=[.1,.2,.4,.6])
        ax.grid(alpha=.2)
        ax.legend(title="CNN depth",loc="lower right",fontsize=9)
        if hierarchical:
            ax.set_yscale("log")
            ax.set_ylim(.002,150)
        else:
            ax.set_ylim(-2,103)
    n = sum(r["epochs"] == 1000 for r in raw)
    fig.suptitle(
        "BB DEM CNN: raw and hierarchical performance\n"
        f"Raw: validation-best from latest completed runs ({n} at 1,000 epochs; {len(raw)-n} at 800)\n"
        "Hierarchical: best on displayed evaluation shots, not independently confirmed; 1,000-epoch evaluations pending\n"
        "Checkpoints and shot banks can differ between columns; raw: linear scale, hierarchical: log scale\n"
        "8,192 shots; marginal 95% Wilson CI; hollow triangle: zero-failure upper bound; gaps: no result",
        fontsize=11)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def render(original, training, evaluations):
    raw = raw_selection(original, training)
    hierarchical = hierarchical_selection(evaluations)
    assert len(raw) == len({(r['code'],r['depth'],r['p']) for r in raw})
    assert len(hierarchical) == len({(r['code'],r['depth'],r['p']) for r in hierarchical})
    emit_csv(prior.OUT / "bb_dem_latest_raw.csv",raw)
    emit_csv(prior.OUT / "bb_dem_latest_hierarchical.csv",hierarchical)
    figure(raw,hierarchical,PLOTS / "overview.png")
    removed = []
    # The user requested cleanup only within the BB circuit-level DEM CNN plots.
    for path in sorted(PLOTS.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".png", ".pdf", ".svg"):
            relative = str(path.relative_to(PLOTS))
            if relative not in KEEP:
                path.unlink()
                removed.append(relative)
    print(json.dumps(dict(kept=sorted(KEEP),removed=removed,raw_points=len(raw),
                          hierarchical_points=len(hierarchical)),indent=2))
    return raw, hierarchical


def main():
    from scripts import analyze_bb_dem_november4 as campaign
    training, evaluations = [], []
    for path in [*prior.DESTINATIONS.values(), *campaign.DESTINATIONS.values()]:
        a,b,_ = prior.audit_job(path,jobs={**prior.JOBS,**campaign.JOBS})
        training.extend(a); evaluations.extend(b)
    render(prior.baseline(),training,evaluations)


if __name__ == "__main__":
    main()
