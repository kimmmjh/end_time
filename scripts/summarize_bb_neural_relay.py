#!/usr/bin/env python3
"""Audit and plot the BB72/BB144 Neural Relay sweeps (jobs 0 through 3).

Reads the original histories and archived reference histories; never loads or
runs a decoder. Rates in the CSV are fractions, and plots show percentages.
The accompanying interpretation is maintained in the dated Markdown report.
Plots are three curated PNGs under results/plots/neural_relay; no PDF export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

import summarize_bb_circuit_campaign as common


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/analysis"
PLOTS = ROOT / "results/plots/neural_relay"
CAMPAIGNS = {
    "bb72": dict(jobs=("58164810", "58164811"), rounds=6, date="2026_09_13"),
    "bb144": dict(jobs=("58164812", "58164813"), rounds=12, date="2026_09_15"),
}
COMPARISON_STEM = "bb_neural_relay_bb72_bb144_2026_09_15"
Z95 = 1.959963984540054


def relative(path):
    return str(path.relative_to(ROOT))


def wilson(failures, shots):
    rate = failures / shots
    denominator = 1 + Z95**2 / shots
    center = (rate + Z95**2 / (2 * shots)) / denominator
    radius = Z95 * math.sqrt(rate * (1 - rate) / shots + Z95**2 / (4 * shots**2)) / denominator
    # Retain the observed rate at boundaries despite floating-point roundoff.
    return min(rate, max(0, center - radius)), max(rate, min(1, center + radius))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_directory(job, code):
    candidates = [ROOT / f"resdir_{job}", ROOT / "results/bb/circuit/neural_relay/q_equals_p_idle0/orbit" / code / f"resdir_{job}"]
    existing = [p for p in candidates if p.is_dir()]
    assert len(existing) == 1, (job, existing)
    return existing[0]


def paired_interval(evaluation):
    gain = evaluation["paired_gain"]
    if evaluation["rescued"] + evaluation["harmed"]:
        width = evaluation["paired_gain_error"]
        return gain-width, gain+width, "normal_approximation"
    # A logged 0 +/- 0 is a degenerate variance estimate, not evidence of
    # equivalence. With no discordances, P(discordance) has this exact one-sided
    # 95% upper bound; |E(paired difference)| cannot exceed P(discordance).
    bound = 1 - .05**(1/evaluation["shots"])
    return -bound, bound, "zero_discordance_exact_bound"


def holm(rows, *, suffix=""):
    running = 0
    for rank, row in enumerate(sorted(rows, key=lambda r: r["exact_p"])):
        running = max(running, min(1, (len(rows)-rank)*row["exact_p"]))
        row[f"exact_p_holm{suffix}"] = running
        row[f"significant_improvement_holm{suffix}"] = running < .05 and row["paired_gain"] > 0
        row[f"significant_degradation_holm{suffix}"] = running < .05 and row["paired_gain"] < 0


def collect(code):
    rows, histories, training, validation = [], {}, [], []
    for job in CAMPAIGNS[code]["jobs"]:
        resdir = source_directory(job, code)
        assert (resdir / "completed.txt").is_file(), resdir
        snapshots = sorted((resdir / "source_snapshot").rglob("*.py"))
        assert len(snapshots) == 10, resdir
        for source in snapshots:
            local = ROOT / source.relative_to(resdir / "source_snapshot")
            assert source.read_bytes() == local.read_bytes(), source
        assert (resdir / "entrypoint_snapshot.py").read_bytes() == (ROOT / "main.py").read_bytes()
        paths = sorted(resdir.glob("outputs/*/*/history.json"))
        assert len(paths) == 4, paths
        for path in paths:
            h = json.loads(path.read_text())
            c, f = h["config"], h["final"]
            assert c["architecture"] == "bb_neural_relay_bp_circuit"
            assert c["baseline_decoder"] == "relay_min_sum"
            assert c["code"] == code and c["rounds"] == CAMPAIGNS[code]["rounds"]
            assert (c["bp_relay_legs"], c["bp_iterations"], c["bp_relay_solutions"]) == (4, 12, 2)
            assert len(h["train"]) == sum(p["epochs"] for p in h["phases"]) == 100
            assert [t["epoch"] for t in h["train"]] == list(range(100))
            assert len(h["eval"]) == 10 and f["shots"] == 4096 and f["osd_shots"] == 0
            for row in h["train"] + h["eval"] + [f]:
                assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            index, options = common.find_command(resdir, c)
            assert index is not None and int(options["bb_osd_eval_shots"]) == 0
            assert common.read_optional_integer(resdir / f"exit_code_exp_{index}.txt") == 0
            log = path.with_name("training_log.txt").read_text()
            assert "[Selected Best (neural_paired_gain)]" in log and "Traceback" not in log
            selected_lines = [line for line in log.splitlines() if "[Selected Best (neural_paired_gain)]" in line]
            assert len(selected_lines) == 1
            selected_log = selected_lines[0]
            for label, key in (("Accuracy", "neural_accuracy"), ("Relay BP Accuracy", "vanilla_accuracy"),
                               ("Flagged", "neural_flagged"), ("Unflagged Logical", "neural_unflagged"),
                               ("Rescued", "rescued"), ("Harmed", "harmed"), ("Eval Samples", "shots")):
                match = re.search(r"(?:\| |\] )" + re.escape(label) + r": ([0-9.]+)", selected_log)
                assert match and math.isclose(float(match[1]), f[key], abs_tol=5.1e-9), (path, label)
            assert f"Epoch: {h['best_epoch']} |" in selected_log
            for name in ("model.pt", "best_model.pt"):
                assert path.with_name(name).is_file()
            n, gain = f["shots"], f["paired_gain"]
            rescued, harmed = f["rescued"], f["harmed"]
            assert math.isclose(gain, (rescued - harmed) / n, abs_tol=1e-12)
            assert math.isclose(gain, f["neural_accuracy"] - f["vanilla_accuracy"], abs_tol=1e-12)
            halfwidth = Z95 * math.sqrt(((rescued + harmed) / n - gain**2) / (n - 1))
            assert math.isclose(halfwidth, f["paired_gain_error"], abs_tol=1e-12)
            for prefix in ("neural", "vanilla"):
                assert 0 <= f[f"{prefix}_accuracy"] <= f[f"{prefix}_converged"] <= 1
                assert 1 <= f[f"{prefix}_mean_bp_iterations"] <= 48
            assert math.isclose(1 - f["neural_accuracy"], f["neural_flagged"] + f["neural_unflagged"])
            best = next(e for e in h["eval"] if e["epoch"] == h["best_epoch"])
            assert best is max(h["eval"], key=lambda e: (e["paired_gain"], e["neural_accuracy"]))
            train_log = re.findall(r"\[Train\] Epoch: (\d+).*?\| LR: ([0-9.e+-]+) \| Time: ([0-9.]+)s", log)
            assert len(train_log) == 100 and [int(t[0]) for t in train_log] == list(range(100))
            ci_low, ci_high, ci_method = paired_interval(f)
            row = dict(job_id=job, experiment_index=index, code=c["code"], p=c["gate_error_rate"],
                       seed=c["seed"], best_epoch=h["best_epoch"], completed_epochs=len(h["train"]),
                       batch_size=int(options["batch_size"]), batches_per_epoch=int(options["batches"]),
                       training_shots=len(h["train"])*int(options["batch_size"])*int(options["batches"]),
                       validation_shots=h["eval"][0]["shots"],
                       num_detectors=c["num_detectors"], num_mechanisms=c["num_mechanisms"],
                       num_edges=c["num_edges"], num_orbits=c["num_orbits"], trainable_parameters=c["trainable_parameters"],
                       rounds=c["rounds"], relay_legs=c["bp_relay_legs"], iterations_per_leg=c["bp_iterations"],
                       solutions=c["bp_relay_solutions"], memory_first=c["bp_relay_memory_strength"],
                       memory_min=c["bp_relay_memory_min"], memory_max=c["bp_relay_memory_max"],
                       graph_fingerprint=c["graph_fingerprint"], **f)
            for prefix in ("neural", "vanilla"):
                count = round(n * (1 - f[f"{prefix}_accuracy"]))
                row[f"{prefix}_failures"] = count
                row[f"{prefix}_ler"] = count / n
                row[f"{prefix}_ler_wilson_low"], row[f"{prefix}_ler_wilson_high"] = wilson(count, n)
            row.update(paired_ci95_low=ci_low, paired_ci95_high=ci_high, paired_ci95_method=ci_method,
                       exact_p=binomtest(rescued, rescued+harmed).pvalue if rescued+harmed else 1,
                       relative_ler_reduction=gain/row["vanilla_ler"],
                       iteration_reduction=1-f["neural_mean_bp_iterations"]/f["vanilla_mean_bp_iterations"],
                       first_train_loss=h["train"][0]["total"], last_train_loss=h["train"][-1]["total"],
                       mean_first10_train_loss=float(np.mean([t["total"] for t in h["train"][:10]])),
                       mean_last10_train_loss=float(np.mean([t["total"] for t in h["train"][-10:]])),
                       selected_validation_ler=1-best["neural_accuracy"],
                       last_validation_ler=1-h["eval"][-1]["neural_accuracy"],
                       positive_validation_evaluations=sum(e["paired_gain"] > 0 for e in h["eval"]),
                       validation_evaluations=len(h["eval"]),
                       zero_neural_success_validations=sum(e["neural_accuracy"] == 0 for e in h["eval"]),
                       last_validation_paired_gain=h["eval"][-1]["paired_gain"],
                       mean_training_epoch_seconds=float(np.mean(common.parse_epoch_times(path.with_name("training_log.txt")))),
                       source_history=relative(path), source_commit=(resdir/"git_commit.txt").read_text().strip())
            rows.append(row)
            histories[row["p"]] = h
            for t, (_, lr, seconds) in zip(h["train"], train_log):
                training.append(dict(code=code, p=row["p"], seed=row["seed"], **t,
                                     learning_rate=float(lr), epoch_seconds=float(seconds)))
            for e in h["eval"]:
                assert e["shots"] == 1024 and e["osd_shots"] == 0
                assert math.isclose(e["paired_gain"], (e["rescued"]-e["harmed"])/e["shots"], abs_tol=1e-12)
                lo, hi, method = paired_interval(e)
                validation.append(dict(code=code, p=row["p"], seed=row["seed"], **e,
                                       paired_ci95_low=lo, paired_ci95_high=hi, paired_ci95_method=method))
    rows.sort(key=lambda r: r["p"])
    assert [r["p"] for r in rows] == [.001, .002, .003, .004, .005, .006, .008, .01]
    # Holm correction over the eight final paired comparisons.
    holm(rows)
    return rows, histories, training, validation


def historical(rows, histories):
    for row in rows:
        c = histories[row["p"]]["config"]
        for tag, folder in (("legacy_raw", f"no_osd/q_equals_p_idle0/orbit/{row['code']}"),
                            ("legacy_osd", f"q_equals_p_idle0/orbit/{row['code']}")):
            matches = []
            for path in (ROOT / "results/bb/circuit" / folder).rglob("history.json"):
                h = json.loads(path.read_text())
                old = h["config"]
                if old["gate_error_rate"] == row["p"] and old["seed"] == row["seed"]:
                    assert old["graph_fingerprint"] == c["graph_fingerprint"]
                    allowed = {"architecture", "bb_osd_method", "bb_osd_order", "checkpoint_selection_metric"}
                    assert all(old[k] == c[k] for k in old.keys() & c.keys() - allowed)
                    if h.get("final"):
                        assert len(h["train"]) == 100 and h["final"]["shots"] == 4096
                    if tag == "legacy_osd" and h.get("final"):
                        assert old["bb_osd_method"] == "OSD_0" and h["final"]["osd_shots"] == 4096
                    matches.append((path, h))
            assert len(matches) <= 1
            row[f"{tag}_neural_ler"] = None
            row[f"{tag}_vanilla_ler"] = None
            row[f"{tag}_source"] = None
            row[f"{tag}_status"] = "not_available"
            row[f"{tag}_completed_epochs"] = None
            row[f"{tag}_shots"] = None
            if matches:
                path, h = matches[0]
                row[f"{tag}_source"] = relative(path)
                row[f"{tag}_completed_epochs"] = len(h["train"])
                row[f"{tag}_status"] = "complete" if h.get("final") else "partial_no_final"
                if not h.get("final"):
                    continue
                suffix = "_osd_accuracy" if tag == "legacy_osd" else "_accuracy"
                row[f"{tag}_shots"] = h["final"]["osd_shots" if tag == "legacy_osd" else "shots"]
                for prefix in ("neural", "vanilla"):
                    row[f"{tag}_{prefix}_ler"] = 1 - h["final"][prefix+suffix]
                row[f"{tag}_source"] = relative(path)


def training_plot(histories, code):
    """One PNG per code: all eight losses and paired validation gains."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), layout="constrained")
    bounds = [paired_interval(e)[:2] for h in histories.values() for e in h["eval"]]
    gain_limits = (min(-2, 100*min(b[0] for b in bounds)-5),
                   max(2, 100*max(b[1] for b in bounds)+3))
    for i, (p, h) in enumerate(sorted(histories.items())):
        top, column = (i // 4)*2, i % 4
        loss_ax, gain_ax = axes[top, column], axes[top+1, column]
        loss_ax.plot([t["epoch"] for t in h["train"]], [t["total"] for t in h["train"]],
                     color="#75619A")
        loss_ax.set(title=f"p={p:g} | selected epoch {h['best_epoch']}", ylabel="Training loss")
        epochs = np.array([e["epoch"] for e in h["eval"]])
        gains = np.array([e["paired_gain"]*100 for e in h["eval"]])
        intervals = np.array([paired_interval(e)[:2] for e in h["eval"]])*100
        gain_ax.axhline(0, color="#576574", lw=1)
        gain_ax.fill_between(epochs, intervals[:, 0], intervals[:, 1],
                             color="#007A87", alpha=.15, label="Pointwise paired 95% interval")
        gain_ax.plot(epochs, gains, "o-", ms=4, color="#007A87", label="Validation LER reduction")
        best = next(e for e in h["eval"] if e["epoch"] == h["best_epoch"])
        gain_ax.scatter(best["epoch"], best["paired_gain"]*100, marker="*", s=100,
                        color="#C97727", zorder=3, label="Selected checkpoint")
        last = h["eval"][-1]
        if all(e["neural_accuracy"] == e["vanilla_accuracy"] == 0 for e in h["eval"]):
            note = "Both: 0/1,024 successes at every eval"
        else:
            note = (f"Last LER: Neural {100*(1-last['neural_accuracy']):.2f}%"
                    f" | Relay {100*(1-last['vanilla_accuracy']):.2f}%")
        gain_ax.text(.03, .03, note, transform=gain_ax.transAxes, fontsize=8, va="bottom",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        gain_ax.set(ylabel="Validation gain (pp)", ylim=gain_limits, xticks=[9, 29, 49, 69, 99])
        for ax in (loss_ax, gain_ax):
            ax.axvline(h["best_epoch"], color="#C97727", ls="--", alpha=.7)
            ax.set_xlabel("Epoch (zero-based)")
            ax.grid(alpha=.15)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle(f"{code.upper()}: all eight training runs\n"
                 "Gain = Relay LER - Neural LER; positive favors learning | 1,024 fresh paired validation shots/evaluation",
                 fontsize=14)
    fig.savefig(PLOTS/f"training_{code}.png", dpi=180)
    plt.close(fig)


def comparison_plot(all_rows):
    """Final paired comparison and historical references in a single PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="row", layout="constrained")
    for column, code in enumerate(CAMPAIGNS):
        rows = [r for r in all_rows if r["code"] == code]
        x = np.array([r["p"]*100 for r in rows])
        ax = axes[0, column]
        for prefix, label, color in (("neural", "Neural Relay", "#007A87"),
                                     ("vanilla", "Relay (no learning)", "#576574")):
            y = np.array([r[f"{prefix}_ler"]*100 for r in rows])
            lo = np.array([r[f"{prefix}_ler_wilson_low"]*100 for r in rows])
            hi = np.array([r[f"{prefix}_ler_wilson_high"]*100 for r in rows])
            ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="o-", capsize=3, label=label, color=color)
        for tag, label, color in (("legacy_raw", "Neural BP T12", "#A16BBD"),
                                  ("legacy_osd", "Neural BP + OSD-0", "#C97727")):
            selected = [r for r in rows if r[f"{tag}_neural_ler"] is not None]
            positives = [r for r in selected if r[f"{tag}_neural_ler"] > 0]
            ax.plot([r["p"]*100 for r in positives], [r[f"{tag}_neural_ler"]*100 for r in positives],
                    "s--", color=color, alpha=.85, label=label)
            for r in selected:
                if r[f"{tag}_neural_ler"] == 0:
                    shots = r[f"{tag}_shots"]
                    bound = (1-.05**(1/shots))*100
                    ax.scatter(r["p"]*100, bound, marker="v", color=color)
                    ax.annotate(f"0/{shots:,}; 95% upper bound", (r["p"]*100, bound),
                                xytext=(8, -5), textcoords="offset points", fontsize=8)
        ax.set(yscale="log", ylim=(.04, 120), ylabel="Final block LER (%)",
               title=f"{code.upper()} | {rows[0]['rounds']} noisy rounds")
        ax.legend(fontsize=8, loc="lower right")
        ax = axes[1, column]
        ax.axhline(0, color="#999999", lw=1)
        for r in rows:
            color = "#B14B42" if r["significant_degradation_holm_all16"] else "#007A87"
            ax.errorbar(r["p"]*100, r["paired_gain"]*100,
                        yerr=[[100*(r["paired_gain"]-r["paired_ci95_low"])],
                              [100*(r["paired_ci95_high"]-r["paired_gain"])]],
                        fmt="o", color=color, capsize=4)
            if r["exact_p_holm_all16"] < .05:
                ax.annotate("*", (r["p"]*100, r["paired_ci95_high"]*100+1), ha="center", fontsize=15)
        ax.set(ylabel="LER reduction vs Relay (pp)", ylim=(-5, 30))
        for ax in axes[:, column]:
            ax.set_xlabel("Physical error rate p (%)")
            ax.set_xticks(x)
            ax.grid(alpha=.15)
    fig.suptitle("Neural Relay: final LER and gain vs non-neural Relay | 4,096 shots/point | * Holm over 16 tests\n"
                 "Historical comparisons are descriptive; code-specific rounds and training budgets differ",
                 fontsize=12)
    fig.savefig(PLOTS/"overview.png", dpi=180)
    plt.close(fig)


def write_manifest(code, stem):
    entries = []
    for job in CAMPAIGNS[code]["jobs"]:
        resdir = source_directory(job, code)
        for path in sorted(resdir.rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
            entries.append(dict(job_id=job, file=path.relative_to(resdir).as_posix(),
                                bytes=path.stat().st_size, sha256=digest.hexdigest()))
    write_csv(ANALYSIS/f"{stem}_manifest.csv", entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=[*CAMPAIGNS, "all"], default="all",
                        help="Default updates both codes and all three PNGs; a single code updates its data and training PNG.")
    args = parser.parse_args()
    codes = list(CAMPAIGNS) if args.code == "all" else [args.code]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    all_rows = []
    for code in codes:
        rows, histories, training, validation = collect(code)
        historical(rows, histories)
        stem = f"bb_neural_relay_{code}_{CAMPAIGNS[code]['date']}"
        write_csv(ANALYSIS/f"{stem}.csv", rows)
        write_csv(ANALYSIS/f"{stem}_train.csv", sorted(training, key=lambda r: (r["p"], r["epoch"])))
        write_csv(ANALYSIS/f"{stem}_validation.csv", sorted(validation, key=lambda r: (r["p"], r["epoch"])))
        training_plot(histories, code)
        write_manifest(code, stem)
        all_rows.extend(rows)
        print(f"{code}: audited {len(rows)} completed runs, {sum(r['shots'] for r in rows):,} final shots.")
        print("Holm-significant improvements:", [r["p"] for r in rows if r["significant_improvement_holm"]])
        print("Holm-significant degradations:", [r["p"] for r in rows if r["significant_degradation_holm"]])
    if args.code == "all":
        holm(all_rows, suffix="_all16")
        write_csv(ANALYSIS/f"{COMPARISON_STEM}.csv", all_rows)
        comparison_plot(all_rows)
    print("Wrote final/train/validation/manifest CSVs and curated PNGs in results/plots/neural_relay/.")


if __name__ == "__main__":
    main()
