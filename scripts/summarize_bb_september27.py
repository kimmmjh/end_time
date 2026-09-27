#!/usr/bin/env python3
"""Audit the retained September 27 CNN import and regenerate CSVs and two PNGs.

Reads archived server artifacts only; does not train or rerun any decoder.
Original files are checked against the import manifest before analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bb_code import BBCodeSpec

ANALYSIS = ROOT / "results/analysis"
CNN = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn/bb72/resdir_58793761"
PLOTS = ROOT / "results/plots/september_2026_update"
STEM = "bb_update_2026_09_27"
Z = 1.959963984540054


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ANALYSIS / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, tolerance=1e-10):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def wilson(count, n):
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return max(0., center - half), min(1., center + half)


def holm(rows):
    previous = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r["paired_exact_p"])):
        previous = min(1., max(previous, (len(rows) - rank) * row["paired_exact_p"]))
        row["paired_holm_p"] = previous


def paired(reference, candidate):
    rescued = int((~reference & candidate).sum())
    harmed = int((reference & ~candidate).sum())
    # Both evaluated policies retain every already-valid correction. A success
    # therefore cannot be lost; the gain is a Bernoulli rescue proportion.
    assert harmed == 0
    n = reference.size
    low, high = wilson(rescued, n)
    return dict(rescued=rescued, harmed=harmed, paired_gain=rescued / n,
                paired_gain_low=low, paired_gain_high=high,
                paired_gain_interval="Wilson rescue proportion; monotone policy",
                paired_exact_p=binomtest(rescued, rescued + harmed).pvalue if rescued else 1.)


def audit_manifest():
    rows = read_csv(ANALYSIS / "bb_import_2026_09_27_manifest.csv")
    for row in rows:
        path = ROOT / row["archive_file"]
        assert path.stat().st_size == int(row["bytes"])
        assert sha256(path) == row["sha256"], path
    return dict(original_files=len(rows), bytes=sum(int(r["bytes"]) for r in rows),
                jobs=len({r["job_id"] for r in rows}), all_original_hashes_verified=True)


def check_job(job):
    assert (job / "completed.txt").is_file() and not (job / "failed.txt").exists()
    for index in range(4):
        assert (job / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
    # Only BBCodeSpec from the current checkout is used for rescoring capacity
    # shots. Decoder implementations may evolve after these archived runs.
    # The manifest checks every retained archived file against its original hash.
    source = job / "source_snapshot/src/bb_code.py"
    assert sha256(source) == sha256(ROOT / "src/bb_code.py"), source


def xz(pauli):
    return ((pauli == 1) | (pauli == 2)).astype(np.uint8), ((pauli == 2) | (pauli == 3)).astype(np.uint8)


def score_pauli(correction, truth, syndrome, code):
    cx, cz = xz(correction)
    tx, tz = xz(truth)
    predicted = np.concatenate(((cz @ code.hx.T) % 2, (cx @ code.hz.T) % 2), axis=1)
    converged = (predicted == syndrome).all(axis=1)
    trivial = ~np.concatenate((((cx ^ tx) @ code.logicals_z.T) % 2,
                               ((cz ^ tz) @ code.logicals_x.T) % 2), axis=1).any(axis=1)
    return converged & trivial, converged


def cnn_results():
    check_job(CNN)
    code = BBCodeSpec.from_name("bb72")
    finals, training, validation, histories = [], [], [], {}
    for path in sorted(CNN.rglob("history.json")):
        h = json.loads(path.read_text())
        c, f, p = h["config"], h["final"], h["config"]["error_rate"]
        assert c["cnn_width"] == 64 and c["cnn_depth"] == 2 and c["code"] == "bb72"
        assert c["noise_model"] == "capacity" and c["channel"] == "depolarizing"
        assert c["checkpoint_selection"] == "raw_logical_accuracy" and c["osd_extra_bp_iterations"] == 0
        assert c["graph_fingerprint"] == hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
        assert [t["epoch"] for t in h["train"]] == list(range(100))
        assert [e["epoch"] for e in h["eval"]] == list(range(4, 100, 5))
        selected = max(h["eval"], key=lambda e: e["raw"]["logical_accuracy"])
        assert selected["epoch"] == h["best_epoch"] == f["epoch"]
        assert f["shots"] == 65536
        assert all((path.parent / name).is_file() for name in ("model.pt", "best_model.pt"))
        for e in [*h["eval"], f]:
            close(e["osd_minus_raw_gain"], e["raw"]["logical_error_rate"] - e["osd"]["logical_error_rate"])
            close(e["osd_minus_raw_gain"], e["rescued"] / e["shots"])
            assert e["harmed"] == 0 and e["osd"]["syndrome_convergence"] == 1.
        with np.load(path.with_name("final_shots.npz"), allow_pickle=False) as bank:
            b = {key: bank[key] for key in bank.files}
        assert json.loads(str(b["metadata_json"])) == c
        for key in ("syndrome", "pauli", "raw_correction", "osd_correction"):
            assert b[key].shape == (65536, 72)
        tx, tz = xz(b["pauli"])
        assert np.array_equal(b["syndrome"], np.concatenate(((tz @ code.hx.T) % 2, (tx @ code.hz.T) % 2), axis=1))
        row = dict(code="bb72", p=p, seed=c["seed"], depth=2, width=64, parameters=c["parameter_count"],
                   epochs=100, training_shots=819200, shots=f["shots"], best_epoch_zero_based=h["best_epoch"],
                   best_epoch_one_based=h["best_epoch"] + 1,
                   first_loss=h["train"][0]["total"], last_loss=h["train"][-1]["total"],
                   first10_mean_loss=float(np.mean([t["total"] for t in h["train"][:10]])),
                   last10_mean_loss=float(np.mean([t["total"] for t in h["train"][-10:]])),
                   selected_validation_raw_ler=selected["raw"]["logical_error_rate"],
                   last_validation_raw_ler=h["eval"][-1]["raw"]["logical_error_rate"],
                   last5_validation_raw_ler=float(np.mean([e["raw"]["logical_error_rate"] for e in h["eval"][-5:]])),
                   osd_call_fraction=f["osd_call_fraction"], nn_batch_seconds=f["nn_batch_seconds"],
                   osd_with_transfer_seconds=f["osd_with_transfer_seconds"], source=path.relative_to(ROOT).as_posix())
        outcomes = {}
        for method in ("raw", "osd"):
            success, converged = score_pauli(b[f"{method}_correction"], b["pauli"], b["syndrome"], code)
            assert np.array_equal(success, b[f"{method}_success"].astype(bool))
            failures = int((~success).sum())
            flagged, unflagged = int((~converged).sum()), int((converged & ~success).sum())
            close(f[method]["logical_error_rate"], failures / success.size)
            close(f[method]["flagged_failure_rate"], flagged / success.size)
            close(f[method]["unflagged_logical_failure_rate"], unflagged / success.size)
            low, high = wilson(failures, success.size)
            row.update({f"{method}_{key}": value for key, value in dict(failures=failures, flagged=flagged,
                        unflagged=unflagged, ler=failures / success.size, ler_low=low, ler_high=high).items()})
            outcomes[method] = success
            if method == "raw":
                close((~converged).mean(), f["osd_call_fraction"])
                assert np.array_equal(b["raw_correction"][converged], b["osd_correction"][converged])
        row.update(paired(outcomes["raw"], outcomes["osd"]))
        row["osd_relative_ler_reduction"] = row["paired_gain"] / row["raw_ler"]
        close(row["rescued"], f["rescued"])
        finals.append(row)
        histories[p] = h
        for t in h["train"]:
            training.append(dict(code="bb72", p=p, seed=c["seed"], **t))
        for e in h["eval"]:
            validation.append(dict(code="bb72", p=p, seed=c["seed"], epoch=e["epoch"], shots=e["shots"],
                                   raw_ler=e["raw"]["logical_error_rate"], osd_ler=e["osd"]["logical_error_rate"],
                                   osd_call_fraction=e["osd_call_fraction"], paired_gain=e["osd_minus_raw_gain"],
                                   selected=e["epoch"] == h["best_epoch"]))
    finals.sort(key=lambda r: r["p"])
    holm(finals)
    assert [r["p"] for r in finals] == [.02, .04, .06, .08]
    write_csv(f"{STEM}_cnn_final.csv", finals)
    write_csv(f"{STEM}_cnn_train.csv", sorted(training, key=lambda r: (r["p"], r["epoch"])))
    write_csv(f"{STEM}_cnn_validation.csv", sorted(validation, key=lambda r: (r["p"], r["epoch"])))
    return finals, histories


def references(cnn):
    old_neural = [r for r in read_csv(ANALYSIS / "bb_neural_bp_depolarizing_orbit.csv") if r["code"] == "bb72"]
    classical = [r for r in read_csv(ANALYSIS / "bb_campaign_2026_08_classical.csv") if r["code"] == "bb72"]
    rows = []
    for new in cnn:
        for old in old_neural:
            if float(old["p"]) != new["p"]:
                continue
            for tag, name in (("vanilla", "BP4 T12"), ("neural", "Neural BP4 T12")):
                rows.append(dict(p=new["p"], method=name, ler=float(old[f"{tag}_logical_error_rate"]),
                                 shots=int(old["eval_samples"]), source=old["source_resdir"], comparison="separate shot banks and training budgets"))
        for old in classical:
            if float(old["p"]) == new["p"] and old["method"] in {"bposd_0", "bposd_cs7"}:
                rows.append(dict(p=new["p"], method={"bposd_0": "CSS BP2+OSD-0", "bposd_cs7": "CSS BP2+OSD-CS7"}[old["method"]],
                                 ler=float(old["logical_error_rate"]), shots=int(old["samples"]), source=old["source_file"],
                                 comparison="separate shot banks; X/Z split; different OSD convention and BP budget"))
    write_csv(f"{STEM}_capacity_references.csv", rows)
    return rows


def error_curve(ax, rows, *, label, color, prefix="", **kwargs):
    x = np.array([r["p"] * 100 for r in rows])
    y = np.array([r[prefix + "ler"] * 100 for r in rows])
    low = np.array([r[prefix + "ler_low"] * 100 for r in rows])
    high = np.array([r[prefix + "ler_high"] * 100 for r in rows])
    ax.errorbar(x, y, yerr=np.maximum([y - low, high - y], 0), fmt="o-", capsize=3,
                color=color, label=label, **kwargs)


def cnn_plot(rows, refs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True, layout="constrained")
    error_curve(axes[0], rows, label="Tanner CNN", color="#BA6842", prefix="raw_")
    for ax in axes:
        error_curve(ax, rows, label="Tanner CNN + direct OSD-0", color="#007C83", prefix="osd_")
    for ax, methods in zip(axes, (("BP4 T12", "Neural BP4 T12"), ("CSS BP2+OSD-0", "CSS BP2+OSD-CS7"))):
        for name, color in zip(methods, ("#626B75", "#8F62A5")):
            selected = [r for r in refs if r["method"] == name]
            ax.plot([r["p"] * 100 for r in selected], [r["ler"] * 100 for r in selected], "s--", color=color, label=name + " (previous)")
        ax.set(yscale="log", ylim=(.07, 80), xlabel="Depolarizing data-qubit error probability p (%)", xticks=[2, 4, 6, 8])
        ax.grid(alpha=.18, which="both")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set(ylabel="Block LER (%)", title="Joint Pauli output / BP4 references")
    axes[1].set(title="Separate X/Z BP-OSD references")
    fig.suptitle("BB72 code capacity | Tanner CNN depth 2, width 64 | 65,536 fresh shots/point\n"
                 "One raw-LER-selected checkpoint for CNN and CNN+OSD. Error bars: 95% Wilson.\n"
                 "Previous curves use different samples, training budgets and OSD implementations.", fontsize=11)
    fig.savefig(PLOTS / "tanner_cnn.png", dpi=180)
    plt.close(fig)


def training_plot(histories):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, layout="constrained")
    for col, (p, h) in enumerate(sorted(histories.items())):
        x = [t["epoch"] + 1 for t in h["train"]]
        for key, color in (("total", "#222222"), ("syndrome", "#C98427"), ("logical", "#5475A9")):
            axes[0, col].plot(x, [t[key] for t in h["train"]], color=color, label=key, lw=1.3)
        for tag, color, label in (("raw", "#BA6842", "CNN"), ("osd", "#007C83", "CNN+OSD")):
            axes[1, col].plot([e["epoch"] + 1 for e in h["eval"]],
                              [e[tag]["logical_error_rate"] * 100 for e in h["eval"]], "o-", ms=3, color=color, label=label + " validation")
            axes[1, col].scatter(h["best_epoch"] + 1, h["final"][tag]["logical_error_rate"] * 100,
                                 marker="D", s=55, color=color, edgecolors="black", zorder=5, label=label + " fresh test")
        for ax in axes[:, col]:
            ax.axvline(h["best_epoch"] + 1, ls=":", color=".4", lw=1)
            ax.grid(alpha=.18)
        axes[0, col].set(title=f"p={p:g} | selected epoch {h['best_epoch'] + 1}", yscale="log", ylabel="Training loss" if col == 0 else "")
        axes[1, col].set(yscale="log", xlabel="Epoch (1-based)", ylabel="Block LER (%)" if col == 0 else "")
    axes[0, 0].legend(fontsize=8)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("BB72 Tanner CNN: all four completed 100-epoch runs\n"
                 "Validation: 4,096 fresh shots each; diamonds: 65,536-shot selected-checkpoint test", fontsize=12)
    fig.savefig(PLOTS / "tanner_cnn_training.png", dpi=170)
    plt.close(fig)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest()
    cnn, histories = cnn_results()
    refs = references(cnn)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cnn_plot(cnn, refs)
    training_plot(histories)
    audit.update(cnn_experiments=len(cnn), cnn_final_shots=sum(r["shots"] for r in cnn),
                 cnn_saved_corrections_independently_rescored=True,
                 removed_plain_bp_jobs=[58793753, 58793754, 58793756, 58793757, 58793759])
    (ANALYSIS / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    for row in cnn:
        print(f"CNN p={row['p']}: raw={row['raw_ler']:.6%}, OSD={row['osd_ler']:.6%}, selected epoch={row['best_epoch_one_based']}")
    print("Wrote four analysis CSVs, audit JSON, and two PNGs. Markdown interpretation is maintained separately.")


if __name__ == "__main__":
    main()
