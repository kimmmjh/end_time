#!/usr/bin/env python3
"""Audit the imported October 10 jobs without training or rerunning decoders.

CNN corrections are rescored. Circuit results with omitted chunks are explicitly
aggregate-only. Inputs are archived under results; historical reports stay intact.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_october10_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bb_code import BBCodeSpec
from scripts.summarize_bb_september27 import score_pauli, wilson, close, holm

JOBS = (58939141, 58939143, 58939147, 58962932, 58962935, 58962937,
        59122447, 59122448, 59122449, 59122450, 59122451, 59122452)
OUT = ROOT / "results/analysis"
BASELINE_PLOT = ROOT / "results/plots/bb/circuit/library_bp_osd/lowp_million.png"
STEM = "bb_update_2026_10_10"


def job_directory(job_id):
    """Locate an intact job folder in the organized result archive."""
    if job_id in JOBS[:3]:
        group = "replicates" if job_id == 58939147 else "bb144"
        parent = ROOT / "results/bb/circuit/library_bp_osd" / group
    else:
        group = {58962932: "bb72", 58962935: "bb144", 58962937: "replicates",
                 59122447: "bb72", 59122448: "bb144", 59122449: "bb72",
                 59122450: "bb144", 59122451: "bb72", 59122452: "bb144"}[job_id]
        parent = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn/400_epochs" / group
    return parent / f"resdir_{job_id}"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def paired(old, new):
    rescued, harmed = int((~old & new).sum()), int((old & ~new).sum())
    n = len(old)
    gain = (rescued - harmed) / n
    se = math.sqrt(max(0., (rescued + harmed) / n - gain * gain) / (n - 1)) if n > 1 else 0.
    return dict(shots=n, rescued=rescued, harmed=harmed, ler_reduction=gain,
                paired_se=se, gain_ci95_low=gain - 1.96 * se,
                gain_ci95_high=gain + 1.96 * se,
                paired_exact_p=binomtest(rescued, rescued + harmed).pvalue if rescued + harmed else 1.)


def audit_bank(path, config, metrics):
    code = BBCodeSpec.from_name(config["code"])
    with np.load(path, allow_pickle=False) as data:
        b = {key: data[key] for key in data.files}
    assert json.loads(str(b["metadata_json"])) == config
    n = metrics["shots"]
    for key in ("syndrome", "pauli", "raw_correction", "osd_correction"):
        assert b[key].shape == (n, code.n) and b[key].dtype == np.uint8
    assert np.isin(b["syndrome"], (0, 1)).all()
    for key in ("pauli", "raw_correction", "osd_correction"):
        assert np.isin(b[key], (0, 1, 2, 3)).all()
    # Recompute truth syndrome, not just the saved success flags.
    truth_ok, _ = score_pauli(b["pauli"], b["pauli"], b["syndrome"], code)
    assert truth_ok.all()
    digest = hashlib.sha256(b["pauli"].tobytes() + b["syndrome"].tobytes()).hexdigest()
    row, outcomes = {"shot_bank_sha256": digest}, {}
    for method in ("raw", "osd"):
        success, valid = score_pauli(b[f"{method}_correction"], b["pauli"], b["syndrome"], code)
        assert np.array_equal(success, b[f"{method}_success"].astype(bool))
        failures, flagged = int((~success).sum()), int((~valid).sum())
        close(metrics[method]["logical_error_rate"], failures / n)
        close(metrics[method]["flagged_failure_rate"], flagged / n)
        close(metrics[method]["unflagged_logical_failure_rate"], (failures - flagged) / n)
        lo, hi = wilson(failures, n)
        row.update({f"{method}_{k}": v for k, v in dict(failures=failures, ler=failures / n,
                   ci95_low=lo, ci95_high=hi, flagged=flagged, unflagged=failures - flagged).items()})
        outcomes[method] = success
        if method == "raw":
            close(metrics["osd_call_fraction"], flagged / n)
            assert np.array_equal(b["raw_correction"][valid], b["osd_correction"][valid])
        else:
            assert valid.all()
    gain = paired(outcomes["raw"], outcomes["osd"])
    assert gain["harmed"] == metrics["harmed"] == 0
    assert gain["rescued"] == metrics["rescued"]
    close(gain["ler_reduction"], metrics["osd_minus_raw_gain"])
    return row, outcomes


def analyze_cnn():
    finals, status, training, validation, resume, banks, histories = [], [], [], [], [], {}, {}
    for job_id in JOBS[3:]:
        job = job_directory(job_id)
        assert sha(job / "source_snapshot/src/bb_code.py") == sha(ROOT / "src/bb_code.py")
        paths = sorted(job.glob("outputs/*/*/history.json"))
        assert len(paths) == 4
        specs = csv_read_tsv(job / "experiments.tsv")
        for path in paths:
            h = read(path)
            c = h["config"]
            key = (c["code"], c["error_rate"], c["seed"], c["cnn_depth"])
            meta = dict(job=job_id, code=key[0], p=key[1], seed=key[2], depth=key[3])
            spec = next(s for s in specs if f"--seed={key[2]}" in s["arguments"])
            index = spec["experiment_index"]
            exit_path = job / f"exit_code_exp_{index}.txt"
            completed = len(h["train"])
            assert [r["epoch"] for r in h["train"]] == list(range(completed))
            assert c["cnn_width"] == 64 and c["noise_model"] == "capacity"
            assert c["channel"] == "depolarizing" and c["checkpoint_selection"] == "raw_logical_accuracy"
            assert c["osd"] == "direct_hard_centered_osd0_v1" and c["osd_extra_bp_iterations"] == 0
            code = BBCodeSpec.from_name(key[0])
            assert c["graph_fingerprint"] == hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
            assert c["parameter_count"] == 54404 + (key[3] - 1) * 106880
            for phase in h["phases"]:
                assert all(phase[k] == v for k, v in dict(batch_size=64, batches=128,
                    eval_batches=64, eval_every=5, final_eval_batches=1024, learning_rate=.0003).items())
            selected = max(h["eval"], key=lambda e: e["raw"]["logical_accuracy"])
            assert selected["epoch"] == h["best_epoch"]
            status.append({**meta, "experiment": index, "completed_epochs": completed,
                "has_final": "final" in h, "exit_code": exit_path.read_text().strip() if exit_path.exists() else "missing",
                "source": str(path.relative_to(ROOT))})
            for r in h["train"]:
                training.append({**meta, **r})
            for r in h["eval"]:
                validation.append({**meta, "epoch": r["epoch"] + 1, "shots": r["shots"],
                                   "raw_ler": r["raw"]["logical_error_rate"], "osd_ler": r["osd"]["logical_error_rate"]})
            if "final" not in h:
                continue
            assert completed == 400 and exit_path.read_text().strip() == "0"
            f = h["final"]
            assert f["epoch"] == selected["epoch"] and f["shots"] == 65536
            scored, outcomes = audit_bank(path.with_name("final_shots.npz"), c, f)
            row = {**meta, "width": 64, "parameters": c["parameter_count"], "epochs": completed,
                   "training_shots": completed * 8192, "shots": f["shots"], "best_epoch": f["epoch"] + 1,
                   "schedule": "100+300 cosine restart" if key[3] == 2 else "400 cosine from scratch",
                   "osd_call_fraction": f["osd_call_fraction"],
                   # Despite its name, the trainer sums this timer across all evaluation batches.
                   "nn_us_per_shot_amortized": f["nn_batch_seconds"] * 1e6 / f["shots"],
                   "osd_us_per_shot_amortized": f["osd_with_transfer_seconds"] * 1e6 / f["shots"],
                   "last50_mean_loss": float(np.mean([r["total"] for r in h["train"][-50:]])),
                   "previous50_mean_loss": float(np.mean([r["total"] for r in h["train"][-100:-50]])),
                   "last5_val_raw_ler": float(np.mean([r["raw"]["logical_error_rate"] for r in h["eval"][-5:]])),
                   "source": str(path.relative_to(ROOT)), **scored}
            finals.append(row)
            banks[key] = outcomes
            histories[key] = h
            if "resume_comparison" in h:
                comp = h["resume_comparison"]
                refrow, refout = audit_bank(path.with_name("reference_final_shots.npz"), c, comp["reference"])
                assert refrow["shot_bank_sha256"] == scored["shot_bank_sha256"]
                source = comp["source"]
                original = ROOT / source["checkpoint"].split("/end_time/")[1]
                assert sha(original) == source["sha256"]
                original_h = read(original.with_name("history.json"))
                assert h["train"][:100] == original_h["train"]
                assert h["eval"][:20] == original_h["eval"]
                for method in ("raw", "osd"):
                    pair = paired(refout[method], outcomes[method])
                    for name in ("rescued", "harmed", "ler_reduction", "paired_se"):
                        close(pair[name], comp["paired"][method][name])
                    resume.append({**meta, "method": method, "old_ler_same_bank": refrow[f"{method}_ler"],
                                   "new_ler": row[f"{method}_ler"], **pair})
            print(f"Audited CNN {job_id} {key}: raw={row['raw_ler']:.6g}, OSD={row['osd_ler']:.6g}", flush=True)
    holm(resume)
    depth_pairs = []
    for a, b in itertools.combinations(finals, 2):
        if (a["code"], a["p"], a["seed"]) != (b["code"], b["p"], b["seed"]):
            continue
        if a["depth"] > b["depth"]:
            a, b = b, a
        ka = (a["code"], a["p"], a["seed"], a["depth"])
        kb = (b["code"], b["p"], b["seed"], b["depth"])
        for method in ("raw", "osd"):
            pair = dict(code=a["code"], p=a["p"], seed=a["seed"], from_depth=a["depth"],
                        to_depth=b["depth"], method=method, old_ler=a[f"{method}_ler"], new_ler=b[f"{method}_ler"])
            if a["shot_bank_sha256"] == b["shot_bank_sha256"]:
                pair.update(comparison="paired identical bank", **paired(banks[ka][method], banks[kb][method]))
            else:
                pair.update(comparison="different banks; descriptive only",
                            ler_reduction=a[f"{method}_ler"] - b[f"{method}_ler"])
            depth_pairs.append(pair)
    holm([r for r in depth_pairs if "paired_exact_p" in r])
    for name, rows in (("cnn_final", finals), ("cnn_status", status), ("cnn_train", training),
                       ("cnn_validation", validation), ("cnn_resume_pairs", resume), ("cnn_depth_pairs", depth_pairs)):
        write(name, rows)
    return finals, status, histories, resume, depth_pairs


def csv_read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def analyze_baselines():
    status, rows, comparisons = [], [], []
    for job_id in JOBS[:3]:
        job = job_directory(job_id)
        for path in sorted(job.glob("exp_*/results.json")):
            report = read(path)
            c = report["config"]
            exp = path.parent
            meta = dict(job=job_id, experiment=exp.name, code=c["code"], p=c["p"], seed=c["seed"])
            log = (job / f"log_{exp.name}.txt").read_text()
            status.append({**meta, "reported_status": report["status"], "shots_completed": report["shots_completed"],
                "target_shots": c["shots"], "chunk_files": len(list((exp / "chunks").glob("*.npz"))),
                "oom_in_log": "oom_kill" in log or "Out Of Memory" in log,
                "time_limit_in_log": "TIME LIMIT" in log, "source": str(path.relative_to(ROOT))})
            assert c["circuit_policy"] == "bravyi_8tick_plus_memory_v1"
            assert c["max_iterations"] == 1000 and c["ms_scaling_factor"] == 1.
            assert c["detector_inputs"] == ["x", "xz"] and c["library_version"] == "2.4.1"
            for source, digest in c["source_sha256"].items():
                snapshot = job / ("entrypoint_snapshot.py" if source.startswith("scripts/") else f"source_snapshot/{source}")
                assert sha(snapshot) == digest
            assert sha(exp / "shots.npz") == report["shot_bank_sha256"]
            for name, key in (("circuit.stim", "circuit_sha256"), ("detector_error_model.dem", "dem_sha256")):
                assert sha(exp / name) == c[key]
            if report["status"] != "complete":
                assert not report["decoders"] and report["shots_completed"] == 0
                continue
            n = c["shots"]
            assert report["shots_completed"] == n
            summary = {r["decoder"]: r for r in csv_read(exp / "summary.csv")}
            for name, r in report["decoders"].items():
                assert r["shots"] == n
                assert r["failures"] == r["flagged_failures"] + r["unflagged_failures"]
                assert 0 <= r["logical_mismatches"] <= r["failures"] <= n
                close(r["syndrome_convergence"], 1 - r["flagged_failures"] / n)
                close(r["mean_bp_iterations"], r["bp_iteration_sum"] / n)
                for metric, count in (("block_failure", "failures"), ("logical_z_mismatch", "logical_mismatches")):
                    close(r[f"{metric}_rate"], r[count] / n)
                    lo, hi = wilson(r[count], n)
                    close(r[f"{metric}_ci95_low"], lo)
                    close(r[f"{metric}_ci95_high"], hi)
                for k, v in r.items():
                    close(float(summary[name][k]), v)
                if not name.endswith("_bp"):
                    assert r["flagged_failures"] == r["harmed"] == 0
                    assert r["logical_mismatches"] == r["failures"]
                    ref = report["decoders"][name.split("_")[0] + "_bp"]
                    for metric, rescued, harmed, count in (("block", "rescued", "harmed", "failures"),
                            ("logical", "logical_rescued", "logical_harmed", "logical_mismatches")):
                        gain = (r[rescued] - r[harmed]) / n
                        close(gain, (ref[count] - r[count]) / n)
                        discordant = r[rescued] + r[harmed]
                        comparisons.append({**meta, "decoder": name, "metric": metric, "ler_reduction": gain,
                            "rescued": r[rescued], "harmed": r[harmed],
                            "paired_exact_p": binomtest(r[rescued], discordant).pvalue if discordant else 1.,
                            "audit": "server aggregate counts; no chunk rescoring"})
                rows.append({**meta, "decoder": name, **r, "audit": "aggregate arithmetic and CSV checked; chunks omitted"})
    holm(comparisons)
    write("baseline_status", status)
    write("baseline_final", rows)
    write("baseline_vs_bp_pairs", comparisons)
    return rows, status


def figures(finals, histories, baseline):
    from scripts.plot_bb_cnn_all import main as plot_all
    plot_all()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    styles = {"bp": ("#7c8795", "o"), "bposd0": ("#3478c6", "s"), "bposd_cs3": ("#c95235", "^")}
    for ax, mode in zip(axes, ("x", "xz")):
        for method, (color, marker) in styles.items():
            rr = sorted([r for r in baseline if r["decoder"] == f"{mode}_{method}"], key=lambda r: r["p"])
            ax.errorbar([r["p"] for r in rr], [r["logical_z_mismatch_rate"] for r in rr],
                yerr=[[r["logical_z_mismatch_rate"]-r["logical_z_mismatch_ci95_low"] for r in rr],
                      [r["logical_z_mismatch_ci95_high"]-r["logical_z_mismatch_rate"] for r in rr]],
                color=color, marker=marker, capsize=3, label=method)
        ax.set(title=f"BB72 | {mode.upper()} input", xlabel="Circuit p", ylabel="Logical-Z mismatch rate", yscale="log")
        ax.set_xticks([.001, .002])
        ax.grid(alpha=.2, which="both")
        ax.legend()
    fig.suptitle("Circuit BP/OSD | 1,000,000 shots per p | server aggregates, chunks omitted")
    fig.tight_layout()
    BASELINE_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASELINE_PLOT, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = [{"job": job, "path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
                for job in JOBS for p in sorted(job_directory(job).rglob("*")) if p.is_file()]
    manifest_path = OUT / f"{STEM}_manifest.csv"
    if manifest_path.exists():
        assert [{k: str(v) for k, v in r.items()} for r in manifest] == csv_read(manifest_path), "Imported originals changed"
    else:
        write("manifest", manifest)
    finals, status, histories, resume, depth_pairs = analyze_cnn()
    baseline, baseline_status = analyze_baselines()
    figures(finals, histories, baseline)
    audit = dict(jobs=list(JOBS), imported_files=len(manifest), imported_bytes=sum(r["bytes"] for r in manifest),
        cnn_completed=len(finals), cnn_incomplete=len(status)-len(finals),
        cnn_final_shots_rescored=sum(r["shots"] for r in finals), cnn_reference_shots_rescored=len(resume)//2*65536,
        cnn_unique_final_banks=len({r["shot_bank_sha256"] for r in finals}),
        depth_comparison_rows=len(depth_pairs), paired_depth_rows=sum(isinstance(r.get("paired_exact_p"), (int, float)) for r in depth_pairs),
        baseline_completed=sum(r["reported_status"] == "complete" for r in baseline_status),
        baseline_zero_saved_shots=sum(r["shots_completed"] == 0 for r in baseline_status),
        baseline_audit="Aggregate arithmetic, CSV, source/circuit/DEM/shot-bank hashes verified. No chunks: corrections not rescored.",
        inference_limits=["Single seed per deep-model code/p; depth 2 has 3 seeds only at p=.06.",
            "Depth 2: 100+300 LR restart; deeper models: fresh 400-epoch cosine. Equal shots, unequal compute/schedules.",
            "Paired exact binomial p-values Holm-adjusted separately for resume and depth comparison families.",
            "Zero p-values in CSV mean numerical underflow. Paired gain intervals are normal approximations.",
            "Zero observed failures are not zero risk; Wilson intervals are reported."])
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
