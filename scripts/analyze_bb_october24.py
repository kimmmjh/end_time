#!/usr/bin/env python3
"""Archive and audit the depth 6--8 campaign; rescore saved shots, no inference.

Run with --archive once to move the seven imported root folders. Subsequent
runs verify the relocation manifest and regenerate statistics and figures.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_october24_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bb_october24_cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_bb_october10 import audit_bank, csv_read_tsv, paired, read, sha
from scripts.summarize_bb_september27 import close, holm, score_pauli, wilson
from src.bb_code import BBCodeSpec
from scripts.bb_cnn_plot_osd import STANDARD, interval, report_index, standard_metrics

BASE = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn"
OUT = ROOT / "results/analysis"
PLOTS = ROOT / "results/plots/bb/code_capacity/tanner_cnn"
STEM = "bb_update_2026_10_24"
JOBS = {59707467: ("bb144", 5), 59765839: ("bb144", 8),
        59765840: ("bb72", 6), 59795406: ("bb144", 6),
        59892117: ("bb72", 7), 59928152: ("bb144", 7), 59948099: ("bb72", 8)}


def destination(job):
    code, depth = JOBS[job]
    parent = BASE / ("continued_training" if depth == 5 else f"800_epochs/syndrome_osd0/{code}")
    return parent / f"resdir_{job}"


def write_csv(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def relocate(move):
    manifest = OUT / f"{STEM}_relocation.csv"
    if move and not manifest.exists():
        rows = []
        for job in JOBS:
            source, dest = ROOT / f"resdir_{job}", destination(job)
            assert source.is_dir() and not dest.exists(), (source, dest)
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    rows.append(dict(job=job, old_path=str(p.relative_to(ROOT)),
                                     new_path=str((dest / p.relative_to(source)).relative_to(ROOT)),
                                     bytes=p.stat().st_size, sha256=sha(p)))
        write_csv("relocation", rows)  # Preserve original hashes before moving.
        for job in JOBS:
            dest = destination(job)
            dest.parent.mkdir(parents=True, exist_ok=True)
            (ROOT / f"resdir_{job}").rename(dest)
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {p for job in JOBS for p in destination(job).rglob("*") if p.is_file()} == {
        ROOT / r["new_path"] for r in rows}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"], p
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows),
                relocation_hashes_verified=True)


def analyze():
    finals, statuses, training, validation, pairs, resume_pairs = [], [], [], [], [], []
    banks, histories = {}, {}
    for job, (code_name, depth) in JOBS.items():
        folder = destination(job)
        assert (folder / "completed.txt").exists() and not (folder / "failed.txt").exists()
        assert sha(folder / "source_snapshot/src/bb_code.py") == sha(ROOT / "src/bb_code.py")
        specs = csv_read_tsv(folder / "experiments.tsv")
        paths = sorted(folder.glob("outputs/*/*/history.json"))
        assert len(paths) == len(specs) == (1 if depth == 5 else 4)
        metadata = (folder / "job_metadata.txt").read_text()
        started = dt.datetime.fromisoformat(metadata.split("Started: ")[1].splitlines()[0].replace("Z", "+00:00"))
        for path in paths:
            h = read(path)
            c, f = h["config"], h["final"]
            key = (c["code"], c["error_rate"], c["cnn_depth"])
            assert key[0] == code_name and key[2] == depth
            spec = next(s for s in specs if f"--seed={c['seed']}" in s["arguments"])
            index = spec["experiment_index"]
            args = dict(a[2:].split("=", 1) for a in shlex.split(spec["arguments"]) if "=" in a)
            assert args["code"] == code_name and float(args["p"]) == key[1]
            assert int(args["depth" if depth == 5 else "bb_cnn_depth"]) == depth
            assert (folder / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
            log = (folder / f"log_exp_{index}.txt").read_text()
            assert "Traceback (most recent call last)" not in log and "[Selected best by raw LER]" in log
            assert [r["epoch"] for r in h["train"]] == list(range(800))
            assert c["cnn_width"] == 64 and c["channel"] == "depolarizing" and c["noise_model"] == "capacity"
            assert c["osd"] == ("direct_hard_centered_osd0_v1" if depth == 5 else "syndrome_osd0_v1")
            assert c["checkpoint_selection"] == "raw_logical_accuracy" and c["osd_extra_bp_iterations"] == 0
            code = BBCodeSpec.from_name(code_name)
            assert c["graph_fingerprint"] == hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
            assert c["parameter_count"] == 54404 + (depth - 1)*106880
            phase = h["phases"][-1]
            for name, value in dict(batch_size=64, batches=128, eval_batches=64,
                                    eval_every=5, final_eval_batches=1024).items():
                assert phase[name] == value
            if depth > 5:
                assert len(h["phases"]) == 1 and phase["epochs"] == 800
                assert phase["learning_rate"] == .0003 and args["bb_cnn_osd"] == "syndrome"
            best = max(h["eval"], key=lambda r: r["raw"]["logical_accuracy"])
            assert best["epoch"] == h["best_epoch"] == f["epoch"] and f["shots"] == 65536
            ckpt = torch.load(path.with_name("model.pt"), map_location="cpu", weights_only=False)
            assert ckpt["epoch"] == 799 and ckpt["best_epoch"] == f["epoch"] and ckpt["config"] == c
            for name in ("train", "eval", "phases", "final"):
                assert ckpt["history"][name] == h[name]
            for name in ("model_state_dict", "best_model_state_dict"):
                assert all(torch.isfinite(t).all() for t in ckpt[name].values())
            selected = torch.load(path.with_name("best_model.pt"), map_location="cpu", weights_only=False)
            assert selected["epoch"] == f["epoch"]
            for name, value in ckpt["best_model_state_dict"].items():
                assert torch.equal(value, selected["model_state_dict"][name])
            del ckpt, selected
            score, outcomes = audit_bank(path.with_name("final_shots.npz"), c, f)
            finished = (folder / f"finished_exp_{index}.txt").read_text().split("Finished: ")[1].splitlines()[0]
            elapsed = (dt.datetime.fromisoformat(finished.replace("Z", "+00:00")) - started).total_seconds()/3600
            meta = dict(job=job, experiment=int(index), code=code_name, p=key[1], depth=depth,
                        seed=c["seed"], source=str(path.relative_to(ROOT)))
            row = dict(**meta, width=64, parameters=c["parameter_count"], epochs=800,
                       training_shots=800*8192, shots=65536, best_epoch=f["epoch"]+1,
                       osd_convention=c["osd"], schedule="400+400 restart" if depth == 5 else "fresh 800 cosine",
                       osd_call_fraction=f["osd_call_fraction"],
                       nn_us_per_shot_amortized=f["nn_batch_seconds"]*1e6/f["shots"],
                       osd_us_per_shot_amortized=f["osd_with_transfer_seconds"]*1e6/f["shots"],
                       completion_observed_hours=elapsed,
                       loss_last50=float(np.mean([t["total"] for t in h["train"][-50:]])),
                       loss_previous50=float(np.mean([t["total"] for t in h["train"][-100:-50]])), **score)
            finals.append(row)
            statuses.append(dict(**meta, epochs=800, exit_code=0, best_epoch=f["epoch"]+1,
                                 source_code_hash_verified=True, checkpoint_and_bank_verified=True))
            training.extend({**meta, **t} for t in h["train"])
            validation.extend(dict(**meta, epoch=e["epoch"]+1, raw_ler=e["raw"]["logical_error_rate"],
                                   osd_ler=e["osd"]["logical_error_rate"]) for e in h["eval"])
            banks[key], histories[key] = outcomes, h
            if depth == 5:
                assert phase["start_epoch"] == 400 and phase["epochs"] == 400 and phase["learning_rate"] == .0001
                source = h["resume_comparison"]["source"]
                suffix = Path(source["checkpoint"]).parts
                start = next(i for i, v in enumerate(suffix) if v.startswith("resdir_"))
                source_path = BASE.joinpath("continued_training", *suffix[start:])
                assert sha(source_path) == source["sha256"]
                original = torch.load(source_path, map_location="cpu", weights_only=False)
                assert h["train"][:400] == original["history"]["train"]
                del original
                ref_score, ref_out = audit_bank(path.with_name("reference_final_shots.npz"), c,
                                                h["resume_comparison"]["reference"])
                assert ref_score["shot_bank_sha256"] == score["shot_bank_sha256"]
                for method in ("raw", "osd"):
                    p = paired(ref_out[method], outcomes[method])
                    for name in ("rescued", "harmed", "ler_reduction", "paired_se", "shots"):
                        close(p[name], h["resume_comparison"]["paired"][method][name])
                    resume_pairs.append(dict(**meta, method=method, old_ler=ref_score[f"{method}_ler"],
                                             new_ler=score[f"{method}_ler"], **p))
            print(f"Audited {code_name} d{depth} p={key[1]} raw={score['raw_ler']:.6%} OSD={score['osd_ler']:.6%}", flush=True)
    for a, b in itertools.combinations([r for r in finals if r["depth"] > 5], 2):
        if (a["code"], a["p"], a["seed"]) != (b["code"], b["p"], b["seed"]):
            continue
        a, b = sorted((a, b), key=lambda r: r["depth"])
        assert a["shot_bank_sha256"] == b["shot_bank_sha256"]
        for method in ("raw", "osd"):
            pairs.append(dict(code=a["code"], p=a["p"], from_depth=a["depth"], to_depth=b["depth"],
                              method=method, old_ler=a[f"{method}_ler"], new_ler=b[f"{method}_ler"],
                              **paired(banks[(a["code"], a["p"], a["depth"])][method],
                                       banks[(b["code"], b["p"], b["depth"])][method])))
    assert len(finals) == 25 and len(pairs) == 48
    holm(pairs)
    holm(resume_pairs)
    for suffix, rows in (("cnn_final", finals), ("cnn_status", statuses), ("cnn_train", training),
                         ("cnn_validation", validation), ("cnn_depth_pairs", pairs), ("cnn_resume_pairs", resume_pairs)):
        write_csv(suffix, rows)
    return finals, pairs, resume_pairs, histories


def posthoc(finals):
    folder = destination(59707467)
    assert (folder / "exit_code_standard_osd0.txt").read_text().strip() == "0"
    paths = list(folder.rglob("report.json"))
    assert len(paths) == 1
    report = read(paths[0])
    run = ROOT / next(r["source"] for r in finals if r["depth"] == 5)
    for filename, key in (("model.pt", "checkpoint_sha256"), ("final_shots.npz", "shot_bank_sha256"),
                          ("history.json", "history_sha256")):
        assert sha(run.with_name(filename)) == report[key]
    with np.load(paths[0].with_name("paired_shots.npz")) as f:
        b = dict(f)
    with np.load(run.with_name("final_shots.npz")) as f:
        for key in ("syndrome", "pauli", "raw_correction", "raw_success"):
            assert np.array_equal(f[key], b[key])
        assert np.array_equal(f["osd_correction"], b["residual_osd0_correction"])
    code = BBCodeSpec.bb144()
    rows, outcomes = [], {}
    for name in ("raw", "residual_osd0", "syndrome_osd0"):
        success, valid = score_pauli(b[f"{name}_correction"], b["pauli"], b["syndrome"], code)
        assert np.array_equal(success, b[f"{name}_success"].astype(bool))
        failures = int((~success).sum())
        close(failures/len(success), report["metrics"][name]["logical_error_rate"])
        if name != "raw":
            assert valid.all()
        lo, hi = wilson(failures, len(success))
        rows.append(dict(code="bb144", depth=5, p=.04, method=name, failures=failures,
                         shots=len(success), ler=failures/len(success), ci95_low=lo, ci95_high=hi))
        outcomes[name] = success
    pair = paired(outcomes["residual_osd0"], outcomes["syndrome_osd0"])
    for key in ("rescued", "harmed", "ler_reduction", "paired_se"):
        close(pair[key], report["paired"]["standard_vs_residual"][key])
    write_csv("depth5_standard_osd", rows)
    write_csv("depth5_standard_vs_residual", [pair])
    return rows, pair


def references(finals):
    index = report_index()
    latest = {}
    for p in BASE.rglob("history.json"):
        h = read(p)
        c = h.get("config", {})
        if c.get("cnn_depth") != 5 or "final" not in h:
            continue
        key = (c["code"], c["error_rate"])
        if key not in latest or len(h["train"]) > len(latest[key][1]["train"]):
            latest[key] = (p, h)
    rows = []
    for (code, p), (path, h) in sorted(latest.items()):
        f = h["final"]
        result = standard_metrics(path, index)
        row = dict(code=code, p=p, depth=5, epochs=len(h["train"]), osd_convention=STANDARD,
                   raw_ler=f["raw"]["logical_error_rate"], recorded_osd_ler=f["osd"]["logical_error_rate"],
                   osd_ler=None, osd_status="missing", osd_source="",
                   source=str(path.relative_to(ROOT)))
        if result is not None:
            metric, report_path, status = result
            count, lo, hi = interval(metric["logical_error_rate"], f["shots"])
            row.update(osd_ler=metric["logical_error_rate"], osd_failures=count,
                       osd_ci95_low=lo, osd_ci95_high=hi, osd_status=status,
                       osd_source=str(report_path.relative_to(ROOT)))
        rows.append(row)
    assert len(rows) == 8
    write_csv("depth5_references", rows)
    return rows


def plot(finals, refs, standard, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {6: "#2864b4", 7: "#d98116", 8: "#14856f"}
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i, code in enumerate(("bb72", "bb144")):
        for j, method in enumerate(("raw", "osd")):
            ax = axes[i, j]
            for depth, color in colors.items():
                rows = sorted((r for r in finals if r["code"] == code and r["depth"] == depth), key=lambda r: r["p"])
                ax.plot([r["p"]*100 for r in rows], [r[f"{method}_ler"]*100 or np.nan for r in rows],
                        color=color, label=f"Depth {depth}" + (", syndrome OSD" if method == "osd" else ""))
                for r in rows:
                    y, lo, hi = (r[f"{method}_{k}"]*100 for k in ("ler", "ci95_low", "ci95_high"))
                    if y:
                        ax.errorbar(r["p"]*100, y, yerr=[[y-lo], [hi-y]], fmt="o", color=color, capsize=3)
                    else:
                        ax.plot(r["p"]*100, hi, marker="v", mfc="none", color=color, ms=8)
            old = sorted((r for r in refs if r["code"] == code), key=lambda r: r["p"])
            if method == "raw":
                ax.plot([r["p"]*100 for r in old], [r["raw_ler"]*100 for r in old], color="#777777",
                        ls="--", marker="x", label="Depth 5 (different schedule)")
            else:
                ax.plot([r["p"]*100 for r in old], [r["recorded_osd_ler"]*100 or np.nan for r in old],
                        color="#777777", ls="--", marker="s", ms=4,
                        label="Depth 5, residual OSD (different schedule)")
                available = [r for r in old if r["osd_status"] != "missing"]
                if available:
                    ax.plot([r["p"]*100 for r in old],
                            [r["osd_ler"]*100 if r["osd_ler"] else np.nan for r in old],
                            color="#777777", ls="-", marker="*", label="Depth 5, syndrome OSD (available points)")
                    for r in available:
                        y, lo, hi = (r[f"osd_{k}"]*100 for k in ("ler", "ci95_low", "ci95_high"))
                        if y:
                            ax.errorbar(r["p"]*100, y, yerr=[[y-lo], [hi-y]], fmt="*", color="#444444", ms=12, capsize=3)
                        else:
                            ax.plot(r["p"]*100, hi, marker="v", mfc="none", color="#444444")
                ax.text(.97, .03, f'Depth 5: residual OSD shown at all 4 points\nSyndrome OSD available at {len(available)}/4 points',
                        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
            ax.set(title=f"{code.upper()} | {'CNN alone' if method == 'raw' else 'CNN + OSD-0 (both conventions)'}",
                   xlabel="Physical error rate p (%)", ylabel="Block LER (%)", yscale="log", xticks=[2, 4, 6, 8])
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Depth 6–8: 800 epochs, width 64, 65,536 final shots\n95% Wilson intervals; hollow triangles = upper bounds for zero failures", fontsize=13)
    fig.savefig(PLOTS / "depth678.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, code in enumerate(("bb72", "bb144")):
        for j, p in enumerate((.02, .04, .06, .08)):
            ax = axes[i, j]
            for depth, color in colors.items():
                h = histories[(code, p, depth)]
                e = h["eval"]
                y = np.array([r["raw"]["logical_error_rate"] for r in e])*100
                smooth = np.convolve(y, np.ones(5)/5, mode="valid")
                ax.plot([r["epoch"]+1 for r in e[4:]], smooth, label=f"Depth {depth}", color=color)
            ax.set(title=f"{code.upper()} p={p:.0%}", xlabel="Epoch", ylabel="Validation raw LER (%)", yscale="log")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Raw validation LER: trailing mean of 5 evaluations (25 epochs)")
    fig.savefig(PLOTS / "depth678_validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = relocate(args.archive)
    finals, pairs, resume, histories = analyze()
    standard, standard_pair = posthoc(finals)
    refs = references(finals)
    plot(finals, refs, standard, histories)
    audit.update(completed_jobs=7, completed_models=25, depth678_models=24,
                 final_model_shots_rescored=25*65536, reference_model_shots_rescored=65536,
                 posthoc_model_shots_rescored=3*65536, depth_tests=48,
                 unique_depth678_shot_banks=len({r["shot_bank_sha256"] for r in finals if r["depth"] > 5}),
                 same_bank_comparisons=True, training_steps_added=0, inference_steps_added=0,
                 standard_vs_residual=standard_pair,
                 depth_significance=[dict(from_depth=a, to_depth=b, method=m,
                     improved=int(sum(r["paired_holm_p"]<.05 and r["ler_reduction"]>0 for r in pairs
                                  if (r["from_depth"],r["to_depth"],r["method"])==(a,b,m))),
                     worsened=int(sum(r["paired_holm_p"]<.05 and r["ler_reduction"]<0 for r in pairs
                                  if (r["from_depth"],r["to_depth"],r["method"])==(a,b,m))))
                     for a,b in ((6,7),(7,8),(6,8)) for m in ("raw","osd")])
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
