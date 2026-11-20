#!/usr/bin/env python3
"""Audit the Nov 3 follow-up download, archive immutable snapshots, and update plots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_dem_october28 as base
from scripts import analyze_bb_hierarchical_dem as hier

STEM = "bb_dem_2026_11_03"
OUT = ROOT / "results/analysis"
DEM = ROOT / "results/bb/circuit/standard/dem_cnn"
PLOTS = ROOT / "results/plots/bb/circuit/standard/dem_cnn"
JOBS = {60688777: "train", 60688778: "train", 60694880: "evaluate",
        60694884: "train", 60701066: "evaluate", 60701325: "train"}
DESTINATIONS = {f"resdir_{job}_followup_{mode}": DEM / "followup_2026_11_02" / mode / f"resdir_{job}_followup_{mode}"
                for job, mode in JOBS.items()}
read = base.read


def write_csv(suffix, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def key(config):
    return config["code"], config["depth"], config["noise_profile"]["base_error_rate"]


def flatten_result(result):
    row = dict(shots=result["shots"], status=result["status"],
               global_call_fraction=result["global_call_fraction"],
               amortized_ms_per_shot=1000*result["hierarchical_seconds"]/result["shots"],
               rescued=result["hierarchical_vs_raw"]["rescued"], harmed=result["hierarchical_vs_raw"]["harmed"])
    for arm in hier.ARMS:
        m = result["metrics"][arm]
        row.update({f"{arm}_failures": m["failures"], f"{arm}_rate": m["block_failure_rate"],
                    f"{arm}_low": m["block_failure_wilson95"][0],
                    f"{arm}_high": m["block_failure_wilson95"][1],
                    f"{arm}_syndrome_rate": m["syndrome_failure_rate"]})
    return row


def audit_job(directory, jobs=None):
    metadata = (directory / "job_metadata.txt").read_text()
    job = int(re.search(r"^Job ID: (\d+)$", metadata, re.M)[1])
    mode = (JOBS if jobs is None else jobs)[job]
    array, task = map(int, re.search(r"^Array job/task: (\d+)/(\d+)$", metadata, re.M).groups())
    assert array == (60688774 if mode == "train" else 60688775)
    assert f"mode: {mode}" in metadata
    manifest = read(directory / "source_manifest.json")
    assert manifest == read(ROOT / "scripts/manifests/bb_dem_followup_2026_11_02.json")
    selection = [(e, w) for e in manifest["experiments"] if e["action"] == mode
                 for w in (("latest",) if mode == "train" else ("best", "latest"))]
    ids = list(range(task*4, min(task*4+4, len(selection))))
    assert sorted(int(p.name.split("_")[-1]) for p in directory.glob("worker_*")) == ids
    completed = (directory / "completed.txt").exists()
    assert not (directory / "failed.txt").exists()
    train, evaluations, snapshots = [], [], []
    for path in sorted((directory / "source_snapshot").rglob("*")):
        if path.is_file():
            rel = path.relative_to(directory / "source_snapshot")
            current = ROOT / rel
            snapshots.append(dict(job=job, file=str(rel), sha256=base.sha(path),
                                  matches_current=current.is_file() and base.sha(current) == base.sha(path)))
    for worker in ids:
        w = directory / f"worker_{worker}"
        entry, weights = selection[worker]
        original = read(ROOT / entry["imported_run"] / "history.json")
        c = original["config"]
        assert all(c[k] == v for k, v in entry["expected_config"].items())
        code, depth, p = key(c)
        meta = dict(job=job, worker=worker, code=code, depth=depth, p=p,
                    source=str(w.relative_to(ROOT)))
        command = shlex.split((directory / f"command_worker_{worker}.txt").read_text())
        for opt, val in (("--worker", str(worker)), ("--mode", mode), ("--shots", "8192"),
                         ("--target-epochs", "1000"), ("--learning-rate", "0.00003")):
            assert command[command.index(opt)+1] == val
        log = (directory / f"log_worker_{worker}.txt").read_text()
        assert "Traceback (most recent call last)" not in log
        exit_path = directory / f"exit_code_worker_{worker}.txt"
        if completed:
            assert exit_path.read_text().strip() == "0"
        else:
            assert not exit_path.exists()
        if mode == "train":
            paths = list(w.glob("outputs/*/*/history.json"))
            assert len(paths) == 1
            history = read(paths[0])
            plan = read(w / "resume_plan.json")
            assert plan["source"] == entry and plan["weights"] == "latest"
            assert plan["resume"]["completed_epochs"] == 800 and plan["resume"]["additional_epochs"] == 200
            assert plan["resume"]["resume_learning_rate"] == 3e-5
            assert plan["resume"]["prior_best_epoch"] == original["best_epoch"]+1
            assert history["config"] == c
            n = len(history["train"])
            assert 800 < n <= 1000 and [r["epoch"] for r in history["train"]] == list(range(n))
            assert history["train"][:800] == original["train"]
            assert history["eval"][:len(original["eval"])] == original["eval"]
            assert history["phases"][:-1] == original["phases"]
            phase = history["phases"][-1]
            assert phase["start_epoch"] == 800 and phase["epochs"] == 200
            assert phase["schedule"] == "cosine_restart" and phase["learning_rate"] == 3e-5
            for row in history["train"]:
                assert all(math.isfinite(v) for v in row.values())
                assert abs(row["total"]-row["syndrome"]-row["logical"]-.1*row["mechanism"]) < 1e-6
            for row in history["eval"]:
                base.audit_evaluation(row, 1024)
            best = max(history["eval"], key=lambda r:r["raw"]["logical_accuracy"])
            assert best["epoch"] == history["best_epoch"]
            assert best["raw"]["logical_accuracy"] == history["best_raw_accuracy"]
            for log_text in (log, paths[0].with_name("training_log.txt").read_text()):
                for label, field in (("Train", "train"), ("Validation", "eval")):
                    records = [dict(epoch=int(epoch), **json.loads(v)) for epoch, v in
                               re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$", log_text, re.M)]
                    assert records == [r for r in history[field] if r["epoch"] >= 800]
                final_records = re.findall(r"^\[Final selected best\] (.+)$", log_text, re.M)
                if completed:
                    assert len(final_records) == 1 and json.loads(final_records[0]) == history["final"]
                else:
                    assert not final_records and "final" not in history
            assert completed == (n == 1000 and "final" in history)
            row = dict(**meta, epochs=n, status="complete" if completed else "partial_snapshot",
                       previous_best_epoch=original["best_epoch"]+1, best_epoch=history["best_epoch"]+1,
                       best_updated=history["best_epoch"] >= 800,
                       old_final_rate=original["final"]["raw"]["logical_error_rate"],
                       old_final_count=original["final"]["raw"]["failures"],
                       last_validation_epoch=history["eval"][-1]["epoch"]+1,
                       last_validation_rate=history["eval"][-1]["raw"]["logical_error_rate"],
                       old_loss_last50=statistics.mean(r["logical"] for r in original["train"][-50:]),
                       new_loss_last50=statistics.mean(r["logical"] for r in history["train"][-50:]))
            if completed:
                f = history["final"]
                base.audit_evaluation(f, 8192)
                assert f["epoch"] == history["best_epoch"]
                row.update(shots=8192, block_count=f["raw"]["failures"], block_rate=f["raw"]["logical_error_rate"],
                           block_low=hier.wilson(f["raw"]["failures"], 8192)[0],
                           block_high=hier.wilson(f["raw"]["failures"], 8192)[1],
                           delta_pp=100*(f["raw"]["logical_error_rate"]-row["old_final_rate"]))
            train.append((row, history))
        else:
            m, result = read(w / "metadata.json"), read(w / "results.json")
            assert m["config"] == c and m["source"]["weights"] == weights
            assert m["source"]["selected_epoch"] == (original["best_epoch"]+1 if weights == "best" else 800)
            assert m["requested_shots"] == 8192 and m["seed"] == 20261101 and m["batch_size"] == 4
            assert m["confidence_threshold"] == .9 and m["active_candidates_only"]
            assert m["backend"] == "direct_hard_centered_osd0_v1" and not m["compare_direct_osd"]
            assert m["sampling"] == "fresh_exact_circuit"
            assert m["source"]["checkpoint"].endswith(entry["source_run"] + ("/best_model.pt" if weights == "best" else "/model.pt"))
            logs = [json.loads(line) for line in log.splitlines() if line.startswith('{"status":')]
            assert [r["shots"] for r in logs] == list(range(64, logs[-1]["shots"]+1, 64))
            for r in logs + [result]:
                hier.verify_result(r)
                assert r["status"] == ("complete" if r["shots"] == 8192 else "running")
            if completed:
                assert result == logs[-1] and result["shots"] == 8192
            else:
                assert result["status"] == "running" and logs[-1]["shots"] <= result["shots"] < logs[-1]["shots"]+64
                for arm in hier.ARMS:
                    extra = result["metrics"][arm]["failures"]-logs[-1]["metrics"][arm]["failures"]
                    assert 0 <= extra <= result["shots"]-logs[-1]["shots"]
            row = dict(**meta, weights=weights, selected_epoch=m["source"]["selected_epoch"],
                       sampler_seed=m["sampler_seed"], checkpoint_sha256=m["source"]["sha256"],
                       **flatten_result(result))
            evaluations.append((row, {r["shots"]:r for r in logs}))
    return train, evaluations, snapshots


def archive(move):
    path = OUT / f"{STEM}_relocation.csv"
    if not path.exists():
        assert move, "First import requires --archive"
        rows = []
        for name, dest in DESTINATIONS.items():
            source = ROOT / name
            assert source.is_dir() and not dest.exists()
            audit_job(source)
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    rows.append(dict(old_path=str(p.relative_to(ROOT)), new_path=str((dest / p.relative_to(source)).relative_to(ROOT)),
                                     bytes=p.stat().st_size, sha256=base.sha(p)))
        write_csv("relocation", rows)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for name, dest in DESTINATIONS.items():
        source = ROOT / name
        if move and source.exists():
            assert not dest.exists(), "Refusing to merge or overwrite an archived snapshot"
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.rename(dest)
    assert {str(p.relative_to(ROOT)) for d in DESTINATIONS.values() for p in d.rglob("*") if p.is_file()} == {r["new_path"] for r in rows}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and base.sha(p) == r["sha256"]
    return dict(folders=len(DESTINATIONS), files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), hashes_verified=True)


def baseline():
    histories = {}
    for folder in ("800_epochs", "resumed_800_epochs"):
        for p in (DEM / folder).rglob("history.json"):
            h = read(p)
            assert len(h["train"]) == 800 and "final" in h
            histories[key(h["config"])] = h
    assert len(histories) == 32
    return histories


def common_comparisons(evaluations):
    rows = []
    for code, depth, p in sorted({(r["code"],r["depth"],r["p"]) for r, _ in evaluations}):
        pair = {r["weights"]:(r,logs) for r,logs in evaluations if (r["code"],r["depth"],r["p"]) == (code,depth,p)}
        assert set(pair) == {"best", "latest"}
        a, b = pair["best"][0], pair["latest"][0]
        assert a["sampler_seed"] == b["sampler_seed"]
        n = max(set(pair["best"][1]) & set(pair["latest"][1]))
        row = dict(code=code, depth=depth, p=p, shots=n,
                   status="complete" if n == 8192 else "partial_common_logged_prefix",
                   best_epoch=a["selected_epoch"], latest_epoch=b["selected_epoch"])
        for weight in ("best", "latest"):
            result = pair[weight][1][n]
            m = result["metrics"]["hierarchical"]
            row.update({f"{weight}_failures":m["failures"], f"{weight}_rate":m["block_failure_rate"],
                        f"{weight}_low":m["block_failure_wilson95"][0],f"{weight}_high":m["block_failure_wilson95"][1]})
        row["latest_minus_best_pp"] = 100*(row["latest_rate"]-row["best_rate"])
        rows.append(row)
    return rows


def plots(original, training, evaluations, comparisons, *, csv_writer=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    colors = base.COLORS
    emit_csv = write_csv if csv_writer is None else csv_writer
    latest = dict(original)
    final = dict(original)
    partial = set()
    for row, h in training:
        k = key(h["config"])
        latest[k] = h
        if row["status"] == "complete":
            final[k] = h
        else:
            partial.add(k)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    final_rows = []
    for ax, code in zip(axes, ("bb72", "bb144")):
        for depth, color in colors.items():
            keys = sorted(k for k in final if k[:2] == (code, depth))
            x = [100*k[2] for k in keys]
            old = [100*original[k]["final"]["raw"]["logical_error_rate"] for k in keys]
            y = [100*final[k]["final"]["raw"]["logical_error_rate"] for k in keys]
            ax.plot(x, old, ":", color=color, alpha=.4)
            ax.plot(x, y, color=color, label=f"D{depth}")
            for k, xp, yp in zip(keys, x, y):
                h = final[k]; f = h["final"]; n = f["shots"]
                lo, hi = hier.wilson(f["raw"]["failures"], n)
                extended = len(h["train"]) == 1000
                ax.errorbar(xp, yp, yerr=[[max(0,yp-100*lo)], [max(0,100*hi-yp)]],
                            fmt="s" if extended else "o", color=color, mfc=color if extended else "white", capsize=3)
                final_rows.append(dict(code=k[0], depth=k[1], p=k[2], epochs=len(h["train"]),
                                       selected_epoch=h["best_epoch"]+1, shots=n, failures=f["raw"]["failures"],
                                       rate=yp/100, low=lo, high=hi))
        ax.set(title=code.upper(), xlabel="Per-location error rate p (%)", ylabel="Raw block failure (%)",
               xticks=[.1,.2,.4,.6], ylim=(-2,103))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right")
    n_extended = sum(len(h["train"]) == 1000 for h in final.values())
    fig.suptitle(f"Latest completed raw evaluations: {n_extended} runs at 1,000 epochs + {len(final)-n_extended} at 800 epochs\n"
                 "Filled squares: 1,000-epoch runs; open circles: 800; dotted: old 800 baseline\n"
                 "8,192 final shots; validation-selected best weights; 95% Wilson intervals", fontsize=12)
    fig.savefig(PLOTS / "overview.png", dpi=160); plt.close(fig)
    emit_csv("raw_latest_completed", final_rows)

    for kind, filename in (("validation", "validation.png"), ("loss", "logical_loss.png")):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
        for i, code in enumerate(("bb72", "bb144")):
            for j, p in enumerate((.001,.002,.004,.006)):
                ax = axes[i,j]
                for depth, color in colors.items():
                    k = (code,depth,p); h = latest[k]
                    rows = h["eval"] if kind == "validation" else h["train"]
                    window = 5 if kind == "validation" else 25
                    values = [100*r["raw"]["logical_error_rate"] if kind == "validation" else r["logical"] for r in rows]
                    smooth = np.convolve(values,np.ones(window)/window,mode="valid")
                    ax.plot([r["epoch"]+1 for r in rows[window-1:]], smooth, color=color,
                            ls="--" if k in partial else "-", marker="x" if k in partial else None,
                            markevery=[-1] if k in partial else None,
                            label=f"D{depth}"+(f" partial {len(h['train'])}" if k in partial else ""))
                ax.axvline(800, color="gray", ls=":", alpha=.65)
                if kind == "loss":
                    ax.axhline(math.log(2), color="gray", ls=":", lw=1)
                ax.set(title=f"{code.upper()} p={p:.1%}", xlabel="Epoch",
                       ylabel="Raw block failure (%)" if kind == "validation" else "Logical parity loss")
                if kind == "validation":
                    lo,hi = ax.get_ylim(); ax.set_ylim(max(0,lo),min(100.1,hi))
                ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
        fig.suptitle(("Raw validation | trailing 5-evaluation mean" if kind == "validation"
                      else "Training logical parity loss | trailing 25-epoch mean")+
                     "\nVertical line: continuation starts at 800; dashed/x: partial download; panel scales differ", fontsize=13)
        fig.savefig(PLOTS / filename, dpi=160); plt.close(fig)

    complete = [r for r,h in training if r["status"] == "complete"]
    complete.sort(key=lambda r:(r["code"],r["depth"],r["p"]))
    fig, ax = plt.subplots(figsize=(12,max(7,.43*len(complete)+1.5)), layout="constrained")
    for i,r in enumerate(complete):
        ax.barh(i,r["delta_pp"],color="#26754b" if r["delta_pp"] < 0 else "#bd6558",
                hatch="///" if not r["best_updated"] else None, alpha=.75)
        ax.text(1.2,i,f"{100*r['old_final_rate']:.2f} → {100*r['block_rate']:.2f}%   "
                f"best {r['previous_best_epoch']} → {r['best_epoch']}", va="center", fontsize=9)
    ax.set_yticks(range(len(complete)),[f"{r['code'].upper()} D{r['depth']}  p={r['p']:.1%}" for r in complete])
    ax.invert_yaxis(); ax.axvline(0,color="gray",lw=1)
    ax.set_xlim(min(-5.2,min(r["delta_pp"] for r in complete)-.8),7)
    ax.set_xticks(range(math.floor(min(-5,min(r["delta_pp"] for r in complete))),2))
    ax.set_xlabel("New minus old raw final failure (percentage points; negative = lower)")
    ax.grid(axis="x",alpha=.2)
    ax.set_title(f"800 → 1,000 epochs: {len(complete)} completed continuations\n"
                 "Point differences on different shot banks; hatched = unchanged best weights, not training gain")
    fig.savefig(PLOTS / "continuation_comparison.png", dpi=160); plt.close(fig)

    old_rows = []
    with (OUT / "bb_dem_2026_11_02_hierarchical_summary.csv").open() as stream:
        for r in csv.DictReader(stream):
            old_rows.append({k:(v if k == "code" else float(v)) for k,v in r.items()
                             if k in ("code","depth","p","raw_rate","raw_low","raw_high", "hierarchical_rate",
                                      "hierarchical_low","hierarchical_high","global_call_fraction","amortized_ms_per_shot")})
    # Repeated best evaluations use the same source/seed and reproduce the old totals;
    # retain one complete baseline per point instead of counting the repeats twice.
    for row,_ in evaluations:
        if row["weights"] == "best" and row["status"] == "complete":
            old = next((r for r in old_rows if (r["code"],r["depth"],r["p"]) ==
                        (row["code"],row["depth"],row["p"])), None)
            if old is None:
                old_rows.append(dict(row))
            else:
                assert all(old[f"{a}_rate"] == row[f"{a}_rate"] for a in ("raw","hierarchical"))
    latest_eval = [r for r,_ in evaluations if r["weights"] == "latest"]
    combined = [dict(r,weights="best",status="complete",shots=8192) for r in old_rows] + latest_eval
    emit_csv("hierarchical_plot_data", combined)
    for diagnostic, filename in ((False,"depth_comparison.png"),(True,"depth_diagnostics.png")):
        fig, axes = plt.subplots(2,2,figsize=(13,9),layout="constrained")
        for i,code in enumerate(("bb72","bb144")):
            for j in range(2):
                ax = axes[i,j]
                field = (("global_call_fraction","amortized_ms_per_shot") if diagnostic else ("raw_rate","hierarchical_rate"))[j]
                scale = 1 if field == "amortized_ms_per_shot" else 100
                for depth,color in colors.items():
                    rows = sorted([r for r in old_rows if r["code"] == code and r["depth"] == depth],key=lambda r:r["p"])
                    if not rows: continue
                    x = [100*r["p"] for r in rows]; y = [scale*r[field] for r in rows]
                    ax.plot(x,[v if v > 0 else np.nan for v in y],color=color,label=f"D{depth} best")
                    for r,xp,yp in zip(rows,x,y):
                        if diagnostic:
                            ax.plot(xp,yp,"o",color=color)
                        elif yp > 0:
                            arm = field.removesuffix("_rate")
                            ax.errorbar(xp,yp,yerr=[[max(0,yp-100*r[f'{arm}_low'])],[max(0,100*r[f'{arm}_high']-yp)]],fmt="o",color=color,capsize=2)
                        else:
                            arm = field.removesuffix("_rate")
                            ax.scatter(xp,100*r[f"{arm}_high"],marker="v",facecolors="none",edgecolors=color)
                for r in latest_eval:
                    if r["code"] != code: continue
                    color = colors[r["depth"]]; xp,yp = r["p"]*100,scale*r[field]
                    complete_point = r["status"] == "complete"
                    kw = dict(fmt="s",color=color,mfc=color if complete_point else "white",ms=8,capsize=3,zorder=5)
                    if diagnostic:
                        ax.errorbar(xp,yp,**kw)
                    else:
                        arm = field.removesuffix("_rate")
                        ax.errorbar(xp,yp,yerr=[[max(0,yp-100*r[f'{arm}_low'])],[max(0,100*r[f'{arm}_high']-yp)]],**kw)
                ax.set_title(f"{code.upper()} — "+({"raw_rate":"Raw CNN", "hierarchical_rate":"Hierarchical residual OSD-0",
                             "global_call_fraction":"Global completion calls", "amortized_ms_per_shot":"CNN + gating + OSD time"}[field]))
                ax.set_xlabel("Per-location error rate p (%)")
                ax.set_ylabel("ms / shot (batch amortized)" if field == "amortized_ms_per_shot" else "Calls (%)" if diagnostic else "Block failure (%)")
                ax.set_xticks([.1,.2,.4,.6]); ax.grid(alpha=.2)
                if field == "hierarchical_rate": ax.set_yscale("log"); ax.set_ylim(.002,150)
                elif field != "amortized_ms_per_shot": ax.set_ylim(-2,103)
                handles,labels = ax.get_legend_handles_labels()
                handles += [Line2D([],[],marker="s",color="gray",ls="",label="latest 800 (complete)"),
                            Line2D([],[],marker="s",color="gray",mfc="white",ls="",label="latest 800 (partial)")]
                ax.legend(handles=handles,fontsize=8,loc="lower right" if not diagnostic else "best")
        fig.suptitle("Existing 800-epoch best baseline + newly received latest-weight evaluations\n"
                     "Circles: complete best; squares: latest; hollow squares: partial snapshots\n"+
                     ("Timing is amortized, not single-shot latency" if diagnostic else "95% Wilson intervals; hollow downward triangle = zero failures, plotted at CI upper bound"),fontsize=12)
        fig.savefig(PLOTS / "hierarchical" / filename,dpi=160); plt.close(fig)

    columns = 2 if len(comparisons) <= 4 else 3
    rows = math.ceil(len(comparisons)/columns)
    fig, axes = plt.subplots(rows,columns,figsize=(6*columns,4*rows),layout="constrained",squeeze=False)
    for ax,r in zip(axes.flat,comparisons):
        for i,w in enumerate(("best","latest")):
            y = 100*r[f"{w}_rate"]
            ax.errorbar(i,y,yerr=[[max(0,y-100*r[f'{w}_low'])],[max(0,100*r[f'{w}_high']-y)]],
                        fmt="o",color="#3478c6" if i == 0 else "#d98116",ms=8,capsize=5,
                        mfc="white" if r["status"] != "complete" else None)
            ax.annotate(f"{r[f'{w}_failures']:,}/{r['shots']:,}\n{y:.3f}%",(i,100*r[f"{w}_high"]),
                        xytext=(0,10),textcoords="offset points",ha="center",fontsize=10)
        lo = 100*min(r["best_low"],r["latest_low"]); hi = 100*max(r["best_high"],r["latest_high"])
        span = max(hi-lo,.1); ax.set_ylim(max(0,lo-.3*span),min(100.1,hi+.7*span))
        # Keep labels within the panel even when failures are 100%.
        if hi >= 99.99: ax.set_ylim(lo-.2*span,100+.9*span)
        ax.set_xticks([0,1],[f"Best (epoch {r['best_epoch']})",f"Latest (epoch {r['latest_epoch']})"])
        ax.set_xlim(-.6,1.6); ax.set_ylabel("Hierarchical block failure (%)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y",alpha=.2)
        ax.set_title(f"{r['code'].upper()} D{r['depth']} p={r['p']:.1%}\n"+
                     ("Complete: 8,192 shots each" if r["status"] == "complete" else f"PARTIAL: common logged prefix, {r['shots']:,} shots each"))
    for ax in list(axes.flat)[len(comparisons):]:
        ax.set_visible(False)
    fig.suptitle("Best versus latest: checkpoint selection can matter after residual OSD\n"
                 "Same sampler settings; shot banks absent, so no verified paired significance test; panel scales differ",fontsize=12)
    fig.savefig(PLOTS / "hierarchical/best_vs_latest.png",dpi=160); plt.close(fig)


def report(training, evaluations, comparisons, info):
    complete = sorted([r for r,h in training if r["status"] == "complete"],key=lambda r:(r["code"],r["depth"],r["p"]))
    partial = [r for r,h in training if r["status"] != "complete"]
    lines = ["# 2026-11-03: 추가 학습 및 best/latest hierarchical 평가", "",
             "## 수신 상태", "",
             "새 폴더 6개를 `results/bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`에 정리했다. "
             "Job 안에 여러 코드/depth가 있으므로 job 단위를 유지했다. 원래 폴더 이름과 파일 내용은 보존했다.", "",
             "- 추가 학습 계획 21개 중 **12개는 1,000 epoch 완료**, **4개는 다운로드 당시 진행 중**, **5개는 아직 로컬 미수신**이다.",
             "- Hierarchical 계획 22개 checkpoint 평가 중 **4개 완료**, **4개 중간 집계**, **14개 미수신**이다. Best/latest 각각 별도 평가로 센다.",
             "- 진행 중이라는 표시는 저장된 snapshot의 상태다. 서버에서 지금도 실행 중인지, 이후 완료/중단됐는지는 이 다운로드로 알 수 없다.", "",
             "| Job | 종류 | 로컬 수신 상태 |", "|---|---|---|"]
    for job,mode in JOBS.items():
        subset = [r for r,h in (training if mode == "train" else evaluations) if r["job"] == job]
        n = sum(r["status"] == "complete" for r in subset)
        lines.append(f"| {job} | {mode} | 완료 {n}, 중간 snapshot {len(subset)-n} |")
    lines += ["", "## 800 → 1,000 epoch: 완료된 raw 평가", "",
              "기존 latest weights와 AdamW state에서 LR=3e-5의 새 cosine phase로 200 epochs를 추가했다. "
              "아래는 각 run의 **raw validation-selected best**를 8,192 fresh circuit shots로 평가한 결과다. "
              "OSD 없는 raw block failure이며, 이전/이후 shot bank가 달라 paired 비교가 아니다.", "",
              "| 코드 | D | p (%) | 이전 raw (%) | 추가 학습 후 raw (%) | 변화 (%p) | Best epoch 전→후 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in complete:
        lines.append(f"| {r['code']} | {r['depth']} | {100*r['p']:.1f} | {100*r['old_final_rate']:.3f} | "
                     f"{100*r['block_rate']:.3f} | {r['delta_pp']:+.3f} | {r['previous_best_epoch']}→{r['best_epoch']} |")
    lines += ["", "해석:", "",
              "- BB72 D8 p=0.2%는 67.834→63.477% (−4.358%p), BB144 D6 p=0.2%는 87.146→84.058% (−3.088%p)로 낮아졌다. 두 모델 모두 best가 epoch 985로 갱신됐다. 이번 수신분에서는 추가 학습의 이득이 가장 뚜렷한 점추정이다.",
              "- BB72 D4/D6 p=0.1%와 BB144 D6 p=0.1%도 raw 실패율이 낮아졌지만 모든 설정에서 개선되는 것은 아니다. BB72 D6 p=0.2%는 best가 바뀌었어도 final은 53.418→53.540%로 비슷하다.",
              "- **BB72 D4 p=0.2%, D6 p=0.4%, D8 p=0.1%는 best가 그대로**다. Final 변화는 같은 선택 가중치를 새 샷으로 다시 평가한 차이이며 추가 학습 효과로 해석하지 않는다. `.pt` 미수신이므로 가중치 자체 동일성은 로컬에서 확인하지 못했다.",
              "- 위 추가 학습 모델의 hierarchical 평가는 아직 없다. Raw 개선이 residual OSD 후에도 유지되는지는 알 수 없다.", "",
              "## 진행 중이던 학습 snapshot", "",
              "Final plot에는 아래 미완료 run의 새 final 점을 넣지 않았다. Validation/loss 곡선에만 dashed/x로 추가했다.", "",
              "| 코드 | D | p (%) | 수신 epoch / 1,000 | Best epoch | 마지막 validation epoch | 마지막 raw validation (%) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in partial:
        lines.append(f"| {r['code']} | {r['depth']} | {100*r['p']:.1f} | {r['epochs']} | {r['best_epoch']} | "
                     f"{r['last_validation_epoch']} | {100*r['last_validation_rate']:.3f} |")
    lines += ["", "아직 미수신인 train workers 16–20: BB72 D12 p=0.4%/0.6%, BB144 D12 p=0.1%/0.2%/0.4%. "
              "진행/완료 여부를 추정하지 않는다.", "", "## 기존 800-epoch 모델: hierarchical best vs latest", "",
              "추가 학습한 가중치가 아니라 **기존 800-epoch run**의 best와 latest 비교다. "
              "같은 model/noise/seed/batch 설정을 확인했다. 미완료 pair는 로그에 공통으로 존재하는 마지막 샷 수로 맞췄다. "
              "`.npz`가 없어 실제 paired shot 일치 및 discordance 기반 유의성 검정은 하지 않았다.", "",
              "| 코드 | D | p (%) | 샷 수/가중치 | 상태 | Best epoch | Best 실패율 (%) | Latest epoch | Latest 실패율 (%) | 차이 (%p) |",
              "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['code']} | {r['depth']} | {100*r['p']:.1f} | {r['shots']:,} | "
                     f"{'완료' if r['status']=='complete' else '중간 공통 prefix'} | {r['best_epoch']} | "
                     f"{100*r['best_rate']:.3f} | {r['latest_epoch']} | {100*r['latest_rate']:.3f} | {r['latest_minus_best_pp']:+.3f} |")
    lines += ["", "- **BB144 D4 p=0.4% 중간 결과**: 같은 6,016샷 집계에서 best(epoch 5) 5,588 failures, latest(epoch 800) 5,366 failures다. 92.886→89.195%, −3.690%p. Raw 성공률 동률 때문에 선택된 초기 best가 OSD 후에도 최적이라는 보장은 없다는 점을 보여주는 잠정 결과다. 전체 8,192샷 결과로 다시 확인해야 한다.",
              "- BB144 D4 p=0.2%는 153→150 failures로 차이가 작다. BB72 D4 p=0.6%는 7,796→7,848로 latest 점추정이 더 나쁘다. Latest가 항상 더 좋다는 결과가 아니다.",
              "- BB144 D4 p=0.6%는 공통 6,080샷에서 둘 다 100% 실패다. Latest에서도 개선이 관측되지 않았다.",
              "- 수신한 모든 평가 집계에서 hierarchical syndrome failure=0, raw 대비 harmed=0이다. 높은 실패율은 syndrome-valid correction의 logical class 오류다.",
              "- 완료된 best 재평가 2개는 기존 Nov 2 집계와 raw/hierarchical 실패 수가 같다. 독립 반복 실험처럼 합산하지 않았다.", "",
              "실제 최신 `results.json`의 개별 진행량은 다음과 같다. 위 공통-prefix 비교와 샷 수가 다를 수 있다.", "",
              "| Job | Worker | 코드/D/p | Weights | 상태 | 수신 shots | Hier failures | ms/shot |",
              "|---|---:|---|---|---|---:|---:|---:|"]
    for r,_ in evaluations:
        lines.append(f"| {r['job']} | {r['worker']} | {r['code']}/D{r['depth']}/{r['p']:.1%} | {r['weights']} | "
                     f"{r['status']} | {r['shots']:,} | {r['hierarchical_failures']:,} | {r['amortized_ms_per_shot']:.1f} |")
    lines += ["", "시간은 batch4의 CNN/gating/전송/OSD 누적시간÷샷 수이며 단일 샷 latency가 아니다.", "",
              "## 검증 및 한계", "",
              f"- 새 {info['folders']}개 폴더, {info['files']}개 파일 ({info['bytes']:,} bytes)을 이동 전후 SHA-256과 크기로 확인했다.",
              "- Training source config와 원래 800-epoch train/eval prefix, 새 phase/learning rate, 연속 epoch, loss 합, validation/최종 로그 및 best 선택을 대조했다.",
              "- 평가 config/source 선택, 로그의 누적 failure 및 Wilson95 구간, raw→hierarchical rescued/harmed 관계를 검증했다. 중간 results.json이 64샷 간격 로그보다 앞선 것은 정상이며 각 증가량을 확인했다.",
              f"- Source snapshot {info['source_snapshot_files']}개 중 {info['matching_source_snapshots']}개가 현재 파일과 일치한다. 차이는 여섯 job 모두 결과 폴더 접미사 제거 이후 달라진 `scripts/run_bb_dem_followup.sh`다.",
              f"- 새 `.pt` {info['checkpoint_files']}개, `.npz` {info['shot_files']}개. 따라서 checkpoint hash/가중치, 샷별 correction, paired best/latest 검증은 불가능하다. 이번 검증은 metadata/history/log의 내부 일관성 검증이다.",
              "- 현재 overview는 완료 run만 사용하며 **1,000 epoch 12개 + 800 epoch 20개**가 섞여 있다. 따라서 depth별 동일 학습 예산 비교로 해석하면 안 된다. 원래 800-epoch 선은 점선으로 남겼다.",
              "- 중간 다운로드는 불변 snapshot으로 보관했다. 서버 완료 후 같은 job의 새 파일을 다시 받아 최신 상태로 재분석해야 한다.", "",
              "## 갱신한 플롯", "",
              "- [Raw overview](../plots/bb/circuit/standard/dem_cnn/overview.png): 완료된 새 12개 final 반영, 이전 800 baseline 점선.",
              "- [Validation](../plots/bb/circuit/standard/dem_cnn/validation.png) · [Logical loss](../plots/bb/circuit/standard/dem_cnn/logical_loss.png): 12개 완료 및 4개 중간 continuation 반영.",
              "- [추가 학습 전후](../plots/bb/circuit/standard/dem_cnn/continuation_comparison.png): 12개 raw final 변화, best 미변경 모델 hatch 표시.",
              "- [Hierarchical depth 비교](../plots/bb/circuit/standard/dem_cnn/hierarchical/depth_comparison.png) · [시간/호출률](../plots/bb/circuit/standard/dem_cnn/hierarchical/depth_diagnostics.png): 기존 complete best baseline에 새 latest 평가 추가. 중간 결과는 빈 사각형.",
              "- [Best/latest 상세 비교](../plots/bb/circuit/standard/dem_cnn/hierarchical/best_vs_latest.png): 완료/중간 pair 분리, 공통 logged-prefix 샷 수와 Wilson95 표시.", "",
              f"[학습 CSV]({STEM}_training.csv) · [평가 CSV]({STEM}_evaluations.csv) · [공통-prefix 비교]({STEM}_best_latest.csv) · [검증 JSON]({STEM}_audit.json)", "",
              "재생성: `python scripts/analyze_bb_dem_november3.py`. 최초 정리는 `--archive`. 학습이나 추론을 실행하지 않는다.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    if (OUT / "bb_dem_2026_11_05_replacement.csv").exists():
        from scripts.analyze_bb_dem_november5 import main as latest_main
        print("Partial downloads were replaced; rebuilding the November 5 report and combined plot.")
        return latest_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",action="store_true")
    args = parser.parse_args()
    info = archive(args.archive)
    training, evaluations, snapshots = [], [], []
    for directory in DESTINATIONS.values():
        a,b,c = audit_job(directory)
        training.extend(a); evaluations.extend(b); snapshots.extend(c)
    original = baseline()
    comparisons = common_comparisons(evaluations)
    write_csv("training",[r for r,h in training])
    write_csv("evaluations",[r for r,logs in evaluations])
    write_csv("best_latest",comparisons)
    write_csv("snapshots",snapshots)
    progress = [dict(job=row["job"],worker=row["worker"],code=row["code"],depth=row["depth"],
                     p=row["p"],weights=row["weights"],**flatten_result(r))
                for row,logs in evaluations for r in logs.values()]
    write_csv("evaluation_progress",progress)
    info.update(training_received=len(training), training_complete=sum(r["status"]=="complete" for r,h in training),
                training_partial=sum(r["status"]!="complete" for r,h in training), training_missing=21-len(training),
                evaluation_received=len(evaluations),evaluation_complete=sum(r["status"]=="complete" for r,h in evaluations),
                evaluation_partial=sum(r["status"]!="complete" for r,h in evaluations), evaluation_missing=22-len(evaluations),
                evaluation_shots_received=sum(r["shots"] for r,h in evaluations), evaluation_progress_rows=len(progress),
                source_snapshot_files=len(snapshots), matching_source_snapshots=sum(r["matches_current"] for r in snapshots),
                checkpoint_files=sum(len(list(d.rglob("*.pt"))) for d in DESTINATIONS.values()),
                shot_files=sum(len(list(d.rglob("*.npz"))) for d in DESTINATIONS.values()), independent_shot_audit=False)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(info,indent=2)+"\n")
    plots(original,training,evaluations,comparisons)
    report(training,evaluations,comparisons,info)
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    main()
