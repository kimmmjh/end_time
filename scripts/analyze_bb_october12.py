#!/usr/bin/env python3
"""Audit the three resumed CNN jobs, rescore saved banks, and plot comparisons.

No training or inference. These are the jobs' original residual-OSD results,
not the separate post-hoc standard syndrome OSD evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_october12_matplotlib")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_bb_october10 import audit_bank, csv_read_tsv, paired, sha, read
from scripts.summarize_bb_september27 import close, holm
from scripts.continue_bb_tanner_cnn import resolve_checkpoint
from src.bb_code import BBCodeSpec

JOBS = (59622483, 59622488, 59622490)
BASE = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn/continued_training"
OUT = ROOT / "results/analysis"
STEM = "bb_update_2026_10_12"


def write_csv(suffix, rows):
    path = OUT / f"{STEM}_{suffix}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def manifest_audit():
    with (OUT / f"{STEM}_relocation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = {p for job in JOBS for p in (BASE / f"resdir_{job}").rglob("*") if p.is_file()}
    assert files == {ROOT / row["new_path"] for row in rows}
    for row in rows:
        path = ROOT / row["new_path"]
        assert path.stat().st_size == int(row["bytes"])
        assert sha(path) == row["sha256"], path
    return {"files": len(rows), "bytes": sum(int(row["bytes"]) for row in rows),
            "all_relocation_hashes_verified": True}


def analyze():
    audit = manifest_audit()
    finals, pairs, status, training, validation, timing = [], [], [], [], [], []
    banks, histories = {}, {}
    for job in JOBS:
        folder = BASE / f"resdir_{job}"
        assert (folder / "completed.txt").is_file() and not (folder / "failed.txt").exists()
        assert sha(folder / "source_snapshot/src/bb_code.py") == sha(ROOT / "src/bb_code.py")
        specs = csv_read_tsv(folder / "experiments.tsv")
        paths = sorted(folder.glob("outputs/*/*/history.json"))
        assert len(paths) == len(specs) == 4
        for path in paths:
            h = read(path)
            c = h["config"]
            key = (c["code"], c["error_rate"], c["cnn_depth"])
            spec = next(s for s in specs if f"--code={key[0]}" in s["arguments"]
                        and f"--p={key[1]} " in s["arguments"] and f"--depth={key[2]}" in s["arguments"])
            index = spec["experiment_index"]
            assert (folder / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
            log = (folder / f"log_exp_{index}.txt").read_text()
            assert "Traceback (most recent call last)" not in log
            assert "[Resume comparison]" in log and "[Selected best by raw LER]" in log
            assert c["architecture"] == "bb_tanner_cnn" and c["cnn_width"] == 64
            assert c["noise_model"] == "capacity" and c["channel"] == "depolarizing"
            assert c["osd"] == "direct_hard_centered_osd0_v1" and c["osd_extra_bp_iterations"] == 0
            assert c["checkpoint_selection"] == "raw_logical_accuracy"
            code = BBCodeSpec.from_name(key[0])
            assert c["graph_fingerprint"] == hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
            assert c["parameter_count"] == 54404 + (key[2] - 1) * 106880
            phase = h["phases"][-1]
            for name, value in dict(batch_size=64, batches=128, eval_batches=64, eval_every=5,
                                    final_eval_batches=1024, learning_rate=.0001,
                                    component_probability_policy="softmax_marginal_clamp_v1",
                                    schedule="cosine_restart", compare_resume=True).items():
                assert phase[name] == value, (path, name)
            comp = h["resume_comparison"]
            assert comp["source"] == phase["resume_source"]
            source = comp["source"]
            source_job = int(next(p for p in Path(source["checkpoint"]).parts if p.startswith("resdir_")).split("_")[1])
            checkpoint = resolve_checkpoint(ROOT, source_job, key[0], key[1], c["seed"], depth=key[2])
            assert sha(checkpoint) == source["sha256"]
            old = torch.load(checkpoint, map_location="cpu", weights_only=False)
            new = torch.load(path.with_name("model.pt"), map_location="cpu", weights_only=False)
            assert old["config"] == new["config"] == c
            start = old["epoch"] + 1
            end = new["epoch"] + 1
            recovery = start == 114
            assert (start, end) == ((114, 400) if recovery else (400, 800))
            assert end == len(h["train"]) and [row["epoch"] for row in h["train"]] == list(range(end))
            assert phase["start_epoch"] == source["completed_epochs"] == start
            assert phase["epochs"] == end - start
            assert h["train"][:start] == old["history"]["train"]
            assert h["eval"][:len(old["history"]["eval"])] == old["history"]["eval"]
            for name in ("train", "eval", "phases", "final", "resume_comparison"):
                assert new["history"][name] == h[name]
            selected = max(h["eval"], key=lambda e: e["raw"]["logical_accuracy"])
            assert selected["epoch"] == h["best_epoch"] == new["best_epoch"] == h["final"]["epoch"] == comp["selected_epoch"]
            assert old["best_epoch"] == comp["reference"]["epoch"] == source["best_epoch"]
            close(selected["raw"]["logical_accuracy"], h["best_raw_accuracy"])
            for name in ("model_state_dict", "best_model_state_dict"):
                assert all(torch.isfinite(t).all() for t in new[name].values())
            same_weights = all(torch.equal(t, old["best_model_state_dict"][name])
                               for name, t in new["best_model_state_dict"].items())
            for label, metric in (("final", h["final"]), ("reference", comp["reference"])):
                assert metric["shots"] == 65536
            meta = dict(job=job, experiment=int(index), code=key[0], p=key[1], depth=key[2],
                        seed=c["seed"], source_job=source_job, from_epochs=start, to_epochs=end,
                        recovery=recovery, source=str(path.relative_to(ROOT)))
            score, outcomes = audit_bank(path.with_name("final_shots.npz"), c, h["final"])
            reference, reference_outcomes = audit_bank(path.with_name("reference_final_shots.npz"), c, comp["reference"])
            assert score["shot_bank_sha256"] == reference["shot_bank_sha256"]
            if same_weights:
                for method in ("raw", "osd"):
                    assert np.array_equal(reference_outcomes[method], outcomes[method])
            row = {**meta, "width": 64, "parameters": c["parameter_count"], "shots": 65536,
                   "training_shots": end*8192, "added_training_shots": (end-start)*8192,
                   "best_epoch": h["best_epoch"]+1, "reference_best_epoch": old["best_epoch"]+1,
                   "best_weights_unchanged": same_weights, "osd_call_fraction": h["final"]["osd_call_fraction"],
                   "old_raw_same_bank": reference["raw_ler"], "old_osd_same_bank": reference["osd_ler"],
                   "loss_last50": float(np.mean([t["total"] for t in h["train"][-50:]])),
                   "loss_previous50": float(np.mean([t["total"] for t in h["train"][-100:-50]])), **score}
            finals.append(row)
            status.append({**meta, "exit_code": 0, "latest_epoch_verified": end,
                           "best_epoch": h["best_epoch"]+1, "best_weights_unchanged": same_weights,
                           "source_checkpoint_hash_verified": True})
            for method in ("raw", "osd"):
                pair = paired(reference_outcomes[method], outcomes[method])
                for name in ("rescued", "harmed", "ler_reduction", "paired_se", "shots"):
                    close(pair[name], comp["paired"][method][name])
                pairs.append({**meta, "method": method, "old_ler_same_bank": reference[f"{method}_ler"],
                              "new_ler": score[f"{method}_ler"],
                              "relative_ler_reduction": pair["ler_reduction"]/reference[f"{method}_ler"], **pair})
            for t in h["train"]:
                training.append({**meta, **t})
            for e in h["eval"]:
                validation.append({**meta, "epoch": e["epoch"]+1,
                                   "raw_ler": e["raw"]["logical_error_rate"],
                                   "osd_ler": e["osd"]["logical_error_rate"]})
            for label, metric in (("frozen_reference", comp["reference"]), ("selected", h["final"])):
                timing.append({**meta, "model": label, "osd_call_fraction": metric["osd_call_fraction"],
                               "nn_us_per_shot": metric["nn_batch_seconds"]*1e6/65536,
                               "osd_us_per_shot": metric["osd_with_transfer_seconds"]*1e6/65536})
            banks[key] = outcomes
            histories[key] = h
            print(f"Audited {job} {key}: {start}->{end} epochs, raw={score['raw_ler']:.7f}, osd={score['osd_ler']:.7f}", flush=True)
            del old, new
    holm(pairs)  # One family: all 24 pre-specified old/new decoder comparisons.
    depth_pairs = []
    for a, b in itertools.combinations(finals, 2):
        if (a["code"], a["p"], a["seed"]) != (b["code"], b["p"], b["seed"]):
            continue
        if a["depth"] > b["depth"]:
            a, b = b, a
        same_bank = a["shot_bank_sha256"] == b["shot_bank_sha256"]
        for method in ("raw", "osd"):
            row = dict(code=a["code"], p=a["p"], method=method, from_depth=a["depth"], to_depth=b["depth"],
                       from_epochs=a["to_epochs"], to_epochs=b["to_epochs"],
                       old_ler=a[f"{method}_ler"], new_ler=b[f"{method}_ler"], same_bank=same_bank,
                       comparison="paired identical bank" if same_bank else "different banks and training budgets; descriptive")
            if same_bank:
                row.update(paired(banks[(a["code"], a["p"], a["depth"])][method],
                                  banks[(b["code"], b["p"], b["depth"])][method]))
            depth_pairs.append(row)
    holm([r for r in depth_pairs if r["same_bank"]])
    for suffix, rows in (("cnn_final", finals), ("cnn_status", status), ("cnn_resume_pairs", pairs),
                         ("cnn_depth_pairs", depth_pairs), ("cnn_train", training),
                         ("cnn_validation", validation), ("cnn_timing", timing)):
        write_csv(suffix, rows)
    audit.update(jobs=list(JOBS), completed_models=len(finals), resumed_400_to_800=sum(not r["recovery"] for r in finals),
                 recovered_114_to_400=sum(r["recovery"] for r in finals),
                 final_model_shots_rescored=len(finals)*65536, reference_model_shots_rescored=len(finals)*65536,
                 source_checkpoint_hashes_verified=len(finals), paired_resume_tests=len(pairs),
                 paired_depth_tests=sum(r["same_bank"] for r in depth_pairs),
                 best_weights_unchanged=sum(r["best_weights_unchanged"] for r in finals),
                 raw_significant_after_holm=int(sum(r["method"]=="raw" and r["paired_holm_p"]<.05 for r in pairs)),
                 osd_significant_after_holm=int(sum(r["method"]=="osd" and r["paired_holm_p"]<.05 for r in pairs)),
                 osd_implementation="direct_hard_centered_osd0_v1", added_training_steps_by_analysis=0)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return finals, pairs, depth_pairs, histories, audit



if __name__ == "__main__":
    finals, pairs, depth_pairs, histories, audit = analyze()
    from scripts.plot_bb_cnn_all import main as plot_all
    plot_all()
    print(json.dumps(audit, indent=2))
