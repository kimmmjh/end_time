#!/usr/bin/env python3
"""Audit the September 27 CNN campaign and regenerate CSVs and the combined CNN figure.

Reads archived server artifacts only; does not train or rerun any decoder.
Original files are checked against the import manifest before analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bb_code import BBCodeSpec

ANALYSIS = ROOT / "results/analysis"
CNN = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn"
PLOTS = ROOT / "results/plots/bb/code_capacity/tanner_cnn"
STEM = "bb_update_2026_09_27"
Z = 1.959963984540054


def original_jobs():
    """Keep the fixed 100-epoch campaign separate from later CNN runs.

    Original source jobs may be kept at the repository root for transfer
    to the cluster. Their original archive manifest still identifies the files.
    """
    jobs = []
    for job_id, group in ((58793761, "bb72"), (58793763, "bb144"),
                          (58793764, "bb72"), (58793765, "bb144"),
                          (58793766, "replicates")):
        archived = CNN / group / f"resdir_{job_id}"
        jobs.append(archived if archived.is_dir() else ROOT / archived.name)
    return jobs


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
    assert len(rows) == len({r["archive_file"] for r in rows})
    jobs = {job.name: job for job in original_jobs()}
    def locate(row):
        parts = Path(row["archive_file"]).parts
        index = next(i for i, part in enumerate(parts) if part.startswith("resdir_"))
        return jobs[parts[index]].joinpath(*parts[index + 1:])
    assert {locate(row) for row in rows} == {
        p for job in jobs.values() for p in job.rglob("*") if p.is_file()
    }, "Manifest must cover every original 100-epoch CNN file"
    for row in rows:
        path = locate(row)
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
    with (job / "experiments.tsv").open() as handle:
        specs = list(csv.DictReader(handle, delimiter="\t"))
    assert len(specs) == 4
    for spec in specs:
        args = dict(token[2:].split("=", 1) for token in shlex.split(spec["arguments"]) if "=" in token)
        for key, value in dict(epochs="100", batch_size="64", batches="128", eval_batches="64",
                               eval_every="5", final_eval_batches="1024", lr="0.0003").items():
            assert args[key] == value, (job, key)


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
    jobs = sorted(original_jobs())
    assert {j.name for j in jobs} == {f"resdir_{i}" for i in (58793761, 58793763, 58793764, 58793765, 58793766)}
    for job in jobs:
        check_job(job)
    for name in ("models/_bb_tanner_cnn.py", "src/_bb_tanner_cnn_experiment.py",
                 "src/_direct_osd.py", "src/_bb_metrics.py"):
        assert len({sha256(job / "source_snapshot" / name) for job in jobs}) == 1, name
    finals, training, validation, histories, banks = [], [], [], {}, {}
    # Later 400-epoch jobs live in a separate subtree of the same CNN archive.
    for path in sorted(p for job in jobs for p in job.rglob("history.json")):
        h = json.loads(path.read_text())
        c, f, p = h["config"], h["final"], h["config"]["error_rate"]
        code = BBCodeSpec.from_name(c["code"])
        depth = c["cnn_depth"]
        key = (c["code"], p, c["seed"], depth)
        assert key not in histories
        assert c["cnn_width"] == 64 and depth in (1, 2)
        assert c["parameter_count"] == {1: 54404, 2: 161284}[depth]
        assert c["noise_model"] == "capacity" and c["channel"] == "depolarizing"
        assert c["checkpoint_selection"] == "raw_logical_accuracy" and c["osd_extra_bp_iterations"] == 0
        assert c["hard_decision"] == "component_marginal_gt_half" and c["osd"] == "direct_hard_centered_osd0_v1"
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
        assert all(e["shots"] == 4096 for e in h["eval"])
        with np.load(path.with_name("final_shots.npz"), allow_pickle=False) as bank:
            b = {key: bank[key] for key in bank.files}
        assert json.loads(str(b["metadata_json"])) == c
        for key in ("syndrome", "pauli", "raw_correction", "osd_correction"):
            assert b[key].shape == (65536, code.n)
            assert b[key].dtype == np.uint8
        tx, tz = xz(b["pauli"])
        assert np.array_equal(b["syndrome"], np.concatenate(((tz @ code.hx.T) % 2, (tx @ code.hz.T) % 2), axis=1))
        key = (c["code"], p, c["seed"], depth)
        bank_hash = hashlib.sha256(b["pauli"].tobytes() + b["syndrome"].tobytes()).hexdigest()
        row = dict(code=c["code"], p=p, seed=c["seed"], depth=depth, width=64, parameters=c["parameter_count"],
                   shot_bank_sha256=bank_hash,
                   epochs=100, training_shots=819200, shots=f["shots"], best_epoch_zero_based=h["best_epoch"],
                   best_epoch_one_based=h["best_epoch"] + 1,
                   first_loss=h["train"][0]["total"], last_loss=h["train"][-1]["total"],
                   first10_mean_loss=float(np.mean([t["total"] for t in h["train"][:10]])),
                   last10_mean_loss=float(np.mean([t["total"] for t in h["train"][-10:]])),
                   selected_validation_raw_ler=selected["raw"]["logical_error_rate"],
                   last_validation_raw_ler=h["eval"][-1]["raw"]["logical_error_rate"],
                   last5_validation_raw_ler=float(np.mean([e["raw"]["logical_error_rate"] for e in h["eval"][-5:]])),
                   first_validation_raw_ler=h["eval"][0]["raw"]["logical_error_rate"],
                   first_validation_osd_ler=h["eval"][0]["osd"]["logical_error_rate"],
                   last_validation_osd_ler=h["eval"][-1]["osd"]["logical_error_rate"],
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
        histories[key] = h
        banks[key] = dict(hash=bank_hash, **outcomes)
        for t in h["train"]:
            training.append(dict(code=c["code"], p=p, seed=c["seed"], depth=depth, **t))
        for e in h["eval"]:
            validation.append(dict(code=c["code"], p=p, seed=c["seed"], depth=depth, epoch=e["epoch"], shots=e["shots"],
                                   raw_ler=e["raw"]["logical_error_rate"], osd_ler=e["osd"]["logical_error_rate"],
                                   osd_call_fraction=e["osd_call_fraction"], paired_gain=e["osd_minus_raw_gain"],
                                   selected=e["epoch"] == h["best_epoch"]))
    order = lambda r: (int(r["code"][2:]), r["depth"], r["p"], r["seed"])
    finals.sort(key=order)
    holm(finals)
    assert len(finals) == 20
    for code in ("bb72", "bb144"):
        for depth in (1, 2):
            assert [r["p"] for r in finals if r["code"] == code and r["depth"] == depth and primary(r)] == [.02, .04, .06, .08]
    write_csv(f"{STEM}_cnn_final.csv", finals)
    write_csv(f"{STEM}_cnn_train.csv", sorted(training, key=lambda r: (*order(r), r["epoch"])))
    write_csv(f"{STEM}_cnn_validation.csv", sorted(validation, key=lambda r: (*order(r), r["epoch"])))
    return finals, histories, banks


def primary(row):
    return row["seed"] == int(row["code"][2:]) * 100000 + 1000 + round(row["p"] * 1000)


def depth_comparison(banks):
    rows = []
    for (code, p, seed, depth), first in sorted(banks.items()):
        if depth != 1:
            continue
        second = banks[(code, p, seed, 2)]
        assert first["hash"] == second["hash"], "Depth comparison requires identical truth and syndrome banks"
        for method in ("raw", "osd"):
            a, b = first[method], second[method]
            rescued, harmed = int((~a & b).sum()), int((a & ~b).sum())
            n = a.size
            gain = (rescued - harmed) / n
            se = math.sqrt(((rescued + harmed) / n - gain * gain) / (n - 1))
            rows.append(dict(code=code, p=p, seed=seed, method=method, shots=n,
                             identical_shot_bank=True, shot_bank_sha256=first["hash"],
                             depth1_ler=float((~a).mean()), depth2_ler=float((~b).mean()),
                             rescued=rescued, harmed=harmed, paired_gain=gain,
                             paired_gain_low=gain-Z*se, paired_gain_high=gain+Z*se,
                             paired_gain_interval="paired normal approximation; shot uncertainty only",
                             paired_exact_p=binomtest(rescued, rescued+harmed).pvalue if rescued+harmed else 1.))
    assert len(rows) == 16
    holm(rows)  # One family: eight code/p pairs times raw and OSD.
    write_csv(f"{STEM}_cnn_depth_pairs.csv", rows)
    return rows


def seed_summary(finals):
    rows = []
    for code in ("bb72", "bb144"):
        selected = [r for r in finals if r["code"] == code and r["depth"] == 2 and r["p"] == .06]
        assert len(selected) == 3 and len({r["shot_bank_sha256"] for r in selected}) == 3
        for method in ("raw", "osd"):
            rates = np.array([r[f"{method}_ler"] for r in selected])
            n = sum(r["shots"] for r in selected)
            failures = sum(r[f"{method}_failures"] for r in selected)
            rows.append(dict(code=code, p=.06, depth=2, method=method, seeds=3, shots=n,
                             failures=failures, mean_ler=float(rates.mean()), pooled_ler=failures/n,
                             min_ler=float(rates.min()), max_ler=float(rates.max()),
                             sample_sd_ler=float(rates.std(ddof=1)),
                             uncertainty="between-run SD includes training and test-bank variation; only three seeds"))
    write_csv(f"{STEM}_cnn_seeds.csv", rows)
    return rows


def references(cnn):
    old_neural = read_csv(ANALYSIS / "bb_neural_bp_depolarizing_orbit.csv")
    classical = read_csv(ANALYSIS / "bb_campaign_2026_08_classical.csv")
    rows = []
    for new in cnn:
        if new["depth"] != 2 or not primary(new):
            continue
        for old in old_neural:
            if float(old["p"]) != new["p"] or old["code"] != new["code"]:
                continue
            for tag, name in (("vanilla", "BP4 T12"), ("neural", "Neural BP4 T12")):
                rows.append(dict(code=new["code"], p=new["p"], method=name, ler=float(old[f"{tag}_logical_error_rate"]),
                                 shots=int(old["eval_samples"]), source=old["source_resdir"], comparison="separate shot banks and training budgets"))
        for old in classical:
            if old["code"] == new["code"] and float(old["p"]) == new["p"] and old["method"] in {"bposd_0", "bposd_cs7"}:
                rows.append(dict(code=new["code"], p=new["p"], method={"bposd_0": "CSS BP2+OSD-0", "bposd_cs7": "CSS BP2+OSD-CS7"}[old["method"]],
                                 ler=float(old["logical_error_rate"]), shots=int(old["samples"]), source=old["source_file"],
                                 comparison="separate shot banks; X/Z split; different OSD convention and BP budget"))
    write_csv(f"{STEM}_capacity_references.csv", rows)
    return rows


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest()
    cnn, histories, banks = cnn_results()
    depths = depth_comparison(banks)
    seed_summary(cnn)
    refs = references(cnn)
    from scripts.plot_bb_cnn_all import main as plot_all
    plot_all()
    audit.update(cnn_experiments=len(cnn), cnn_final_shots=sum(r["shots"] for r in cnn),
                 cnn_saved_corrections_independently_rescored=True,
                 cnn_decoder_and_evaluation_snapshots_identical=True,
                 unique_final_shot_banks=len({b["hash"] for b in banks.values()}),
                 unique_final_shots=len({b["hash"] for b in banks.values()}) * 65536,
                 depth_pairs_with_identical_shots=len(depths)//2,
                 depth_tests_holm_family_size=len(depths),
                 raw_to_osd_tests_holm_family_size=len(cnn),
                 removed_plain_bp_jobs=[58793753, 58793754, 58793756, 58793757, 58793759])
    (ANALYSIS / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    for row in cnn:
        print(f"{row['code']} depth={row['depth']} p={row['p']} seed={row['seed']}: raw={row['raw_ler']:.6%}, OSD={row['osd_ler']:.6%}, epoch={row['best_epoch_one_based']}")
    print("Wrote analysis CSVs, audit JSON, and the combined CNN PNG. Markdown interpretation is maintained separately.")


if __name__ == "__main__":
    main()
