#!/usr/bin/env python3
"""Audit the September 30 BB72 library baseline imports; save CSVs and one PNG.

Final statistics come from complete server aggregates. Download completeness is
tracked separately. Available corrections are independently rescored without
rerunning BP/OSD. Missing originals are never fabricated in the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "results/analysis"
ARCHIVE = ROOT / "results/bb/circuit/library_bp_osd/bb72"
PLOT = ROOT / "results/plots/bb/circuit/library_bp_osd/overview.png"
STEM = "bb_library_bp_osd_2026_09_30"
JOBS = ("58939136", "58939138")
MODES = ("x", "xz")
METHODS = ("bp", "bposd0", "bposd_cs3")
Z = 1.959963984540054


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(suffix, rows):
    with (ANALYSIS / f"{STEM}{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0., c - h), min(1., c + h)


def manifest_audit():
    rows = read_csv(ANALYSIS / f"{STEM}_manifest.csv")
    actual = {p.relative_to(ROOT).as_posix() for job in JOBS
              for p in (ARCHIVE / f"resdir_{job}").rglob("*") if p.is_file()}
    assert actual == {r["archive_file"] for r in rows}
    for row in rows:
        path = ROOT / row["archive_file"]
        assert path.stat().st_size == int(row["bytes"]), path
        assert sha(path) == row["sha256"], path
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), hashes_verified=True)


def check_summary(exp, report):
    c = report["config"]
    assert report["status"] == "complete"
    assert report["shots_completed"] == c["shots"] == 100000
    expected = dict(code="bb72", rounds=6, max_iterations=1000,
                    circuit_policy="bravyi_8tick_plus_memory_v1", bp_method="min_sum",
                    ms_scaling_factor=1., schedule="parallel", osd_method="OSD_CS", osd_order=3,
                    detector_inputs=["x", "xz"], library="ldpc", library_version="2.4.1",
                    num_observables=12, num_detectors=504, detector_frames=7, input_state="logical_plus",
                    observable_basis="logical_X", error_component="logical_Z", noisy_idle=True,
                    posterior_reseeding=False, custom_message_clipping=False,
                    metric_time_unit="full_memory_experiment")
    assert all(c[k] == v for k, v in expected.items())
    csv_rows = {r["decoder"]: r for r in read_csv(exp / "summary.csv")}
    assert set(csv_rows) == set(report["decoders"]) == {f"{m}_{d}" for m in MODES for d in METHODS}
    for key, row in report["decoders"].items():
        assert row["shots"] == c["shots"]
        assert 0 <= row["logical_mismatches"] <= row["failures"] <= row["shots"]
        assert row["failures"] == row["flagged_failures"] + row["unflagged_failures"]
        close(row["block_failure_rate"], row["failures"] / row["shots"])
        close(row["logical_z_mismatch_rate"], row["logical_mismatches"] / row["shots"])
        close(row["syndrome_convergence"], 1 - row["flagged_failures"] / row["shots"])
        close(row["mean_bp_iterations"], row["bp_iteration_sum"] / row["shots"])
        for metric, count in (("block_failure", "failures"), ("logical_z_mismatch", "logical_mismatches")):
            lo, hi = wilson(row[count], row["shots"])
            close(row[f"{metric}_ci95_low"], lo)
            close(row[f"{metric}_ci95_high"], hi)
        for field, value in row.items():
            close(csv_rows[key][field], value)
        assert csv_rows[key]["status"] == "complete" and csv_rows[key]["code"] == c["code"]
        close(csv_rows[key]["p"], c["p"])
        if not key.endswith("_bp"):
            assert row["flagged_failures"] == row["harmed"] == 0
            assert row["logical_mismatches"] == row["failures"]


def problem_matrices(exp, config, missing):
    problems = {}
    regenerated = None
    for mode in MODES:
        names = [f"{mode}_check.npz", f"{mode}_logical.npz"]
        if all((exp / n).exists() for n in names):
            check, logical = (sp.load_npz(exp / name) for name in names)
        else:
            # Reconstruct only in memory, with the exact source and circuit hash.
            from src.bb_paper_baselines import make_memory_circuit, decoding_problem
            if regenerated is None:
                regenerated = make_memory_circuit(config["code"], p=config["p"], rounds=config["rounds"])
                assert hashlib.sha256(str(regenerated).encode()).hexdigest() == config["circuit_sha256"]
            problem = decoding_problem(regenerated, mode)
            check, logical = problem.check, problem.logical
            for name in names:
                if not (exp / name).exists():
                    missing.append(dict(archive_file=(exp / name).relative_to(ROOT).as_posix(),
                                        status="missing", role="matrix; reconstructed in memory after source/circuit checks"))
            with np.load(exp / f"{mode}_prior.npz", allow_pickle=False) as prior:
                np.testing.assert_array_equal(prior["detector_indices"], problem.detector_indices)
                np.testing.assert_allclose(prior["probabilities"], problem.priors, rtol=1e-12, atol=1e-15)
        with np.load(exp / f"{mode}_prior.npz", allow_pickle=False) as prior:
            indices = prior["detector_indices"].copy()
            assert len(prior["probabilities"]) == check.shape[1]
        assert check.shape == (len(indices), logical.shape[1]) and logical.shape[0] == 12
        problems[mode] = check, logical, indices
    return problems


def rescore_chunks(exp, report, missing):
    c = report["config"]
    n = c["shots"]
    assert sha(exp / "shots.npz") == report["shot_bank_sha256"]
    with np.load(exp / "shots.npz", allow_pickle=False) as bank:
        assert json.loads(str(bank["metadata_json"])) == c
        detectors = np.unpackbits(bank["detectors_packed"], axis=1, count=c["num_detectors"], bitorder="little")
        truth = np.unpackbits(bank["observables_packed"], axis=1, count=c["num_observables"], bitorder="little")
    assert len(detectors) == len(truth) == n
    problems = problem_matrices(exp, c, missing)
    present = np.zeros(n, dtype=bool)
    outcomes = {key: {field: np.zeros(n, dtype=bool) for field in ("success", "converged", "logical_mismatch")}
                for key in report["decoders"]}
    iterations = {mode: np.zeros(n, dtype=np.int32) for mode in MODES}
    valid_chunks, invalid_chunks = 0, 0
    paths = sorted((exp / "chunks").glob("chunk_*.npz"))
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as packed:
                arrays = {k: packed[k] for k in packed.files}
        except (ValueError, EOFError, OSError, zipfile.BadZipFile) as exc:
            invalid_chunks += 1
            missing.append(dict(archive_file=path.relative_to(ROOT).as_posix(), status="unreadable",
                                role=f"chunk: {type(exc).__name__}: {exc}"))
            continue
        start, stop = int(arrays["start"]), int(arrays["stop"])
        assert path.name == f"chunk_{start:09d}_{stop:09d}.npz"
        assert 0 <= start < stop <= n and not present[start:stop].any()
        present[start:stop] = True
        valid_chunks += 1
        for mode, (check, logical, indices) in problems.items():
            step = arrays[f"{mode}_iterations"]
            assert step.shape == (stop - start,) and np.all((0 <= step) & (step <= c["max_iterations"]))
            iterations[mode][start:stop] = step
            for method in METHODS:
                key = f"{mode}_{method}"
                bits = np.unpackbits(arrays[f"{key}_correction_packed"], axis=1,
                                     count=check.shape[1], bitorder="little")
                correction = sp.csr_matrix(bits)
                predicted_syndrome = (correction @ check.T).toarray() % 2
                prediction = (correction @ logical.T).toarray() % 2
                converged = np.all(predicted_syndrome == detectors[start:stop, indices], axis=1)
                mismatch = np.any(prediction != truth[start:stop], axis=1)
                values = dict(success=converged & ~mismatch, converged=converged, logical_mismatch=mismatch)
                np.testing.assert_array_equal(prediction, arrays[f"{key}_prediction"])
                for field, value in values.items():
                    np.testing.assert_array_equal(value, arrays[f"{key}_{field}"])
                    outcomes[key][field][start:stop] = value
                if method != "bp":
                    assert converged.all()
                    bp_key = f"{mode}_bp"
                    valid_bp = outcomes[bp_key]["converged"][start:stop]
                    np.testing.assert_array_equal(arrays[f"{key}_correction_packed"][valid_bp],
                                                  arrays[f"{bp_key}_correction_packed"][valid_bp])
    # This fixed campaign used one invocation and chunks of 128 samples.
    assert len(report["invocations"]) == 1 and report["invocations"][0]["chunk_size"] == 128
    for start in range(0, n, 128):
        path = exp / "chunks" / f"chunk_{start:09d}_{min(start + 128, n):09d}.npz"
        if not path.exists():
            missing.append(dict(archive_file=path.relative_to(ROOT).as_posix(), status="missing", role="chunk"))
    checked = int(present.sum())
    for key, saved in report["decoders"].items():
        mode = key.split("_", 1)[0]
        actual = outcomes[key]
        bp = outcomes[f"{mode}_bp"]
        values = dict(shots=checked, failures=int((~actual["success"][present]).sum()),
                      flagged_failures=int((~actual["converged"][present]).sum()),
                      logical_mismatches=int(actual["logical_mismatch"][present].sum()),
                      rescued=int((actual["success"] & ~bp["success"] & present).sum()),
                      harmed=int((bp["success"] & ~actual["success"] & present).sum()),
                      logical_rescued=int((bp["logical_mismatch"] & ~actual["logical_mismatch"] & present).sum()),
                      logical_harmed=int((actual["logical_mismatch"] & ~bp["logical_mismatch"] & present).sum()),
                      bp_iteration_sum=int(iterations[mode][present].sum()))
        for field, value in values.items():
            if checked == n:
                assert saved[field] == value, (exp, key, field, saved[field], value)
            else:
                bound = (n - checked) * (c["max_iterations"] if field == "bp_iteration_sum" else 1)
                assert value <= saved[field] <= value + bound, (exp, key, field)
    return dict(chunk_files=len(paths), valid_chunks=valid_chunks, unreadable_chunks=invalid_chunks,
                expected_chunks=math.ceil(n / 128), independently_rescored_shots=checked,
                full_correction_audit=checked == n, shot_bank_hash_verified=True), outcomes


def pair_row(c, reference, candidate, metric, rescued, harmed, *, evidence):
    n = c["shots"]
    gain = (rescued - harmed) / n
    se = math.sqrt(max(0., (rescued + harmed - n * gain * gain) / (n * (n - 1))))
    return dict(code=c["code"], p=c["p"], metric=metric, reference=reference, candidate=candidate,
                shots=n, rescued=rescued, harmed=harmed, error_rate_reduction=gain,
                paired_se=se, approximate_ci95_low=gain - Z * se, approximate_ci95_high=gain + Z * se,
                exact_mcnemar_p=float(binomtest(rescued, rescued + harmed).pvalue) if rescued + harmed else 1.,
                evidence=evidence)


def plot(rows):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    colors = {"bp": "#667085", "bposd0": "#2878b5", "bposd_cs3": "#d97706"}
    names = {"bp": "BP", "bposd0": "BP + OSD-0", "bposd_cs3": "BP + OSD-CS3"}
    for ax, prefix, title in zip(axes, ("logical_z_mismatch", "block_failure"),
                               ("Logical-Z prediction error", "Block decoding failure")):
        for mode in MODES:
            for method in METHODS:
                selected = sorted((r for r in rows if r["decoder"] == f"{mode}_{method}"), key=lambda r: r["p"])
                p = np.array([r["p"] for r in selected])
                y = np.array([r[f"{prefix}_rate"] for r in selected])
                lo = np.array([r[f"{prefix}_ci95_low"] for r in selected])
                hi = np.array([r[f"{prefix}_ci95_high"] for r in selected])
                ax.errorbar(p, y, yerr=np.array([y - lo, hi - y]), color=colors[method],
                            marker="o" if mode == "x" else "^", ls="-" if mode == "x" else "--",
                            linewidth=1.7, markersize=4.5, capsize=2,
                            label=f"{names[method]} | {'X only' if mode == 'x' else 'joint XZ'}")
        ax.set(title=title, xlabel="Physical circuit error probability p", yscale="log", ylim=(1e-5, 1.25))
        ax.set_xticks(np.arange(.001, .009, .001))
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, which="major", alpha=.25)
    axes[0].set_ylabel("Error probability per 6-round memory experiment")
    axes[0].text(.03, .97, "At least one of 12 logical-X\nobservable flips predicted incorrectly",
                 transform=axes[0].transAxes, va="top", fontsize=10)
    axes[1].text(.03, .97, "Invalid syndrome OR\nwrong logical prediction", transform=axes[1].transAxes,
                 va="top", fontsize=10)
    fig.suptitle("BB72 circuit baseline | unscaled min-sum BP, at most 1000 iterations", fontsize=15, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .033, "100,000 shared shots per p; 95% Wilson intervals. OSD has identical values in both panels.",
             ha="center", fontsize=9)
    fig.text(.5, .006, "Final server aggregates are complete. p=0.008 has an incomplete local chunk download; other 7 points fully rescored.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.87, wspace=.15)
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true",
                        help="Validate final JSON/CSV aggregates and redraw the PNG without rescoring chunks.")
    args = parser.parse_args()
    if args.plots_only:
        rows = []
        for job in JOBS:
            for index in range(4):
                exp = ARCHIVE / f"resdir_{job}" / f"exp_{index}"
                report = json.loads((exp / "results.json").read_text())
                check_summary(exp, report)
                rows.extend(dict(p=report["config"]["p"], decoder=key, **result)
                            for key, result in report["decoders"].items())
        plot(rows)
        print(f"Validated {len(rows)} final rows and updated {PLOT.relative_to(ROOT)}")
        return
    audit = dict(manifest=manifest_audit(), experiments=[], notes=[
        "Final curves use complete server aggregates, never the incomplete downloaded subset.",
        "Independent rescoring verifies stored corrections; it does not rerun BP/OSD or prove agreement with published curves.",
    ])
    rows, pairs, missing = [], [], []
    for job_id in JOBS:
        job = ARCHIVE / f"resdir_{job_id}"
        assert (job / "completed.txt").exists()
        for name in ("job_metadata.txt", "submitted_script.slurm", *(f"command_exp_{i}.txt" for i in range(4)),
                     *(f"exit_code_exp_{i}.txt" for i in range(4)),
                     "source_snapshot/src/bb_code.py", "source_snapshot/src/bb_paper_baselines.py"):
            if not (job / name).exists():
                missing.append(dict(archive_file=(job / name).relative_to(ROOT).as_posix(), status="missing", role="job provenance"))
        for exp in sorted(job.glob("exp_*")):
            report = json.loads((exp / "results.json").read_text())
            c = report["config"]
            check_summary(exp, report)
            for name, expected in c["source_sha256"].items():
                assert sha(ROOT / name) == expected, f"Rescoring source changed: {name}"
                snapshot = job / ("entrypoint_snapshot.py" if name.startswith("scripts/") else f"source_snapshot/{name}")
                if snapshot.exists():
                    assert sha(snapshot) == expected
            for name, field in (("circuit.stim", "circuit_sha256"), ("detector_error_model.dem", "dem_sha256")):
                if (exp / name).exists():
                    assert sha(exp / name) == c[field]
                else:
                    missing.append(dict(archive_file=(exp / name).relative_to(ROOT).as_posix(), status="missing", role="circuit provenance"))
            detail, outcomes = rescore_chunks(exp, report, missing)
            audit["experiments"].append(dict(job_id=job_id, experiment=exp.name, p=c["p"],
                                               status=report["status"], final_shots=c["shots"], **detail))
            for key, result in report["decoders"].items():
                mode = key.split("_", 1)[0]
                rows.append(dict(job_id=job_id, experiment=exp.name, code=c["code"], p=c["p"], seed=c["seed"],
                                 decoder=key, status=report["status"], rounds=c["rounds"],
                                 metric_time_unit=c["metric_time_unit"], **result,
                                 independently_rescored_shots=detail["independently_rescored_shots"],
                                 full_correction_audit=detail["full_correction_audit"],
                                 detector_rows=report["problems"][mode]["detectors"],
                                 dem_mechanisms=report["problems"][mode]["mechanisms"],
                                 source=(exp / "results.json").relative_to(ROOT).as_posix()))
                if key.endswith("_bp"):
                    continue
                reference = f"{mode}_bp"
                for metric, count, rescue, harm in (("block_failure", "failures", "rescued", "harmed"),
                                                    ("logical_z_mismatch", "logical_mismatches", "logical_rescued", "logical_harmed")):
                    assert report["decoders"][reference][count] - result[count] == result[rescue] - result[harm]
                    pairs.append(pair_row(c, reference, key, metric, result[rescue], result[harm], evidence="full_server_aggregate"))
            if detail["full_correction_audit"]:
                comparisons = [(f"x_{m}", f"xz_{m}") for m in METHODS]
                comparisons += [(f"{mode}_bposd0", f"{mode}_bposd_cs3") for mode in MODES]
                for reference, candidate in comparisons:
                    before, after = (outcomes[key]["logical_mismatch"] for key in (reference, candidate))
                    pairs.append(pair_row(c, reference, candidate, "logical_z_mismatch",
                                          int((before & ~after).sum()), int((~before & after).sum()),
                                          evidence="independently_rescored_complete_chunks"))
            print(f"{job_id}/{exp.name} p={c['p']:g}: final {c['shots']}, independently rescored "
                  f"{detail['independently_rescored_shots']}, invalid chunks {detail['unreadable_chunks']}", flush=True)
    assert len(rows) == 48 and len({r["p"] for r in rows}) == 8
    previous = 0.
    for index, row in enumerate(sorted(pairs, key=lambda r: r["exact_mcnemar_p"])):
        previous = min(1., max(previous, (len(pairs) - index) * row["exact_mcnemar_p"]))
        row["holm_p"] = previous
        row["holm_family_size"] = len(pairs)
    audit.update(unique_final_shots=800000,
                 independently_rescored_shots=sum(r["independently_rescored_shots"] for r in audit["experiments"]),
                 decoder_rows=len(rows), paired_tests=len(pairs), missing_or_unreadable_files=len(missing))
    write_csv("", rows)
    write_csv("_paired", pairs)
    write_csv("_missing", missing)
    (ANALYSIS / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    plot(rows)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("experiments", "notes")}, indent=2))


if __name__ == "__main__":
    main()
