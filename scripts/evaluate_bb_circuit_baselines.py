#!/usr/bin/env python3
"""Paired library BP / BP+OSD-0 / BP+OSD-CS3 on a saved BB memory experiment.

CPU workers decode disjoint portions of the same exact Stim circuit shot bank.
Each chunk saves all corrections for independent rescoring and safe resumption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp

from src.bb_code import BBCodeSpec
from src.bb_paper_baselines import (
    CIRCUIT_POLICY, SOURCE_COMMIT, METHODS, PairedBpOsd,
    decoding_problem, make_memory_circuit,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("bb72", "bb144"), required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--detector-inputs", nargs="+", choices=("x", "xz"), default=["x", "xz"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for field in ("shots", "workers", "chunk_size", "max_iterations"):
        if getattr(args, field) < 1:
            parser.error(f"{field} must be positive")
    if not 0 < args.p < .5 or args.seed < 0:
        parser.error("require 0 < p < .5 and seed >= 0")
    if args.rounds is not None and args.rounds < 1:
        parser.error("rounds must be positive")
    args.detector_inputs = sorted(set(args.detector_inputs))
    return args


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_npz(path, **values):
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **values)
    temp.replace(path)


def save_json(path, report):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def interval(failures, shots):
    z = 1.959963984540054
    p = failures / shots
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    radius = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots**2)) / denominator
    return max(0., center - radius), min(1., center + radius)


_DECODERS = {}
_DETECTORS = 0
_OBSERVABLES = 0


def init_worker(problems, iterations, detectors, observables):
    global _DECODERS, _DETECTORS, _OBSERVABLES
    _DECODERS = {mode: PairedBpOsd(problem, iterations) for mode, problem in problems.items()}
    _DETECTORS, _OBSERVABLES = detectors, observables


def decode_chunk(task):
    start, det_packed, obs_packed = task
    detectors = np.unpackbits(det_packed, axis=1, count=_DETECTORS, bitorder="little")
    observables = np.unpackbits(obs_packed, axis=1, count=_OBSERVABLES, bitorder="little")
    arrays = {"start": np.array(start), "stop": np.array(start + len(detectors))}
    for mode, decoder in _DECODERS.items():
        outcomes, iterations = decoder.score(detectors[:, decoder.problem.detector_indices], observables)
        arrays[f"{mode}_iterations"] = iterations
        for method, result in outcomes.items():
            arrays.update({f"{mode}_{method}_{field}": value for field, value in result.items()})
    return arrays


def accumulate(totals, arrays, modes):
    for mode in modes:
        reference = arrays[f"{mode}_bp_success"]
        reference_mismatch = arrays[f"{mode}_bp_logical_mismatch"]
        iterations = arrays[f"{mode}_iterations"]
        for method in METHODS:
            name = f"{mode}_{method}"
            success = arrays[f"{name}_success"]
            converged = arrays[f"{name}_converged"]
            row = totals.setdefault(name, dict(shots=0, failures=0, flagged_failures=0,
                logical_mismatches=0, rescued=0, harmed=0, bp_iteration_sum=0,
                logical_rescued=0, logical_harmed=0))
            row["shots"] += len(success)
            row["failures"] += int((~success).sum())
            row["flagged_failures"] += int((~converged).sum())
            row["logical_mismatches"] += int(arrays[f"{name}_logical_mismatch"].sum())
            row["logical_rescued"] += int((reference_mismatch & ~arrays[f"{name}_logical_mismatch"]).sum())
            row["logical_harmed"] += int((~reference_mismatch & arrays[f"{name}_logical_mismatch"]).sum())
            row["rescued"] += int((success & ~reference).sum())
            row["harmed"] += int((~success & reference).sum())
            row["bp_iteration_sum"] += int(iterations.sum())
            if method != "bp" and (row["flagged_failures"] or row["harmed"]):
                raise RuntimeError("OSD validity/preservation invariant failed.")


def write_report(directory, report, totals):
    report["decoders"] = {}
    for name, counts in totals.items():
        n = counts["shots"]
        low, high = interval(counts["failures"], n)
        logical_low, logical_high = interval(counts["logical_mismatches"], n)
        report["decoders"][name] = {**counts,
            "block_failure_rate": counts["failures"] / n,
            "block_failure_ci95_low": low, "block_failure_ci95_high": high,
            "logical_z_mismatch_rate": counts["logical_mismatches"] / n,
            "logical_z_mismatch_ci95_low": logical_low,
            "logical_z_mismatch_ci95_high": logical_high,
            "syndrome_convergence": 1 - counts["flagged_failures"] / n,
            "unflagged_failures": counts["failures"] - counts["flagged_failures"],
            "mean_bp_iterations": counts["bp_iteration_sum"] / n,
            "paired_accuracy_gain": (counts["rescued"] - counts["harmed"]) / n,
            "paired_logical_accuracy_gain": (counts["logical_rescued"] - counts["logical_harmed"]) / n,
        }
    save_json(directory / "results.json", report)
    if totals:
        rows = [dict(code=report["config"]["code"], p=report["config"]["p"],
                     decoder=name, status=report["status"], **values)
                for name, values in report["decoders"].items()]
        temp = directory / "summary.csv.tmp"
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temp.replace(directory / "summary.csv")


def run(args):
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"{args.out} exists; use --resume for its saved bank.")
    if args.resume and not (args.out / "results.json").exists():
        raise FileNotFoundError("--resume requires an existing results.json")
    version = importlib.metadata.version("ldpc")
    if version != "2.4.1":
        raise RuntimeError(f"This campaign pins ldpc==2.4.1; installed {version}")
    code = BBCodeSpec.from_name(args.code)
    rounds = code.d if args.rounds is None else args.rounds
    circuit = make_memory_circuit(code, p=args.p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    problems = {mode: decoding_problem(circuit, mode) for mode in args.detector_inputs}
    sources = ("scripts/evaluate_bb_circuit_baselines.py", "src/bb_paper_baselines.py", "src/bb_code.py")
    config = dict(code=args.code, p=args.p, rounds=rounds, shots=args.shots, seed=args.seed,
        circuit_policy=CIRCUIT_POLICY, schedule_source_commit=SOURCE_COMMIT,
        detector_inputs=args.detector_inputs, max_iterations=args.max_iterations,
        bp_method="min_sum", ms_scaling_factor=1., schedule="parallel",
        osd_method="OSD_CS", osd_order=3, additional_osd0=True, library="ldpc", library_version=version,
        posterior_reseeding=False, custom_message_clipping=False,
        noisy_idle=True, physical_noise="prep/measurement p; CNOT DEP2(p); idle DEP1(p)",
        observable_basis="logical_X", error_component="logical_Z", num_observables=code.k,
        num_detectors=circuit.num_detectors, detector_frames=rounds + 1,
        input_state="logical_plus", metric_time_unit="full_memory_experiment",
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
        dem_sha256=hashlib.sha256(str(dem).encode()).hexdigest(),
        source_sha256={name: file_hash(ROOT / name) for name in sources})
    if args.resume:
        report = json.loads((args.out / "results.json").read_text())
        if report["config"] != config:
            raise ValueError("Saved physics/decoder/source configuration differs; refusing to mix runs.")
        if file_hash(args.out / "shots.npz") != report["shot_bank_sha256"]:
            raise ValueError("Saved shot bank checksum mismatch.")
        with np.load(args.out / "shots.npz", allow_pickle=False) as bank:
            det_packed, obs_packed = bank["detectors_packed"], bank["observables_packed"]
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "chunks").mkdir()
        (args.out / "circuit.stim").write_text(str(circuit))
        (args.out / "detector_error_model.dem").write_text(str(dem))
        for mode, problem in problems.items():
            sp.save_npz(args.out / f"{mode}_check.npz", problem.check)
            sp.save_npz(args.out / f"{mode}_logical.npz", problem.logical)
            save_npz(args.out / f"{mode}_prior.npz", probabilities=problem.priors,
                     detector_indices=problem.detector_indices)
        det_packed, obs_packed = circuit.compile_detector_sampler(seed=args.seed).sample(
            args.shots, separate_observables=True, bit_packed=True)
        save_npz(args.out / "shots.npz", detectors_packed=det_packed, observables_packed=obs_packed,
                 metadata_json=json.dumps(config), bitorder="little")
        report = dict(config=config, status="running", shots_completed=0,
            shot_bank_sha256=file_hash(args.out / "shots.npz"), decoders={},
            versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "stim", "ldpc")},
            problems={mode: dict(detectors=g.check.shape[0], mechanisms=g.check.shape[1])
                      for mode, g in problems.items()}, invocations=[])
    totals, completed = {}, 0
    for path in sorted((args.out / "chunks").glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            if int(chunk["start"]) != completed or not completed < int(chunk["stop"]) <= args.shots:
                raise ValueError(f"Noncontiguous/invalid checkpoint chunk: {path}")
            accumulate(totals, chunk, args.detector_inputs)
            completed = int(chunk["stop"])
    report.update(status="running", shots_completed=completed)
    report["invocations"].append(dict(workers=args.workers, chunk_size=args.chunk_size, resumed=args.resume))
    write_report(args.out, report, totals)
    tasks = ((start, det_packed[start:start + args.chunk_size], obs_packed[start:start + args.chunk_size])
             for start in range(completed, args.shots, args.chunk_size))
    initargs = (problems, args.max_iterations, circuit.num_detectors, code.k)
    pool = None
    started = time.perf_counter()
    try:
        if completed == args.shots:
            results = iter(())
        elif args.workers == 1:
            init_worker(*initargs)
            results = map(decode_chunk, tasks)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=initargs)
            results = pool.imap(decode_chunk, tasks, chunksize=1)
        for arrays in results:
            start, stop = int(arrays["start"]), int(arrays["stop"])
            save_npz(args.out / "chunks" / f"chunk_{start:09d}_{stop:09d}.npz", **arrays)
            accumulate(totals, arrays, args.detector_inputs)
            report["shots_completed"] = stop
            report["invocations"][-1]["wall_seconds"] = time.perf_counter() - started
            write_report(args.out, report, totals)
            print(f"{args.code} p={args.p:g}: {stop}/{args.shots} paired shots", flush=True)
        if pool:
            pool.close(); pool.join(); pool = None
    except BaseException:
        report["status"] = "interrupted"
        write_report(args.out, report, totals)
        raise
    finally:
        if pool:
            pool.terminate(); pool.join()
    report["status"] = "complete"
    write_report(args.out, report, totals)
    return report


if __name__ == "__main__":
    run(parse_args())
