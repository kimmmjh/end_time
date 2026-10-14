#!/usr/bin/env python3
"""Reevaluate frozen CNN best weights on saved final shots; never train.

Compare raw, hard-centred residual OSD-0, and published syndrome OSD-0.
Both OSD arms repair only invalid CSS sectors, using the same marginals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models._bb_tanner_cnn import BBTannerCNN
from src._bb_metrics import aggregate_bb_outcomes, bb_correction_outcomes
from src._direct_osd import DirectOSD0, SyndromeOSD0
from src.bb_code import BBCodeSpec

POLICY = "paired_frozen_cnn_syndrome_osd0_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired(before, after):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    difference = after.astype(float) - before.astype(float)
    return {"ler_reduction": float(difference.mean()),
            "paired_se": float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1 else 0.,
            "rescued": int(np.count_nonzero(after & ~before)),
            "harmed": int(np.count_nonzero(before & ~after))}


def repair(syndrome, px, pz, code, decoders):
    x, z = (px > .5).astype(np.uint8), (pz > .5).astype(np.uint8)
    sx, sz = syndrome[:, code.num_x_checks:], syndrome[:, :code.num_x_checks]
    need_x = np.any(x @ code.hz.T % 2 != sx, axis=1)
    need_z = np.any(z @ code.hx.T % 2 != sz, axis=1)
    for needed, hard, syn, prob, decoder in (
            (need_x, x, sx, px, decoders[0]), (need_z, z, sz, pz, decoders[1])):
        if needed.any():
            hard[needed] = decoder.decode_batch(syn[needed], prob[needed])
    return x + 3*z - 2*x*z


@torch.inference_mode()
def evaluate_run(run_dir, output_dir, *, device="cpu", batch_size=64, max_shots=None):
    """Read-only source; output directory must not already exist."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if batch_size < 1 or (max_shots is not None and max_shots < 1):
        raise ValueError("batch_size and max_shots must be positive.")
    checkpoint_path, bank_path = run_dir / "model.pt", run_dir / "final_shots.npz"
    history = json.loads((run_dir / "history.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (checkpoint.get("format") != "bb_tanner_cnn_v1"
            or config.get("architecture") != "bb_tanner_cnn"
            or config.get("noise_model") != "capacity"):
        raise ValueError("Expected a capacity Tanner CNN checkpoint.")
    if config != history["config"]:
        raise ValueError("Checkpoint and history configs differ.")
    if history.get("final", {}).get("epoch") != checkpoint["best_epoch"]:
        raise ValueError("Saved final bank must use this checkpoint's selected best epoch.")
    state = checkpoint.get("best_model_state_dict")
    if state is None or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Missing or non-finite selected best weights.")
    code = BBCodeSpec.from_name(config["code"])
    fingerprint = hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()
    if config["graph_fingerprint"] != fingerprint:
        raise ValueError("Checkpoint graph differs from the current code.")
    with np.load(bank_path, allow_pickle=False) as saved:
        if json.loads(str(saved["metadata_json"].item())) != config:
            raise ValueError("Final bank metadata differs from the checkpoint.")
        total = len(saved["syndrome"])
        count = min(total, max_shots) if max_shots is not None else total
        if total != history["final"]["shots"] or count == 0:
            raise ValueError("Final bank shot count is inconsistent or empty.")
        bank = {key: saved[key][:count].copy() for key in (
            "syndrome", "pauli", "raw_correction", "osd_correction", "raw_success", "osd_success")}
    if bank["syndrome"].shape != (count, code.num_checks) or bank["pauli"].shape != (count, code.n):
        raise ValueError("Wrong final bank dimensions.")
    if not np.isin(bank["syndrome"], [0, 1]).all() or not np.isin(bank["pauli"], [0, 1, 2, 3]).all():
        raise ValueError("Invalid syndrome or Pauli values.")
    true_x, true_z = np.isin(bank["pauli"], (1, 2)), np.isin(bank["pauli"], (2, 3))
    expected = np.concatenate((true_z @ code.hx.T % 2, true_x @ code.hz.T % 2), axis=1)
    if not np.array_equal(expected, bank["syndrome"]):
        raise ValueError("Final bank truth and syndrome disagree.")

    device = torch.device(device)
    model = BBTannerCNN(code, width=config["cnn_width"], depth=config["cnn_depth"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    prior = torch.tensor(config["channel_probabilities"], dtype=torch.float32, device=device)
    scoring = {key: torch.tensor(getattr(code, key)) for key in ("hx", "hz", "logicals_x", "logicals_z")}
    decoders = {"residual_osd0": (DirectOSD0(code.hz), DirectOSD0(code.hx)),
                "syndrome_osd0": (SyndromeOSD0(code.hz), SyndromeOSD0(code.hx))}
    outcomes = {key: [] for key in ("raw", *decoders)}
    corrections = {key: [] for key in outcomes}
    times = {key: 0. for key in ("inference", *decoders)}
    started = time.perf_counter()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        syndrome, truth = bank["syndrome"][start:end], bank["pauli"][start:end]
        tick = time.perf_counter()
        logits = model(torch.tensor(syndrome, dtype=torch.float32, device=device), channel_probabilities=prior)
        qx, qz = model.component_probabilities(logits)
        px, pz = qx.cpu().numpy(), qz.cpu().numpy()
        if (not np.isfinite(px).all() or not np.isfinite(pz).all()
                or ((px < 0) | (px > 1)).any() or ((pz < 0) | (pz > 1)).any()):
            raise ValueError("Invalid model component probabilities.")
        x, z = (px > .5).astype(np.uint8), (pz > .5).astype(np.uint8)
        batch_corrections = {"raw": x + 3*z - 2*x*z}
        times["inference"] += time.perf_counter() - tick
        for name, decoder_pair in decoders.items():
            tick = time.perf_counter()
            batch_corrections[name] = repair(syndrome, px, pz, code, decoder_pair)
            times[name] += time.perf_counter() - tick
        for name, correction in batch_corrections.items():
            outcome = bb_correction_outcomes(torch.from_numpy(correction), torch.from_numpy(syndrome),
                                            torch.from_numpy(truth), **scoring)
            if name != "raw" and not outcome.syndrome_converged.all():
                raise RuntimeError(f"{name} returned an invalid syndrome.")
            outcomes[name].append(outcome)
            corrections[name].append(correction)
        if start == 0 or end == count or end % (batch_size * 64) == 0:
            print(f"  {end}/{count} shots, {time.perf_counter()-started:.1f}s", flush=True)

    corrections = {name: np.concatenate(values) for name, values in corrections.items()}
    successes = {name: torch.cat([row.success for row in rows]).numpy() for name, rows in outcomes.items()}
    # Audit the historical flags independently before comparing with regenerated outputs.
    for branch in ("raw", "osd"):
        old = bb_correction_outcomes(torch.from_numpy(bank[f"{branch}_correction"]),
                                    torch.from_numpy(bank["syndrome"]), torch.from_numpy(bank["pauli"]), **scoring)
        if not np.array_equal(old.success.numpy(), bank[f"{branch}_success"].astype(bool)):
            raise ValueError(f"Stored {branch} success flags fail rescoring.")
    report = {
        "policy": POLICY, "config": config, "source_run": str(run_dir),
        "checkpoint_sha256": sha256(checkpoint_path), "shot_bank_sha256": sha256(bank_path),
        "history_sha256": sha256(run_dir / "history.json"),
        "implementation_sha256": {p.name: sha256(p) for p in (
            Path(__file__), ROOT / "src/_direct_osd.py", ROOT / "models/_bb_tanner_cnn.py")},
        "completed_epochs": checkpoint["epoch"] + 1, "selected_epoch": checkpoint["best_epoch"],
        "weights": "best_model_state_dict", "training_steps": 0, "extra_bp_iterations": 0,
        "osd_order": 0, "osd_policy": "only_invalid_sectors", "tie_break": "original_column_order",
        "shots": count, "source_shots": total, "partial": count != total,
        "device": str(device), "torch_version": str(torch.__version__), "batch_size": batch_size,
        "metrics": {name: asdict(aggregate_bb_outcomes(rows)) for name, rows in outcomes.items()},
        "paired": {"standard_vs_residual": paired(successes["residual_osd0"], successes["syndrome_osd0"]),
                   "standard_vs_raw": paired(successes["raw"], successes["syndrome_osd0"])},
        "historical_reproduction": {
            name: {"correction_mismatch_shots": int(np.any(corrections[name] != bank[f"{old}_correction"], axis=1).sum()),
                   "success_mismatch_shots": int(np.count_nonzero(successes[name] != bank[f"{old}_success"].astype(bool)))}
            for name, old in (("raw", "raw"),
                              ("syndrome_osd0" if config.get("osd") == "syndrome_osd0_v1"
                               else "residual_osd0", "osd"))},
        "seconds": times,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "paired_shots.npz", syndrome=bank["syndrome"], pauli=bank["pauli"],
                        **{f"{name}_correction": value for name, value in corrections.items()},
                        **{f"{name}_success": value for name, value in successes.items()})
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def discover_runs(roots, *, latest_only=False, require_complete=False):
    """Select completed runs before checking inputs, never fall back to older weights."""
    histories = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Directory does not exist: {root}")
        for path in root.rglob("history.json"):
            history = json.loads(path.read_text())
            if history.get("config", {}).get("architecture") == "bb_tanner_cnn":
                histories[path.parent.resolve()] = history
    missing, candidates = [], []
    for run, history in histories.items():
        if "final" not in history:
            missing.append({"run": str(run), "missing": ["completed final evaluation"]})
        else:
            candidates.append((run, history))
    if latest_only:
        groups = {}
        for run, history in candidates:
            key = json.dumps(history["config"], sort_keys=True)
            groups.setdefault(key, []).append((run, history))
        candidates = []
        for group in groups.values():
            epoch = max(len(h["train"]) for _, h in group)
            latest = sorted((run, h) for run, h in group if len(h["train"]) == epoch)
            if len(latest) > 1:
                # The same archived job may also exist at the server root.
                # Distinct tied runs are ambiguous; identical copies are safe.
                signatures = {tuple(sha256(run / name) if (run / name).is_file() else None
                                    for name in ("model.pt", "final_shots.npz", "history.json"))
                              for run, _ in latest}
                if len(signatures) != 1:
                    raise ValueError(f"Ambiguous latest runs at {epoch} epochs: {[str(r) for r, _ in latest]}")
            candidates.append(latest[0])
    runs, absent_selected = [], []
    for run, _ in candidates:
        absent = [name for name in ("model.pt", "final_shots.npz") if not (run / name).is_file()]
        if absent:
            absent_selected.append({"run": str(run), "missing": absent})
        else:
            runs.append(run)
    if require_complete and absent_selected:
        raise ValueError(f"Selected runs lack required inputs: {absent_selected}")
    return sorted(runs), missing + absent_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="Run directories or result trees containing history.json")
    parser.add_argument("--auto-roots", action="store_true", help="Discover repo-root resdir_* and the organized CNN archive")
    parser.add_argument("--latest-only", action="store_true", help="Keep the longest completed training per exact model configuration")
    parser.add_argument("--require-complete", action="store_true", help="Refuse missing inputs in selected completed runs")
    parser.add_argument("--expected-runs", type=int, help="Require this many eligible runs before sharding")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True, help="Separate output tree; existing runs are never overwritten")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-shots", type=int, help="Explicit partial-bank smoke test; omitted means the full bank")
    parser.add_argument("--dry-run", action="store_true", help="List completed CNN runs and missing inputs without inference")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count.")
    roots = list(args.roots)
    if args.auto_roots:
        roots.extend(p for p in ROOT.glob("resdir_*") if p.is_dir())
        archive = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn"
        if archive.is_dir():
            roots.append(archive)
    if not roots:
        parser.error("Supply input roots or --auto-roots.")
    try:
        runs, missing = discover_runs(roots, latest_only=args.latest_only, require_complete=args.require_complete)
    except ValueError as error:
        parser.error(str(error))
    if not runs:
        parser.error("No evaluable CNN runs found.")
    if args.expected_runs is not None and len(runs) != args.expected_runs:
        parser.error(f"Expected {args.expected_runs} eligible runs, found {len(runs)}; check source directories.")
    total_runs = len(runs)
    runs = runs[args.shard_index::args.shard_count]
    summaries = []
    for run in sorted(runs):
        job = next((p.name for p in run.parents if p.name.startswith("resdir_")), "standalone")
        destination = args.output_root / job / run.parent.name / run.name
        print(f"{run}\n  -> {destination}", flush=True)
        if not args.dry_run:
            report = evaluate_run(run, destination, device=args.device, batch_size=args.batch_size, max_shots=args.max_shots)
            summaries.append({"run": str(run), "report": str(destination / "report.json"),
                              "shots": report["shots"], "metrics": report["metrics"], "paired": report["paired"]})
            print(json.dumps({key: value["logical_error_rate"] for key, value in report["metrics"].items()}), flush=True)
    print(f"Eligible runs: {total_runs}; this shard: {len(runs)}; skipped runs: {len(missing)}", flush=True)
    for item in missing:
        print(f"SKIP {item['run']}: missing {', '.join(item['missing'])}")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary_name = "summary.json" if args.shard_count == 1 else f"summary_shard_{args.shard_index}.json"
        (args.output_root / summary_name).write_text(json.dumps({"runs": summaries, "skipped": missing,
            "latest_only": args.latest_only, "eligible_runs": total_runs,
            "shard_index": args.shard_index, "shard_count": args.shard_count}, indent=2) + "\n")


if __name__ == "__main__":
    main()
