"""Resolve only complete, matching syndrome-OSD results for a saved CNN run."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD = "syndrome_osd0_v1"
POLICY = "paired_frozen_cnn_syndrome_osd0_v1"
RESIDUAL = "direct_hard_centered_osd0_v1"


@lru_cache(maxsize=None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_index(roots=None):
    roots = roots if roots is not None else [ROOT / "results/analysis",
        ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn"]
    index = {}
    for root in roots:
        for path in Path(root).rglob("report.json"):
            report = json.loads(path.read_text())
            if report.get("policy") != POLICY or report.get("partial", True):
                continue
            key = (report["checkpoint_sha256"], report["shot_bank_sha256"])
            index.setdefault(key, []).append((path, report))
    return index


def standard_metrics(history_path, index):
    """Return verified metrics/provenance, or None; never fall back to residual."""
    path = Path(history_path)
    history = json.loads(path.read_text())
    config, final = history["config"], history["final"]
    if config.get("osd") == STANDARD:
        return final["osd"], path, "native"
    # Lightweight downloads retain history/logs but omit weights and shot banks.
    # Recorded results remain plottable; a posthoc substitution cannot be verified.
    if not all(path.with_name(name).is_file() for name in ("model.pt", "final_shots.npz")):
        return None
    key = (sha(path.with_name("model.pt")), sha(path.with_name("final_shots.npz")))
    matches = []
    for source, report in index.get(key, []):
        if (report["config"] != config or report["selected_epoch"] != final["epoch"]
                or report["shots"] != final["shots"] or report["source_shots"] != final["shots"]
                or report["history_sha256"] != sha(path)):
            continue
        assert report["weights"] == "best_model_state_dict" and report["training_steps"] == 0
        assert report["osd_order"] == 0 and report["osd_policy"] == "only_invalid_sectors"
        metric = report["metrics"]["syndrome_osd0"]
        with np.load(source.with_name("paired_shots.npz"), allow_pickle=False) as bank:
            success = bank["syndrome_osd0_success"].astype(bool)
            assert len(success) == final["shots"]
            assert abs((~success).mean() - metric["logical_error_rate"]) < 1e-12
        matches.append((metric, source, "posthoc"))
    if len(matches) > 1:
        # Numerical/platform differences must not be silently cherry-picked.
        assert all(m[0] == matches[0][0] for m in matches), "Conflicting OSD reports"
    return matches[0] if matches else None


def interval(rate, shots):
    count = round(rate * shots)
    assert abs(rate - count/shots) < 1e-12
    z = 1.959963984540054
    denominator = 1 + z*z/shots
    center = (rate + z*z/(2*shots))/denominator
    half = z*np.sqrt(rate*(1-rate)/shots + z*z/(4*shots*shots))/denominator
    return count, max(0., center-half), min(1., center+half)


def residual_metrics(history_path, index):
    """Read residual results, distinguishing a hashed report from shot rescoring.

    Lightweight posthoc downloads may omit paired_shots.npz. Accept their
    reported aggregates only when all three original input hashes match locally;
    retain an explicit report-only provenance instead of claiming a local rescore.
    """
    path = Path(history_path)
    history = json.loads(path.read_text())
    config, final = history["config"], history["final"]
    if config.get("osd") == RESIDUAL:
        return final["osd"], path, "native"
    if not all(path.with_name(n).is_file() for n in ("model.pt", "final_shots.npz")):
        return None
    key = (sha(path.with_name("model.pt")), sha(path.with_name("final_shots.npz")))
    matches = []
    for source, report in index.get(key, []):
        if (report["config"] != config or report["selected_epoch"] != final["epoch"]
                or report["history_sha256"] != sha(path)
                or report["shots"] != final["shots"] or report["source_shots"] != final["shots"]
                or "residual_osd0" not in report["metrics"]):
            continue
        assert report["weights"] == "best_model_state_dict" and report["training_steps"] == 0
        assert report["osd_order"] == 0 and report["osd_policy"] == "only_invalid_sectors"
        metric = report["metrics"]["residual_osd0"]
        interval(metric["logical_error_rate"], final["shots"])
        assert metric["samples"] == final["shots"]
        status = "posthoc_report_inputs_hash_verified"
        if source.with_name("paired_shots.npz").is_file():
            with np.load(source.with_name("paired_shots.npz"), allow_pickle=False) as bank:
                success = bank["residual_osd0_success"].astype(bool)
                assert len(success) == final["shots"]
                assert abs((~success).mean() - metric["logical_error_rate"]) < 1e-12
            status = "posthoc_shots_verified"
        matches.append((metric, source, status))
    if len(matches) > 1:
        assert all(m[0] == matches[0][0] for m in matches), "Conflicting residual OSD reports"
    return matches[0] if matches else None
