import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import run_bb_dem_sample_scaling as scaling


def test_launcher_previews_both_experiments_without_weights():
    launcher = scaling.REPO / "run_bb_dem_sample_scaling.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    env = {**os.environ, "BB_DRY_RUN": "1", "BB_REPO_ROOT": str(scaling.REPO),
           "BB_PYTHON": sys.executable}
    env.pop("BB_SCALE_RUN_DIR", None)
    output = subprocess.run(["bash", str(launcher)], env=env, check=True,
                            text=True, capture_output=True).stdout
    assert output.count('"total_samples": 2000384') == 2
    assert '"experiment": "bb72_d8_p0.002"' in output
    assert '"experiment": "bb144_d6_p0.002"' in output


def recipe():
    return json.loads(scaling.MANIFEST.read_text())


def checkpoint(campaign, entry):
    return dict(format="bb_dem_cnn_v1", config=entry["expected_config"], epoch=999,
                best_epoch=984, best_model_state_dict={"weight": torch.ones(1)},
                optimizer_state_dict={"state": {"kept": True}},
                history={"train": [{"epoch": i} for i in range(1000)],
                         "phases": [{"batch_size": 4, "batches": 128}]})


def test_manifest_matches_archived_training_metadata():
    campaign = recipe()
    for entry in campaign["experiments"]:
        history_path = scaling.REPO / entry["imported_run"] / "history.json"
        if not history_path.exists():
            pytest.skip("Downloaded source histories are not present in this checkout")
        history = json.loads(history_path.read_text())
        assert entry["expected_config"] == history["config"]
        assert len(history["train"]) == entry["source_epochs"] == 1000
        assert history["best_epoch"] + 1 == entry["source_best_epoch"] == 985
        assert history["phases"][-1]["batch_size"] * history["phases"][-1]["batches"] == 512
    assert campaign["target_epoch"] * 512 == 2000384


def test_source_validation_rejects_wrong_schedule_and_sample_count():
    campaign = recipe()
    entry = campaign["experiments"][0]
    saved = checkpoint(campaign, entry)
    assert scaling.validate_checkpoint(saved, campaign, entry, initial=True) == 1000
    wrong = copy.deepcopy(saved)
    wrong["config"]["depth"] = 12
    with pytest.raises(ValueError, match="configuration"):
        scaling.validate_checkpoint(wrong, campaign, entry, initial=True)
    wrong = copy.deepcopy(saved)
    wrong["history"]["phases"][0]["batch_size"] = 8
    with pytest.raises(ValueError, match="accounting"):
        scaling.validate_checkpoint(wrong, campaign, entry)
    saved["epoch"] = 1000
    saved["history"]["train"].append({"epoch": 1000})
    with pytest.raises(ValueError, match="continuous"):
        scaling.validate_checkpoint(saved, campaign, entry)
    saved["scheduler_state_dict"] = {"last_epoch": 128, "T_max": 2907*128}
    assert scaling.validate_checkpoint(saved, campaign, entry) == 1001


def test_campaign_recovers_mid_training_and_uses_separate_final_bank(tmp_path, monkeypatch):
    from scripts import resume_bb_dem_cnn
    campaign = recipe()
    entry = campaign["experiments"][0]
    source = tmp_path / entry["source_run"] / "model.pt"
    source.parent.mkdir(parents=True)
    torch.save(checkpoint(campaign, entry), source)
    original = source.read_bytes()
    checked = []
    monkeypatch.setattr(resume_bb_dem_cnn, "verify_source_stopped", checked.append)
    args = SimpleNamespace(source_root=tmp_path, campaign_root=tmp_path / "resdir_999", stop_file=tmp_path / "STOP")
    segments, evaluations = [], []
    cache = {}

    def train(path, output, end_epoch, campaign, entry, stop):
        saved = torch.load(path, weights_only=False)
        start = scaling.validate_checkpoint(saved, campaign, entry)
        segments.append((start, end_epoch))
        # Simulate a graceful stop during the first segment.
        end = 1200 if len(segments) == 1 else end_epoch
        saved["epoch"] = end-1
        saved["history"]["train"] = [{"epoch": i} for i in range(end)]
        saved["scheduler_state_dict"] = {"last_epoch": (end-1000)*128, "T_max": 2907*128}
        output.mkdir(parents=True, exist_ok=True)
        torch.save(saved, output / "model.pt")
        if end == 1200:
            args.stop_file.touch()

    def evaluate(path, weights, directory, campaign, seed, stop):
        key = (str(path), weights, seed)
        if key not in cache:
            evaluations.append(key)
            cache[key] = {"status": "complete", "shots": campaign["shots"],
                          "metrics": {"hierarchical": {"failures": 10 if '3907' in str(path) else 20}}}
        return cache[key]

    monkeypatch.setattr(scaling, "train_segment", train)
    monkeypatch.setattr(scaling, "run_evaluation", evaluate)
    assert not scaling.run(args, campaign, entry)
    args.stop_file.unlink()
    assert scaling.run(args, campaign, entry)
    assert segments == [(1000, 1953), (1200, 1953), (1953, 2930), (2930, 3907)]
    assert len(evaluations) == 9
    assert all(e[2] == campaign["selection_seed"] for e in evaluations[:-1])
    assert evaluations[-1][2] == campaign["final_seed"]
    directory = args.campaign_root / entry["name"]
    result = json.loads((directory / "completed.json").read_text())
    assert result["selected"]["epoch"] == 3907 and result["selected"]["weights"] == "best"
    assert len(list((directory / "checkpoints").glob("*.pt"))) == 4
    assert source.read_bytes() == original
    assert len(checked) == 1
    # A completed campaign is idempotent and retrains nothing.
    assert scaling.run(args, campaign, entry)
    assert len(segments) == 4 and len(evaluations) == 9


def test_evaluation_reuses_only_matching_complete_attempt(tmp_path):
    campaign = recipe()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    directory = tmp_path / "evaluation"
    attempt = directory / "attempt_001"
    attempt.mkdir(parents=True)
    signature = dict(checkpoint_sha256=scaling.digest(checkpoint), weights="best",
                     seed=campaign["selection_seed"], shots=8192, confidence=.9)
    scaling.write_json(directory / "attempt_001.json", signature)
    scaling.write_json(attempt / "results.json", {"status": "complete", "shots": 8192})
    result = scaling.run_evaluation(checkpoint, "best", directory, campaign,
                                    campaign["selection_seed"], lambda: True)
    assert result["shots"] == 8192
    assert scaling.run_evaluation(checkpoint, "latest", directory, campaign,
                                  campaign["selection_seed"], lambda: True) is None
    scaling.write_json(attempt / "results.json", {"status": "running", "shots": 4096})
    assert scaling.run_evaluation(checkpoint, "best", directory, campaign,
                                  campaign["selection_seed"], lambda: True) is None
