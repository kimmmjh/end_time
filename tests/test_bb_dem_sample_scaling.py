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


def test_worker_lock_rejects_another_process_and_releases(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "60828644")
    # The old flock file left by the failed run is harmless.
    (tmp_path / "worker.lock").touch()
    with scaling.worker_lock(tmp_path):
        owner = json.loads((tmp_path / "worker.lock.d/owner.json").read_text())
        assert owner["job_id"] == "60828644" and owner["pid"] == os.getpid()
        code = (
            "from pathlib import Path\n"
            "import sys\n"
            "from scripts.run_bb_dem_sample_scaling import worker_lock\n"
            "with worker_lock(Path(sys.argv[1])):\n"
            "    raise AssertionError('Concurrent worker acquired the lock')\n"
        )
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                               cwd=scaling.REPO, capture_output=True, text=True)
        assert child.returncode != 0 and "Worker lock already exists" in child.stderr
        assert json.loads((tmp_path / "worker.lock.d/owner.json").read_text()) == owner
    assert not (tmp_path / "worker.lock.d").exists()
    with scaling.worker_lock(tmp_path):
        pass


def test_worker_lock_releases_on_exception_and_does_not_steal_stale_lock(tmp_path):
    with pytest.raises(ValueError, match="simulated failure"):
        with scaling.worker_lock(tmp_path):
            raise ValueError("simulated failure")
    lock = tmp_path / "worker.lock.d"
    assert not lock.exists()
    # A process killed between mkdir and metadata creation leaves an empty lock.
    lock.mkdir()
    with pytest.raises(RuntimeError, match="verify that its Slurm job"):
        with scaling.worker_lock(tmp_path):
            pytest.fail("A stale lock must not be stolen automatically")
    assert lock.is_dir()


def test_launcher_previews_both_experiments_without_weights():
    launcher = scaling.REPO / "run_bb_dem_sample_scaling.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    env = {**os.environ, "BB_DRY_RUN": "1", "BB_REPO_ROOT": str(scaling.REPO),
           "BB_PYTHON": sys.executable, "BB_SCALE_WORKERS": "0,1"}
    env.pop("BB_SCALE_RUN_DIR", None)
    output = subprocess.run(["bash", str(launcher)], env=env, check=True,
                            text=True, capture_output=True).stdout
    assert output.count('"total_samples": 2000384') == 2
    assert '"experiment": "bb72_d8_p0.002"' in output
    assert '"experiment": "bb144_d6_p0.002"' in output
    env.pop("BB_SCALE_WORKERS")
    output = subprocess.run(["bash", str(launcher)], env=env, check=True,
                            text=True, capture_output=True).stdout
    assert '"experiment": "bb72_d8_p0.002"' not in output
    assert '"continuous": true' in output


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


def test_continuation_resumes_training_and_unfinished_evaluation(tmp_path, monkeypatch):
    campaign = recipe()
    entry = campaign["experiments"][1]
    directory = tmp_path / entry["name"]
    baseline = directory / "checkpoints/epoch_3907.pt"
    baseline.parent.mkdir(parents=True)
    saved = checkpoint(campaign, entry)
    saved.update(epoch=3906, scheduler_state_dict={"last_epoch": 2907*128, "T_max": 2907*128})
    saved["history"]["train"] = [{"epoch": i} for i in range(3907)]
    torch.save(saved, baseline)
    original = baseline.read_bytes()
    scaling.write_json(directory / "comparison.json", [])
    args = SimpleNamespace(continuation_lr=1e-5, continuation_every=1000)
    stopped = [False]
    segments, evaluated, cache = [], [], {}

    def train(path, output, end_epoch, campaign, entry, stop, *, continuation):
        saved = torch.load(path, weights_only=False)
        start = scaling.validate_checkpoint(saved, campaign, entry, continuation=continuation)
        segments.append((start, end_epoch))
        # Simulate an epoch-boundary stop, then a complete milestone.
        end = 4000 if len(segments) == 1 else end_epoch
        saved.update(epoch=end-1, scheduler_type="constant",
                     scheduler_state_dict={"last_epoch": (end-3907)*128, "base_lrs": [1e-5]})
        saved["optimizer_state_dict"]["param_groups"] = [{"lr": 1e-5}]
        saved["history"]["train"] = [{"epoch": i} for i in range(end)]
        output.mkdir(parents=True, exist_ok=True)
        torch.save(saved, output / "model.pt")
        if end == 4000 or end == 5907:
            stopped[0] = True

    def evaluate(path, weights, directory, campaign, seed, stop):
        key = (str(path), weights)
        if key in cache:
            return cache[key]
        if stop():
            return None
        evaluated.append(key)
        cache[key] = {"metrics": {"hierarchical": {"failures": 10}}}
        # Stop after best, before latest: the next job must finish evaluation first.
        if len(evaluated) == 1:
            stopped[0] = True
        return cache[key]

    monkeypatch.setattr(scaling, "train_segment", train)
    monkeypatch.setattr(scaling, "run_evaluation", evaluate)
    for _ in range(3):
        stopped[0] = False
        assert not scaling.run_continuation(args, campaign, entry, directory, lambda: stopped[0])
    assert segments == [(3907, 4907), (4000, 4907), (4907, 5907)]
    assert [weights for _, weights in evaluated] == ["best", "latest"]
    records = json.loads((directory / "continuous/comparison.json").read_text())
    assert len(records) == 2 and records[-1]["epoch"] == 4907
    assert records[-1]["cumulative_samples"] == 4907 * 512
    assert baseline.read_bytes() == original
    latest = torch.load(directory / "continuous/training/model.pt", weights_only=False)
    assert latest["optimizer_state_dict"]["state"] == {"kept": True}
    bad = copy.deepcopy(latest)
    bad["scheduler_state_dict"]["last_epoch"] -= 128
    with pytest.raises(ValueError, match="constant-LR"):
        scaling.validate_checkpoint(bad, campaign, entry,
                                    continuation=scaling.continuation_recipe(args, campaign))
    args.continuation_lr = 2e-5
    with pytest.raises(ValueError, match="settings changed"):
        scaling.run_continuation(args, campaign, entry, directory, lambda: False)


@pytest.mark.parametrize("mode, expected_status, submits", [
    ("deadline", 0, True), ("failure", 1, False), ("cancel", 0, False),
    ("manual", 0, False), ("submission_failure", 1, True),
])
def test_launcher_chains_only_graceful_deadline_stops(tmp_path, mode, expected_status, submits):
    # Exercise the actual Bash launcher with local scheduler stubs; no jobs submitted.
    root = tmp_path / "repo"
    root.mkdir()
    for folder in ("models", "src", "scripts"):
        (root / folder).mkdir()
    (root / "models/model.py").touch()
    (root / "src/train.py").touch()
    for name in ("run_bb_dem_sample_scaling.py", "evaluate_bb_hierarchical_dem_cnn.py", "resume_bb_dem_cnn.py"):
        (root / "scripts" / name).touch()
    manifest = root / "scripts/manifests/bb_dem_sample_scaling.json"
    manifest.parent.mkdir()
    manifest.write_text(scaling.MANIFEST.read_text())
    campaign = root / "recovered"
    campaign.mkdir()
    (campaign / "source_manifest.json").write_text(manifest.read_text())
    activate = tmp_path / "envs/nde/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    srun = "exit 1" if mode == "failure" else (
        'kill -TERM "$PPID"; exit 75' if mode == "cancel" else 'kill -USR1 "$PPID"; exit 75')
    for name, body in {
        "module": "exit 0", "fakepython": "exit 0", "git": "exit 1", "srun": srun,
        "sbatch": 'printf "%s\\n" "$@" > "$TEST_SUBMISSION"\n' + (
            "exit 1" if mode == "submission_failure" else "echo 777"),
    }.items():
        path = commands / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = {key: val for key, val in os.environ.items() if not key.startswith(("BB_", "SLURM_"))}
    env.update(PATH=str(commands) + os.pathsep + env["PATH"], PSCRATCH=str(tmp_path),
               SLURM_JOB_ID="999", SLURM_NTASKS="1", BB_REPO_ROOT=str(root),
               BB_SCALE_RUN_DIR=str(campaign), BB_PYTHON=str(commands / "fakepython"),
               BB_SCALE_AUTO_RESUBMIT="0" if mode == "manual" else "1",
               TEST_SUBMISSION=str(tmp_path / "submission"))
    result = subprocess.run(["bash", str(scaling.REPO / "run_bb_dem_sample_scaling.slurm")],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == expected_status, result.stdout + result.stderr
    submission = tmp_path / "submission"
    assert submission.exists() == submits
    if submits:
        argv = submission.read_text().splitlines()
        assert "--dependency=afterok:999" in argv
        assert "--ntasks-per-node=1" in argv and "--time=48:00:00" in argv
    if mode == "deadline":
        assert (root / "resdir_999/next_job_id.txt").read_text().strip() == "777"
    if mode == "cancel":
        assert (campaign / "STOP_CONTINUOUS").exists()
