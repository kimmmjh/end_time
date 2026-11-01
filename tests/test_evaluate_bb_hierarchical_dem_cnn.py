from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from models._bb_dem_cnn import BBDemCNN
from scripts.evaluate_bb_hierarchical_dem_cnn import (
    DEFAULT_MANIFEST, aggregate, evaluate, load_weights, main, paired,
    parse_args, resolve_checkpoint, source_entry,
)
from src._bb_circuit_experiment import _graph_fingerprint
from src.bb_circuit_data import BBCircuitGenerator
from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION


@pytest.fixture
def checkpoint(tmp_path):
    generator = BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003,
                                   circuit_noise_model="standard", batch_size=2, seed=37)
    model = BBDemCNN(generator.graph, width=4, depth=1)
    config = dict(architecture="bb_dem_cnn", noise_model="circuit", code="bb72",
                  rounds=1, width=4, depth=1, circuit_schema_version=CIRCUIT_SCHEMA_VERSION,
                  noise_profile=asdict(generator.noise_profile),
                  graph_fingerprint=_graph_fingerprint(generator.graph))
    saved = dict(format="bb_dem_cnn_v1", config=config, epoch=9, best_epoch=4,
                 model_state_dict=model.state_dict(),
                 best_model_state_dict={k: v.clone() for k, v in model.state_dict().items()})
    saved["model_state_dict"]["head.2.bias"] = torch.tensor([1.])
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    return path, saved


def test_best_selection_does_not_use_latest_weights(checkpoint):
    path, saved = checkpoint
    _, best, info = load_weights(path, "best")
    _, latest, latest_info = load_weights(path, "latest")
    assert best["head.2.bias"].item() == 0
    assert latest["head.2.bias"].item() == 1
    assert info["selected_epoch"] == 5 and latest_info["selected_epoch"] == 10
    assert len(info["sha256"]) == 64
    with pytest.raises(ValueError, match="manifest"):
        load_weights(path, "best", {"depth": 12})
    saved["best_model_state_dict"] = None
    torch.save(saved, path)
    with pytest.raises(ValueError, match="best weights"):
        load_weights(path, "best")
    saved["epoch"] = saved["best_epoch"]
    torch.save(saved, path)
    assert load_weights(path, "best")[2]["selected_epoch"] == 5


def test_source_resolution_handles_server_imported_and_embedded_best(tmp_path):
    entry = dict(source_run="resdir_123/outputs/date/run", imported_run="results/resdir_123/outputs/date/run")
    server, imported = tmp_path / entry["source_run"], tmp_path / entry["imported_run"]
    imported.mkdir(parents=True)
    (imported / "best_model.pt").touch()
    assert resolve_checkpoint(entry, tmp_path, tmp_path, "best")[0] == imported / "best_model.pt"
    server.mkdir(parents=True)
    (server / "model.pt").touch()
    assert resolve_checkpoint(entry, tmp_path, tmp_path, "best")[0] == server / "model.pt"
    (server / "best_model.pt").touch()
    assert resolve_checkpoint(entry, tmp_path, tmp_path, "best")[0] == server / "best_model.pt"
    assert resolve_checkpoint(entry, tmp_path, tmp_path, "latest")[0] == server / "model.pt"


def test_manifest_covers_only_completed_campaign():
    entries = json.loads(DEFAULT_MANIFEST.read_text())["experiments"]
    assert len(entries) == 24
    assert {e["task"] for e in entries} == {0, 1, 2, 3, 4, 6}
    for task in (0, 1, 2, 3, 4, 6):
        for index, rate in enumerate((.001, .002, .004, .006)):
            e = source_entry(DEFAULT_MANIFEST, task, index)
            assert e["expected_config"]["noise_profile"]["base_error_rate"] == rate
            assert e["training_epochs"] == 800
    with pytest.raises(ValueError, match="manifest experiment"):
        source_entry(DEFAULT_MANIFEST, 5, 0)


def test_missing_weights_preview_and_strict_preflight(tmp_path):
    argv = ["--root", str(tmp_path), "--task", "0", "--index", "0", "--dry-run"]
    assert not main(argv)["checkpoint_exists"]
    with pytest.raises(FileNotFoundError, match="Metadata/history"):
        main(argv + ["--require-checkpoint"])


def test_paired_metrics_and_wilson_intervals():
    before = np.array([True, False, False, True])
    after = np.array([False, True, True, True])
    result = paired(before, after)
    assert result["rescued"] == 2 and result["harmed"] == 1 and result["accuracy_gain"] == .25
    result = aggregate(before, np.array([True, True, False, True]), before)
    assert result["block_failure_rate"] == .5
    assert result["syndrome_failure_rate"] == .25 and result["unflagged_failure_rate"] == .25
    assert result["block_failure_wilson95"][0] < .5 < result["block_failure_wilson95"][1]


def test_real_evaluation_is_paired_evaluation_only_and_does_not_overwrite(checkpoint, tmp_path):
    path, saved = checkpoint
    original = path.read_bytes()
    output = tmp_path / "evaluation"
    args = parse_args(["--checkpoint", str(path), "--output", str(output), "--device", "cpu",
                       "--shots", "3", "--batch-size", "2", "--compare-direct-osd", "--save-corrections"])
    result = evaluate(args, path)
    assert result["status"] == "complete" and result["shots"] == 3
    assert all(m["shots"] == 3 for m in result["metrics"].values())
    for name in ("hierarchical", "direct_osd"):
        assert result["metrics"][name]["syndrome_failure_rate"] == 0
    assert path.read_bytes() == original
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"]["selected_epoch"] == 5
    banks = sorted((output / "shots").glob("*.npz"))
    assert len(banks) == 2
    with np.load(banks[-1]) as bank:
        assert bank["detectors"].shape[0] == 1
        assert "raw_correction" in bank and "hierarchical_correction" in bank
    with pytest.raises(FileExistsError):
        evaluate(args, path)
    saved["config"]["graph_fingerprint"] = "invalid"
    torch.save(saved, path)
    args.output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="does not match"):
        evaluate(args, path)
    assert not args.output.exists()


def test_slurm_syntax_and_all_dry_run_tasks():
    import os
    import sys
    root = Path(__file__).resolve().parents[1]
    launcher = root / "run_bb_dem_hierarchical_eval.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    env = {**os.environ, "BB_DRY_RUN": "1", "BB_PYTHON": sys.executable, "BB_REPO_ROOT": str(root)}
    env.pop("SLURM_ARRAY_TASK_ID", None)
    output = subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True, text=True).stdout
    assert output.count('"checkpoint_exists"') == 24
    assert output.count('"shots": 8192') == 24


@pytest.mark.parametrize("options", [["--shots", "0"], ["--confidence", "nan"], ["--batch-size", "0"]])
def test_cli_rejects_bad_budgets(options):
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--dry-run", *options])
