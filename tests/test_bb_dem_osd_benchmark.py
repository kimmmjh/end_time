from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from models._bb_dem_cnn import BBDemCNN
from scripts import benchmark_bb_dem_osd as bench
from scripts.run_bb_dem_sample_scaling import worker_lock
from src._bb_circuit_experiment import _graph_fingerprint
from src.bb_circuit_data import BBCircuitGenerator
from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION


def test_source_uses_extension_and_refuses_active_training(tmp_path):
    entry = {"name":"bb144_d6_p0.002"}
    folder = tmp_path / entry["name"]
    base = folder / "training/model.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base")
    args = bench.parse_args(["--campaign-root", str(tmp_path), "--output", str(tmp_path/"output")])
    args.output.mkdir()
    assert bench.source_checkpoint(args, entry) == base
    extension = folder / "continuous/training/model.pt"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"extension")
    with worker_lock(folder):
        with pytest.raises(RuntimeError, match="Worker lock already exists"):
            bench.snapshot_checkpoint(args, entry)
        assert not (args.output/"checkpoint.pt").exists()
    path, info = bench.snapshot_checkpoint(args, entry)
    assert path.read_bytes() == b"extension"
    assert extension.read_bytes() == b"extension"
    assert info["original_path"] == str(extension)
    assert not (folder/"worker.lock.d").exists()


def test_cpu_smoke_with_real_circuit_and_frozen_weights(tmp_path):
    gen = BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003,
                            circuit_noise_model="standard", batch_size=2, seed=37)
    model = BBDemCNN(gen.graph, width=4, depth=1)
    config = dict(architecture="bb_dem_cnn", noise_model="circuit", code="bb72",
                  rounds=1, width=4, depth=1, circuit_schema_version=CIRCUIT_SCHEMA_VERSION,
                  noise_profile=asdict(gen.noise_profile), graph_fingerprint=_graph_fingerprint(gen.graph))
    path = tmp_path/"model.pt"
    torch.save(dict(format="bb_dem_cnn_v1", config=config, epoch=9, best_epoch=9,
                    model_state_dict=model.state_dict(), best_model_state_dict=model.state_dict()), path)
    before = path.read_bytes()
    output = tmp_path/"check"
    result = bench.main(["--checkpoint", str(path), "--output", str(output), "--device", "cpu",
                         "--shots", "3", "--batch-size", "2", "--repeats", "2"])
    assert result["status"] == "complete" and result["shots"] == 3
    assert result["cuda_tested"] is False
    assert result["correction_mismatch_shots"] == {"reference_cpu":0, "packed_cpu":0}
    assert result["metrics"]["reference_cpu"] == result["metrics"]["packed_cpu"]
    assert path.read_bytes() == before == (output/"checkpoint.pt").read_bytes()
    metadata = json.loads((output/"metadata.json").read_text())
    assert metadata["confidence"] == .9 and metadata["source"]["selected_epoch"] == 10
    with pytest.raises(FileExistsError):
        bench.main(["--checkpoint", str(path), "--output", str(output)])


def test_benchmark_launcher_previews_two_workers_and_never_submits(tmp_path):
    launcher = bench.REPO/"run_bb_dem_osd_benchmark.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith("BB_")}
    env.update(BB_DRY_RUN="1", BB_PYTHON=sys.executable, BB_SCALE_RUN_DIR=str(tmp_path))
    output = subprocess.run(["bash", str(launcher)], env=env, check=True,
                            capture_output=True, text=True).stdout
    assert output.count('"training": false') == 2
    assert '"experiment": "bb72_d8_p0.002"' in output
    assert '"experiment": "bb144_d6_p0.002"' in output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option,value", [("--confidence","nan"),("--shots","0"),("--repeats","0")])
def test_bad_benchmark_arguments(option, value):
    with pytest.raises(SystemExit):
        bench.parse_args(["--output","unused",option,value])
