import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_bb_dem_followup import DEFAULT_MANIFEST, main, workers

ROOT = Path(__file__).resolve().parents[1]


def test_partition_covers_all_32_sources_once_and_pairs_weights():
    data = json.loads(DEFAULT_MANIFEST.read_text())
    entries = data["experiments"]
    assert len(entries) == len({(e["task"], e["index"]) for e in entries}) == 32
    train, evaluate = workers(DEFAULT_MANIFEST,"train"), workers(DEFAULT_MANIFEST,"evaluate")
    assert len(train) == 21 and len(evaluate) == 22
    assert {w for e,w in train} == {"latest"}
    for a,b in zip(evaluate[::2],evaluate[1::2]):
        assert a[0] == b[0] and (a[1],b[1]) == ("best","latest")
    assert {e["source_run"] for e,w in train}.isdisjoint({e["source_run"] for e,w in evaluate})
    for e in entries:
        assert e["training_epochs"] == 800
        if e["task"] in (5,7):
            assert e["source_run"].startswith("resdir_60428756/" if e["task"] == 5 else "resdir_60636624/")
    selection = {(e["expected_config"]["code"],e["expected_config"]["depth"],e["index"]):e["action"] for e in entries}
    assert selection["bb72",12,3] == selection["bb144",12,2] == "train"
    assert selection["bb144",8,2] == selection["bb144",4,1] == "evaluate"
    assert all(selection["bb144",d,3] == "evaluate" for d in (4,6,8,12))


def test_missing_weights_preview_and_execution_guard(tmp_path):
    argv = ["--root",str(tmp_path),"--mode","train","--worker","0","--dry-run"]
    plan = main(argv)
    assert not plan["checkpoint_exists"] and plan["additional_epochs"] == 200
    assert "command" not in plan
    with pytest.raises(FileNotFoundError):
        main(argv + ["--require-checkpoint"])
    assert not list(tmp_path.iterdir())


def test_evaluation_pair_uses_same_shots_and_seed(tmp_path):
    plans = [main(["--root",str(tmp_path),"--mode","evaluate","--worker",str(i),"--dry-run"]) for i in (0,1)]
    for plan,weight in zip(plans,("best","latest")):
        command = plan["command"]
        assert command[command.index("--weights")+1] == weight
        assert command[command.index("--shots")+1] == "8192"
        assert command[command.index("--seed")+1] == "20261101"
    assert plans[0]["source"] == plans[1]["source"]


@pytest.mark.parametrize("mode,name,count", [
    ("train","run_bb_dem_continue.slurm",21),
    ("evaluate","run_bb_dem_hierarchical_plateau.slurm",22),
])
def test_launchers_preview_all_workers_and_tail_without_gpu(mode,name,count):
    for path in (ROOT/name,ROOT/"scripts/run_bb_dem_followup.sh"):
        subprocess.run(["bash","-n",str(path)],check=True)
    env = {**os.environ,"BB_DRY_RUN":"1","BB_REPO_ROOT":str(ROOT),"BB_PYTHON":sys.executable}
    env.pop("SLURM_ARRAY_TASK_ID",None)
    result = subprocess.run(["bash",str(ROOT/name)],env=env,check=True,text=True,capture_output=True)
    assert result.stdout.count('"checkpoint_exists"') == count
    env["SLURM_ARRAY_TASK_ID"] = "5"
    result = subprocess.run(["bash",str(ROOT/name)],env=env,check=True,text=True,capture_output=True)
    assert result.stdout.count('"checkpoint_exists"') == count-20
    env["SLURM_ARRAY_TASK_ID"] = "6"
    assert subprocess.run(["bash",str(ROOT/name)],env=env,capture_output=True).returncode != 0
