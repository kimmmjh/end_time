import json

import pytest
import torch

from scripts.resume_bb_dem_cnn import find_source, plan_resume


def fixture_checkpoint(tmp_path, completed=500, final=False):
    source = tmp_path / "resdir_123"
    run = source / "outputs/2026-10-28/run"
    run.mkdir(parents=True)
    (source / "job_metadata.txt").write_text("Job ID: 123\nArray job/task: 60310800/5\n")
    c = dict(architecture="bb_dem_cnn",code="bb144",depth=8,width=64,rounds=12,
             noise_model="circuit",noise_profile=dict(name="standard",base_error_rate=.001),seed=14402001,
             loss_weights=dict(syndrome_weight=1.,logical_weight=1.,mechanism_weight=.1))
    phase = dict(batch_size=4,batches=128,eval_batches=256,eval_every=5,final_eval_batches=2048,
                 weight_decay=.0001,gradient_clip=1.,edge_chunk_size=4096,osd_eval_shots=0,gradient_checkpoint=True)
    history = dict(train=[dict(epoch=i) for i in range(completed)],phases=[phase])
    if final:
        history["final"] = dict(shots=8192)
    saved = dict(format="bb_dem_cnn_v1",config=c,history=history,epoch=completed-1,
                 best_epoch=399,optimizer_state_dict=dict(param_groups=[dict(lr=.000091)]))
    # The .pt is authoritative even if the separate JSON was written later.
    (run / "history.json").write_text(json.dumps(dict(config=c,train=[{}]*600)))
    torch.save(saved,run / "model.pt")
    return source,run,saved


def test_resume_uses_checkpoint_epoch_and_saved_lr(tmp_path):
    source,run,_ = fixture_checkpoint(tmp_path)
    assert find_source(tmp_path,60310800,8) == source
    plan = plan_resume(tmp_path,source,8,.001,800)
    args = dict(a[2:].split("=",1) for a in plan["command"] if a.startswith("--") and "=" in a)
    assert plan["additional_epochs"] == 300
    assert args["epochs"] == "300" and float(args["lr"]) == .000091
    assert args["load_model"] == str(run / "model.pt")
    assert args["bb_osd_eval_shots"] == "0"
    assert args["bb_cnn_depth"] == "8" and args["batch_size"] == "4"


def test_completed_training_without_final_does_not_add_epochs(tmp_path):
    source,_,_ = fixture_checkpoint(tmp_path,completed=800)
    with pytest.raises(ValueError,match="evaluation-only"):
        plan_resume(tmp_path,source,8,.001,800)


def test_already_completed_is_skipped(tmp_path):
    source,_,_ = fixture_checkpoint(tmp_path,completed=800,final=True)
    plan = plan_resume(tmp_path,source,8,.001,800)
    assert plan["status"] == "already_complete" and not plan["command"]


def test_missing_latest_checkpoint_and_mismatched_checkpoint_are_rejected(tmp_path):
    source,run,saved = fixture_checkpoint(tmp_path)
    saved["config"]["depth"] = 12
    torch.save(saved,run / "model.pt")
    with pytest.raises(ValueError,match="configuration"):
        plan_resume(tmp_path,source,8,.001,800)
    (run / "model.pt").unlink()
    with pytest.raises(ValueError,match="latest model.pt"):
        plan_resume(tmp_path,source,8,.001,800)
