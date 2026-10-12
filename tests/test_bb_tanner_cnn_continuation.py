import json

import pytest
import torch

from scripts.continue_bb_tanner_cnn import build_command, resolve_checkpoint


def source(directory):
    directory.mkdir(parents=True)
    config = dict(architecture="bb_tanner_cnn", code="bb72", error_rate=.06, seed=7201060,
                  cnn_depth=2, cnn_width=64, noise_model="capacity", channel="depolarizing",
                  gradient_clip=1., weight_decay=1e-4, syndrome_loss_weight=1.,
                  logical_loss_weight=1., pauli_loss_weight=.1, x_error_rate=None, z_error_rate=None)
    history = {"train": [{"epoch": epoch} for epoch in range(100)], "final": {},
               "phases": [dict(batch_size=64, batches=128, eval_batches=64,
                               eval_every=5, final_eval_batches=1024)]}
    (directory / "history.json").write_text(json.dumps({"config": config, **history}))
    checkpoint = directory / "model.pt"
    torch.save(dict(format="bb_tanner_cnn_v1", config=config, epoch=99,
                    history=history, best_model_state_dict={}), checkpoint)
    return checkpoint


def test_find_archived_checkpoint_and_compute_additional_epochs(tmp_path):
    checkpoint = source(tmp_path / "results/bb/code_capacity/depolarizing/tanner_cnn/bb72/"
                        "resdir_58793761/outputs/date/model with spaces")
    found = resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060)
    assert found == checkpoint
    command = build_command(found, repo_root=tmp_path, code="bb72", p=.06, seed=7201060)
    assert "--epochs=300" in command
    assert f"--load_model={checkpoint}" in command  # Path remains a single argv element.
    assert "--bb_cnn_compare_resume" in command
    assert "--final_eval_batches=1024" in command
    with pytest.raises(ValueError, match="exceed"):
        build_command(found, repo_root=tmp_path, code="bb72", p=.06, seed=7201060, target_epochs=100)
    with pytest.raises(ValueError, match="Expected 400"):
        build_command(found, repo_root=tmp_path, code="bb72", p=.06, seed=7201060, expected_epochs=400)
    with pytest.raises(ValueError, match="configuration"):
        build_command(found, repo_root=tmp_path, code="bb144", p=.06, seed=7201060)


def test_missing_and_ambiguous_sources_fail_instead_of_starting_fresh(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060)
    original = source(tmp_path / "resdir_58793761/outputs/date/model")
    assert resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060) == original
    source(tmp_path / "results/bb/code_capacity/depolarizing/tanner_cnn/bb72/"
           "resdir_58793761/outputs/date/model")
    with pytest.raises(ValueError, match="found 2"):
        resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060)
    assert resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060, source_root=tmp_path) == original
    original.unlink()
    with pytest.raises(FileNotFoundError, match="Latest checkpoint"):
        resolve_checkpoint(tmp_path, 58793761, "bb72", .06, 7201060, source_root=tmp_path)


def test_find_mixed_code_continuation_archive(tmp_path):
    checkpoint = source(tmp_path / "results/bb/code_capacity/depolarizing/tanner_cnn/"
                        "continued_training/resdir_59622488/outputs/date/model")
    assert resolve_checkpoint(tmp_path, 59622488, "bb72", .06, 7201060) == checkpoint


def test_deep_recovery_requires_explicit_permission_and_exact_completed_epochs(tmp_path):
    checkpoint = source(tmp_path / "results/bb/code_capacity/depolarizing/tanner_cnn/400_epochs/"
                        "bb72/resdir_59122449/outputs/date/model")
    saved = torch.load(checkpoint, weights_only=False)
    saved["config"]["cnn_depth"] = 5
    saved["epoch"] = 113
    saved["history"]["train"] = [{"epoch": i} for i in range(114)]
    del saved["history"]["final"]
    torch.save(saved, checkpoint)
    checkpoint.with_name("history.json").write_text(json.dumps({"config": saved["config"], **saved["history"]}))
    found = resolve_checkpoint(tmp_path, 59122449, "bb72", .06, 7201060, depth=5)
    assert found == checkpoint
    kwargs = dict(repo_root=tmp_path, code="bb72", p=.06, seed=7201060, depth=5,
                  expected_epochs=114, target_epochs=400, learning_rate=1e-4)
    with pytest.raises(ValueError, match="allow-incomplete"):
        build_command(found, **kwargs)
    command = build_command(found, **kwargs, allow_incomplete=True)
    assert "--epochs=286" in command and "--bb_cnn_depth=5" in command
    assert "--lr=0.0001" in command and "--bb_cnn_compare_resume" in command
    with pytest.raises(ValueError, match="configuration"):
        build_command(found, **{**kwargs, "depth": 4}, allow_incomplete=True)
    with pytest.raises(ValueError, match="Expected 400"):
        build_command(found, **{**kwargs, "expected_epochs": 400}, allow_incomplete=True)
    saved["model_state_dict"] = {"weight": torch.tensor([float("nan")])}
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="Non-finite weights"):
        build_command(found, **kwargs, allow_incomplete=True)


def test_completed_deep_model_continues_to_800(tmp_path):
    checkpoint = source(tmp_path / "completed")
    saved = torch.load(checkpoint, weights_only=False)
    saved["config"]["cnn_depth"] = 4
    saved["epoch"] = 399
    saved["history"]["train"] = [{"epoch": i} for i in range(400)]
    torch.save(saved, checkpoint)
    command = build_command(checkpoint, repo_root=tmp_path, code="bb72", p=.06, seed=7201060,
                            depth=4, expected_epochs=400, target_epochs=800, learning_rate=1e-4)
    assert "--epochs=400" in command and "--bb_cnn_depth=4" in command
