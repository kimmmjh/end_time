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
