import copy
import json

import numpy as np
import pytest
import torch

from models._bb_tanner_cnn import BBTannerCNN
from src._bb_loss import DegeneracyAwareBPLoss
from src._bb_tanner_cnn_experiment import BBTannerCNNTrainer
from src.bb_code import BBCodeSpec
from src.bb_data_generator import BBCodeCapacityGenerator


def trainer(directory, *, load=None, depth=2, epochs=1):
    code = BBCodeSpec.bb72()
    model = BBTannerCNN(code, width=4, depth=depth)
    criterion = DegeneracyAwareBPLoss(
        **{key: torch.tensor(getattr(code, key)) for key in ("hx", "hz", "logicals_x", "logicals_z")},
        deep_supervision_weight=0.,
    )
    return BBTannerCNNTrainer(
        model=model, code=code, criterion=criterion, device=torch.device("cpu"),
        train_generator=BBCodeCapacityGenerator(code, .06, seed=4),
        eval_generator=BBCodeCapacityGenerator(code, .06, seed=5),
        output_directory=directory, config={"architecture": "bb_tanner_cnn", "depth": depth},
        epochs=epochs, batches=2, batch_size=4, eval_batches=1, eval_every=1,
        final_eval_batches=2, save_model=True, load_model_path=load,
    )


def test_train_paired_evaluation_saved_bank_and_latest_resume(tmp_path):
    torch.manual_seed(41)
    original = trainer(tmp_path / "first")
    before = copy.deepcopy(original.model.state_dict())
    original.train()
    assert any(not torch.equal(before[key], value) for key, value in original.model.state_dict().items())
    history = json.loads((original.directory / "history.json").read_text())
    final = history["final"]
    assert final["shots"] == 8
    assert final["osd"]["syndrome_convergence"] == 1
    assert final["harmed"] == 0  # Valid raw corrections are never replaced.
    with np.load(original.directory / "final_shots.npz", allow_pickle=False) as bank:
        assert bank["syndrome"].shape == (8, original.code.num_checks)
        assert bank["raw_success"].mean() == final["raw"]["logical_accuracy"]
        assert bank["osd_success"].mean() == final["osd"]["logical_accuracy"]
        assert np.count_nonzero(bank["osd_success"] & ~bank["raw_success"].astype(bool)) == final["rescued"]
        raw, osd = bank["raw_correction"], bank["osd_correction"]
        raw_x, raw_z = np.isin(raw, (1, 2)), np.isin(raw, (2, 3))
        valid = np.all(raw_x @ original.code.hz.T % 2 == bank["syndrome"][:, original.code.num_x_checks:], axis=1)
        valid &= np.all(raw_z @ original.code.hx.T % 2 == bank["syndrome"][:, :original.code.num_x_checks], axis=1)
        np.testing.assert_array_equal(raw[valid], osd[valid])
    latest = torch.load(original.directory / "model.pt", weights_only=False)
    selected = torch.load(original.directory / "best_model.pt", weights_only=False)
    assert selected["epoch"] == latest["best_epoch"] == 0
    continued = trainer(tmp_path / "resume", load=original.directory / "model.pt")
    assert continued.start_epoch == 1
    assert continued.train_generator.state_dict()["rng_state"] == latest["train_generator_state"]["rng_state"]
    assert continued.eval_generator.state_dict()["rng_state"] == latest["eval_generator_state"]["rng_state"]
    torch.testing.assert_close(torch.get_rng_state(), latest["torch_rng_state"])
    continued.train()
    assert [row["epoch"] for row in continued.history["train"]] == [0, 1]
    assert (continued.directory / "best_model.pt").is_file()
    with pytest.raises(ValueError, match="depth"):
        trainer(tmp_path / "incompatible", load=original.directory / "model.pt", depth=1)


def test_selection_uses_raw_cnn_not_osd_and_final_restores_selected_weights(tmp_path, monkeypatch):
    instance = trainer(tmp_path / "selection", epochs=2)
    real_evaluate = instance.evaluate
    calls = []
    snapshots = []

    def evaluate(batches, *, save_shots=False):
        row = real_evaluate(batches, save_shots=save_shots)
        calls.append(save_shots)
        if not save_shots:
            snapshots.append(copy.deepcopy(instance.model.state_dict()))
            row["raw"]["logical_accuracy"] = [.75, .5][len(snapshots) - 1]
            row["osd"]["logical_accuracy"] = [.75, 1.][len(snapshots) - 1]
        else:
            for key, value in instance.model.state_dict().items():
                torch.testing.assert_close(value, snapshots[0][key])
        return row

    monkeypatch.setattr(instance, "evaluate", evaluate)
    instance.train()
    assert calls == [False, False, True]
    assert instance.best_epoch == 0
    selected = torch.load(instance.directory / "best_model.pt", weights_only=False)
    assert selected["epoch"] == 0
    assert len(selected["history"]["train"]) == 1
    for key, value in instance.model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][key])
    # Resuming best_model.pt must retain that checkpoint if later validation ties it.
    resumed = trainer(tmp_path / "resume_best", load=instance.directory / "best_model.pt")
    monkeypatch.setattr(resumed, "evaluate", lambda *a, **kw: {
        "raw": {"logical_accuracy": .5, "logical_error_rate": .5, "syndrome_convergence": 1.},
        "osd": {"logical_accuracy": 1., "logical_error_rate": 0.},
        "osd_call_fraction": 0., "osd_minus_raw_gain": .5, "shots": 4,
    })
    resumed.train()
    retained = torch.load(resumed.directory / "best_model.pt", weights_only=False)
    assert retained["epoch"] == 0
