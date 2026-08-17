import torch
from torch import nn

from src._trainer import Trainer


def _make_optimizer_and_scheduler(model, *, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=2,
    )
    return optimizer, scheduler


def test_resume_restores_history_epoch_and_optimizer_but_restarts_scheduler(tmp_path):
    original_model = nn.Linear(3, 2)
    original_optimizer, original_scheduler = _make_optimizer_and_scheduler(
        original_model, epochs=3
    )
    original_trainer = Trainer(
        model=original_model,
        loss_function=nn.CrossEntropyLoss(),
        optimizers=[original_optimizer],
        schedulers=[original_scheduler],
        batch_size=2,
        epochs=3,
        batches=2,
        save_directory=str(tmp_path / "original"),
    )

    # Populate AdamW moments so the test verifies more than weight loading.
    loss = original_model(torch.randn(2, 3)).square().mean()
    loss.backward()
    original_optimizer.step()
    original_scheduler.step()
    original_trainer.history = {
        "epoch": [1, 2, 3],
        "loss": [1.0, 0.8, 0.6],
        "accuracy": [0.4, 0.5, 0.7],
    }
    original_trainer.save_model(
        path=str(tmp_path), model_name="checkpoint", epoch=2
    )

    resumed_model = nn.Linear(3, 2)
    resumed_optimizer, resumed_scheduler = _make_optimizer_and_scheduler(
        resumed_model, epochs=2
    )
    fresh_lr = resumed_optimizer.param_groups[0]["lr"]
    resumed_trainer = Trainer(
        model=resumed_model,
        loss_function=nn.CrossEntropyLoss(),
        optimizers=[resumed_optimizer],
        schedulers=[resumed_scheduler],
        batch_size=2,
        epochs=2,
        batches=2,
        load_model_path=str(tmp_path / "checkpoint.pt"),
        save_directory=str(tmp_path / "resumed"),
    )

    assert resumed_trainer.start_epoch == 3
    assert resumed_trainer._num_epochs == 5
    assert resumed_trainer.history == original_trainer.history
    assert resumed_trainer.resume_epochs == [3]
    assert resumed_scheduler.last_epoch == 0
    assert resumed_optimizer.param_groups[0]["lr"] == fresh_lr

    original_state = next(iter(original_optimizer.state.values()))
    resumed_state = next(iter(resumed_optimizer.state.values()))
    torch.testing.assert_close(resumed_state["exp_avg"], original_state["exp_avg"])
    torch.testing.assert_close(
        resumed_state["exp_avg_sq"], original_state["exp_avg_sq"]
    )

    resumed_trainer.history["epoch"].append(4)
    resumed_trainer.history["loss"].append(0.5)
    resumed_trainer.history["accuracy"].append(0.75)
    resumed_trainer.save_plots(path=str(tmp_path / "resumed"))

    assert (tmp_path / "resumed" / "loss_curve.png").is_file()
    assert (tmp_path / "resumed" / "accuracy_curve.png").is_file()


def test_resume_upgrades_legacy_history_without_epoch_axis(tmp_path):
    model = nn.Linear(2, 2)
    optimizer, scheduler = _make_optimizer_and_scheduler(model, epochs=1)
    checkpoint = {
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "scheduler_states": [scheduler.state_dict()],
        "history": {"loss": [1.0, 0.5], "accuracy": [0.25, 0.75]},
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)

    resumed_model = nn.Linear(2, 2)
    resumed_optimizer, resumed_scheduler = _make_optimizer_and_scheduler(
        resumed_model, epochs=1
    )
    trainer = Trainer(
        model=resumed_model,
        loss_function=nn.CrossEntropyLoss(),
        optimizers=[resumed_optimizer],
        schedulers=[resumed_scheduler],
        batch_size=2,
        epochs=1,
        batches=2,
        load_model_path=str(checkpoint_path),
        save_directory=str(tmp_path / "legacy_resume"),
    )

    assert trainer.history["epoch"] == [1, 2]
    assert trainer.start_epoch == 2
    assert trainer._num_epochs == 3
