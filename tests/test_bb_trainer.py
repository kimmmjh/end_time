import torch

from models import EquivariantNeuralBP4
from src._bb_loss import DegeneracyAwareBPLoss
from src._bb_trainer import BBNeuralBPTrainer
from src.bb_code import BBCodeSpec
from src.bb_data_generator import BBCodeCapacityGenerator


def _configuration():
    return {
        "architecture": "bb_neural_bp",
        "code": "bb72",
        "graph_fingerprint": "test-bb72-graph",
        "bp_iterations": 1,
        "bp_residual_hidden_dim": 4,
        "bp_parameter_sharing": "orbit",
        "bp_residual_scale": 2.0,
        "bp_max_relaxation_delta": 0.5,
        "bp_deep_supervision_weight": 0.0,
        "bb_syndrome_loss_weight": 1.0,
        "bb_logical_loss_weight": 1.0,
        "bb_pauli_loss_weight": 0.1,
        "bb_weight_decay": 1e-4,
        "channel": "depolarizing",
        "error_rate": 0.06,
        "x_error_rate": None,
        "z_error_rate": None,
    }


def _components(code):
    model = EquivariantNeuralBP4(
        code.hx,
        code.hz,
        edge_orbits=code.edge_orbit,
        iterations=1,
        residual_hidden_dim=4,
    )
    criterion = DegeneracyAwareBPLoss(
        hx=torch.tensor(code.hx),
        hz=torch.tensor(code.hz),
        logicals_x=torch.tensor(code.logicals_x),
        logicals_z=torch.tensor(code.logicals_z),
        deep_supervision_weight=0.0,
    )
    train_generator = BBCodeCapacityGenerator(code, 0.06, seed=3)
    eval_generator = BBCodeCapacityGenerator(code, 0.06, seed=4)
    return model, criterion, train_generator, eval_generator


def _trainer(tmp_path, *, load_model_path=None):
    code = BBCodeSpec.bb72()
    model, criterion, train_generator, eval_generator = _components(code)
    trainer = BBNeuralBPTrainer(
        model=model,
        code=code,
        train_generator=train_generator,
        eval_generator=eval_generator,
        criterion=criterion,
        device=torch.device("cpu"),
        epochs=1,
        batches=2,
        batch_size=2,
        eval_batches=1,
        eval_every=1,
        final_eval_batches=1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        output_directory=tmp_path,
        experiment_config=_configuration(),
        save_model=True,
        load_model_path=load_model_path,
    )
    trainer._plot_history = lambda: None
    return trainer


def test_one_epoch_checkpoint_and_resume_round_trip(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train()

    latest = tmp_path / "model.pt"
    best = tmp_path / "best_model.pt"
    assert latest.is_file()
    assert best.is_file()
    assert (tmp_path / "history.json").is_file()
    try:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(latest, map_location="cpu")
    assert checkpoint["epoch"] == 0
    assert checkpoint["best_epoch"] == 0

    resumed = _trainer(tmp_path / "resumed", load_model_path=latest)
    assert resumed.start_epoch == 1
    assert resumed.history["train_epoch"] == [0]
    assert resumed.optimizer.param_groups[0]["weight_decay"] == 1e-4
