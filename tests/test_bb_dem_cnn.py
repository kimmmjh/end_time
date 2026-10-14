import copy
import json
import sys

import numpy as np
import pytest
import torch

from models._bb_dem_cnn import BBDemCNN, DEMConvolution
from src._bb_circuit_loss import CircuitDegeneracyAwareLoss
from src._bb_circuit_metrics import score_corrections
from src._bb_dem_cnn_experiment import BBDemCNNTrainer
from src.bb_circuit_data import BBCircuitGenerator


@pytest.fixture(scope="module")
def generator():
    return BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003,
                              circuit_noise_model="standard", batch_size=2, seed=71)


def test_sparse_convolution_matches_explicit_role_matrices_and_chunked_gradients():
    torch.manual_seed(8)
    conv = DEMConvolution(3, 2, edge_chunk_size=2).double()
    source, target, orbit = [torch.tensor(v) for v in ([0, 1, 2, 0], [0, 0, 1, 1], [0, 1, 0, 1])]
    degree = torch.tensor([2., 2., 0.], dtype=torch.double)
    features = torch.randn(2, 3, 3, dtype=torch.double, requires_grad=True)
    # Independent dense reference: one explicit W_out diag(g_r) W_in per edge.
    rows = []
    for node in range(3):
        value = features.new_zeros(2, 3)
        for e in range(4):
            if target[e] == node:
                weight = conv.output.weight @ torch.diag(conv.gates.weight[orbit[e]]) @ conv.input.weight
                value = value + features[:, source[e]] @ weight.T
        rows.append(value / degree[node].clamp_min(1).sqrt() + conv.output.bias)
    expected = torch.stack(rows, dim=1)
    actual = conv(features, source, target, orbit, degree)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (features, *conv.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (features, *conv.parameters()))
    for first, second in zip(actual_grad, expected_grad):
        torch.testing.assert_close(first, second)


def test_real_dem_translation_equivariance_and_joint_mechanisms(generator):
    graph = generator.graph
    coordinates = graph.detector_coordinates
    lookup = {tuple(row): i for i, row in enumerate(coordinates)}
    shifted = coordinates.copy()
    shifted[:, 1] = (shifted[:, 1] + 1) % graph.ell
    shifted[:, 2] = (shifted[:, 2] + 2) % graph.m
    detector_map = np.array([lookup[tuple(row)] for row in shifted])
    columns = graph.check_matrix.tocsc()
    supports = [tuple(columns.indices[columns.indptr[i]:columns.indptr[i + 1]])
                for i in range(graph.num_mechanisms)]
    lookup_m = {support: i for i, support in enumerate(supports)}
    assert len(lookup_m) == graph.num_mechanisms
    mechanism_map = np.array([lookup_m[tuple(sorted(detector_map[list(s)]))] for s in supports])
    # Joint hyperedges really contain both detector sectors, not independent X/Z graphs.
    assert any(len(set(coordinates[list(s), 0])) == 2 and len(s) > 2 for s in supports)
    np.testing.assert_allclose(graph.priors[mechanism_map], graph.priors)
    torch.manual_seed(24)
    model = BBDemCNN(graph, width=4, depth=2).eval()
    # Amplify the learned output so the check cannot pass on the constant prior alone.
    with torch.no_grad():
        model.head[-1].weight.normal_(std=.4)
    detectors = torch.randint(2, (2, graph.num_detectors)).float()
    moved = torch.empty_like(detectors)
    moved[:, detector_map] = detectors
    expected = model(detectors)
    actual = model(moved)
    torch.testing.assert_close(actual[:, mechanism_map], expected, atol=2e-6, rtol=2e-6)


def test_depth_receptive_field_and_checkpointed_training(generator):
    graph = generator.graph
    adjacent = torch.tensor(graph.check_matrix[:, 0].toarray().ravel()).bool()
    for depth in (1, 2):
        torch.manual_seed(12)
        model = BBDemCNN(graph, width=4, depth=depth, edge_chunk_size=2048)
        detectors = torch.zeros(1, graph.num_detectors, requires_grad=True)
        model(detectors)[0, 0].backward()
        far_gradient = detectors.grad[0, ~adjacent].abs().sum().item()
        assert (far_gradient == 0) if depth == 1 else (far_gradient > 1e-12)
    batch = generator.sample_dem()
    plain = BBDemCNN(graph, width=4, depth=2, gradient_checkpoint=False)
    checkpointed = BBDemCNN(graph, width=4, depth=2, gradient_checkpoint=True)
    checkpointed.load_state_dict(plain.state_dict())
    criterion = CircuitDegeneracyAwareLoss(check_matrix=graph.check_matrix,
                                         observable_matrix=graph.observable_matrix)
    assert score_corrections(batch.mechanisms.numpy(), detectors=batch.detectors.numpy(),
                             observables=batch.observables.numpy(), check_matrix=graph.check_matrix,
                             observable_matrix=graph.observable_matrix).success.all()
    outputs = []
    for model in (plain, checkpointed):
        outputs.append(model(batch.detectors))
        losses = criterion(outputs[-1], batch.detectors, batch.mechanisms)
        assert torch.isfinite(losses.total)
        losses.total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.first.gates.weight.grad.abs().sum() > 0
    torch.testing.assert_close(*outputs)
    for first, second in zip(plain.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(first.grad, second.grad)


def make_trainer(directory, generator, *, load=None, depth=2, epochs=1):
    graph = generator.graph
    train = BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003, circuit_noise_model="standard",
                              batch_size=2, seed=9 if load is None else 19, graph=graph)
    evaluation = BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003, circuit_noise_model="standard",
                                   batch_size=2, seed=10 if load is None else 20, graph=graph)
    return BBDemCNNTrainer(
        model=BBDemCNN(graph, width=4, depth=depth), train_generator=train, eval_generator=evaluation,
        criterion=CircuitDegeneracyAwareLoss(check_matrix=graph.check_matrix,
                                            observable_matrix=graph.observable_matrix),
        device=torch.device("cpu"), output_directory=directory,
        config={"architecture": "bb_dem_cnn", "depth": depth},
        epochs=epochs, batches=2, eval_batches=1, eval_every=1, final_eval_batches=2,
        osd_eval_shots=3, save_model=True, load_model_path=load)


def test_training_saved_circuit_bank_paired_osd_and_resume(tmp_path, generator, monkeypatch):
    trainer = make_trainer(tmp_path / "first", generator)
    before = copy.deepcopy(trainer.model.state_dict())
    # Evaluation must never obtain latent DEM labels.
    monkeypatch.setattr(trainer.eval_generator, "sample_dem",
                        lambda *a, **kw: pytest.fail("Evaluation used DEM labels"))
    trainer.train()
    assert any(not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items())
    history = json.loads((trainer.directory / "history.json").read_text())
    final = history["final"]
    assert final["shots"] == 4 and final["osd_shots"] == 3
    assert final["osd"]["syndrome_convergence"] == 1 and final["harmed"] == 0
    with np.load(trainer.directory / "final_shots.npz") as bank:
        raw = trainer._score(bank["raw_correction"], bank["detectors"], bank["observables"])
        np.testing.assert_array_equal(raw.success, bank["raw_success"])
        assert raw.accuracy == final["raw"]["logical_accuracy"]
        idx = bank["osd_indices"]
        repaired = trainer._score(bank["osd_correction"], bank["detectors"][idx], bank["observables"][idx])
        assert repaired.syndrome_converged.all()
        np.testing.assert_array_equal(repaired.success, bank["osd_success"])
        valid = raw.syndrome_converged[idx]
        np.testing.assert_array_equal(bank["osd_correction"][valid], bank["raw_correction"][idx][valid])
        assert final["paired_raw_accuracy"] == raw.success[idx].mean()
    latest = torch.load(trainer.directory / "model.pt", weights_only=False)
    resumed = make_trainer(tmp_path / "resumed", generator, load=trainer.directory / "model.pt")
    assert resumed.start_epoch == 1
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, latest["model_state_dict"][key])
    for key, value in resumed.optimizer.state_dict()["state"].items():
        torch.testing.assert_close(value["exp_avg"], latest["optimizer_state_dict"]["state"][key]["exp_avg"])
    resumed.train()
    assert [row["epoch"] for row in resumed.history["train"]] == [0, 1]
    with pytest.raises(ValueError, match="depth"):
        make_trainer(tmp_path / "bad", generator, load=trainer.directory / "model.pt", depth=1)


def test_best_selection_uses_raw_validation_and_never_final(tmp_path, generator, monkeypatch):
    trainer = make_trainer(tmp_path, generator, epochs=2)
    real_evaluate, snapshots = trainer.evaluate, []

    def evaluate(batches, *, save_shots=False):
        row = real_evaluate(batches, save_shots=save_shots)
        if save_shots:
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, snapshots[0][key])
        else:
            snapshots.append(copy.deepcopy(trainer.model.state_dict()))
            row["raw"]["logical_accuracy"] = [.75, .5][len(snapshots) - 1]
            row["osd"]["logical_accuracy"] = [.75, 1.][len(snapshots) - 1]
        return row

    monkeypatch.setattr(trainer, "evaluate", evaluate)
    trainer.train()
    assert trainer.best_epoch == 0
    best = torch.load(trainer.directory / "best_model.pt", weights_only=False)
    assert best["epoch"] == 0 and len(best["history"]["train"]) == 1
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][key])


@pytest.mark.parametrize("architecture", ["bb_dem_cnn", "bb_tanner_cnn"])
def test_cli_routes_circuit_cnn_with_noise_profile(monkeypatch, architecture):
    import main
    received = []
    monkeypatch.setattr(main, "run_bb_dem_cnn_experiment", received.append)
    monkeypatch.setattr(sys, "argv", ["main.py", "--code=bb72", f"--architecture={architecture}",
                                     "--noise_model=circuit", "--bb_circuit_noise_model=standard",
                                     "--p=.003", "--amp_dtype=none"])
    main.main()
    assert len(received) == 1
    assert received[0].rounds == 6
    assert received[0].measurement_error_rate == .003


@pytest.mark.parametrize("extra", ["--noise_model=capacity", "--bb_cnn_width=1", "--bp_relay_legs=2",
                                  "--bb_osd_eval_shots=-1", "--bb_cnn_compare_resume", "--amp_dtype=fp16"])
def test_cli_rejects_incompatible_dem_cnn_options(monkeypatch, extra):
    import main
    monkeypatch.setattr(sys, "argv", ["main.py", "--code=bb72", "--architecture=bb_dem_cnn",
                                     "--noise_model=circuit", "--amp_dtype=none", extra])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_invalid_inputs_and_llr_sign(generator):
    model = BBDemCNN(generator.graph, width=4, depth=1)
    with pytest.raises(ValueError, match="binary"):
        model(torch.full((1, generator.num_detectors), .5))
    with pytest.raises(ValueError, match="Expected detectors"):
        model(torch.zeros(1, 3))
    llr = torch.tensor([[-2., 0., 2.]])
    np.testing.assert_array_equal(model.hard_decision(llr), [[1, 0, 0]])
    torch.testing.assert_close(model.probabilities(llr), torch.sigmoid(-llr))
