"""Relay memory, per-shot candidate selection, gradients and paired scoring."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models._neural_relay_bp2 import NeuralRelayBP2
from src._bb_circuit_loss import CircuitDegeneracyAwareLoss
from src._bb_circuit_trainer import BBCircuitTrainer


def graph(check=((1, 1),), priors=(5.0, 1.0)):
    matrix = np.asarray(check)
    rows, columns = np.nonzero(matrix)
    return SimpleNamespace(
        num_detectors=matrix.shape[0], num_mechanisms=matrix.shape[1],
        num_observables=1, edge_detector=rows, edge_mechanism=columns,
        edge_orbit=np.zeros(rows.size, dtype=np.int64), num_orbits=1,
        prior_log_odds=np.asarray(priors), check_matrix=sp.csr_matrix(matrix),
        observable_matrix=sp.csr_matrix(np.ones((1, matrix.shape[1]))),
    )


def model(**kwargs):
    return NeuralRelayBP2(graph(), hidden_dim=4, orbit_embedding_dim=2, **kwargs)


def test_memory_bias_applied_once_and_extrinsic_excluded_before_clipping():
    decoder = NeuralRelayBP2(graph(((1,), (1,)), (5.0,)), iterations=1)
    decoder._check_update = lambda variable, syndrome: torch.full_like(variable, 20)
    variable, check, posterior = decoder._relay_iteration(
        torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[9.]]),
        torch.zeros(1, 2), torch.tensor([[.25]]), False,
    )
    # Bias = 5 + .25*(9-5) = 6, once per variable. Incoming sum = 40.
    assert torch.equal(posterior, torch.tensor([[30.]]))
    assert torch.equal(variable, torch.tensor([[26., 26.]]))
    assert torch.equal(check, torch.tensor([[20., 20.]]))


def test_zero_initialized_neural_and_non_neural_relay_are_identical():
    decoder = model(iterations=3, relay_legs=3, relay_solutions=2).eval()
    syndrome = torch.tensor([[0.], [1.], [1.]])
    memory = decoder.sample_memory(3, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        assert torch.equal(
            decoder(syndrome, neural=True, memory=memory),
            decoder(syndrome, neural=False, memory=memory),
        )
    neural = decoder.decode(syndrome, memory=memory)
    vanilla = decoder.decode(syndrome, neural=False, memory=memory)
    for field in ("correction", "posterior", "converged", "iterations", "legs", "solutions"):
        assert torch.equal(getattr(neural, field), getattr(vanilla, field))


def test_relay_carries_marginals_resets_edges_and_keeps_cheapest_valid_candidate():
    decoder = model(iterations=1, relay_legs=3, relay_solutions=3)
    outputs = [torch.tensor([[-2., 10.]]), torch.tensor([[.1, -20.]]), torch.ones(1, 2)]
    seen = []

    def iteration(variable, check, posterior, syndrome, memory, neural):
        seen.append(posterior.clone())
        assert torch.equal(variable, torch.tensor([[5., 1.]]))
        assert torch.count_nonzero(check) == 0
        return variable, check, outputs[len(seen) - 1]

    decoder._relay_iteration = iteration
    result = decoder.decode(torch.ones(1, 1))
    assert torch.equal(seen[1], outputs[0])
    assert torch.equal(seen[2], outputs[1])
    # Original-prior costs are 5 for 10 and 1 for 01; the last leg is invalid.
    assert torch.equal(result.correction, torch.tensor([[0, 1]], dtype=torch.uint8))
    assert torch.equal(result.posterior, outputs[1])
    assert result.converged.item()
    assert result.solutions.item() == 2
    assert result.iterations.item() == result.legs.item() == 3


def test_decode_stops_individual_shots_at_first_valid_iteration():
    decoder = model(iterations=4, relay_legs=3, relay_solutions=1)
    sizes = []

    def iteration(variable, check, posterior, syndrome, memory, neural):
        sizes.append(posterior.shape[0])
        output = torch.ones_like(posterior)
        output[0, 0] = -1
        return variable, check, output

    decoder._relay_iteration = iteration
    result = decoder.decode(torch.ones(2, 1))
    assert sizes == [2, 1]
    assert result.iterations.tolist() == [1, 2]
    assert result.legs.tolist() == [1, 1]
    assert result.converged.all()


def test_no_solution_remains_flagged_and_exhausts_budget():
    decoder = model(iterations=2, relay_legs=3)
    decoder._relay_iteration = lambda v, c, p, s, m, n: (v, c, torch.ones_like(p))
    result = decoder.decode(torch.ones(2, 1))
    assert not result.converged.any()
    assert result.iterations.tolist() == [6, 6]
    assert result.legs.tolist() == [3, 3]
    assert not result.correction.any()


def test_training_crosses_leg_boundaries_and_checkpoint_gradients_match():
    torch.manual_seed(7)
    direct = model(iterations=2, relay_legs=3, gradient_checkpoint=False)
    with torch.no_grad():
        direct.residual_network[-1].weight.normal_(std=.05)
    checkpointed = copy.deepcopy(direct)
    checkpointed.gradient_checkpoint = True
    syndrome = torch.tensor([[1.], [0.]])
    memory = direct.sample_memory(2).requires_grad_()
    criterion = CircuitDegeneracyAwareLoss(
        check_matrix=graph().check_matrix, observable_matrix=graph().observable_matrix,
    )
    gradients = []
    for decoder in (direct, checkpointed):
        posterior, history = decoder(syndrome, return_all=True, memory=memory)
        assert history.shape == (2, 6, 2)
        loss = criterion(posterior, syndrome, torch.tensor([[1., 0.], [0., 0.]]), history).total
        loss.backward()
        grads = [p.grad.clone() for p in decoder.parameters()]
        assert all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        gradients.append(grads)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right)
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    # Terminal output has a derivative through the prior leg's memory, not a detach.
    terminal_memory = direct.sample_memory(2).requires_grad_()
    terminal = direct(syndrome, memory=terminal_memory)
    derivative = torch.autograd.grad(terminal.sum(), terminal_memory)[0]
    assert derivative[1].abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"relay_legs": 0}, {"relay_solutions": 5},
    {"relay_memory_min": .7, "relay_memory_max": .2},
    {"relay_memory_strength": float("nan")}, {"relay_memory_min": -1.},
])
def test_invalid_relay_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)


def test_paired_evaluation_uses_same_memory_and_isolates_training_rng():
    decoder = model(iterations=2, relay_legs=3)
    seen_memory = []
    original_decode = decoder.decode

    def record_decode(*args, **kwargs):
        seen_memory.append(kwargs["memory"].clone())
        return original_decode(*args, **kwargs)

    decoder.decode = record_decode
    trainer = BBCircuitTrainer.__new__(BBCircuitTrainer)
    trainer.model = decoder
    trainer.device = torch.device("cpu")
    trainer.experiment_config = {"seed": 91}
    trainer.osd_eval_shots = 0
    trainer.bp_reference_iterations = 5
    trainer._osd = None
    trainer.eval_generator = SimpleNamespace(
        graph=graph(), sample_circuit=lambda **kwargs: SimpleNamespace(
            detectors=torch.tensor([[1.], [0.]]), observables=torch.tensor([[1.], [0.]])
        ),
    )
    rng_before = torch.get_rng_state().clone()
    evaluation = trainer.evaluate(2)
    assert len(seen_memory) == 4
    assert torch.equal(seen_memory[0], seen_memory[1])
    assert torch.equal(seen_memory[2], seen_memory[3])
    assert not torch.equal(seen_memory[0][1:], seen_memory[2][1:])
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert evaluation.paired_gain == 0
    assert evaluation.neural_mean_bp_iterations == evaluation.vanilla_mean_bp_iterations
    assert evaluation.neural_mean_relay_legs == evaluation.vanilla_mean_relay_legs
    assert evaluation.osd_shots == 0
    assert set(evaluation.bp_baselines) == {"bp_2", "bp_5"}
    assert all(row["shots"] == 4 for row in evaluation.bp_baselines.values())


def test_checkpoint_rejects_changed_relay_semantics():
    trainer = BBCircuitTrainer.__new__(BBCircuitTrainer)
    # No Relay metadata means legacy behavior on both sides.
    trainer.experiment_config = {"circuit_noise_model": "legacy"}
    trainer._validate_checkpoint_config({})
    trainer.experiment_config.update({
        "bp_relay_legs": 3, "bp_relay_solutions": 2,
        "bp_relay_memory_strength": .125, "bp_relay_memory_min": -.24,
        "bp_relay_memory_max": .66,
    })
    saved = copy.deepcopy(trainer.experiment_config)
    trainer._validate_checkpoint_config(saved)
    with pytest.raises(ValueError, match="bp_relay_legs"):
        trainer._validate_checkpoint_config({})
    saved["bp_relay_memory_min"] = -.1
    with pytest.raises(ValueError, match="bp_relay_memory_min"):
        trainer._validate_checkpoint_config(saved)
