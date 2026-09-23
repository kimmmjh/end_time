"""First-valid BP inference, shared-budget evaluation, and OSD fallback."""

import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models._equivariant_neural_bp2 import EquivariantNeuralBP2
from models._neural_relay_bp2 import NeuralRelayBP2
from src._bb_circuit_metrics import score_corrections
from src._bb_circuit_trainer import BBCircuitTrainer, BP_EVALUATION_POLICY


def graph():
    return SimpleNamespace(
        num_detectors=1, num_mechanisms=2, num_observables=1,
        edge_detector=np.array([0, 0]), edge_mechanism=np.array([0, 1]),
        edge_orbit=np.array([0, 0]), num_orbits=1,
        prior_log_odds=np.array([5., 1.]),
        check_matrix=sp.csr_matrix([[1, 1]], dtype=np.uint8),
        observable_matrix=sp.csr_matrix([[1, 0]], dtype=np.uint8),
    )


def model(cls=EquivariantNeuralBP2):
    return cls(graph(), iterations=2, hidden_dim=4, orbit_embedding_dim=0,
               gradient_checkpoint=False)


def test_multiple_caps_freeze_first_valid_shots_and_continue_only_failures():
    decoder = model()
    sizes = []

    def iteration(variable, check, posterior, syndrome, neural):
        sizes.append(syndrome.shape[0])
        output = torch.ones_like(posterior)
        if len(sizes) <= 2:
            output[0, 0] = -float(len(sizes))
        return variable, check, output

    decoder._iteration = iteration
    results = decoder.decode_bp_budgets(torch.ones(3, 1), budgets=(1, 4, 1))
    assert sizes == [3, 2, 1, 1]
    assert results[1].converged.tolist() == [True, False, False]
    assert results[1].iterations.tolist() == [1, 1, 1]
    assert results[4].converged.tolist() == [True, True, False]
    assert results[4].iterations.tolist() == [1, 2, 4]
    assert results[4].posterior.tolist() == [[-1., 1.], [-2., 1.], [1., 1.]]
    assert results[4].correction.tolist() == [[1, 0], [1, 0], [0, 0]]
    # Returning a later snapshot must not mutate the earlier budget's result.
    assert results[1].posterior.tolist() == [[-1., 1.], [1., 1.], [1., 1.]]


def test_early_convergence_finishes_all_future_budgets_without_extra_iterations():
    decoder = model()
    original = decoder._iteration
    calls = []

    def iteration(*args):
        calls.append(1)
        return original(*args)

    decoder._iteration = iteration
    results = decoder.decode_bp_budgets(torch.tensor([[1.], [0.]]), budgets=(2, 1000))
    assert len(calls) == 1
    for result in results.values():
        assert result.converged.all()
        assert result.iterations.tolist() == [1, 1]
        assert result.correction.tolist() == [[0, 1], [0, 0]]
    assert results[2].posterior.data_ptr() != results[1000].posterior.data_ptr()


def test_stopping_checks_syndrome_without_peeking_at_logical_truth():
    decoder = model()
    result = decoder.decode_bp(torch.ones(1, 1), max_iterations=1000)
    assert result.iterations.item() == 1
    assert result.converged.item()
    outcomes = score_corrections(
        result.correction.numpy(), detectors=np.array([[1]]),
        observables=np.array([[1]]), check_matrix=graph().check_matrix,
        observable_matrix=graph().observable_matrix,
    )
    assert outcomes.unflagged_failure == 1.0


@pytest.mark.parametrize("budgets", [(), (0,), (-1,), (1.5,), (True,)])
def test_invalid_iteration_caps_rejected(budgets):
    with pytest.raises(ValueError, match="positive integer"):
        model().decode_bp_budgets(torch.ones(1, 1), budgets=budgets)


@pytest.mark.parametrize("syndrome", [torch.empty(0, 1), torch.ones(1, 2),
                                    torch.tensor([[float("nan")]]), torch.tensor([[.5]])])
def test_invalid_detector_input_rejected(syndrome):
    with pytest.raises(ValueError, match="syndrome"):
        model().decode_bp(syndrome)


def test_ordinary_reference_ignores_learned_terms_and_relay_memory():
    decoder = model(NeuralRelayBP2)
    syndrome = torch.tensor([[0.], [1.]])
    before = decoder.decode_bp(syndrome)
    with torch.no_grad():
        decoder.relaxation_raw.fill_(10)
        decoder.residual_network[-1].bias.fill_(-100)
    decoder.sample_memory = lambda *args, **kwargs: pytest.fail("Ordinary BP used Relay memory")
    decoder._residual = lambda *args: pytest.fail("Ordinary BP used neural residual")
    after = decoder.decode_bp(syndrome)
    for name in ("correction", "posterior", "converged", "iterations"):
        assert torch.equal(getattr(before, name), getattr(after, name))


def test_training_still_unrolls_every_step_and_backpropagates():
    decoder = model()
    detector = torch.tensor([[0.], [1.]])
    posterior, history = decoder(detector, neural=True, return_all=True)
    assert history.shape == (2, decoder.iterations, 2)
    history.sum().backward()
    assert decoder.residual_network[-1].weight.grad.abs().sum() > 0
    assert not decoder.decode_bp(detector, neural=True).posterior.requires_grad
    assert torch.equal(posterior, history[:, -1])


def test_trainer_records_two_paired_plain_bp_caps_on_the_same_shots():
    trainer = BBCircuitTrainer.__new__(BBCircuitTrainer)
    trainer.model = model()
    trainer.device = torch.device("cpu")
    trainer.bp_reference_iterations = 5
    trainer.osd_eval_shots = 0
    trainer._osd = None
    trainer.experiment_config = {}
    samples = []

    def sample_circuit(**kwargs):
        samples.append(1)
        return SimpleNamespace(detectors=torch.tensor([[0.], [1.]]),
                               observables=torch.zeros(2, 1))

    trainer.eval_generator = SimpleNamespace(graph=graph(), sample_circuit=sample_circuit)
    result = trainer.evaluate(2)
    assert len(samples) == 2
    assert result.paired_gain == 0
    assert result.neural_mean_bp_iterations == result.vanilla_mean_bp_iterations == 1
    assert set(result.bp_baselines) == {"bp_2", "bp_5"}
    for row in result.bp_baselines.values():
        assert row["shots"] == 4
        assert row["logical_error_rate"] == row["paired_gain"] == 0
        assert row["syndrome_convergence"] == row["mean_bp_iterations"] == 1
    assert "bp_5 (first valid, max 5)" in trainer._format(result, 0, "test")
    assert asdict(result)["bp_baselines"]["bp_5"]["early_stopping"]


def test_osd_runs_only_on_syndrome_failures_and_preserves_valid_corrections():
    trainer = BBCircuitTrainer.__new__(BBCircuitTrainer)
    calls = []

    def decode_batch(detectors, *, posterior):
        calls.append((detectors.copy(), posterior.copy()))
        return np.array([[1, 0]], dtype=np.uint8)

    trainer._osd = SimpleNamespace(decode_batch=decode_batch)
    detectors = np.array([[0], [1], [1]], dtype=np.uint8)
    correction = np.array([[0, 0], [0, 0], [0, 1]], dtype=np.uint8)
    posterior = torch.tensor([[5., 1.], [3., 2.], [4., -1.]])
    result = trainer._postprocess_unconverged(
        detectors, correction, posterior, np.array([True, False, True]),
    )
    assert len(calls) == 1
    assert calls[0][0].tolist() == [[1]]
    assert calls[0][1].tolist() == [[3., 2.]]
    assert result.tolist() == [[0, 0], [1, 0], [0, 1]]
    assert correction[1].tolist() == [0, 0]
    trainer._postprocess_unconverged(detectors, result, posterior, np.ones(3, dtype=bool))
    assert len(calls) == 1


def test_checkpoint_rejects_old_evaluation_policy_and_changed_reference_budget():
    trainer = BBCircuitTrainer.__new__(BBCircuitTrainer)
    trainer.experiment_config = {
        "circuit_noise_model": "legacy", "bp_evaluation_policy": BP_EVALUATION_POLICY,
        "bb_bp_reference_iterations": 1000,
    }
    trainer._validate_checkpoint_config(copy.deepcopy(trainer.experiment_config))
    with pytest.raises(ValueError, match="bp_evaluation_policy"):
        trainer._validate_checkpoint_config({})
    changed = {**trainer.experiment_config, "bb_bp_reference_iterations": 12}
    with pytest.raises(ValueError, match="bb_bp_reference_iterations"):
        trainer._validate_checkpoint_config(changed)
