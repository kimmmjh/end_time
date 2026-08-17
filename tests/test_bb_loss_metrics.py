import numpy as np
import torch

from src._bb_loss import DegeneracyAwareBPLoss
from src._bb_metrics import aggregate_bb_outcomes, bb_shot_outcomes
from src.bb_code import BBCodeSpec
from src.bb_data_generator import BBCodeCapacityGenerator


def _hard_logits(pauli):
    pauli = torch.as_tensor(pauli, dtype=torch.long)
    logits = torch.full((*pauli.shape, 4), -20.0)
    logits.scatter_(-1, pauli.unsqueeze(-1), 20.0)
    return logits


def _outcomes(code, logits, batch):
    return bb_shot_outcomes(
        logits,
        batch.syndrome,
        batch.pauli,
        hx=torch.tensor(code.hx),
        hz=torch.tensor(code.hz),
        logicals_x=torch.tensor(code.logicals_x),
        logicals_z=torch.tensor(code.logicals_z),
    )


def test_block_metrics_distinguish_success_flagged_and_unflagged_failure():
    code = BBCodeSpec.bb72()
    generator = BBCodeCapacityGenerator(code, 0.08)
    truth = np.zeros((1, code.n), dtype=np.int64)
    batch = generator.batch_from_pauli(truth)

    identity = _outcomes(code, _hard_logits(truth), batch)
    assert identity.success.item()

    # A different representative in the same stabilizer coset is also valid.
    stabilizer = truth.copy()
    stabilizer[0, code.hx[0].astype(bool)] = 1
    degenerate = _outcomes(code, _hard_logits(stabilizer), batch)
    assert degenerate.success.item()
    assert not degenerate.pauli_correct.all()

    single_x = truth.copy()
    single_x[0, 0] = 1
    flagged = _outcomes(code, _hard_logits(single_x), batch)
    assert flagged.flagged_failure.item()
    assert not flagged.success.item()

    logical_x = truth.copy()
    logical_x[0, code.logicals_x[0].astype(bool)] = 1
    unflagged = _outcomes(code, _hard_logits(logical_x), batch)
    assert unflagged.syndrome_converged.item()
    assert unflagged.unflagged_logical_failure.item()
    assert not unflagged.success.item()

    aggregate = aggregate_bb_outcomes([identity, degenerate, flagged, unflagged])
    assert aggregate.samples == 4
    assert aggregate.logical_accuracy == 0.5
    assert aggregate.flagged_failure_rate == 0.25
    assert aggregate.unflagged_logical_failure_rate == 0.25


def test_coset_loss_accepts_a_stabilizer_degenerate_correction_and_has_gradients():
    code = BBCodeSpec.bb72()
    generator = BBCodeCapacityGenerator(code, 0.08, seed=11)
    truth = np.zeros((1, code.n), dtype=np.int64)
    batch = generator.batch_from_pauli(truth)
    correction = truth.copy()
    correction[0, code.hx[0].astype(bool)] = 1
    logits = _hard_logits(correction).requires_grad_()

    criterion = DegeneracyAwareBPLoss(
        hx=torch.tensor(code.hx),
        hz=torch.tensor(code.hz),
        logicals_x=torch.tensor(code.logicals_x),
        logicals_z=torch.tensor(code.logicals_z),
        pauli_weight=0.0,
        deep_supervision_weight=0.0,
    )
    losses = criterion(logits, batch.syndrome, batch.pauli)
    assert losses.syndrome.item() < 1e-4
    assert losses.logical.item() < 1e-4
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_is_finite_on_sampled_batch():
    code = BBCodeSpec.bb72()
    batch = BBCodeCapacityGenerator(code, 0.1, seed=12).sample(5)
    logits = torch.randn(5, code.n, 4, requires_grad=True)
    criterion = DegeneracyAwareBPLoss(
        hx=torch.tensor(code.hx),
        hz=torch.tensor(code.hz),
        logicals_x=torch.tensor(code.logicals_x),
        logicals_z=torch.tensor(code.logicals_z),
    )
    losses = criterion(logits, batch.syndrome, batch.pauli)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert torch.isfinite(logits.grad).all()
