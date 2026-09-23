import math

import numpy as np
import pytest
import torch

from models._bb_tanner_cnn import BBTannerCNN, TannerShiftConv
from src._direct_osd import DirectOSD0
from src.bb_code import BBCodeSpec


@pytest.mark.parametrize("name", ["bb72", "bb144"])
@pytest.mark.parametrize("to_qubits", [True, False])
def test_shift_convolution_matches_actual_tanner_edges(name, to_qubits):
    code = BBCodeSpec.from_name(name)
    conv = TannerShiftConv(code, 1, 1, to_qubits=to_qubits)
    with torch.no_grad():
        for orbit, kernel in enumerate(conv.kernels):
            kernel.weight.fill_(orbit + 1)
    values = torch.arange(2 * code.cells, dtype=torch.float32).reshape(1, 2, 1, code.ell, code.m)
    expected = np.zeros(code.n)
    flat = values.numpy().reshape(-1)
    for check, qubit, orbit in zip(code.edge_check_index, code.edge_qubit_index, code.edge_orbit):
        source, target = (check, qubit) if to_qubits else (qubit, check)
        expected[target] += flat[source] * (orbit + 1)
    actual = conv(values).detach().numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected / math.sqrt(6), rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("name", ["bb72", "bb144"])
def test_joint_cnn_is_translation_equivariant_and_differentiable(name):
    torch.manual_seed(31)
    code = BBCodeSpec.from_name(name)
    model = BBTannerCNN(code, width=4, depth=2)
    syndrome = torch.randint(2, (2, code.num_checks)).float()
    prior = torch.rand(2, code.n, 4)
    prior /= prior.sum(-1, keepdim=True)
    def shift(value):
        return torch.roll(value, (1, 2), dims=(-2, -1))
    moved_syndrome = shift(syndrome.reshape(2, 2, code.ell, code.m)).reshape_as(syndrome)
    moved_prior = shift(prior.reshape(2, 2, code.ell, code.m, 4).permute(0, 1, 4, 2, 3))
    moved_prior = moved_prior.permute(0, 1, 3, 4, 2).reshape_as(prior)
    output = model(syndrome, channel_probabilities=prior)
    actual = model(moved_syndrome, channel_probabilities=moved_prior)
    expected = shift(output.reshape(2, 2, code.ell, code.m, 4).permute(0, 1, 4, 2, 3))
    expected = expected.permute(0, 1, 3, 4, 2).reshape_as(output)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    for weights in (model.first_conv.kernels[0].weight, model.blocks[0].qubit_to_check.kernels[0].weight):
        assert weights.grad is not None and torch.isfinite(weights.grad).all()
        assert weights.grad.abs().sum() > 0


def test_depth_expands_receptive_field_beyond_the_same_six_bit_patch():
    torch.manual_seed(14)
    code = BBCodeSpec.bb72()
    adjacent = torch.tensor(np.concatenate((code.hx, code.hz), axis=0)[:, 0]).bool()
    magnitudes = []
    for depth in (1, 2):
        model = BBTannerCNN(code, width=4, depth=depth)
        syndrome = torch.zeros(1, code.num_checks, requires_grad=True)
        model(syndrome, channel_probabilities=torch.tensor([.94, .02, .02, .02]))[0, 0, 1].backward()
        magnitudes.append(syndrome.grad[0, ~adjacent].abs().sum().item())
    assert magnitudes[0] == 0
    assert magnitudes[1] > 1e-12


def test_hard_decision_uses_shared_component_marginals_not_pauli_argmax():
    logits = torch.tensor([[[.35, .30, .30, .05]]]).log()
    assert logits.argmax(-1).item() == 0
    assert BBTannerCNN.hard_decision(logits).item() == 1


def test_direct_osd_changes_uncertain_bit_and_preserves_valid_hard_decisions():
    decoder = DirectOSD0([[1, 1]])
    np.testing.assert_array_equal(decoder.decode([1], [.01, .49]), [0, 1])
    np.testing.assert_array_equal(decoder.decode([0], [.99, .49]), [1, 1])
    np.testing.assert_array_equal(decoder.decode([1], [.99, .01]), [1, 0])


def test_direct_osd_handles_dependent_rows_and_rejects_impossible_syndromes():
    matrix = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    decoder = DirectOSD0(matrix)
    for value in range(8):
        truth = np.array([(value >> bit) & 1 for bit in range(3)], dtype=np.uint8)
        syndrome = matrix @ truth % 2
        correction = decoder.decode(syndrome, [.8, .2, .45])
        np.testing.assert_array_equal(matrix @ correction % 2, syndrome)
    with pytest.raises(ValueError, match="not in the image"):
        decoder.decode([1, 0, 0], [.1, .1, .1])


def test_direct_osd_repairs_real_rank_deficient_bb_checks():
    code = BBCodeSpec.bb72()
    rng = np.random.default_rng(73)
    for check in (code.hx, code.hz):
        errors = rng.integers(0, 2, (5, code.n), dtype=np.uint8)
        syndromes = errors @ check.T % 2
        probabilities = rng.uniform(.01, .99, (5, code.n))
        correction = DirectOSD0(check).decode_batch(syndromes, probabilities)
        np.testing.assert_array_equal(correction @ check.T % 2, syndromes)
