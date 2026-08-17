import itertools

import numpy as np
import pytest
import torch

from models._equivariant_neural_bp import EquivariantNeuralBP4
from src.bb_code import BBCodeSpec


def _empty_hz(num_qubits):
    return np.zeros((0, num_qubits), dtype=np.uint8)


def test_one_iteration_matches_exact_tree_marginal_and_vanilla_path():
    # One X check imposes z(q0) xor z(q1) = 1.  Its Tanner graph is a tree, so
    # one sum-product iteration must equal direct enumeration exactly.
    hx = np.array([[1, 1]], dtype=np.uint8)
    model = EquivariantNeuralBP4(hx, _empty_hz(2), iterations=1)
    syndrome = torch.tensor([[1]], dtype=torch.uint8)
    p = 0.12

    neural_logits = model(syndrome, p=p)
    vanilla_logits = model(syndrome, p=p, neural=False)
    torch.testing.assert_close(neural_logits, vanilla_logits, rtol=0.0, atol=0.0)

    prior = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0])
    # I,X,Y,Z anticommute with an X check as 0,0,1,1.
    anti = np.array([0, 0, 1, 1], dtype=np.uint8)
    expected = np.zeros((2, 4), dtype=np.float64)
    normalization = 0.0
    for first, second in itertools.product(range(4), repeat=2):
        weight = prior[first] * prior[second]
        if (anti[first] ^ anti[second]) == 1:
            normalization += weight
            expected[0, first] += weight
            expected[1, second] += weight
    expected /= normalization

    torch.testing.assert_close(
        neural_logits.exp(),
        torch.tensor(expected, dtype=neural_logits.dtype).unsqueeze(0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_orbit_tied_updates_are_cyclic_translation_equivariant():
    torch.manual_seed(17)
    size = 5
    hx = np.zeros((size, size), dtype=np.uint8)
    edge_orbits = []
    for check in range(size):
        hx[check, check] = 1
        hx[check, (check + 1) % size] = 1
        # IDs must follow row-major nonzeros, including the wrapped row whose
        # neighbor column zero appears before its diagonal column.
    for check, qubit in np.argwhere(hx):
        edge_orbits.append(0 if qubit == check else 1)

    model = EquivariantNeuralBP4(
        hx,
        _empty_hz(size),
        edge_orbits=edge_orbits,
        iterations=3,
        residual_hidden_dim=11,
    )
    # Exercise the learned path, not only its BP initialization.
    with torch.no_grad():
        for residual in model.residual_mlps:
            torch.nn.init.normal_(residual.network[-1].weight, std=0.08)
            torch.nn.init.normal_(residual.network[-1].bias, std=0.03)
        model.relaxation_raw.normal_(std=0.2)

    syndrome = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32)
    logits = model(syndrome, p=0.08)
    shifted = model(torch.roll(syndrome, shifts=1, dims=1), p=0.08)

    torch.testing.assert_close(
        shifted,
        torch.roll(logits, shifts=1, dims=1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_bb72_edge_metadata_and_model_are_jointly_translation_equivariant():
    torch.manual_seed(23)
    code = BBCodeSpec.bb72()
    model = EquivariantNeuralBP4(
        code.hx,
        code.hz,
        edge_orbits=code.edge_orbit,
        iterations=2,
        residual_hidden_dim=7,
    )
    with torch.no_grad():
        for residual in model.residual_mlps:
            torch.nn.init.normal_(residual.network[-1].weight, std=0.04)
            torch.nn.init.normal_(residual.network[-1].bias, std=0.02)
        model.relaxation_raw.normal_(std=0.1)

    syndrome = torch.randint(0, 2, (2, code.num_checks)).float()
    permutation = code.translation_permutations(dx=2, dy=-1)
    translated_syndrome = torch.empty_like(syndrome)
    translated_syndrome[:, permutation["checks"]] = syndrome

    logits = model(syndrome, p=0.07)
    translated_logits = model(translated_syndrome, p=0.07)
    expected = torch.empty_like(logits)
    expected[:, permutation["qubits"], :] = logits
    torch.testing.assert_close(translated_logits, expected, rtol=1e-5, atol=1e-6)


def test_shapes_custom_priors_and_gradients():
    hx = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    hz = hx.copy()  # Every X/Z row overlap is zero or two, hence CSS-valid.
    # Four Hx edges followed by four Hz edges in row-major order.  Raw orbit
    # zero is intentionally reused: check type still creates distinct sharing.
    model = EquivariantNeuralBP4(
        hx,
        hz,
        edge_orbits=[0, 1, 0, 1, 0, 1, 0, 1],
        iterations=4,
        residual_hidden_dim=9,
    )
    assert model.edge_type_keys == ((0, 0), (0, 1), (1, 0), (1, 1))

    syndrome = torch.tensor(
        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.float32
    )
    prior_logits = torch.randn(3, 4, 4, requires_grad=True)
    final, history = model(syndrome, prior_logits=prior_logits, return_all=True)

    assert final.shape == (3, 4, 4)
    assert history.shape == (3, 4, 4, 4)
    torch.testing.assert_close(final, history[:, -1])
    torch.testing.assert_close(final.exp().sum(dim=-1), torch.ones(3, 4))

    loss = final.square().mean()
    loss.backward()
    assert prior_logits.grad is not None
    assert torch.all(torch.isfinite(prior_logits.grad))
    final_layers = [residual.network[-1] for residual in model.residual_mlps]
    assert all(layer.weight.grad is not None for layer in final_layers)


def test_uniform_vectorized_check_update_matches_irregular_fallback():
    torch.manual_seed(29)
    size = 7
    hx = np.zeros((size, size), dtype=np.uint8)
    for check in range(size):
        hx[check, check] = 1
        hx[check, (check + 1) % size] = 1
        hx[check, (check + 3) % size] = 1
    model = EquivariantNeuralBP4(hx, _empty_hz(size), iterations=1)
    assert model.uniform_check_degree == 3

    messages = torch.log_softmax(torch.randn(4, model.num_edges, 4), dim=-1)
    syndrome = torch.randint(0, 2, (4, model.num_checks)).float()
    vectorized = model._exact_check_update_uniform(messages, syndrome)
    fallback = model._exact_check_update_general(messages, syndrome)

    assert torch.all(torch.isfinite(vectorized))
    torch.testing.assert_close(vectorized, fallback, rtol=1e-6, atol=1e-7)


def test_rejects_non_css_checks_and_misaligned_orbits():
    hx = np.array([[1, 1, 0]], dtype=np.uint8)
    hz = np.array([[1, 0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="commuting CSS"):
        EquivariantNeuralBP4(hx, hz)

    with pytest.raises(ValueError, match="one entry per row-major Tanner edge"):
        EquivariantNeuralBP4(hx, _empty_hz(3), edge_orbits=[0])
