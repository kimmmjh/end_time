import numpy as np
import pytest
import torch

from src.bb_code import BBCodeSpec, gf2_rank
from src.bb_data_generator import BBCodeCapacityGenerator


@pytest.mark.parametrize(
    ("constructor", "n", "k", "distance", "checks_per_type"),
    [
        (BBCodeSpec.bb72, 72, 12, 6, 36),
        (BBCodeSpec.bb144, 144, 12, 12, 72),
    ],
)
def test_published_bb_parameters_and_css_relations(
    constructor, n, k, distance, checks_per_type
):
    code = constructor()

    assert (code.n, code.k, code.d) == (n, k, distance)
    assert code.hx.shape == (checks_per_type, n)
    assert code.hz.shape == (checks_per_type, n)
    assert n - gf2_rank(code.hx) - gf2_rank(code.hz) == k
    np.testing.assert_array_equal((code.hx @ code.hz.T) % 2, 0)

    # All published checks have weight six and every qubit has degree three in
    # each CSS check graph.
    np.testing.assert_array_equal(code.hx.sum(axis=1), 6)
    np.testing.assert_array_equal(code.hz.sum(axis=1), 6)
    np.testing.assert_array_equal(code.hx.sum(axis=0), 3)
    np.testing.assert_array_equal(code.hz.sum(axis=0), 3)


@pytest.mark.parametrize("constructor", [BBCodeSpec.bb72, BBCodeSpec.bb144])
def test_logical_operators_are_canonical_quotient_representatives(constructor):
    code = constructor()

    assert code.logicals_x.shape == (code.k, code.n)
    assert code.logicals_z.shape == (code.k, code.n)
    np.testing.assert_array_equal((code.hz @ code.logicals_x.T) % 2, 0)
    np.testing.assert_array_equal((code.hx @ code.logicals_z.T) % 2, 0)
    np.testing.assert_array_equal(
        (code.logicals_x @ code.logicals_z.T) % 2,
        np.eye(code.k, dtype=np.uint8),
    )

    assert gf2_rank(np.vstack((code.hx, code.logicals_x))) == (
        gf2_rank(code.hx) + code.k
    )
    assert gf2_rank(np.vstack((code.hz, code.logicals_z))) == (
        gf2_rank(code.hz) + code.k
    )


def test_edges_are_row_major_and_cyclic_orbits_are_equivariant():
    code = BBCodeSpec.bb72()
    stacked = np.vstack((code.hx, code.hz))
    expected_checks, expected_qubits = np.nonzero(stacked)

    np.testing.assert_array_equal(code.edge_check_index, expected_checks)
    np.testing.assert_array_equal(code.edge_qubit_index, expected_qubits)
    assert code.edge_index.shape == (2, code.num_edges)
    assert code.num_edges == 12 * code.cells
    np.testing.assert_array_equal(
        np.bincount(code.edge_orbit, minlength=code.num_edge_orbits),
        np.full(code.num_edge_orbits, code.cells),
    )

    permutations = code.translation_permutations(dx=2, dy=-1)
    assert sorted(permutations["checks"].tolist()) == list(range(code.num_checks))
    assert sorted(permutations["qubits"].tolist()) == list(range(code.n))
    np.testing.assert_array_equal(
        code.edge_orbit,
        code.edge_orbit[permutations["edges"]],
    )
    np.testing.assert_array_equal(
        code.edge_check_index[permutations["edges"]],
        permutations["checks"][code.edge_check_index],
    )
    np.testing.assert_array_equal(
        code.edge_qubit_index[permutations["edges"]],
        permutations["qubits"][code.edge_qubit_index],
    )


def test_exact_pauli_to_syndrome_and_logical_targets():
    code = BBCodeSpec.bb72()
    generator = BBCodeCapacityGenerator(code, error_rate=0.1)
    pauli = np.zeros((4, code.n), dtype=np.int64)
    pauli[0, 0] = 1  # X
    pauli[1, code.cells] = 2  # Y
    pauli[2, 7] = 3  # Z
    pauli[3, [3, code.cells + 5]] = (1, 3)

    batch = generator.batch_from_pauli(pauli)
    x = np.isin(pauli, (1, 2)).astype(np.uint8)
    z = np.isin(pauli, (2, 3)).astype(np.uint8)

    np.testing.assert_array_equal(batch.syndrome_x_checks.numpy(), (z @ code.hx.T) % 2)
    np.testing.assert_array_equal(batch.syndrome_z_checks.numpy(), (x @ code.hz.T) % 2)
    np.testing.assert_array_equal(batch.logical_x.numpy(), (x @ code.logicals_z.T) % 2)
    np.testing.assert_array_equal(batch.logical_z.numpy(), (z @ code.logicals_x.T) % 2)
    assert batch.pauli.dtype == torch.long


def test_sampling_is_reproducible_and_distinguishes_xz_correlation_models():
    code = BBCodeSpec.bb72()
    depolarizing_a = BBCodeCapacityGenerator(code, error_rate=0.12, seed=91)
    depolarizing_b = BBCodeCapacityGenerator(code, error_rate=0.12, seed=91)
    independent = BBCodeCapacityGenerator(
        code,
        error_rate=0.12,
        noise_model="independent_xz",
        seed=91,
    )

    batch_a = depolarizing_a.sample(256)
    batch_b = depolarizing_b.sample(256)
    torch.testing.assert_close(batch_a.pauli, batch_b.pauli)
    torch.testing.assert_close(batch_a.syndrome, batch_b.syndrome)

    np.testing.assert_allclose(
        depolarizing_a.channel_probabilities,
        (0.88, 0.04, 0.04, 0.04),
    )
    np.testing.assert_allclose(
        independent.channel_probabilities,
        (0.7744, 0.1056, 0.0144, 0.1056),
    )
    assert np.isclose(independent.channel_probabilities.sum(), 1.0)


def test_generator_state_restores_next_samples_and_rejects_mismatch():
    code = BBCodeSpec.bb72()
    original = BBCodeCapacityGenerator(code, error_rate=0.08, seed=7)
    original.sample(13)
    state = original.state_dict()
    expected_next = original.sample(31)

    resumed = BBCodeCapacityGenerator(code, error_rate=0.08, seed=999)
    resumed.load_state_dict(state)
    resumed_next = resumed.sample(31)
    torch.testing.assert_close(resumed_next.pauli, expected_next.pauli)
    torch.testing.assert_close(resumed_next.syndrome, expected_next.syndrome)

    wrong_channel = BBCodeCapacityGenerator(code, error_rate=0.09, seed=999)
    with pytest.raises(ValueError, match="channel"):
        wrong_channel.load_state_dict(state)

    wrong_code = BBCodeCapacityGenerator(BBCodeSpec.bb144(), error_rate=0.08, seed=999)
    with pytest.raises(ValueError, match="code identity/shape"):
        wrong_code.load_state_dict(state)


def test_torch_buffers_and_input_validation():
    code = BBCodeSpec.from_name("gross")
    buffers = code.torch_buffers()

    assert buffers["hx"].shape == code.hx.shape
    assert buffers["hx"].dtype == torch.float32
    assert buffers["edge_index"].dtype == torch.long
    assert buffers["edge_index"].shape == code.edge_index.shape

    with pytest.raises(ValueError, match="Unknown BB code"):
        BBCodeSpec.from_name("not-a-code")
    with pytest.raises(ValueError, match="noise_model"):
        BBCodeCapacityGenerator(code, 0.1, noise_model="burst")
    with pytest.raises(ValueError, match="shape"):
        BBCodeCapacityGenerator(code, 0.1).batch_from_pauli(
            np.zeros((2, code.n - 1), dtype=np.int64)
        )
