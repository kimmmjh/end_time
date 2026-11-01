from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models._bb_dem_cnn import BBDemCNN
from models._bb_hierarchical_dem_cnn import BBHierarchicalDemCNN
from src._direct_osd import DirectOSD0


@pytest.fixture
def graph():
    # Mixed degrees 1/2/3/0: includes a joint hyperedge and a detector-free fault.
    check = sp.csr_matrix(np.array([[1, 1, 1, 0],
                                   [0, 1, 1, 0],
                                   [0, 0, 1, 0]], dtype=np.uint8))
    edges = check.tocoo()
    return SimpleNamespace(
        num_detectors=3, num_mechanisms=4, num_edges=check.nnz,
        edge_detector=edges.row, edge_mechanism=edges.col,
        edge_orbit=np.arange(check.nnz), num_orbits=check.nnz,
        detector_coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 2]]),
        detector_frames=3, prior_log_odds=np.array([3., 4., 5., 6.]),
        check_matrix=check,
    )


def model_with_llrs(graph, llrs, **kwargs):
    model = BBHierarchicalDemCNN(graph, width=4, **kwargs)
    model.forward = Mock(return_value=torch.tensor(llrs, dtype=torch.float32))
    return model


def test_forward_checkpoint_compatibility_and_gradients(graph):
    base = BBDemCNN(graph, width=4, depth=2, edge_chunk_size=2)
    model = BBHierarchicalDemCNN(graph, width=4, depth=2, edge_chunk_size=2)
    model.load_state_dict(base.state_dict(), strict=True)
    detectors = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    llrs = model(detectors)
    torch.testing.assert_close(llrs, base(detectors))
    torch.nn.functional.binary_cross_entropy_with_logits(-llrs, torch.zeros_like(llrs)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.first.input.weight.grad.abs().sum() > 0


def test_confidence_candidates_and_exact_residual(graph):
    model = model_with_llrs(graph, [[-5., -1., -4., -8.]], edge_chunk_size=2)
    detectors = torch.tensor([[1, 0, 0]])
    local = model.preprocess(detectors)
    # Low-confidence m1 abstains; detector-free m3 is not an active candidate.
    assert local.local_correction.tolist() == [[1, 0, 1, 0]]
    expected = detectors.numpy() ^ ((graph.check_matrix @ local.local_correction.numpy().T).T % 2)
    np.testing.assert_array_equal(local.residual_detectors.numpy(), expected)
    torch.testing.assert_close(local.residual_log_odds, torch.tensor([[5., -1., 4., -8.]]))
    assert not local.log_odds.requires_grad


def test_backend_can_undo_local_decision_and_receives_residual_probabilities(graph):
    model = model_with_llrs(graph, [[-5., 2., 4., 6.]])
    detectors = torch.tensor([[1, 1, 0]])
    backend = Mock()
    # Local chooses m0. Global may undo m0 and instead choose m1.
    backend.decode_batch.return_value = np.array([[1, 1, 0, 0]])
    result = model.decode(detectors, backend=backend)
    syndrome, probabilities = backend.decode_batch.call_args.args
    np.testing.assert_array_equal(syndrome, [[0, 1, 0]])
    np.testing.assert_allclose(probabilities, torch.sigmoid(torch.tensor([[-5., -2., -4., -6.]], dtype=torch.double)))
    assert result.correction.tolist() == [[0, 1, 0, 0]]
    assert result.global_used.tolist() == [True]
    assert result.syndrome_converged.all()


def test_zero_residual_skips_backend_and_mixed_batch_keeps_order(graph):
    model = model_with_llrs(graph, [[-5., 4., 5., 6.], [3., 4., 5., 6.], [3., 4., 5., 6.]])
    detectors = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    backend = Mock(wraps=DirectOSD0(graph.check_matrix))
    result = model.decode(detectors, backend=backend)
    assert result.global_used.tolist() == [False, True, False]
    assert backend.decode_batch.call_args.args[0].shape == (1, 3)
    assert result.correction.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    model = model_with_llrs(graph, [[-5., 4., 5., 6.]])
    result = model.decode(detectors[:1])
    assert model._default_backend is None
    assert not result.global_used.any()


def test_default_backend_satisfies_all_toy_syndromes(graph):
    detectors = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])
    model = model_with_llrs(graph, [[-4., -3., 5., 6.]] * 8)
    result = model.decode(detectors)
    expected = (graph.check_matrix @ result.correction.numpy().T).T % 2
    np.testing.assert_array_equal(expected, detectors.numpy())
    assert result.syndrome_converged.all()


@pytest.mark.parametrize("output, error", [([[0, 0, 0, 0]], RuntimeError),
                                         ([[0, 2, 0, 0]], ValueError),
                                         ([[0, 1]], ValueError)])
def test_invalid_backend_results_are_rejected(graph, output, error):
    model = model_with_llrs(graph, [[3., 4., 5., 6.]])
    backend = Mock()
    backend.decode_batch.return_value = np.array(output)
    with pytest.raises(error):
        model.decode(torch.tensor([[1, 0, 0]]), backend=backend)


@pytest.mark.parametrize("threshold", [0.5, 1., float("nan"), float("inf")])
def test_invalid_confidence(graph, threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        BBHierarchicalDemCNN(graph, confidence_threshold=threshold)


def test_invalid_inputs_and_nonfinite_llrs(graph):
    model = BBHierarchicalDemCNN(graph, width=4)
    with pytest.raises(ValueError, match="binary"):
        model.preprocess(torch.tensor([[0., 0.5, 1.]]))
    with pytest.raises(ValueError, match="Expected detectors"):
        model.preprocess(torch.zeros(0, 3))
    model = model_with_llrs(graph, [[float("nan"), 4., 5., 6.]])
    with pytest.raises(ValueError, match="non-finite"):
        model.decode(torch.zeros(1, 3))


def test_active_candidate_filter_can_be_disabled(graph):
    model = model_with_llrs(graph, [[3., 4., 5., -8.]], active_candidates_only=False)
    result = model.decode(torch.zeros(1, 3))
    assert result.correction.tolist() == [[0, 0, 0, 1]]
    assert not result.global_used.any()


def test_real_bb_circuit_residual_completion():
    from src.bb_circuit_data import BBCircuitGenerator

    generator = BBCircuitGenerator("bb72", rounds=1, gate_error_rate=.003,
                                   circuit_noise_model="standard", batch_size=2, seed=71)
    model = BBHierarchicalDemCNN(generator.graph, width=4, depth=2).eval()
    batch = generator.sample_dem()
    result = model.decode(batch.detectors)
    actual = (generator.graph.check_matrix @ result.correction.numpy().T).T % 2
    np.testing.assert_array_equal(actual, batch.detectors.numpy())
    assert result.global_used.any()
    assert result.syndrome_converged.all()
