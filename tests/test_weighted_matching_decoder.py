import numpy as np
import torch
from panqec.codes import Toric2DCode

from models import NeuralWeightedMatchingDecoder, RecurrentEdgeWeightNetwork
from models.loss_functions import EdgeBCELoss
from src._data_generator import CircuitLevelDataGenerator
from src.stim_utils import generate_toric_memory_circuit


def _small_weighted_decoder(*, seed=7):
    torch.manual_seed(seed)
    lattice_size = 3
    rounds = 3
    code = Toric2DCode(lattice_size)
    circuit = generate_toric_memory_circuit(
        code,
        rounds=rounds,
        gate_error_rate=0.01,
        measurement_error_rate=0.01,
    )
    network = RecurrentEdgeWeightNetwork(
        channels=[4],
        depths=[1],
        lattice_size=lattice_size,
        gru_channels=4,
        gru_layers=1,
        edge_hidden_channels=8,
        edge_chunk_size=64,
    )
    decoder = NeuralWeightedMatchingDecoder(
        network,
        detector_error_model=circuit.detector_error_model(decompose_errors=True),
        lattice_size=lattice_size,
        rounds=rounds,
        num_observables=4,
    )
    return code, circuit, network, decoder


def test_dem_mechanism_targets_reconstruct_detectors_and_observables():
    _, circuit, _, decoder = _small_weighted_decoder()
    dem = circuit.detector_error_model(decompose_errors=True).flattened()
    detectors, observables, mechanisms = dem.compile_sampler(seed=11).sample(
        shots=128,
        return_errors=True,
    )

    targets = decoder.layout.edge_targets(mechanisms)
    arrays = decoder.layout.arrays
    reconstructed_detectors = (targets @ arrays.check_matrix.T.toarray()) & 1
    reconstructed_observables = (targets @ arrays.faults.T.toarray()) & 1

    np.testing.assert_array_equal(reconstructed_detectors, detectors)
    np.testing.assert_array_equal(reconstructed_observables, observables)


def test_zero_delta_decoder_equals_static_mwpm_and_edge_loss_backpropagates():
    code, _, network, decoder = _small_weighted_decoder()
    generator = CircuitLevelDataGenerator(
        code,
        error_rate=0.01,
        measurement_error_rate=0.01,
        rounds=3,
        batch_size=4,
        verbose=False,
        seed=13,
    )
    syndrome, truth, metadata = generator.generate_batch_with_metadata(
        torch.device("cpu")
    )

    decoder.train()
    dummy_predictions = decoder(syndrome)
    edge_logits, edge_targets = decoder.loss_inputs(
        dummy_predictions,
        truth,
        batch_metadata=metadata,
    )
    loss = EdgeBCELoss(entropy_weight=0.01)(edge_logits, edge_targets)
    loss.backward()

    final_layer = network.edge_head[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.weight.grad is not None
    assert torch.all(torch.isfinite(final_layer.weight.grad))
    assert final_layer.weight.grad.abs().sum() > 0

    decoder.eval()
    with torch.no_grad():
        predictions = decoder(syndrome)
    torch.testing.assert_close(
        predictions.argmax(dim=1), decoder.last_baseline_classes
    )


def test_edge_logits_are_equivariant_under_toric_translation():
    _, _, network, decoder = _small_weighted_decoder()
    lattice_size = 3
    rounds = 3
    with torch.no_grad():
        torch.nn.init.normal_(network.edge_head[-1].weight, std=0.1)
        torch.nn.init.normal_(network.edge_head[-1].bias, std=0.1)
    network.eval()

    syndrome = torch.randn(2, 2, rounds, lattice_size**2)
    spatial = syndrome.reshape(2, 2, rounds, lattice_size, lattice_size)
    shifted = torch.roll(spatial, shifts=(1, -1), dims=(3, 4)).reshape_as(
        syndrome
    )
    with torch.no_grad():
        original_logits = network(
            syndrome,
            decoder.edge_endpoints,
            decoder.edge_geometry,
            decoder.base_edge_logits,
        )
        shifted_logits = network(
            shifted,
            decoder.edge_endpoints,
            decoder.edge_geometry,
            decoder.base_edge_logits,
        )

    endpoints = decoder.edge_endpoints.cpu().numpy()
    edge_lookup = {tuple(edge): index for index, edge in enumerate(endpoints)}

    def translate_node(node: int) -> int:
        area = lattice_size**2
        time, remainder = divmod(int(node), 2 * area)
        sector, position = divmod(remainder, area)
        x, y = divmod(position, lattice_size)
        x = (x + 1) % lattice_size
        y = (y - 1) % lattice_size
        return ((time * 2 + sector) * lattice_size + x) * lattice_size + y

    edge_permutation = []
    for node1, node2 in endpoints:
        translated1 = translate_node(node1)
        translated2 = translate_node(node2) if node2 >= 0 else -1
        if translated2 >= 0 and translated1 > translated2:
            translated1, translated2 = translated2, translated1
        edge_permutation.append(edge_lookup[(translated1, translated2)])

    torch.testing.assert_close(
        shifted_logits[:, edge_permutation], original_logits, rtol=1e-5, atol=1e-6
    )

