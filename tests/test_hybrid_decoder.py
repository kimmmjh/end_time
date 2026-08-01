import numpy as np
import torch
from panqec.codes import Toric2DCode
from torch import nn
from torch.nn import functional as F

from models import MatchingResidualDecoder, RecurrentResidualEND2D
from src.stim_utils import generate_toric_memory_circuit


class _RecordingMatching:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.uint8)
        self.last_shots = None
        self.last_enable_correlations = None

    def decode_batch(self, shots, *, enable_correlations=False):
        self.last_shots = np.array(shots, copy=True)
        self.last_enable_correlations = enable_correlations
        return self.predictions[: shots.shape[0]]


class _ConstantResidualDecoder(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_classes))

    def forward(self, syndrome):
        return self.logits.unsqueeze(0).expand(syndrome.shape[0], -1)


class _TableResidualDecoder(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, syndrome):
        return self.logits[: syndrome.shape[0]]


def _roll_flat_lattice(syndrome, *, lattice_size, shifts):
    batch, sectors, rounds, _ = syndrome.shape
    spatial = syndrome.reshape(
        batch, sectors, rounds, lattice_size, lattice_size
    )
    return torch.roll(spatial, shifts=shifts, dims=(3, 4)).reshape_as(syndrome)


def test_recurrent_residual_decoder_is_invariant_and_backpropagates():
    torch.manual_seed(10)
    lattice_size = 3
    model = RecurrentResidualEND2D(
        channels=[5, 6],
        depths=[1, 1],
        lattice_size=lattice_size,
        gru_channels=7,
        gru_layers=2,
    )
    model.eval()
    syndrome = torch.randn(
        2,
        2,
        3,
        lattice_size**2,
        requires_grad=True,
    )

    logits = model(syndrome)
    shifted_logits = model(
        _roll_flat_lattice(syndrome, lattice_size=lattice_size, shifts=(1, -1))
    )

    assert logits.shape == (2, 16)
    torch.testing.assert_close(shifted_logits, logits, rtol=1e-5, atol=1e-6)

    logits.square().mean().backward()
    assert syndrome.grad is not None
    assert torch.all(torch.isfinite(syndrome.grad))
    assert syndrome.grad.abs().sum() > 0


def test_matching_wrapper_preserves_detector_order_and_xor_classes():
    # Observable predictions encode classes 10 (1010) and 5 (0101).
    matching = _RecordingMatching([[1, 0, 1, 0], [0, 1, 0, 1]])
    residual_decoder = _ConstantResidualDecoder()
    with torch.no_grad():
        residual_decoder.logits[0] = 5.0
    decoder = MatchingResidualDecoder(
        residual_decoder,
        matching=matching,
        enable_correlations=False,
    )

    syndrome = torch.tensor(
        [
            [
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
            ],
            [
                [[1, 0], [1, 1]],
                [[0, 1], [0, 1]],
            ],
        ],
        dtype=torch.float32,
    )
    final_logits = decoder(syndrome)

    expected_flat = (
        syndrome.numpy().transpose(0, 2, 1, 3).reshape(syndrome.shape[0], -1)
    )
    np.testing.assert_array_equal(matching.last_shots, expected_flat)
    assert matching.last_enable_correlations is False
    torch.testing.assert_close(
        decoder.last_baseline_classes,
        torch.tensor([10, 5]),
    )
    torch.testing.assert_close(final_logits.argmax(dim=1), torch.tensor([10, 5]))

    true_classes = torch.tensor([3, 12])
    residual_logits, residual_classes = decoder.loss_inputs(
        final_logits, true_classes
    )
    torch.testing.assert_close(
        residual_logits,
        residual_decoder(syndrome),
    )
    torch.testing.assert_close(
        residual_classes,
        torch.tensor([3 ^ 10, 12 ^ 5]),
    )

    F.cross_entropy(residual_logits, residual_classes).backward()
    assert residual_decoder.logits.grad is not None
    assert residual_decoder.logits.grad.abs().sum() > 0


def test_disabled_gate_is_exact_mwpm_for_arbitrary_neural_logits():
    matching = _RecordingMatching([[1, 0, 1, 0], [0, 1, 0, 1]])
    residual_decoder = _TableResidualDecoder(
        [
            [-5.0, 0.0, 1.0, 20.0] + [0.0] * 12,
            [-10.0] + [0.0] * 14 + [30.0],
        ]
    )
    decoder = MatchingResidualDecoder(
        residual_decoder,
        matching=matching,
        enable_correlations=False,
    )

    final_logits = decoder(torch.zeros(2, 2, 1, 2))

    assert not bool(decoder.gate_enabled.item())
    torch.testing.assert_close(final_logits.argmax(dim=1), torch.tensor([10, 5]))


def test_enabled_gate_applies_only_nonzero_residual_above_margin():
    # Use baseline class 5 for the tie cases so absolute-class argmax ordering
    # cannot accidentally make the equality case look correct.
    matching = _RecordingMatching(
        [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]]
    )
    residual_decoder = _TableResidualDecoder(
        [
            [0.0, 1.2] + [0.0] * 14,  # 1.2 / 2 > margin: apply class 1.
            [0.0, 0.8, 0.0] + [0.0] * 13,  # 0.8 / 2 < margin: fallback.
            [0.0, 1.0] + [0.0] * 14,  # Equal to margin: strict fallback.
        ]
    )
    decoder = MatchingResidualDecoder(
        residual_decoder,
        matching=matching,
        enable_correlations=False,
    )
    decoder.configure_gate(enabled=True, temperature=2.0, margin=0.5)

    final_logits = decoder(torch.zeros(3, 2, 1, 2))

    torch.testing.assert_close(
        final_logits.argmax(dim=1),
        torch.tensor([4 ^ 1, 5, 5]),
    )


def test_loss_inputs_use_raw_logits_and_backpropagate_through_enabled_gate():
    matching = _RecordingMatching([[0, 0, 0, 0], [0, 0, 0, 0]])
    residual_decoder = _TableResidualDecoder(
        [[0.0, 2.0] + [0.0] * 14, [1.0, 0.0] + [0.0] * 14]
    )
    decoder = MatchingResidualDecoder(
        residual_decoder,
        matching=matching,
        enable_correlations=False,
    )
    decoder.configure_gate(enabled=True, temperature=3.0, margin=1.5)
    syndrome = torch.zeros(2, 2, 1, 2)

    final_logits = decoder(syndrome)
    loss_logits, loss_targets = decoder.loss_inputs(
        final_logits, torch.tensor([1, 0])
    )

    torch.testing.assert_close(loss_logits, residual_decoder.logits)
    torch.testing.assert_close(loss_targets, torch.tensor([1, 0]))
    F.cross_entropy(loss_logits, loss_targets).backward()
    assert residual_decoder.logits.grad is not None
    assert residual_decoder.logits.grad.abs().sum() > 0


def test_harmful_calibration_keeps_gate_disabled():
    sample_count = 100
    logits = torch.zeros(sample_count, 16)
    logits[:, 1] = 3.0  # Every proposed correction would harm correct MWPM.
    true_classes = torch.zeros(sample_count, dtype=torch.long)
    matching_classes = torch.zeros_like(true_classes)
    decoder = MatchingResidualDecoder(
        _ConstantResidualDecoder(),
        matching=_RecordingMatching(np.zeros((sample_count, 4), dtype=np.uint8)),
        enable_correlations=False,
    )

    stats = decoder.calibrate_gate(logits, true_classes, matching_classes)

    assert stats["enabled"] is False
    assert stats["corrections"] == 0
    assert stats["net_gain"] == 0
    assert not bool(decoder.gate_enabled.item())


def test_single_calibration_sample_cannot_enable_gate():
    logits = torch.zeros(1, 16)
    logits[0, 1] = 10.0
    decoder = MatchingResidualDecoder(
        _ConstantResidualDecoder(),
        matching=_RecordingMatching(np.zeros((1, 4), dtype=np.uint8)),
        enable_correlations=False,
    )

    stats = decoder.calibrate_gate(
        logits,
        true_classes=torch.tensor([1]),
        matching_classes=torch.tensor([0]),
        confidence_z=0.0,
    )

    assert stats["enabled"] is False
    assert not bool(decoder.gate_enabled.item())


def test_helpful_calibration_enables_conservative_residual_gate():
    sample_count = 200
    logits = torch.zeros(sample_count, 16)
    true_classes = torch.zeros(sample_count, dtype=torch.long)
    matching_classes = torch.zeros_like(true_classes)

    # Eighty high-margin class-1 predictions rescue MWPM failures.
    logits[:80, 1] = 4.0
    true_classes[:80] = 1
    # Ten lower-margin class-1 predictions would harm correct MWPM shots.  A
    # conservative threshold should separate these from the rescues.
    logits[80:90, 1] = 1.0

    table = _TableResidualDecoder(logits)
    matching = _RecordingMatching(np.zeros((sample_count, 4), dtype=np.uint8))
    decoder = MatchingResidualDecoder(
        table,
        matching=matching,
        enable_correlations=False,
    )

    stats = decoder.calibrate_gate(logits, true_classes, matching_classes)
    final_logits = decoder(torch.zeros(sample_count, 2, 1, 2))

    assert stats["enabled"] is True
    assert stats["rescues"] == 80
    assert stats["harms"] == 0
    assert stats["net_gain_lcb"] > 0.0
    assert bool(decoder.gate_enabled.item())
    torch.testing.assert_close(final_logits.argmax(dim=1), true_classes)


def test_gate_calibration_survives_state_dict_roundtrip():
    matching_predictions = np.zeros((4, 4), dtype=np.uint8)
    source = MatchingResidualDecoder(
        _ConstantResidualDecoder(),
        matching=_RecordingMatching(matching_predictions),
        enable_correlations=False,
    )
    source.configure_gate(enabled=True, temperature=1.7, margin=0.85)
    restored = MatchingResidualDecoder(
        _ConstantResidualDecoder(),
        matching=_RecordingMatching(matching_predictions),
        enable_correlations=False,
    )

    restored.load_state_dict(source.state_dict())

    assert bool(restored.gate_enabled.item())
    torch.testing.assert_close(restored.gate_temperature, torch.tensor(1.7))
    torch.testing.assert_close(restored.gate_margin, torch.tensor(0.85))


def test_matching_wrapper_pads_missing_observable_columns():
    matching = _RecordingMatching(np.empty((3, 0), dtype=np.uint8))
    decoder = MatchingResidualDecoder(
        _ConstantResidualDecoder(),
        matching=matching,
        num_observables=4,
        enable_correlations=False,
    )

    observables = decoder.matching_observables(
        torch.zeros(3, 2, 1, 4),
    )

    assert observables.shape == (3, 4)
    assert not torch.any(observables)


def test_matching_wrapper_builds_from_real_noisy_toric_circuit():
    lattice_size = 3
    rounds = 2
    code = Toric2DCode(lattice_size)
    circuit = generate_toric_memory_circuit(
        code,
        rounds=rounds,
        gate_error_rate=0.005,
        measurement_error_rate=0.005,
    )
    network = RecurrentResidualEND2D(
        channels=[4],
        depths=[1],
        lattice_size=lattice_size,
        gru_channels=4,
    )
    decoder = MatchingResidualDecoder(
        network,
        circuit=circuit,
        num_observables=2 * code.k,
        enable_correlations=True,
    )
    detectors, true_bits = circuit.compile_detector_sampler(seed=11).sample(
        shots=4,
        separate_observables=True,
    )
    syndrome = torch.tensor(
        detectors.reshape(4, rounds, 2, lattice_size**2).transpose(0, 2, 1, 3),
        dtype=torch.float32,
    )
    true_classes = torch.tensor(
        true_bits.astype(np.uint8) @ np.array([8, 4, 2, 1], dtype=np.uint8),
        dtype=torch.long,
    )

    final_logits = decoder(syndrome)
    residual_logits, residual_classes = decoder.loss_inputs(
        final_logits, true_classes
    )

    assert final_logits.shape == (4, 16)
    assert residual_logits.shape == (4, 16)
    assert residual_classes.shape == (4,)
    assert torch.all(torch.isfinite(final_logits))
