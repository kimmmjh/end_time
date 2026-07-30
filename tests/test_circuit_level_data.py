import numpy as np
import torch
from panqec.codes import Toric2DCode

from models import Decoder
from models._the_end_3d import TransformedEND3D
from models.pooling_layers import TranslationalEquivariantPooling2D
from src._data_generator import CircuitLevelDataGenerator
from src.stim_utils import generate_toric_memory_circuit


def test_noiseless_circuit_is_deterministic():
    code = Toric2DCode(3)
    circuit = generate_toric_memory_circuit(
        code,
        rounds=3,
        gate_error_rate=0.0,
        measurement_error_rate=0.0,
    )

    detectors, observables = circuit.compile_detector_sampler(seed=1).sample(
        shots=16,
        separate_observables=True,
    )

    assert detectors.shape == (16, 3 * 2 * 3**2)
    assert observables.shape == (16, 2 * code.k)
    assert not np.any(detectors)
    assert not np.any(observables)
    circuit.detector_error_model()


def test_noisy_circuit_is_closed_in_time():
    lattice_size = 3
    rounds = 3
    circuit = generate_toric_memory_circuit(
        Toric2DCode(lattice_size),
        rounds=rounds,
        gate_error_rate=0.001,
        measurement_error_rate=0.001,
    )

    dem = circuit.detector_error_model(decompose_errors=True)
    assert circuit.num_detectors == rounds * 2 * lattice_size**2
    assert len(circuit.shortest_graphlike_error()) == lattice_size

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        has_logical = any(target.is_logical_observable_id() for target in targets)
        has_detector = any(target.is_relative_detector_id() for target in targets)
        assert not has_logical or has_detector


def test_generator_matches_model_input_shape():
    code = Toric2DCode(3)
    generator = CircuitLevelDataGenerator(
        code=code,
        error_rate=0.01,
        measurement_error_rate=0.02,
        batch_size=8,
        rounds=2,
        verbose=False,
        seed=2,
    )

    syndromes, classes = generator.generate_batch(torch.device("cpu"))

    assert syndromes.shape == (8, 2, 2, 3**2)
    assert classes.shape == (8,)
    assert syndromes.dtype == torch.float32
    assert classes.dtype == torch.int64
    assert torch.all((syndromes == 0) | (syndromes == 1))
    assert torch.all((0 <= classes) & (classes < 16))


def test_raw_logical_bits_are_available_for_offline_preparation():
    code = Toric2DCode(3)
    generator = CircuitLevelDataGenerator(
        code=code,
        error_rate=0.02,
        measurement_error_rate=0.01,
        batch_size=8,
        rounds=2,
        categorical_classification=False,
        verbose=False,
        seed=3,
    )

    _, logical_bits = generator.generate_batch(torch.device("cpu"))

    assert logical_bits.shape == (8, 2 * code.k)
    assert torch.all((logical_bits == 0) | (logical_bits == 1))


def test_current_decoder_accepts_circuit_batch():
    lattice_size = 2
    generator = CircuitLevelDataGenerator(
        code=Toric2DCode(lattice_size),
        error_rate=0.01,
        measurement_error_rate=0.01,
        batch_size=2,
        rounds=2,
        verbose=False,
        seed=4,
    )
    syndromes, _ = generator.generate_batch(torch.device("cpu"))
    decoder = Decoder(
        network=TransformedEND3D(
            channels=[6],
            depths=[1],
            lattice_size=lattice_size,
        ),
        pooling=TranslationalEquivariantPooling2D(lattice_size),
        ensemble=None,
    )

    output = decoder(syndromes)

    assert output.shape == (2, 16)
    assert torch.all(torch.isfinite(output))
