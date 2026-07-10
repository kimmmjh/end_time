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
