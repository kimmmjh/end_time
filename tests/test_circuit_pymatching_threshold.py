import numpy as np

from scripts.circuit_pymatching_threshold import (
    benchmark_point,
    build_circuit_matching,
    decode_observables,
)


def test_circuit_pymatching_has_no_failures_without_noise():
    row = benchmark_point(
        L=3,
        rounds=3,
        p=0.0,
        q=0.0,
        shots=32,
        batch_size=8,
        seed=123,
    )

    assert row["noise_model"] == "circuit"
    assert row["failures"] == 0
    assert row["accuracy"] == 1.0


def test_circuit_matcher_prediction_has_all_logical_columns():
    circuit, matching = build_circuit_matching(
        L=3,
        rounds=2,
        p=0.002,
        q=0.002,
    )
    detectors, observables = circuit.compile_detector_sampler(seed=7).sample(
        shots=8,
        separate_observables=True,
    )

    predicted = decode_observables(
        matching,
        detectors,
        num_observables=circuit.num_observables,
    )

    assert predicted.shape == observables.shape == (8, 4)
    assert predicted.dtype == np.uint8
    assert np.all((predicted == 0) | (predicted == 1))
