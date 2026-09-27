"""Paper circuit physics, independent BP/OSD references, and resumable artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import stim
from ldpc import BpDecoder, BpOsdDecoder

from src.bb_code import BBCodeSpec
from src.bb_paper_baselines import (
    DecodingProblem, PairedBpOsd, decoding_problem, make_memory_circuit,
)
from scripts import evaluate_bb_circuit_baselines as campaign


@pytest.mark.parametrize("name", ["bb72", "bb144"])
def test_paper_circuit_boundaries_noise_locations_and_observables(name):
    code = BBCodeSpec.from_name(name)
    ideal = make_memory_circuit(code, p=0., rounds=1)
    detectors, observables = ideal.compile_detector_sampler(seed=2).sample(16, separate_observables=True)
    assert not detectors.any() and not observables.any()
    assert observables.shape == (16, code.k)
    noisy = make_memory_circuit(code, p=.003, rounds=1)
    instructions = list(noisy.flattened())
    assert sum(op.name == "TICK" for op in instructions) == 8 * 3
    assert not any(op.name == "H" for op in instructions)
    # Paper ticks 0 and 6: half the data idle; tick 7: all data idle.
    assert sum(len(op.targets_copy()) for op in instructions if op.name == "DEPOLARIZE1") == 2 * code.n
    assert sum(len(op.targets_copy()) // 2 for op in instructions if op.name == "DEPOLARIZE2") == 6 * code.n
    assert noisy.num_detectors == 2 * code.n
    assert noisy.num_observables == code.k
    for mode, rows in (("x", code.n), ("xz", 2 * code.n)):
        problem = decoding_problem(noisy, mode)
        assert problem.check.shape[0] == rows
        assert problem.logical.shape[0] == code.k
        assert np.all((problem.priors > 0) & (problem.priors < .5))


def test_projection_merges_probabilities_and_rejects_invisible_logical_faults():
    circuit = SimpleNamespace(num_detectors=2, num_observables=1,
        get_detector_coordinates=lambda: {0: [0], 1: [1]},
        detector_error_model=lambda **_: stim.DetectorErrorModel(
            "error(0.1) D0 D1 L0\nerror(0.2) D0 L0\nerror(0.3) D1"))
    x = decoding_problem(circuit, "x")
    np.testing.assert_array_equal(x.check.toarray(), [[1]])
    np.testing.assert_array_equal(x.logical.toarray(), [[1]])
    np.testing.assert_allclose(x.priors, [.26])
    assert decoding_problem(circuit, "xz").check.shape == (2, 3)
    circuit.detector_error_model = lambda **_: stim.DetectorErrorModel("error(.1) D1 L0")
    with pytest.raises(ValueError, match="Undetectable logical"):
        decoding_problem(circuit, "x")


@pytest.mark.parametrize("iterations", [1, 10])
def test_paired_outputs_are_standard_bp_and_osd_not_reseeded(iterations):
    h = sp.csr_matrix([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0],
                       [1, 0, 1, 0, 0, 1]], dtype=np.uint8)
    p = np.array([.03, .07, .11, .17, .23, .31])
    problem = DecodingProblem(h, sp.csr_matrix([[1, 0, 0, 0, 0, 0]], dtype=np.uint8), p, np.arange(3))
    decoder = PairedBpOsd(problem, iterations)
    settings = dict(error_channel=p.tolist(), bp_method="ms", ms_scaling_factor=1.,
                    max_iter=iterations, schedule="parallel")
    bp = BpDecoder(h, **settings)
    osd0 = BpOsdDecoder(h, osd_method="OSD_0", osd_order=0, **settings)
    # End with zero to expose stale state after a nonzero/OSD shot.
    for value in [*range(1, 8), 0]:
        syndrome = np.array([value >> bit & 1 for bit in range(3)], dtype=np.uint8)
        corrections, converged, used = decoder.decode(syndrome)
        np.testing.assert_array_equal(corrections[0], bp.decode(syndrome))
        np.testing.assert_array_equal(corrections[1], osd0.decode(syndrome))
        assert converged == bool(bp.converge)
        if value:
            assert used == bp.iter
        else:
            assert used == 0
        for correction in corrections[1:]:
            np.testing.assert_array_equal((h @ correction) % 2, syndrome)


def test_saved_corrections_rescore_and_resume_without_resampling(tmp_path, monkeypatch):
    args = campaign.parse_args(["--code=bb72", "--p=.006", "--rounds=1", "--shots=7",
        "--seed=29", "--max-iterations=1", "--workers=1", "--chunk-size=3", f"--out={tmp_path / 'run'}"])
    original = campaign.decode_chunk

    def interrupt_after_first(task):
        if task[0] > 0:
            raise RuntimeError("simulated worker failure")
        return original(task)

    monkeypatch.setattr(campaign, "decode_chunk", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated"):
        campaign.run(args)
    assert json.loads((args.out / "results.json").read_text())["shots_completed"] == 3
    bank_hash = campaign.file_hash(args.out / "shots.npz")
    first = next((args.out / "chunks").glob("chunk_*.npz"))
    first_hash = campaign.file_hash(first)
    monkeypatch.setattr(campaign, "decode_chunk", original)
    args.resume = True
    report = campaign.run(args)
    assert report["status"] == "complete" and report["shots_completed"] == 7
    assert report["decoders"]["xz_bp"]["flagged_failures"] > 0
    assert campaign.file_hash(args.out / "shots.npz") == bank_hash
    assert campaign.file_hash(first) == first_hash
    with np.load(args.out / "shots.npz") as bank:
        detectors = np.unpackbits(bank["detectors_packed"], axis=1,
                                  count=report["config"]["num_detectors"], bitorder="little")
        truth = np.unpackbits(bank["observables_packed"], axis=1, count=12, bitorder="little")
    totals = {name: 0 for name in report["decoders"]}
    for path in sorted((args.out / "chunks").glob("*.npz")):
        with np.load(path) as chunk:
            start, stop = int(chunk["start"]), int(chunk["stop"])
            for mode in args.detector_inputs:
                h = sp.load_npz(args.out / f"{mode}_check.npz")
                logical = sp.load_npz(args.out / f"{mode}_logical.npz")
                with np.load(args.out / f"{mode}_prior.npz") as metadata:
                    selected = metadata["detector_indices"]
                for method in ("bp", "bposd0", "bposd_cs3"):
                    name = f"{mode}_{method}"
                    correction = np.unpackbits(chunk[f"{name}_correction_packed"], axis=1,
                                                count=h.shape[1], bitorder="little")
                    valid = np.all((h @ correction.T).T % 2 == detectors[start:stop, selected], axis=1)
                    predicted = (logical @ correction.T).T % 2
                    success = valid & np.all(predicted == truth[start:stop], axis=1)
                    np.testing.assert_array_equal(predicted, chunk[f"{name}_prediction"])
                    np.testing.assert_array_equal(success, chunk[f"{name}_success"])
                    totals[name] += int((~success).sum())
    for name, failures in totals.items():
        assert report["decoders"][name]["failures"] == failures
    assert campaign.run(args)["decoders"] == report["decoders"]
    args.p = .005
    with pytest.raises(ValueError, match="configuration differs"):
        campaign.run(args)
