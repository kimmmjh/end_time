import csv
import json

import numpy as np
import pytest

pytest.importorskip("ldpc")

from scripts import bb_classical_baselines as baseline


def test_reproducible_decoder_defaults():
    args = baseline.parse_args(["--p", "0.08"])
    assert args.osd_max_iter == 0
    assert args.lsd_max_iter == 0
    assert args.bp_method == "minimum_sum"
    assert args.osd_ms_scaling_factor == pytest.approx(0.625)
    assert args.lsd_ms_scaling_factor == pytest.approx(0.625)
    assert args.schedule == "parallel"
    assert args.omp_threads == 1


def test_scoring_matches_stabilizer_cosets_and_failure_decomposition():
    code = baseline._load_bb_code_spec().bb72()
    shots = 3
    x_error = np.zeros((shots, code.n), dtype=np.uint8)
    z_error = np.zeros_like(x_error)
    x_error[0, 0] = 1
    x_error[1, 0] = 1
    x_error[2, 0] = 1
    syndrome_x, syndrome_z = baseline.syndromes_for_errors(
        x_error, z_error, hx=code.hx, hz=code.hz
    )

    correction_x = x_error.copy()
    correction_z = z_error.copy()
    # A stabilizer-degenerate correction remains successful.
    correction_x[1] ^= code.hx[0]
    # A logical residual has the right syndrome but must be an unflagged fail.
    correction_x[2] ^= code.logicals_x[0]
    outcomes = baseline.score_css_corrections(
        x_error=x_error,
        z_error=z_error,
        correction_x=correction_x,
        correction_z=correction_z,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        hx=code.hx,
        hz=code.hz,
        logicals_x=code.logicals_x,
        logicals_z=code.logicals_z,
    )
    np.testing.assert_array_equal(outcomes["success"], (True, True, False))
    np.testing.assert_array_equal(
        outcomes["unflagged_logical_failure"], (False, False, True)
    )
    assert not outcomes["flagged_failure"].any()

    correction_x[0] = 0
    flagged = baseline.score_css_corrections(
        x_error=x_error,
        z_error=z_error,
        correction_x=correction_x,
        correction_z=correction_z,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        hx=code.hx,
        hz=code.hz,
        logicals_x=code.logicals_x,
        logicals_z=code.logicals_z,
    )
    assert flagged["flagged_failure"][0]
    assert not flagged["unflagged_logical_failure"][0]


def test_sampling_is_order_independent_and_depolarizing():
    seed = baseline._point_seed(123, "bb72", 0.08)
    first = baseline.sample_depolarizing_errors(shots=50_000, n=2, p=0.09, seed=seed)
    second = baseline.sample_depolarizing_errors(shots=50_000, n=2, p=0.09, seed=seed)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert np.isclose(first[0].mean(), 2 * 0.09 / 3, atol=0.003)
    assert np.isclose(first[1].mean(), 2 * 0.09 / 3, atol=0.003)
    # Y is the joint X/Z component and occurs with probability p/3.
    assert np.isclose((first[0] & first[1]).mean(), 0.09 / 3, atol=0.002)


def test_cli_smoke_writes_three_paired_rows_and_npz(tmp_path):
    csv_path = tmp_path / "baseline.csv"
    npz_path = tmp_path / "outcomes.npz"
    assert (
        baseline.main(
            [
                "--code",
                "bb72",
                "--p",
                "0.04",
                "--shots",
                "6",
                "--osd_max_iter",
                "3",
                "--lsd_max_iter",
                "3",
                "--warmup_shots",
                "1",
                "--progress_every",
                "0",
                "--output",
                str(csv_path),
                "--save_test_bank",
                str(npz_path),
            ]
        )
        == 0
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method"] for row in rows] == list(baseline.METHODS)
    assert all(row["samples"] == "6" for row in rows)
    assert all(row["sector_strategy"] == "css_separated" for row in rows)
    assert all(row["uses_xz_correlation"] == "False" for row in rows)
    assert all(row["sample_sha256"] == rows[0]["sample_sha256"] for row in rows)

    with np.load(npz_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["methods"] == list(baseline.METHODS)
        assert metadata["sector_strategy"] == "css_separated"
        assert metadata["uses_xz_correlation"] is False
        assert archive["bb72_p0p04__x_error"].shape == (6, 72)
        assert archive["bb72_p0p04__syndrome_x"].shape == (6, 36)
        for method in baseline.METHODS:
            success = archive[f"bb72_p0p04__{method}__success"]
            assert success.dtype == np.bool_
            assert success.shape == (6,)
