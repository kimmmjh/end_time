"""Standalone BP campaign: launch targets, paired shot banks, and saved metrics."""

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest

from scripts import evaluate_bb_circuit_bp as baseline


def test_slurm_sweep_launches_only_plain_bp_on_the_expected_grid():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BB_DRY_RUN": "1", "BB_REPO_ROOT": str(root),
                   "BB_BP_SHOTS": "4096", "BB_BP_REFERENCE_ITERATIONS": "1000"}
    points = []
    for job in range(5):
        output = subprocess.check_output(
            ["bash", str(root / f"run_bb_{job}.slurm")], env=environment, text=True,
        )
        commands = output.splitlines()
        assert len(commands) == 4
        for index, command in enumerate(commands):
            argv = shlex.split(command)
            assert argv[:3] == ["python", "-u", str(root / "scripts/evaluate_bb_circuit_bp.py")]
            args = baseline.parse_args(argv[3:])
            assert args.iteration_caps == [12, 1000]
            assert args.shots == 4096 and args.normalisation == 0.625
            assert args.device == "cuda" and args.out == Path(f"exp_{index}")
            assert (args.rounds, args.batch_size) == ((6, 16) if args.code == "bb72" else (12, 8))
            points.append((args.code, args.p, args.seed))
    assert len(set(points)) == 20
    for code in ("bb72", "bb144"):
        assert {p for c, p, _ in points[:16] if c == code} == {
            .001, .002, .003, .004, .005, .006, .008, .010,
        }
        assert sum(c == code and p == .004 for c, p, _ in points) == 3


def test_real_circuit_run_saves_paired_counts_and_never_uses_neural_updates(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary BP benchmark invoked neural/training code")

    monkeypatch.setattr(baseline.EquivariantNeuralBP2, "_residual", forbidden)
    monkeypatch.setattr(baseline.EquivariantNeuralBP2, "forward", forbidden)
    calls = []
    original = baseline.EquivariantNeuralBP2.decode_bp_budgets

    def decode(self, syndrome, *, budgets, neural):
        assert neural is False
        calls.append(syndrome.shape[0])
        return original(self, syndrome, budgets=budgets, neural=neural)

    monkeypatch.setattr(baseline.EquivariantNeuralBP2, "decode_bp_budgets", decode)
    directory = tmp_path / "bp"
    args = baseline.parse_args([
        "--code=bb72", "--p=0.003", "--rounds=1", "--shots=5", "--batch-size=2",
        "--iteration-caps", "1", "7", "--seed=923", "--device=cpu", "--progress-every=1",
        f"--out={directory}",
    ])
    report = baseline.run(args)
    assert calls == [2, 2, 1]
    assert report == json.loads((directory / "results.json").read_text())
    assert report["status"] == "complete" and report["shots_completed"] == 5
    assert not any(report["config"][key] for key in ("neural", "relay", "osd"))
    with (directory / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["decoder"] for row in rows] == ["bp_1", "bp_7"]
    assert all(row["status"] == "complete" and row["shots"] == "5" for row in rows)
    with np.load(directory / "shots.npz", allow_pickle=False) as bank:
        assert bank["detectors"].shape[0] == bank["observables"].shape[0] == 5
        assert bank["observables"].shape[1] == 24
    with np.load(directory / "outcomes.npz", allow_pickle=False) as outcomes:
        short = outcomes["bp_1_success"]
        for cap in (1, 7):
            success = outcomes[f"bp_{cap}_success"]
            converged = outcomes[f"bp_{cap}_converged"]
            iterations = outcomes[f"bp_{cap}_iterations"]
            row = report["decoders"][f"bp_{cap}"]
            assert row["failures"] == (~success).sum()
            assert row["failures"] == row["flagged_failures"] + row["unflagged_failures"]
            assert row["mean_bp_iterations"] == iterations.mean()
            assert ((iterations >= 1) & (iterations <= cap)).all()
            assert not (success & ~converged).any()
            assert row["rescued"] == (success & ~short).sum()
            assert row["harmed"] == (~success & short).sum()
    before = (directory / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        baseline.run(args)
    assert (directory / "results.json").read_bytes() == before


def test_zero_failures_still_have_a_nonzero_upper_confidence_bound():
    success = np.ones(100, dtype=bool)
    result = baseline.summarize(success, success, np.ones(100), success)
    assert result["logical_error_rate"] == 0
    assert result["logical_error_ci95_low"] == pytest.approx(0)
    assert 0 < result["logical_error_ci95_high"] < .04
