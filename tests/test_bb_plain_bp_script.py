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
from scripts import evaluate_bb_circuit_baselines as library_baseline


def test_slurm_sweep_launches_paired_library_baselines_on_the_expected_grid():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BB_DRY_RUN": "1", "BB_REPO_ROOT": str(root),
                   "BB_BASELINE_SHOTS": "4096", "BB_BASELINE_ITERATIONS": "1000",
                   "BB_BASELINE_WORKERS": "2", "BB_BASELINE_RESUME_ROOT": ""}
    points = []
    for job in range(5):
        output = subprocess.check_output(
            ["bash", str(root / f"run_bb_{job}.slurm")], env=environment, text=True,
        )
        commands = output.splitlines()
        assert len(commands) == 4
        for index, command in enumerate(commands):
            argv = shlex.split(command)
            assert argv[:3] == ["python", "-u", str(root / "scripts/evaluate_bb_circuit_baselines.py")]
            args = library_baseline.parse_args(argv[3:])
            assert args.max_iterations == 1000 and args.detector_inputs == ["x", "xz"]
            assert args.shots == 4096 and not hasattr(args, "normalisation")
            assert "--normalisation" not in command
            assert args.workers == 2 and args.out == Path(f"exp_{index}")
            assert args.rounds == (6 if args.code == "bb72" else 12)
            points.append((args.code, args.p, args.seed))
    assert len(set(points)) == 20
    for code in ("bb72", "bb144"):
        assert {p for c, p, _ in points[:16] if c == code} == {
            .001, .002, .003, .004, .005, .006, .007, .008,
        }
        assert sum(c == code and p == .001 for c, p, _ in points) == 2
        assert sum(c == code and p == .002 for c, p, _ in points) == 2


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
    assert report["config"]["decoder"] == "min_sum"
    assert report["config"]["normalisation"] == 1.0
    assert report["config"]["normalisation_applied"] is False
    assert report["config"]["bp_evaluation_policy"] == "first_syndrome_valid_unscaled_v2"
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


def test_library_slurm_default_statistics_gpu_allocation_cpu_workers_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("BB_BASELINE_")}
    env.update(BB_DRY_RUN="1", BB_REPO_ROOT=str(root))
    for job in range(5):
        script = root / f"run_bb_{job}.slurm"
        text = script.read_text()
        assert "#SBATCH --constraint=gpu" in text and "#SBATCH --account=m5328_g\n" in text
        assert "#SBATCH --cpus-per-task=32" in text
        assert "#SBATCH --gpus-per-task=1" in text
        assert "--hint=nomultithread" not in text
        assert "srun --exclusive --exact --nodes=1 --ntasks=1" in text
        assert "        --gpus-per-task=1\n" in text
        output = subprocess.check_output(["bash", str(script)], env=env, text=True)
        for command in output.splitlines():
            args = library_baseline.parse_args(shlex.split(command)[3:])
            assert args.shots == (1_000_000 if job == 4 else 100_000)
            assert args.workers == 16 and not args.resume
    env["BB_BASELINE_RESUME_ROOT"] = str(tmp_path / "previous_job")
    output = subprocess.check_output(["bash", str(root / "run_bb_0.slurm")], env=env, text=True)
    for i, command in enumerate(output.splitlines()):
        args = library_baseline.parse_args(shlex.split(command)[3:])
        assert args.resume and args.out == tmp_path / "previous_job" / f"exp_{i}"


def test_zero_failures_still_have_a_nonzero_upper_confidence_bound():
    success = np.ones(100, dtype=bool)
    result = baseline.summarize(success, success, np.ones(100), success)
    assert result["logical_error_rate"] == 0
    assert result["logical_error_ci95_low"] == pytest.approx(0)
    assert 0 < result["logical_error_ci95_high"] < .04


def test_plain_bp_cli_rejects_normalisation_instead_of_silently_running_scaled_bp():
    with pytest.raises(SystemExit):
        baseline.parse_args(["--code=bb72", "--p=.004", "--seed=1", "--out=unused",
                             "--normalisation=.625"])
