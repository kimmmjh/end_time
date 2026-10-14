import hashlib
import json

import numpy as np
import pytest
import torch

from models._bb_tanner_cnn import BBTannerCNN
from scripts.evaluate_bb_cnn_osd import evaluate_run, repair, sha256
from src._bb_metrics import bb_correction_outcomes
from src._direct_osd import DirectOSD0, SyndromeOSD0
from src.bb_code import BBCodeSpec
from src.bb_data_generator import BBCodeCapacityGenerator


def test_syndrome_osd_uses_error_likelihood_and_zero_free_variables():
    check = [[1, 1, 1]]
    q = [.99, .9, .49]
    # Residual repair retains two certain ones and flips the uncertain third.
    np.testing.assert_array_equal(DirectOSD0(check).decode([1], q), [1, 1, 1])
    # Published syndrome OSD instead chooses the most likely erroneous column.
    np.testing.assert_array_equal(SyndromeOSD0(check).decode([1], q), [1, 0, 0])
    np.testing.assert_array_equal(SyndromeOSD0(check).decode([0], q), [0, 0, 0])
    np.testing.assert_array_equal(SyndromeOSD0(check).decode([1], [.5]*3), [1, 0, 0])
    np.testing.assert_array_equal(SyndromeOSD0(check).decode([1], [0, 1, 1]), [0, 1, 0])


def test_syndrome_osd_dependent_checks_and_impossible_syndrome():
    h = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    decoder = SyndromeOSD0(h)
    for bits in range(8):
        truth = np.array([(bits >> i) & 1 for i in range(3)], dtype=np.uint8)
        result = decoder.decode(h @ truth % 2, [.8, .2, .45])
        np.testing.assert_array_equal(h @ result % 2, h @ truth % 2)
        assert result[1] == 0  # Independent columns 0 and 2 span the image.
    with pytest.raises(ValueError, match="not in the image"):
        decoder.decode([1, 0, 0], [.8, .2, .45])
    with pytest.raises(ValueError, match="Non-finite"):
        decoder.decode([0, 0, 0], [np.nan, .2, .3])


def test_css_fallback_preserves_valid_sectors():
    code = BBCodeSpec.bb72()
    batch = BBCodeCapacityGenerator(code, .2, seed=51).sample(8)
    x, z = batch.x_error.numpy().astype(np.uint8), batch.z_error.numpy().astype(np.uint8)
    # Supply the true X component and deliberately incorrect Z component.
    px, pz = .1 + .8*x, np.full_like(z, .1, dtype=float)
    out = repair(batch.syndrome.numpy().astype(np.uint8), px, pz, code,
                 (SyndromeOSD0(code.hz), SyndromeOSD0(code.hx)))
    np.testing.assert_array_equal(np.isin(out, [1, 2]), x)
    np.testing.assert_array_equal(np.isin(out, [2, 3]) @ code.hx.T % 2, batch.syndrome_x_checks)


def make_run(run, *, syndrome_osd=False):
    run.mkdir()
    torch.manual_seed(5)
    code = BBCodeSpec.bb72()
    model = BBTannerCNN(code, width=4, depth=2).eval()
    generator = BBCodeCapacityGenerator(code, .06, seed=37)
    config = {"architecture": "bb_tanner_cnn", "code": code.name, "noise_model": "capacity",
              "cnn_width": 4, "cnn_depth": 2, "channel_probabilities": generator.channel_probabilities.tolist(),
              "graph_fingerprint": hashlib.sha256(code.hx.tobytes() + code.hz.tobytes()).hexdigest()}
    if syndrome_osd:
        config["osd"] = "syndrome_osd0_v1"
    batch = generator.sample(8)
    syndrome, pauli = batch.syndrome.numpy().astype(np.uint8), batch.pauli.numpy().astype(np.uint8)
    with torch.no_grad():
        logits = model(batch.syndrome, channel_probabilities=batch.channel_probabilities)
        qx, qz = model.component_probabilities(logits)
        raw = model.hard_decision(logits).numpy().astype(np.uint8)
    decoder = SyndromeOSD0 if syndrome_osd else DirectOSD0
    osd = repair(syndrome, qx.numpy(), qz.numpy(), code, (decoder(code.hz), decoder(code.hx)))
    scoring = {k: torch.tensor(getattr(code, k)) for k in ("hx", "hz", "logicals_x", "logicals_z")}
    success = {name: bb_correction_outcomes(torch.from_numpy(value), batch.syndrome, batch.pauli,
                                           **scoring).success.numpy() for name, value in (("raw", raw), ("osd", osd))}
    np.savez_compressed(run / "final_shots.npz", syndrome=syndrome, pauli=pauli, raw_correction=raw,
                        osd_correction=osd, raw_success=success["raw"], osd_success=success["osd"],
                        metadata_json=json.dumps(config))
    # Latest weights intentionally differ; evaluation must choose the best.
    torch.save({"format": "bb_tanner_cnn_v1", "config": config, "epoch": 3, "best_epoch": 1,
                "model_state_dict": {k: torch.zeros_like(v) for k, v in model.state_dict().items()},
                "best_model_state_dict": model.state_dict()}, run / "model.pt")
    (run / "history.json").write_text(json.dumps({"config": config, "final": {"epoch": 1, "shots": 8}}))


@pytest.mark.parametrize("syndrome_osd", [False, True])
def test_posthoc_uses_best_weights_same_shots_and_never_trains(tmp_path, monkeypatch, syndrome_osd):
    source = tmp_path / "source"
    make_run(source, syndrome_osd=syndrome_osd)
    hashes = {p.name: sha256(p) for p in source.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Posthoc evaluation must not construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    report = evaluate_run(source, tmp_path / "out", batch_size=4)
    assert report["training_steps"] == 0 and report["selected_epoch"] == 1
    assert report["shots"] == 8 and not report["partial"]
    assert report["metrics"]["syndrome_osd0"]["syndrome_convergence"] == 1
    assert report["paired"]["standard_vs_raw"]["harmed"] == 0
    assert all(v["correction_mismatch_shots"] == 0 for v in report["historical_reproduction"].values())
    assert ("syndrome_osd0" if syndrome_osd else "residual_osd0") in report["historical_reproduction"]
    assert hashes == {p.name: sha256(p) for p in source.iterdir()}
    with np.load(source / "final_shots.npz") as old, np.load(tmp_path / "out/paired_shots.npz") as new:
        np.testing.assert_array_equal(old["pauli"], new["pauli"])
        np.testing.assert_array_equal(old["syndrome"], new["syndrome"])
    with pytest.raises(FileExistsError):
        evaluate_run(source, tmp_path / "out")
    partial = evaluate_run(source, tmp_path / "partial", max_shots=3)
    assert partial["partial"] and partial["shots"] == 3


def test_posthoc_refuses_wrong_bank_or_selected_epoch(tmp_path):
    source = tmp_path / "source"
    make_run(source)
    history = json.loads((source / "history.json").read_text())
    history["final"]["epoch"] = 2
    (source / "history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError, match="selected best epoch"):
        evaluate_run(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def discovery_run(directory, epochs, *, seed=1, complete=True):
    directory.mkdir(parents=True)
    history = {'config': {'architecture': 'bb_tanner_cnn', 'code': 'bb144',
                          'cnn_depth': 5, 'cnn_width': 64, 'error_rate': .04, 'seed': seed},
               'train': [{'epoch': i} for i in range(epochs)]}
    if complete:
        history['final'] = {'epoch': epochs-1, 'shots': 8}
    (directory/'history.json').write_text(json.dumps(history))
    (directory/'model.pt').write_bytes(b'checkpoint')
    if complete:
        (directory/'final_shots.npz').write_bytes(b'bank')
    return directory


def test_latest_discovery_keeps_seeds_and_refuses_missing_newest_bank(tmp_path):
    from scripts.evaluate_bb_cnn_osd import discover_runs
    old = discovery_run(tmp_path/'old', 400)
    latest = discovery_run(tmp_path/'new', 800)
    other_seed = discovery_run(tmp_path/'other', 400, seed=2)
    incomplete = discovery_run(tmp_path/'incomplete', 900, complete=False)
    runs, skipped = discover_runs([tmp_path], latest_only=True, require_complete=True)
    assert set(runs) == {latest, other_seed}
    assert skipped == [{'run': str(incomplete), 'missing': ['completed final evaluation']}]
    (latest/'final_shots.npz').unlink()
    with pytest.raises(ValueError, match='Selected runs lack required inputs'):
        discover_runs([tmp_path], latest_only=True, require_complete=True)
    runs, skipped = discover_runs([tmp_path], latest_only=True)
    assert runs == [other_seed] and old not in runs  # No silent fallback to 400.


def test_latest_discovery_deduplicates_copies_and_rejects_ambiguous_ties(tmp_path):
    from scripts.evaluate_bb_cnn_osd import discover_runs
    first = discovery_run(tmp_path/'root_copy', 800)
    second = discovery_run(tmp_path/'archive_copy', 800)
    runs, _ = discover_runs([tmp_path, first], latest_only=True, require_complete=True)
    assert len(runs) == 1
    (second/'model.pt').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='Ambiguous latest runs'):
        discover_runs([tmp_path], latest_only=True, require_complete=True)
