import json

import numpy as np

from scripts.bb_cnn_plot_osd import POLICY, STANDARD, report_index, sha, standard_metrics, residual_metrics


def run_and_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    history = {"config": {"osd": "direct_hard_centered_osd0_v1"},
               "final": {"epoch": 4, "shots": 8, "osd": {"logical_error_rate": .5}}}
    path = run / "history.json"
    path.write_text(json.dumps(history))
    (run / "model.pt").write_bytes(b"selected checkpoint")
    (run / "final_shots.npz").write_bytes(b"source bank")
    output = tmp_path / "evaluation"
    output.mkdir()
    report = dict(policy=POLICY, partial=False, config=history["config"], selected_epoch=4,
                  shots=8, source_shots=8, weights="best_model_state_dict", training_steps=0,
                  osd_order=0, osd_policy="only_invalid_sectors", history_sha256=sha(path),
                  checkpoint_sha256=sha(run / "model.pt"), shot_bank_sha256=sha(run / "final_shots.npz"),
                  metrics={"syndrome_osd0": {"logical_error_rate": .125}})
    np.savez(output / "paired_shots.npz", syndrome_osd0_success=[1, 1, 1, 1, 1, 1, 1, 0])
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report))
    return path, report_path, report


def test_plot_replaces_only_with_matching_complete_standard_result(tmp_path):
    path, report_path, _ = run_and_report(tmp_path)
    metric, source, status = standard_metrics(path, report_index([tmp_path]))
    assert metric["logical_error_rate"] == .125  # Not the recorded residual .5.
    assert source == report_path and status == "posthoc"


def test_plot_does_not_fall_back_to_partial_old_epoch_or_other_checkpoint(tmp_path):
    path, report_path, report = run_and_report(tmp_path)
    assert standard_metrics(path, {}) is None
    for field, value in (("partial", True), ("selected_epoch", 3),
                         ("checkpoint_sha256", "old checkpoint"), ("shot_bank_sha256", "other shots"),
                         ("history_sha256", "other history")):
        report_path.write_text(json.dumps({**report, field: value}))
        assert standard_metrics(path, report_index([tmp_path])) is None


def test_native_standard_results_need_no_posthoc_substitution(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"config": {"osd": STANDARD},
                               "final": {"osd": {"logical_error_rate": .25}}}))
    metric, source, status = standard_metrics(path, {})
    assert metric["logical_error_rate"] == .25 and source == path and status == "native"


def test_lightweight_download_does_not_substitute_unverifiable_posthoc(tmp_path):
    path, _, _ = run_and_report(tmp_path)
    index = report_index([tmp_path])
    path.with_name("model.pt").unlink()
    assert standard_metrics(path, index) is None
    path.with_name("model.pt").write_bytes(b"selected checkpoint")
    path.with_name("final_shots.npz").unlink()
    assert standard_metrics(path, index) is None


def test_residual_report_only_requires_matching_original_inputs(tmp_path):
    path, report_path, report = run_and_report(tmp_path)
    history = json.loads(path.read_text())
    history['config']['osd'] = STANDARD
    path.write_text(json.dumps(history))
    # sha() may already have cached the fixture's original history.
    sha.cache_clear()
    report.update(config=history['config'], history_sha256=sha(path))
    report['metrics']['residual_osd0'] = {'samples': 8, 'logical_error_rate': .25}
    report_path.write_text(json.dumps(report))
    report_path.with_name('paired_shots.npz').unlink()
    metric, source, status = residual_metrics(path, report_index([tmp_path]))
    assert metric['logical_error_rate'] == .25 and source == report_path
    assert status == 'posthoc_report_inputs_hash_verified'
    for field, value in (('partial', True), ('history_sha256', 'wrong'),
                         ('selected_epoch', 3), ('source_shots', 16), ('checkpoint_sha256', 'wrong')):
        report_path.write_text(json.dumps({**report, field: value}))
        assert residual_metrics(path, report_index([tmp_path])) is None
    report_path.write_text(json.dumps(report))
    path.with_name('final_shots.npz').unlink()
    assert residual_metrics(path, report_index([tmp_path])) is None
