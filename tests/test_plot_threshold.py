from scripts.plot_threshold import (
    aggregate,
    discover_inputs,
    parse_csv,
    parse_log,
)


def test_plot_keeps_recurrent_and_cnn3d_curves_separate(tmp_path):
    template = (
        "Executed Command: python main.py --architecture={architecture} --L=5 "
        "--p=0.01 --channels 8 8 --depths 1 1 {gru_flags}--lr=0.001\n"
        "[Epoch 0] Loss: 0.1 | Accuracy: 0.9 (±0.01) | Eval Samples: 100\n"
    )
    cnn_log = tmp_path / "cnn.txt"
    cnn_log.write_text(template.format(architecture="cnn3d", gru_flags=""))
    recurrent_log = tmp_path / "recurrent.txt"
    recurrent_log.write_text(
        template.format(
            architecture="convgru",
            gru_flags="--gru_channels 12 --gru_layers 2 --gru_kernel_size 3 ",
        )
    )

    records = [parse_log(cnn_log, "final"), parse_log(recurrent_log, "final")]
    curves, rows = aggregate(
        [record for record in records if record is not None],
        "L_arch_decoder",
    )

    assert len(curves) == 2
    assert len(rows) == 2
    assert {row["architecture"] for row in rows} == {
        "cnn3d",
        "convgru-gc12-gl2-gk3",
    }


def test_plot_keeps_circuit_hybrid_separate_from_phenomenological(tmp_path):
    template = (
        "Executed Command: python main.py --architecture={architecture} "
        "--noise_model={noise_model} --L=5 --p=0.01 "
        "--measurement_error_rate=0.01 --channels 8 --depths 1 "
        "--gru_channels 8 --gru_layers 1 --gru_kernel_size 3\n"
        "[Epoch 0] Loss: 0.1 | Accuracy: 0.9 (±0.01) | Eval Samples: 100\n"
    )
    phenomenological = tmp_path / "phenomenological.txt"
    phenomenological.write_text(
        template.format(
            architecture="convgru",
            noise_model="phenomenological",
        )
    )
    circuit = tmp_path / "circuit.txt"
    circuit.write_text(
        template.format(
            architecture="convgru_mwpm",
            noise_model="circuit",
        )
    )

    records = [
        parse_log(phenomenological, "final"),
        parse_log(circuit, "final"),
    ]
    curves, rows = aggregate(
        [record for record in records if record is not None],
        "L_arch_decoder",
    )

    assert len(curves) == 2
    assert {row["noise_model"] for row in rows} == {
        "phenomenological",
        "circuit",
    }
    assert {row["architecture"] for row in rows} == {
        "convgru-gc8-gl1-gk3",
        "convgru_mwpm-gc8-gl1-gk3-mstandard",
    }


def test_final_metric_prefers_selected_best_independent_evaluation(tmp_path):
    log = tmp_path / "selected_best.txt"
    log.write_text(
        "Executed Command: python main.py --architecture=convgru_mwpm "
        "--noise_model=circuit --L=5 --p=0.01 --loss_fn=ce "
        "--hybrid_calibration_batches=256\n"
        "[Epoch 0] Loss: 0.20 | Accuracy: 0.91 (±0.01) | Eval Samples: 100\n"
        "[Epoch 1] Loss: 0.10 | Accuracy: 0.95 (±0.01) | Eval Samples: 100\n"
        "[Selected Best] Epoch: 0 | Accuracy: 0.925000 "
        "| MWPM Accuracy: 0.920000 | Net Gain: +0.005000 "
        "| Eval Samples: 2048 | Recommended: hybrid\n"
    )

    final = parse_log(log, "final")
    best = parse_log(log, "best")

    assert final is not None
    assert final.epoch == 0
    assert final.loss == 0.20
    assert final.accuracy == 0.925
    assert final.eval_samples == 2048
    assert best is not None
    assert best.epoch == 1
    assert best.accuracy == 0.95
    assert best.eval_samples == 100


def test_plot_separates_unsafe_dynamic_and_ce_gated_hybrid_runs(tmp_path):
    prefix = (
        "Executed Command: python main.py --architecture=convgru_mwpm "
        "--noise_model=circuit --L=5 --p=0.01 --channels 8 --depths 1 "
        "--gru_channels 8 --gru_layers 1 --gru_kernel_size 3 "
    )
    dynamic_log = tmp_path / "dynamic.txt"
    dynamic_log.write_text(
        prefix
        + "--loss_fn=dynamic\n"
        + "[Epoch 0] Loss: 0.1 | Accuracy: 0.80 (±0.01) | Eval Samples: 100\n"
    )
    gated_log = tmp_path / "ce_gated.txt"
    gated_log.write_text(
        prefix
        + "--loss_fn=ce --hybrid_calibration_batches=256\n"
        + "[Epoch 0] Loss: 0.1 | Accuracy: 0.90 (±0.01) | Eval Samples: 100\n"
        + "[Selected Best] Epoch: 0 | Accuracy: 0.91 "
        + "| Eval Samples: 1000 | Recommended: hybrid\n"
    )

    records = [parse_log(dynamic_log, "final"), parse_log(gated_log, "final")]
    curves, rows = aggregate(
        [record for record in records if record is not None],
        "L_arch_decoder",
    )

    assert len(curves) == 2
    assert len(rows) == 2
    assert {row["architecture"] for row in rows} == {
        "convgru_mwpm-gc8-gl1-gk3-mstandard-lossdynamic-gatelegacy",
        "convgru_mwpm-gc8-gl1-gk3-mstandard-lossce-gatecal256",
    }
    assert {row["num_runs"] for row in rows} == {1}
    assert any("loss=dynamic gate=legacy" in label for label in curves)
    assert any("loss=ce gate=cal256" in label for label in curves)


def test_plotter_reads_pymatching_csv(tmp_path):
    path = tmp_path / "circuit_pymatching_standard.csv"
    path.write_text(
        "decoder,L,rounds,p,q,shots,accuracy,failure\n"
        "pymatching,5,5,0.01,0.01,1000,0.9,0.1\n"
    )

    records = parse_csv(path)

    assert len(records) == 1
    assert records[0].decoder == "pymatching"
    assert records[0].eval_samples == 1000
    assert records[0].L == 5


def test_plotter_reuses_aggregated_threshold_csv(tmp_path):
    path = tmp_path / "threshold.csv"
    path.write_text(
        "label,p,failure,accuracy,eval_samples,num_runs\n"
        "L=11,0.03,0.2,0.8,65536,1\n"
    )

    records = parse_csv(path)

    assert len(records) == 1
    assert records[0].decoder == "neural"
    assert records[0].L == 11


def test_plotter_preserves_optimization_plateau_status(tmp_path):
    path = tmp_path / "threshold.csv"
    path.write_text(
        "label,p,failure,accuracy,eval_samples,num_runs,training_status\n"
        "L=13,0.017,0.72,0.28,65536,1,optimization_plateau\n"
    )

    records = parse_csv(path)
    _, rows = aggregate(records, "L_arch_decoder")

    assert records[0].training_status == "optimization_plateau"
    assert rows[0]["training_status"] == "optimization_plateau"


def test_discover_inputs_finds_current_circuit_pymatching_csv(tmp_path):
    path = tmp_path / "circuit_pymatching_standard.csv"
    path.write_text("decoder,L,p,accuracy\n")

    assert discover_inputs([tmp_path]) == [path]
