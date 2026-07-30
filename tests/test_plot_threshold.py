from scripts.plot_threshold import aggregate, parse_log


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
