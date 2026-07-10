from scripts.plot_threshold import parse_csv
from scripts.pymatching_threshold import benchmark_point


def test_pymatching_has_no_failures_without_noise():
    row = benchmark_point(
        L=3,
        rounds=3,
        p=0.0,
        q=0.0,
        shots=64,
        batch_size=16,
        seed=123,
    )

    assert row["failures"] == 0
    assert row["accuracy"] == 1.0


def test_plotter_reads_pymatching_csv(tmp_path):
    path = tmp_path / "pymatching_test.csv"
    path.write_text(
        "decoder,L,rounds,p,q,shots,accuracy,failure\n"
        "pymatching,5,5,0.03,0.03,1000,0.9,0.1\n"
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
