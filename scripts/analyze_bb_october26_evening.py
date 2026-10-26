#!/usr/bin/env python3
"""Import remaining depth 10/12 jobs and frozen depth 6/7/8 OSD reports.

No training or inference. --archive moves four root downloads with SHA-256
manifests; later runs audit the archived files and regenerate tables/plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_oct26_evening_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bb_oct26_evening_cache")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_bb_october26 import analyze, sha
from scripts.bb_cnn_plot_osd import interval, report_index, residual_metrics

OUT = ROOT / "results/analysis"
STEM = "bb_update_2026_10_26_evening"
ARCHIVE = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn"
PLOTS = ROOT / "results/plots/bb/code_capacity/tanner_cnn"
JOBS = ((60255257, "bb72", 10, 0), (60261485, "bb144", 10, 1), (60261486, "bb72", 12, 2))
OSD_DIR = ARCHIVE / "posthoc_osd0/depth678/resdir_60255150"
DESTINATIONS = {job: ARCHIVE / f"800_epochs/residual_osd0/{code}/resdir_{job}"
                for job, code, depth, task in JOBS}
DESTINATIONS[60255150] = OSD_DIR


def read(path):
    return json.loads(path.read_text())


def csv_read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_write(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def archive(move):
    manifest = OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move, "First import requires --archive"
        entries = []
        for job, dest in DESTINATIONS.items():
            src = ROOT / f"resdir_{job}"
            assert src.is_dir() and not dest.exists(), src
            entries.extend(dict(old_path=str(p.relative_to(ROOT)),
                                new_path=str((dest / p.relative_to(src)).relative_to(ROOT)),
                                bytes=p.stat().st_size, sha256=sha(p))
                           for p in sorted(src.rglob("*")) if p.is_file())
        csv_write("relocation", entries)
    entries = csv_read(manifest)
    for job, dest in DESTINATIONS.items():
        src = ROOT / f"resdir_{job}"
        if move and src.is_dir() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
    actual = {p for dest in DESTINATIONS.values() for p in dest.rglob("*") if p.is_file()}
    assert actual == {ROOT / r["new_path"] for r in entries}
    for r in entries:
        path = ROOT / r["new_path"]
        assert path.stat().st_size == int(r["bytes"]) and sha(path) == r["sha256"]
    return dict(files=len(entries), bytes=sum(int(r["bytes"]) for r in entries),
                relocation_hashes_verified=True)


def audit_posthoc():
    from scipy.stats import binomtest
    assert (OSD_DIR / "completed.txt").is_file() and not (OSD_DIR / "failed.txt").exists()
    for i in range(4):
        assert (OSD_DIR / f"exit_code_shard_{i}.txt").read_text().strip() == "0"
        assert "Traceback (most recent call last)" not in (OSD_DIR / f"log_shard_{i}.txt").read_text()
    original = csv_read(OUT / "bb_update_2026_10_24_cnn_final.csv")
    index = report_index([OSD_DIR])
    paths = sorted(OSD_DIR.rglob("report.json"))
    summaries = [r for p in OSD_DIR.rglob("summary_shard_*.json") for r in read(p)["runs"]]
    assert len(paths) == len(summaries) == 24
    rows = []
    for path in paths:
        report = read(path)
        c = report["config"]
        prior = next(r for r in original if r["code"] == c["code"] and int(r["depth"]) == c["cnn_depth"]
                     and float(r["p"]) == c["error_rate"] and int(r["seed"]) == c["seed"])
        history_path = ROOT / prior["source"]
        history = read(history_path)
        result = residual_metrics(history_path, index)
        assert result is not None and result[1] == path
        metric, _, status = result
        assert report["completed_epochs"] == len(history["train"]) == 800
        assert report["extra_bp_iterations"] == 0 and report["shots"] == 65536
        assert report["historical_reproduction"] == {
            m: dict(correction_mismatch_shots=0, success_mismatch_shots=0) for m in ("raw", "syndrome_osd0")}
        for name, filename in (("raw", "raw"), ("syndrome_osd0", "osd")):
            for key in ("logical_error_rate", "flagged_failure_rate", "unflagged_logical_failure_rate"):
                assert report["metrics"][name][key] == history["final"][filename][key]
        summary = next(s for s in summaries if Path(s["report"]).parent.name == path.parent.name)
        assert summary["metrics"] == report["metrics"] and summary["paired"] == report["paired"]
        for name, digest in report["implementation_sha256"].items():
            snapshot = list((OSD_DIR / "source_snapshot").rglob(name))
            assert len(snapshot) == 1 and sha(snapshot[0]) == digest
        row = dict(code=c["code"], p=c["error_rate"], depth=c["cnn_depth"], seed=c["seed"],
                   shots=report["shots"], best_epoch=report["selected_epoch"]+1,
                   source=str(history_path.relative_to(ROOT)), report=str(path.relative_to(ROOT)),
                   checkpoint_verified=True, source_bank_verified=True,
                   posthoc_shots_verified=status == "posthoc_shots_verified", audit_level=status)
        for name in ("raw", "residual_osd0", "syndrome_osd0"):
            m = report["metrics"][name]
            count, lo, hi = interval(m["logical_error_rate"], row["shots"])
            assert m["samples"] == row["shots"] and abs(m["logical_accuracy"]+m["logical_error_rate"]-1) < 1e-12
            assert abs(m["flagged_failure_rate"]+m["unflagged_logical_failure_rate"]-count/row["shots"]) < 1e-12
            row.update({f"{name}_failures": count, f"{name}_ler": count/row["shots"],
                        f"{name}_low": lo, f"{name}_high": hi})
        for comparison, base in (("standard_vs_residual", "residual_osd0"), ("standard_vs_raw", "raw")):
            pair = report["paired"][comparison]
            assert pair["rescued"]-pair["harmed"] == row[f"{base}_failures"]-row["syndrome_osd0_failures"]
            assert abs(pair["ler_reduction"]-(pair["rescued"]-pair["harmed"])/row["shots"]) < 1e-12
        pair = report["paired"]["standard_vs_residual"]
        discordant = pair["rescued"] + pair["harmed"]
        row.update(syndrome_rescued_vs_residual=pair["rescued"], syndrome_harmed_vs_residual=pair["harmed"],
                   reported_paired_p=binomtest(pair["rescued"], discordant).pvalue if discordant else 1.)
        rows.append(row)
    assert len({(r["code"], r["p"], r["depth"]) for r in rows}) == 24
    # Holm correction across the 24 reported paired OSD comparisons.
    maximum = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r["reported_paired_p"])):
        maximum = max(maximum, min(1., (len(rows)-rank)*row["reported_paired_p"]))
        row["reported_paired_p_holm"] = maximum
    rows.sort(key=lambda r: (r["code"], r["depth"], r["p"]))
    csv_write("posthoc_osd", rows)
    return rows


def combined_results(new, posthoc):
    rows = []
    for r in posthoc:
        row = dict(code=r["code"], p=r["p"], depth=r["depth"], shots=r["shots"],
                   source=r["source"], audit_level=r["audit_level"])
        for output, name in (("raw", "raw"), ("osd", "residual_osd0")):
            for field in ("failures", "ler", "low", "high"):
                row[f"{output}_{field}"] = r[f"{name}_{field}"]
        rows.append(row)
    for r in [*new, *csv_read(OUT / "bb_update_2026_10_26_cnn_final.csv")]:
        row = dict(code=r["code"], p=float(r["p"]), depth=int(r["depth"]), shots=int(r["shots"]),
                   source=r["source"], audit_level=r["audit_level"])
        for method in ("raw", "osd"):
            for field, old in (("failures", "failures"), ("ler", "ler"), ("low", "ci95_low"), ("high", "ci95_high")):
                row[f"{method}_{field}"] = float(r[f"{method}_{old}"])
        rows.append(row)
    rows.sort(key=lambda r: (r["code"], r["depth"], r["p"]))
    assert len(rows) == 40 and len({(r["code"],r["depth"],r["p"]) for r in rows}) == 40
    csv_write("depth_comparison", rows)
    return rows


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.plot_bb_cnn_all import COLORS, draw_point
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, code in enumerate(("bb72", "bb144")):
        for j, method in enumerate(("raw", "osd")):
            ax = axes[i, j]
            for depth in (6,7,8,10,12):
                group = [r for r in rows if r["code"] == code and r["depth"] == depth]
                color = COLORS[depth]
                ax.plot([r["p"]*100 for r in group], [r[f"{method}_ler"]*100 or np.nan for r in group],
                        color=color, label=f"Depth {depth}")
                for r in group:
                    draw_point(ax, dict(r, recovery=False), method, color, "o")
            ax.set(title=f'{code.upper()} | {"CNN alone" if method == "raw" else "CNN + residual OSD-0"}',
                   xlabel="Physical error rate p (%)", ylabel="Block LER (%)", yscale="log", xticks=[2,4,6,8])
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
    fig.suptitle("Depth 6/7/8/10/12: width 64, fresh 800 epochs, 65,536 final shots\n"
                 "95% Wilson intervals; hollow triangles = zero-failure upper bounds", fontsize=14)
    fig.savefig(PLOTS / "depth_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for i, code in enumerate(("bb72", "bb144")):
        for j, p in enumerate((.02,.04,.06,.08)):
            ax = axes[i,j]
            for depth in (8,10,12):
                r = next(r for r in rows if r["code"] == code and r["p"] == p and r["depth"] == depth)
                ev = read(ROOT / r["source"])["eval"]
                x = [e["epoch"]+1 for e in ev]
                y = np.convolve([e["raw"]["logical_error_rate"]*100 for e in ev], np.ones(5)/5, mode="valid")
                ax.plot(x[4:],y,color=COLORS[depth],label=f"Depth {depth}")
            ax.set(title=f"{code.upper()} p={p:.0%}", xlabel="Epoch", ylabel="Raw validation LER (%)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Depth 8/10/12 training: trailing mean of 5 validations (25 epochs); linear axes")
    fig.savefig(PLOTS / "depth1012_validation.png", dpi=160)
    # The initial D12 plateau otherwise hides the late-training differences.
    for ax in axes.flat:
        for line in ax.lines:
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            line.set_data(x[x >= 200], y[x >= 200])
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(200, 800)
    fig.suptitle("Depth 8/10/12: epochs 200–800; trailing mean of 5 validations; panel scales differ")
    fig.savefig(PLOTS / "depth1012_validation_late.png", dpi=160)
    plt.close(fig)


def report(rows, new, posthoc, audit):
    lines = ["# 2026-10-26 저녁: depth 10/12 완료 및 depth 6–8 residual OSD", "",
        "이번 결과는 code-capacity depolarizing Tanner CNN이며, circuit-level DEM CNN 결과가 아니다.", "",
        f"새 폴더 4개, {audit['files']}개 파일 ({audit['bytes']:,} bytes)을 이동 전후 SHA-256 검증하여 보관했다. "
        "Job 60255257=BB72 D10, 60261485=BB144 D10, 60261486=BB72 D12; "
        "기존 BB144 D12 job 60255255와 합쳐 depth 10/12의 16개 모델이 모두 완료됐다.", "",
        "새 학습 12개 모두 fresh 800 epochs, width 64, training 6,553,600 shots, "
        "validation 4,096 shots/5 epochs, final 65,536 shots. Raw validation으로 best를 선택했다. "
        "History와 train/final 로그, 종료 코드, source snapshot을 대조했다. "
        "D10/12 checkpoint와 final bank는 다운로드에 없으므로 depth 간 paired 검정은 하지 않았다.", "",
        "Job 60255150은 재학습 없이 기존 D6/7/8 24개 모델의 동일 final bank에 raw/residual/syndrome OSD를 비교했다. "
        "로컬 원본 checkpoint/bank/history의 SHA-256, config, best epoch, 전체 샷 수를 보고서와 대조했다. "
        "재평가 paired_shots.npz는 다운로드에 없어 재평가 aggregate와 paired count는 보고서 기록 기준이다. "
        "Raw 및 syndrome 결과는 이전 기록과 전부 같고, 보고된 shot별 재현 불일치도 모두 0이다.", "",
        "## 같은 residual OSD끼리 비교", "",
        "각 칸은 raw / residual OSD LER(%). 0은 65,536샷에서 실패 0건이라는 뜻이다.", "",
        "| 코드 | p | D6 | D7 | D8 | D10 | D12 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for code in ("bb72", "bb144"):
        for p in (.02,.04,.06,.08):
            group = [next(r for r in rows if r["code"]==code and r["p"]==p and r["depth"]==d) for d in (6,7,8,10,12)]
            lines.append(f"| {code.upper()} | {p:.0%} | " + " | ".join(f"{r['raw_ler']*100:.4f} / {r['osd_ler']*100:.4f}" for r in group) + " |")
    lines += ["", "## 해석", "",
        "BB144의 높은 p에서는 depth를 늘릴수록 raw와 residual OSD가 함께 개선된다. "
        "p=.08 D8→D12: raw 8.9600→5.2429% (41.5% 감소), residual 2.8046→2.2247% (20.7% 감소). "
        "p=.06에서는 raw 2.0126→1.0971% (45.5%), residual 0.3021→0.2701% (10.6%) 감소다. "
        "BB144 p=.04는 D10 residual 13 failures, D12 16 failures로 단조 개선은 아니다.", "",
        "BB72는 D8→D10에서 raw 개선이 있지만 D10→D12는 거의 정체다. "
        "p=.08: raw 12.1292→12.0895%, residual 10.6644→10.6613% (6,989→6,987 failures). "
        "p=.06 residual은 D10 2,424→D12 2,464 failures로 오히려 조금 증가한다. "
        "이 작은 차이만으로 우열을 확정할 수 없다.", "",
        "BB144 p=.02 residual은 D6/7/8에서 각 0 failures, D10에서 1, D12에서 2이다. "
        "실패 0건도 진짜 LER=0이나 100% 성공 보장은 아니다. 95% Wilson 상한은 약 0.005861%이고, "
        "플롯은 그 위치에 빈 역삼각형을 표시한다.", "",
        f"Residual와 syndrome의 보고된 paired count 24개 비교 중 Holm 보정 후 p<.05인 것은 "
        f"{sum(r['reported_paired_p_holm'] < .05 for r in posthoc)}개다. "
        "현재 결과에서 어느 OSD 방식이 일관되게 우월하다는 근거는 없다. "
        "OSD별 곡선을 구분하고 깊이 비교에서는 residual끼리만 비교했다.", "",
        "D8/D10/D12 파라미터 수는 802,564 / 1,016,324 / 1,230,084. "
        "D12는 D10보다 21.0% 크지만 BB72의 추가 이득은 뚜렷하지 않다. "
        "단일 seed와 제한된 final 샷에 대한 결과이며, 낮은 p의 best 선택에는 validation 해상도 한계도 있다.", "",
        "D12는 특히 낮은 p에서 학습 초반 높은 validation LER의 정체 구간을 거친다. "
        "학습 곡선은 전체 구간과 200–800 epoch 확대 그림을 함께 제공한다. "
        "BB144 D10 p=.02의 validation 최저 실패 수 0은 42회 동률이며 첫 epoch 250을 선택했다. "
        "낮은 p에서 checkpoint를 세밀하게 구분하기에는 validation 4,096샷이 부족할 수 있다.", "",
        "## 학습 선택", "", "| 코드 | depth | p | best epoch | best validation 동률 수 | raw 실패 | residual 실패 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(new, key=lambda r:(r["code"],r["depth"],r["p"])):
        lines.append(f"| {r['code']} | {r['depth']} | {r['p']:.0%} | {r['best_epoch']} | {r['best_validation_ties']} | {r['raw_failures']} | {r['osd_failures']} |")
    lines += ["", "## 산출물", "",
        "[최신 전체 플롯](../plots/bb/code_capacity/tanner_cnn/overview.png) · "
        "[D6–12 동일 residual 비교](../plots/bb/code_capacity/tanner_cnn/depth_comparison.png) · "
        "[D8/10/12 validation](../plots/bb/code_capacity/tanner_cnn/depth1012_validation.png) · "
        "[200–800 epoch 확대](../plots/bb/code_capacity/tanner_cnn/depth1012_validation_late.png)", "",
        f"[새 모델 수치]({STEM}_cnn_final.csv) · [OSD 재평가]({STEM}_posthoc_osd.csv) · "
        f"[40개 비교 수치]({STEM}_depth_comparison.csv) · [검증]({STEM}_audit.json) · [이동 기록]({STEM}_relocation.csv)", "",
        "```bash", "python scripts/analyze_bb_october26_evening.py", "python scripts/plot_bb_cnn_all.py", "```", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    audit = archive(args.archive)
    new, jobs = [], []
    for job, code, depth, task in JOBS:
        finals, _, checks = analyze(DESTINATIONS[job], code, depth, job, task, write_results=False)
        new.extend(finals)
        jobs.append(dict(job=job, **checks))
    csv_write("cnn_final", new)
    posthoc = audit_posthoc()
    audit.update(jobs=jobs, new_trained_models=12, posthoc_models=24,
                 posthoc_input_hashes_verified=True, posthoc_shots_verified=False,
                 training_steps_added=0, inference_steps_added=0)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    rows = combined_results(new, posthoc)
    plots(rows)
    report(rows, new, posthoc, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
