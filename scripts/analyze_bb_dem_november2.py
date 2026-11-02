#!/usr/bin/env python3
"""Import six November 2 result folders; audit and plot frozen hierarchical/DEM results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_hierarchical_dem as hier
from scripts import analyze_bb_dem_october28 as base

STEM = "bb_dem_2026_11_02"
OUT = ROOT / "results/analysis"
DEM = ROOT / "results/bb/circuit/standard/dem_cnn"
HIER_JOBS = {60636229: "bb72", 60636226: "bb72", 60636230: "bb144",
             60640605: "bb72", 60641950: "bb144", 60641994: "bb72"}
DESTINATIONS = {f"resdir_{job}_hierarchical": DEM / "hierarchical" / code / f"resdir_{job}_hierarchical"
                for job, code in HIER_JOBS.items() if job != 60636229}
DESTINATIONS["resdir_60636624"] = DEM / "resumed_800_epochs/bb144/resdir_60636624"


def write_csv(suffix, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def training_audit(directory):
    """Audit complete resumed runs, retaining full phase history for plotting."""
    job = int(directory.name.removeprefix("resdir_"))
    assert (directory / "completed.txt").is_file() and not (directory / "failed.txt").exists()
    finals, training, validation = [], [], []
    histories = sorted(directory.glob("outputs/*/*/history.json"))
    assert len(histories) == 4
    for path in histories:
        h = base.read(path)
        c, f = h["config"], h["final"]
        p = c["noise_profile"]["base_error_rate"]
        index = (.001, .002, .004, .006).index(p)
        assert c["code"] == "bb144" and c["depth"] in (8, 12)
        assert c["architecture"] == "bb_dem_cnn" and c["noise_profile"]["name"] == "standard"
        assert [r["epoch"] for r in h["train"]] == list(range(800))
        assert all(math.isfinite(v) for r in h["train"] for v in r.values())
        for row in h["train"]:
            assert abs(row["total"]-row["syndrome"]-row["logical"]-.1*row["mechanism"]) < 1e-6
        for row in h["eval"]:
            base.audit_evaluation(row, 1024)
        base.audit_evaluation(f, 8192)
        best = max(h["eval"], key=lambda r:r["raw"]["logical_accuracy"])
        assert f["epoch"] == h["best_epoch"] == best["epoch"]
        assert h["best_raw_accuracy"] == best["raw"]["logical_accuracy"]
        start = h["phases"][-1]["start_epoch"]
        plan = base.read(directory / f"resume_exp_{index}.json")
        assert plan["completed_epochs"] == start and plan["additional_epochs"] == 800-start
        assert h["phases"][-1]["epochs"] == 800-start and h["phases"][-1]["schedule"] == "cosine_restart"
        if job == 60636624:
            assert c["depth"] == 12
            previous = next(base.read(q) for q in (DEM / "interrupted/bb144/resdir_60428753").glob("outputs/*/*/history.json")
                            if base.read(q)["config"]["noise_profile"]["base_error_rate"] == p)
            assert previous["config"] == c and len(previous["train"]) == start
            assert h["train"][:start] == previous["train"]
            assert h["eval"][:len(previous["eval"])] == previous["eval"]
        assert (directory / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
        for logpath in (directory / f"log_exp_{index}.txt", path.with_name("training_log.txt")):
            log = logpath.read_text()
            assert "Traceback (most recent call last)" not in log
            for label, key in (("Train", "train"), ("Validation", "eval")):
                records = [dict(epoch=int(epoch), **json.loads(value)) for epoch, value in
                           re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$", log, re.M)]
                assert records == [r for r in h[key] if r["epoch"] >= start]
            final_log = re.findall(r"^\[Final selected best\] (.+)$", log, re.M)
            assert len(final_log) == 1 and json.loads(final_log[0]) == f
        meta = dict(job=job, code=c["code"], depth=c["depth"], p=p, source=str(path.relative_to(ROOT)))
        row = dict(**meta, epochs=800, shots=8192, best_epoch=f["epoch"]+1, resumed=True,
                   resumed_from_epoch=start, additional_epochs=800-start,
                   prior_best_epoch=plan["prior_best_epoch"], phases=len(h["phases"]),
                   nn_ms_per_shot_amortized=f["nn_seconds"]*1000/f["shots"])
        for short, key in (("block", "logical_error_rate"), ("flagged", "flagged_failure"),
                           ("unflagged", "unflagged_failure"), ("observable", "observable_mismatch_rate")):
            failures, lo, hi = base.interval(f["raw"][key], 8192)
            row.update({f"{short}_rate": f["raw"][key], f"{short}_count": failures,
                        f"{short}_low": lo, f"{short}_high": hi})
        finals.append(row)
        training.extend(dict(**meta, **r) for r in h["train"])
        validation.extend(dict(**meta, epoch=r["epoch"]+1, **r["raw"]) for r in h["eval"])
    assert {r["p"] for r in finals} == {.001, .002, .004, .006}
    return finals, training, validation


def archive(move):
    manifest = OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move, "First import requires --archive"
        rows = []
        for name, destination in DESTINATIONS.items():
            source = ROOT / name
            assert source.is_dir() and not destination.exists(), source
            if name.endswith("_hierarchical"):
                hier.audit(source)
            else:
                training_audit(source)
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    rows.append(dict(old_path=str(p.relative_to(ROOT)),
                                     new_path=str((destination / p.relative_to(source)).relative_to(ROOT)),
                                     bytes=p.stat().st_size, sha256=base.sha(p)))
        write_csv("relocation", rows)
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    for name, destination in DESTINATIONS.items():
        source = ROOT / name
        if move and source.exists():
            assert not destination.exists(), "Refusing to merge existing imports"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    assert {str(p.relative_to(ROOT)) for d in DESTINATIONS.values() for p in d.rglob("*") if p.is_file()} == {r["new_path"] for r in rows}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and base.sha(p) == r["sha256"], p
    return dict(new_folders=len(DESTINATIONS), files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), hashes_verified=True)


def hierarchy_plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = base.COLORS
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for i, code in enumerate(("bb72", "bb144")):
        for depth, color in colors.items():
            group = sorted([r for r in rows if r["code"] == code and r["depth"] == depth], key=lambda r:r["p"])
            if not group:
                continue
            x = np.array([r["p"]*100 for r in group])
            for j, arm in enumerate(("raw", "hierarchical")):
                y = np.array([r[f"{arm}_rate"]*100 for r in group])
                low, high = [np.array([r[f"{arm}_{key}"]*100 for r in group]) for key in ("low", "high")]
                positive = y > 0
                # Break the line at a zero observation. A triangle marks its CI
                # upper bound, not an invented nonzero failure-rate estimate.
                axes[i, j].plot(x, np.where(positive, y, np.nan), color=color, lw=2, label=f"Depth {depth}")
                axes[i, j].errorbar(x[positive], y[positive], yerr=[y[positive]-low[positive], high[positive]-y[positive]],
                                   fmt="o", color=color, capsize=3)
                if (~positive).any():
                    axes[i, j].scatter(x[~positive], high[~positive], marker="v", facecolors="none", edgecolors=color, s=90)
                    axes[i, j].annotate("0 / 8,192\n95% upper = 0.0469%", (x[0], high[0]), xytext=(12, 10),
                                        textcoords="offset points", fontsize=9, color=color)
        for j, ax in enumerate(axes[i]):
            ax.set_title(f"{code.upper()} — {'Raw CNN' if j == 0 else 'Hierarchical residual OSD-0'}")
            ax.set_xlabel("Per-location error rate p (%)")
            ax.set_ylabel("Block failure (%)")
            ax.set_xticks([.1, .2, .4, .6])
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylim(-2, 103)
            else:
                ax.set_yscale("log")
                ax.set_ylim(.002, 150)
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Frozen DEM CNN weights · standard circuit noise · 8,192 shots per model\n"
                 "Raw / hierarchical paired within each model; 95% Wilson intervals; confidence 0.9", fontsize=14)
    fig.savefig(hier.PLOTS / "depth_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, code in enumerate(("bb72", "bb144")):
        for depth, color in colors.items():
            group = sorted([r for r in rows if r["code"] == code and r["depth"] == depth], key=lambda r:r["p"])
            if not group:
                continue
            x = [r["p"]*100 for r in group]
            axes[i, 0].plot(x, [r["global_call_fraction"]*100 for r in group], "o-", color=color, label=f"Depth {depth}")
            axes[i, 1].plot(x, [r["amortized_ms_per_shot"] for r in group], "o-", color=color, label=f"Depth {depth}")
        for j, ax in enumerate(axes[i]):
            ax.set_title(code.upper())
            ax.set_xlabel("Per-location error rate p (%)")
            ax.set_xticks([.1, .2, .4, .6])
            ax.set_ylabel("Shots requiring OSD (%)" if j == 0 else "Amortized decoding ms / shot")
            ax.set_ylim(bottom=0, top=105 if j == 0 else None)
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
    fig.suptitle("Hierarchical decoding: global completion use and measured cost\n"
                 "Batch 4; CNN + gating + CPU transfers + dense OSD; not single-shot latency", fontsize=14)
    fig.savefig(hier.PLOTS / "depth_diagnostics.png", dpi=160)
    plt.close(fig)


def report(hierarchy, raw, info):
    lines = ["# 2026-11-02: 계층형 DEM 평가 전체 결과와 BB144 D12 학습 완료", "",
             "새 폴더 6개를 보관했다. 계층형 평가 5개 job × 4개 모델을 기존 BB72 D4와 합쳐 총 24개 모델을 분석했다. "
             "별도로 BB144 D12의 네 모델이 총 800 epochs와 raw final 평가를 완료했다.", "",
             "## 계층형 디코딩: 최종 block failure (%)", "",
             "각 모델의 raw와 hierarchical은 동일한 새 circuit 8,192샷에서 평가했다. 기존 best weights를 사용하고 재학습하지 않았다. "
             "표의 p는 location별 오류율, 실패율은 전체 memory experiment의 syndrome 또는 observable 불일치다.", "",
             "| 코드 | 깊이 | p=0.1% | p=0.2% | p=0.4% | p=0.6% |",
             "|---|---:|---:|---:|---:|---:|"]
    for code in ("bb72", "bb144"):
        for depth in sorted({r["depth"] for r in hierarchy if r["code"] == code}):
            group = sorted([r for r in hierarchy if r["code"] == code and r["depth"] == depth], key=lambda r:r["p"])
            lines.append(f"| {code.upper()} | {depth} | " + " | ".join(f"{r['hierarchical_rate']*100:.3f}" for r in group)+" |")
    lines += ["", "- BB144 D4 p=0.1%는 0/8,192 failures이며 95% Wilson 상한은 0.0469%다. 실제 오류율이 0이라는 뜻은 아니다. D6는 2/8,192 (0.0244%); 작은 실패 건수만으로 깊이 우열을 단정하지 않는다.",
              "- BB72 p=0.2%/0.4%/0.6%의 점추정에서는 D12가 가장 낮지만, p=0.1%에서는 D8이 가장 낮다. 단조적인 depth 개선은 아니다.",
              "- 모든 24개 모델에서 hierarchical syndrome failure는 0이다. 높은 p에서 남는 큰 실패율은 syndrome-valid correction의 logical class 오류다.",
              "- Raw 대비 모든 점에서 실패율이 감소하거나 같고, 기록된 harmed는 전부 0이다. 고정 confidence=0.9의 local-only 단계만으로는 raw보다 낫지 않다.",
              "- BB144 D4/D6 p=0.4%/0.6%는 학습 완료 후에도 best epoch=5를 사용한다. Validation 동률 선택의 결과이며 이 고-p 결과를 epoch800 weights의 성능으로 해석하지 않는다.",
              "- Direct CNN+OSD 비교와 classical-only baseline은 이번 평가에 없다. 따라서 hierarchy가 일반 CNN+OSD보다 우수하거나 학습이 classical decoding 대비 얼마만큼 기여했는지는 알 수 없다.",
              "- BB144 D8/D12의 hierarchical 결과는 아직 없다. 아래 raw final 값과 혼합해 hierarchical 곡선으로 사용하지 않는다.",
              "", "## 모든 점의 raw 대비 개선 및 비용", "",
              "| 코드 | D | p (%) | Raw (%) | Hier (%) | Hier 실패 | 95% CI (%) | Rescued/Harmed | OSD (%) | ms/shot |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in hierarchy:
        lines.append(f"| {r['code']} | {r['depth']} | {r['p']*100:.1f} | {r['raw_rate']*100:.3f} | "
                     f"{r['hierarchical_rate']*100:.3f} | {r['hierarchical_failures']} | "
                     f"{r['hierarchical_low']*100:.4f}–{r['hierarchical_high']*100:.4f} | "
                     f"{r['rescued']}/{r['harmed']} | {r['global_call_fraction']*100:.2f} | {r['amortized_ms_per_shot']:.2f} |")
    lines += ["", "BB72는 약 17–103 ms/shot, BB144는 약 624–1,769 ms/shot이다. 이는 batch4의 누적 "
              "CNN/gating/전송/dense OSD 시간÷샷 수이며 단일 샷 latency가 아니다. 원래 training final의 NN-only 시간과 직접 speedup을 비교하지 않는다. "
              "이 구현에서 BB144의 global completion 비용이 크므로 정확도 개선과 실행 비용을 함께 판단해야 한다.", "",
              "## BB144 D12: 800 epochs 완료, OSD 없는 raw final", "",
              "| p (%) | 추가 epochs | Best epoch | D4 raw (%) | D6 raw (%) | D8 raw (%) | D12 raw (%) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for p in (.001, .002, .004, .006):
        group = [next(r for r in raw if r["code"] == "bb144" and r["depth"] == d and r["p"] == p) for d in (4,6,8,12)]
        r = group[-1]
        lines.append(f"| {p*100:.1f} | {r['additional_epochs']} | {r['best_epoch']} | "+" | ".join(f"{v['block_rate']*100:.3f}" for v in group)+" |")
    lines += ["", "D12는 p 순서대로 10/2/13/5 epochs를 추가해 800에 도달했다. 선택된 best는 기존 재개 checkpoint에 이미 있던 "
              "739/769/768/5 epochs다. 마지막 재개의 효과로 best가 개선됐다는 결과는 아니며, 이전 미완료 run의 최종 평가가 확보된 것이다. "
              "D8/D12는 여러 cosine restart와 독립 sampler phase를 거친 재개 학습이므로 중단 없는 학습과 동일하지 않다.",
              "", "## 검증과 한계", "",
              f"- 새 {info['new_folders']}개 폴더, {info['files']}개 파일 ({info['bytes']:,} bytes)을 이동하고 SHA-256과 크기를 검증했다.",
              "- 계층형 metadata/원래 training config/best epoch, exit code, 최종 로그와 results.json, 중간 집계의 failure 수·Wilson CI·paired 표준오차를 검증했다.",
              f"- 계층형 전체 {info['hierarchical_models']}개 모델, {info['hierarchical_shots']:,}샷, {info['hierarchical_progress_rows']}개 진행 집계를 확인했다. source snapshot의 현재 파일 일치 여부는 CSV에 남겼다.",
              "- D12 이전 학습 prefix와 재개 계획, 전체 epoch 순서, train/eval 로그, earliest-best 선택 및 final 집계를 확인했다. 원본 데이터/weights는 수정하지 않았다.",
              "- 이번에도 `.npz`/`.pt`가 없다. 샷별 correction 재채점과 checkpoint hash 재검증은 불가능하다. Paired 수치는 저장된 집계의 내부 일관성을 확인한 것이다.",
              "- 같은 code/p의 depth들은 seed가 같아도 샷 파일이 없어 실제 동일 shot bank인지 검증할 수 없다. Depth 간 paired 유의성 검정을 하지 않았고, raw 학습 final과 hierarchical 평가의 raw도 다른 샷이다.",
              "", "## 플롯과 재생성", "",
              "[계층형 전체 depth 비교](../plots/bb/circuit/standard/dem_cnn/hierarchical/depth_comparison.png) · "
              "[호출률·시간](../plots/bb/circuit/standard/dem_cnn/hierarchical/depth_diagnostics.png)", "",
              "[Raw 32개 모델 overview](../plots/bb/circuit/standard/dem_cnn/overview.png) · "
              "[Validation](../plots/bb/circuit/standard/dem_cnn/validation.png) · "
              "[Logical loss](../plots/bb/circuit/standard/dem_cnn/logical_loss.png)", "",
              f"[계층형 CSV]({STEM}_hierarchical_summary.csv) · [Raw final CSV]({STEM}_raw_final.csv) · [Audit JSON]({STEM}_audit.json)", "",
              "`python scripts/analyze_bb_dem_november2.py`로 재생성한다. 학습이나 추론은 실행하지 않는다.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    info = archive(args.archive)
    hierarchy, metrics, progress, snapshots, jobs = [], [], [], [], []
    for job, code in HIER_JOBS.items():
        directory = DEM / "hierarchical" / code / f"resdir_{job}_hierarchical"
        rows, arms, logs, sources, elapsed = hier.audit(directory)
        hierarchy.extend(rows); metrics.extend(arms); progress.extend(logs)
        snapshots.extend(dict(job=job, **s) for s in sources)
        jobs.append(dict(job=job, code=code, depth=rows[0]["depth"], elapsed_seconds=elapsed))
    hierarchy.sort(key=lambda r:(r["code"], r["depth"], r["p"]))
    raw, train, validation, _, _ = base.analyze()
    for job in (60428756, 60636624):
        directory = DEM / "resumed_800_epochs/bb144" / f"resdir_{job}"
        finals, training, evaluations = training_audit(directory)
        raw.extend(finals); train.extend(training); validation.extend(evaluations)
        if job == 60636624:
            for path in sorted((directory / "source_snapshot").rglob("*.py")):
                relative = path.relative_to(directory / "source_snapshot")
                current = ROOT / relative
                snapshots.append(dict(job=job, file=str(relative), sha256=base.sha(path),
                                      matches_current=current.is_file() and base.sha(path)==base.sha(current)))
    for suffix, rows in (("hierarchical_summary", hierarchy), ("hierarchical_metrics", metrics),
                         ("hierarchical_progress", progress), ("raw_final", raw), ("jobs", jobs), ("snapshots", snapshots)):
        write_csv(suffix, rows)
    info.update(hierarchical_models=len(hierarchy), hierarchical_shots=sum(r["shots"] for r in hierarchy),
                hierarchical_progress_rows=len(progress), raw_final_models=len(raw),
                source_snapshot_files=len(snapshots), matching_source_snapshots=sum(r["matches_current"] for r in snapshots),
                imported_npz_files=sum(len(list(d.rglob("*.npz"))) for d in DESTINATIONS.values()),
                imported_checkpoint_files=sum(len(list(d.rglob("*.pt"))) for d in DESTINATIONS.values()),
                independent_shot_audit=False, direct_osd_comparison=False)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(info, indent=2)+"\n")
    hierarchy_plots(hierarchy)
    base.plots(raw, train, validation)
    report(hierarchy, raw, info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
