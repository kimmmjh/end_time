#!/usr/bin/env python3
"""Archive and audit imported hierarchical DEM evaluation; regenerate plots without inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_hierarchical_analysis_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bb_hierarchical_analysis_cache")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis"
STEM = "bb_hierarchical_dem_2026_11_01"
JOB = "resdir_60636229_hierarchical"
ARCHIVE = ROOT / "results/bb/circuit/standard/dem_cnn/hierarchical/bb72" / JOB
PLOTS = ROOT / "results/plots/bb/circuit/standard/dem_cnn/hierarchical"
ARMS = ("raw", "local", "hierarchical")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count(rate, n):
    value = round(rate * n)
    assert 0 <= value <= n and math.isclose(value / n, rate, abs_tol=1e-12)
    return value


def wilson(failures, n):
    p, z = failures / n, 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def verify_result(result):
    n = result["shots"]
    for arm in ARMS:
        m = result["metrics"][arm]
        assert m["shots"] == n
        assert count(m["block_failure_rate"], n) == m["failures"]
        flagged = count(m["syndrome_failure_rate"], n)
        unflagged = count(m["unflagged_failure_rate"], n)
        observable = count(m["observable_mismatch_rate"], n)
        assert flagged + unflagged == m["failures"]
        assert unflagged <= observable <= m["failures"]
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in
                   zip(m["block_failure_wilson95"], wilson(m["failures"], n)))
    h = result["metrics"]["hierarchical"]
    assert h["syndrome_failure_rate"] == 0
    assert h["observable_mismatch_rate"] == h["block_failure_rate"]
    paired = result["hierarchical_vs_raw"]
    rescued, harmed = paired["rescued"], paired["harmed"]
    raw_fails = result["metrics"]["raw"]["failures"]
    assert paired["shots"] == n and 0 <= rescued <= raw_fails and 0 <= harmed <= n-raw_fails
    assert raw_fails - rescued + harmed == h["failures"]
    assert math.isclose(paired["accuracy_gain"], (rescued-harmed)/n, abs_tol=1e-12)
    se = math.sqrt(((rescued+harmed)/n - paired["accuracy_gain"]**2)/(n-1))
    assert math.isclose(paired["paired_se"], se, abs_tol=1e-12)
    assert result["global_call_fraction"] == result["metrics"]["local"]["syndrome_failure_rate"]
    count(result["global_call_fraction"], n)
    assert result["hierarchical_seconds"] > 0 and result["direct_osd_extra_seconds"] == 0


def audit(directory):
    assert (directory / "completed.txt").is_file() and not (directory / "failed.txt").exists()
    job_metadata = (directory / "job_metadata.txt").read_text()
    job = int(re.search(r"^Job ID: (\d+)$", job_metadata, re.M).group(1))
    array, task = map(int, re.search(r"^Array job/task: (\d+)/(\d+)$", job_metadata, re.M).groups())
    assert array == 60636226 and task in (0, 1, 2, 3, 4, 6)
    code = "bb144" if task % 2 else "bb72"
    depth = (4, 6, 8, 12)[task // 2]
    start = datetime.fromisoformat(job_metadata.split("Started: ")[1].splitlines()[0].replace("Z", "+00:00"))
    finish = datetime.fromisoformat((directory / "completed.txt").read_text().split("Finished: ")[1].strip().replace("Z", "+00:00"))
    manifest = read(directory / "source_manifest.json")
    rows, long_rows, progress, snapshots = [], [], [], []
    for path in sorted((directory / "source_snapshot").rglob("*.py")):
        relative = path.relative_to(directory / "source_snapshot")
        current = ROOT / relative
        snapshots.append(dict(file=str(relative), sha256=sha(path),
                              matches_current=current.is_file() and sha(current) == sha(path)))
    assert len(list(directory.glob("experiment_*"))) == 4
    for i, p in enumerate((.001, .002, .004, .006)):
        experiment = directory / f"experiment_{i}"
        metadata, result = read(experiment / "metadata.json"), read(experiment / "results.json")
        c = metadata["config"]
        assert c["code"] == code and c["depth"] == depth and c["width"] == 64
        assert c["rounds"] == (6 if code == "bb72" else 12)
        assert c["noise_profile"]["name"] == "standard" and c["noise_profile"]["base_error_rate"] == p
        assert metadata["format"] == "bb_hierarchical_dem_eval_v1"
        assert metadata["backend"] == "direct_hard_centered_osd0_v1"
        assert metadata["confidence_threshold"] == .9 and metadata["active_candidates_only"]
        assert metadata["sampling"] == "fresh_exact_circuit" and not metadata["compare_direct_osd"]
        assert result["status"] == "complete" and result["shots"] == metadata["requested_shots"] == 8192
        assert metadata["source"]["weights"] == "best"
        assert (directory / f"exit_code_exp_{i}.txt").read_text().strip() == "0"
        entry = next(e for e in manifest["experiments"] if e["task"] == task and e["index"] == i)
        assert all(c[k] == v for k, v in entry["expected_config"].items())
        history = read(ROOT / entry["imported_run"] / "history.json")
        assert history["config"] == c
        assert metadata["source"]["selected_epoch"] == history["best_epoch"] + 1
        command = shlex.split((directory / f"command_exp_{i}.txt").read_text())
        for option, value in (("--index", str(i)), ("--shots", "8192"), ("--confidence", "0.9"), ("--weights", "best")):
            assert command[command.index(option)+1] == value
        log = (directory / f"log_exp_{i}.txt").read_text()
        assert "Traceback (most recent call last)" not in log
        logs = [json.loads(line) for line in log.splitlines() if line.startswith('{"status":')]
        assert logs[-1] == result
        assert [r["shots"] for r in logs] == list(range(64, 8193, 64))
        for r in logs:
            verify_result(r)
            assert r["status"] == ("complete" if r["shots"] == 8192 else "running")
            progress.append(dict(job=job, code=code, depth=depth, p=p, shots=r["shots"],
                                 raw_failures=r["metrics"]["raw"]["failures"],
                                 hierarchical_failures=r["metrics"]["hierarchical"]["failures"],
                                 hierarchical_seconds=r["hierarchical_seconds"]))
        verify_result(result)
        row = dict(job=job, code=c["code"], depth=c["depth"], p=p,
                   confidence=metadata["confidence_threshold"],
                   selected_epoch=metadata["source"]["selected_epoch"],
                   source_checkpoint_sha256=metadata["source"]["sha256"],
                   shot_files=len(list(experiment.rglob("*.npz"))),
                   **result["hierarchical_vs_raw"],
                   global_call_fraction=result["global_call_fraction"],
                   mean_residual_weight=result["mean_residual_weight"], mean_local_flips=result["mean_local_flips"],
                   hierarchical_seconds=result["hierarchical_seconds"],
                   amortized_ms_per_shot=1000*result["hierarchical_seconds"]/result["shots"],
                   prior_unpaired_raw_rate=history["final"]["raw"]["logical_error_rate"],
                   audit_level="metadata_and_logs_only")
        for arm in ARMS:
            m = result["metrics"][arm]
            row.update({f"{arm}_failures": m["failures"], f"{arm}_rate": m["block_failure_rate"],
                        f"{arm}_low": m["block_failure_wilson95"][0], f"{arm}_high": m["block_failure_wilson95"][1],
                        f"{arm}_syndrome_rate": m["syndrome_failure_rate"]})
            long_rows.append(dict(job=job, code=c["code"], depth=c["depth"], p=p, arm=arm,
                                  **{k:v for k,v in m.items() if k != "block_failure_wilson95"},
                                  ci_low=m["block_failure_wilson95"][0], ci_high=m["block_failure_wilson95"][1]))
        row["raw_to_hierarchical_ratio"] = (row["raw_rate"]/row["hierarchical_rate"]
                                            if row["hierarchical_rate"] else None)
        rows.append(row)
    return rows, long_rows, progress, snapshots, (finish-start).total_seconds()


def relocate(move):
    source = ROOT / JOB
    manifest = OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move and source.is_dir() and not ARCHIVE.exists(), "First import requires --archive and a new destination"
        audit(source)  # Validate before relocating the source directory.
        rows = [dict(old_path=str(p.relative_to(ROOT)),
                     new_path=str((ARCHIVE / p.relative_to(source)).relative_to(ROOT)),
                     bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(source.rglob("*")) if p.is_file()]
        write_csv("relocation", rows)
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if move and source.exists():
        assert not ARCHIVE.exists(), "Refusing to merge duplicate imports"
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        source.rename(ARCHIVE)
    actual = {str(p.relative_to(ROOT)) for p in ARCHIVE.rglob("*") if p.is_file()}
    assert actual == {r["new_path"] for r in rows}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"], p
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), hashes_verified=True)


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22, "figure.dpi": 140})
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.array([r["p"]*100 for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = {"raw": "#3478c6", "local": "#d98116", "hierarchical": "#12633c"}
    labels = {"raw": "Raw CNN", "local": "Local proposals only (confidence 0.9)", "hierarchical": "CNN + hierarchical residual OSD-0"}
    for arm in ARMS:
        y = np.array([r[f"{arm}_rate"]*100 for r in rows])
        err = np.array([[r[f"{arm}_rate"]-r[f"{arm}_low"] for r in rows],
                        [r[f"{arm}_high"]-r[f"{arm}_rate"] for r in rows]])*100
        for ax in axes:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=4, lw=2, color=colors[arm],
                        linestyle="--" if arm == "local" else "-", label=labels[arm])
    axes[0].set_ylim(-2, 106)
    axes[0].set_title("Full failure range")
    axes[1].set_yscale("log")
    axes[1].set_ylim(.04, 150)
    axes[1].set_title("Log scale: low-noise improvement")
    for ax in axes:
        ax.set_xlabel("Per-location error rate p (%)")
        ax.set_ylabel("Block failure (%)")
        ax.set_xticks(x)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("BB72 · depth 4 · standard circuit noise · 6 noisy rounds\n"
                 "8,192 paired shots per p; 95% Wilson intervals; no retraining", fontsize=14)
    fig.savefig(PLOTS / "overview.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    values = ([r["global_call_fraction"]*100 for r in rows],
              [r["mean_residual_weight"] for r in rows],
              [r["amortized_ms_per_shot"] for r in rows])
    titles = ("Global decoder use", "Remaining detector events", "Measured decoding cost")
    units = ("Shots requiring OSD (%)", "Mean residual syndrome weight", "Amortized ms / shot (batch 4)")
    for ax, y, title, unit in zip(axes, values, titles, units):
        ax.plot(x, y, "o-", color="#12633c", lw=2)
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel("Per-location error rate p (%)")
        ax.set_xticks(x)
        ax.set_ylim(0, max(y)*1.22)
    axes[0].set_ylim(0, 110)
    axes[0].set_yticks(range(0, 101, 20))
    fig.suptitle("BB72 depth 4 · confidence 0.9\nCNN + gating + transfers + OSD timing; not single-shot latency", fontsize=14)
    fig.savefig(PLOTS / "diagnostics.png")
    plt.close(fig)


def report(rows, audit_info):
    lines = ["# 2026-11-01: BB72 depth 4 hierarchical DEM CNN 평가", "",
             "기존 DEM CNN best checkpoint를 재학습 없이 사용했다. Standard circuit noise, noisy rounds=6, "
             "width=64, local confidence=0.9이며 p별 raw/local/hierarchical은 동일한 새 circuit 8,192샷을 사용했다.", "",
             "## 최종 block failure", "",
             "전체 memory experiment에서 syndrome 또는 observable이 틀린 샷의 비율이다. Round당·논리 큐빗당 오류율이 아니다.", "",
             "| p (%) | Raw (%) | Local only (%) | Hierarchical (%) | Hierarchical 실패/8192 | 95% Wilson CI (%) | Raw/Hier 비율 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['p']*100:.1f} | {r['raw_rate']*100:.3f} | {r['local_rate']*100:.3f} | "
                     f"{r['hierarchical_rate']*100:.3f} | {r['hierarchical_failures']} | "
                     f"{r['hierarchical_low']*100:.3f}–{r['hierarchical_high']*100:.3f} | {r['raw_to_hierarchical_ratio']:.2f}× |")
    lines += ["", "## 해석", "",
              "- p=0.1%에서 1,374→9 failures, p=0.2%에서 4,426→260 failures로 감소했다. 기존 weights에 residual decoding을 추가하는 것만으로 raw 대비 큰 개선이 관측됐다.",
              "- 모든 p에서 hierarchical syndrome failure는 0이다. 남은 실패는 syndrome을 만족하지만 logical class가 틀린 경우다. p=0.4%/0.6%에서는 여전히 47.668%/95.166%가 실패한다.",
              "- Confidence 0.9의 local proposal만 적용하면 모든 p에서 raw보다 낫지 않다. Global completion이 최종 성능 개선에 핵심적이며, local 단계의 독립 성능이 좋아졌다는 결과는 아니다.",
              "- `compare_direct_osd=false`이므로 일반 CNN+OSD-0 대비 계층화의 추가 이득은 측정하지 않았다. Classical-only baseline도 없어 학습된 reliability의 독립적 기여를 확정할 수 없다.",
              "- 동일 weight가 그대로 사용되는 추론 실험이며 새 prediction architecture를 학습한 결과가 아니다. 첫 low-p 점도 실패 9건이므로 점추정 비율만으로 일반화하지 않고 CI를 함께 본다.",
              "- 이번에 받은 것은 array task 0 (BB72 D4) 하나뿐이다. 다른 깊이/BB144 hierarchical 결과나 threshold를 추정하지 않는다.",
              "", "## Paired 개선 및 비용", "",
              "| p (%) | Rescued | Harmed | 정확도 증가 (pp) | OSD 호출 (%) | 평균 residual weight | 평균 local flips | ms/shot |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['p']*100:.1f} | {r['rescued']} | {r['harmed']} | {r['accuracy_gain']*100:.3f} | "
                     f"{r['global_call_fraction']*100:.3f} | {r['mean_residual_weight']:.3f} | {r['mean_local_flips']:.3f} | {r['amortized_ms_per_shot']:.2f} |")
    lines += ["", "Rescued/harmed는 저장된 paired 집계다. 네 p 모두 harmed=0이지만, 다른 샷에서도 항상 보존된다는 보장은 아니다. "
              "ms/shot은 batch4 누적 decoding 시간/샷 수이며 단일 샷 latency가 아니다. CNN·gating·전송·OSD를 포함하고 sampling/scoring/파일 IO는 제외한다. "
              "별도 raw-only timing이 없으므로 OSD 추가 시간이나 speedup은 추정하지 않는다.", "",
              f"작업 로그 경과 시간은 {audit_info['job_elapsed_seconds']/60:.2f}분이며 네 p worker는 병렬로 실행됐다.",
              "", "## 검증과 보관", "",
              f"- `{ARCHIVE.relative_to(ROOT)}`로 {audit_info['files']}개 파일을 이동하고 전체 SHA-256/크기를 재검증했다.",
              "- 네 worker의 exit code=0, 완료 표시, 최종 로그와 results.json 일치, metadata와 원본 training history의 config/best epoch 일치를 확인했다.",
              "- 512개 중간/최종 로그 집계에서 failure 수, syndrome/logical 분해, Wilson CI, paired gain 및 표준오차를 재계산했다.",
              f"- 소스 snapshot {audit_info['source_snapshots']}개 중 현재 소스와 일치: {audit_info['matching_source_snapshots']}개.",
              "- `.pt`와 `.npz`가 다운로드되지 않았다. 저장된 checkpoint SHA-256은 서버 기록이며 여기서 weights hash를 재검증한 것은 아니다. 샷별 pairing, correction parity 및 logical 결과의 독립 재채점은 수행하지 못했다.",
              "- 이전 학습 final raw와 이번 raw는 새 샷이므로 paired 비교가 아니다. 별도 CSV에 이전 값만 참고용으로 남겼다.", "",
              "## 파일", "",
              "[성능 플롯](../plots/bb/circuit/standard/dem_cnn/hierarchical/overview.png) · "
              "[호출률·잔여 syndrome·시간](../plots/bb/circuit/standard/dem_cnn/hierarchical/diagnostics.png)", "",
              f"[요약 CSV]({STEM}_summary.csv) · [상세 metric CSV]({STEM}_metrics.csv) · [검증 JSON]({STEM}_audit.json)", "",
              "재생성: `python scripts/analyze_bb_hierarchical_dem.py` (학습·추론 없이 저장 결과만 사용).", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    archive_info = relocate(args.archive)
    rows, metrics, progress, snapshots, elapsed = audit(ARCHIVE)
    for suffix, data in (("summary", rows), ("metrics", metrics), ("progress", progress), ("snapshots", snapshots)):
        write_csv(suffix, data)
    info = dict(**archive_info, job_elapsed_seconds=elapsed, experiments=len(rows),
                total_shots=sum(r["shots"] for r in rows), progress_rows_verified=len(progress),
                source_snapshots=len(snapshots), matching_source_snapshots=sum(r["matches_current"] for r in snapshots),
                shot_files=sum(r["shot_files"] for r in rows), checkpoint_files=len(list(ARCHIVE.rglob("*.pt"))),
                independent_shot_audit=False, direct_osd_comparison=False,
                audit_level="metadata_and_logs_only")
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(info, indent=2)+"\n")
    plots(rows)
    report(rows, info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
