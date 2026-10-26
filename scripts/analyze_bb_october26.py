#!/usr/bin/env python3
"""Archive the lightweight BB144 depth-12 download and audit reported metrics.

Use --archive on first import. No inference or shot-level rescoring is performed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_october26_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bb_october26_cache")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bb_cnn_plot_osd import interval

OUT = ROOT / "results/analysis"
STEM = "bb_update_2026_10_26"
SOURCE = ROOT / "resdir_60255255"
DEST = ROOT / "results/bb/code_capacity/depolarizing/tanner_cnn/800_epochs/residual_osd0/bb144/resdir_60255255"
PLOTS = ROOT / "results/plots/bb/code_capacity/tanner_cnn"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def archive(move):
    manifest = OUT / f"{STEM}_relocation.csv"
    if move and not manifest.exists():
        assert SOURCE.is_dir() and not DEST.exists()
        rows = [dict(old_path=str(p.relative_to(ROOT)),
                     new_path=str((DEST / p.relative_to(SOURCE)).relative_to(ROOT)),
                     bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(SOURCE.rglob("*")) if p.is_file()]
        write_csv("relocation", rows)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        SOURCE.rename(DEST)
    rows = list(csv.DictReader(manifest.open()))
    assert {ROOT / r["new_path"] for r in rows} == {p for p in DEST.rglob("*") if p.is_file()}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"], p
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), relocation_hashes_verified=True)


def analyze(dest=DEST, code="bb144", depth=12, job=60255255, task=3, *, write_results=True):
    DEST = dest
    assert (DEST / "completed.txt").is_file() and not (DEST / "failed.txt").exists()
    metadata = (DEST / "job_metadata.txt").read_text()
    assert f"Array job/task: 60255255/{task}" in metadata
    start = dt.datetime.fromisoformat(metadata.split("Started: ")[1].splitlines()[0].replace("Z", "+00:00"))
    finish = dt.datetime.fromisoformat((DEST / "completed.txt").read_text().split("Finished: ")[1].strip().replace("Z", "+00:00"))
    snapshots = list((DEST / "source_snapshot").rglob("*.py"))
    for p in snapshots:
        assert sha(p) == sha(ROOT / p.relative_to(DEST / "source_snapshot")), p
    specs = list(csv.DictReader((DEST / "experiments.tsv").open(), delimiter="\t"))
    paths = sorted(DEST.glob("outputs/*/*/history.json"))
    assert len(paths) == len(specs) == 4
    finals, training, validation, histories = [], [], [], {}
    for path in paths:
        h = read(path)
        c, f = h["config"], h["final"]
        assert c["code"] == code and c["cnn_depth"] == depth and c["cnn_width"] == 64
        assert c["noise_model"] == "capacity" and c["channel"] == "depolarizing"
        assert c["osd"] == "direct_hard_centered_osd0_v1" and c["osd_policy"] == "only_invalid_sectors"
        assert c["parameter_count"] == 1230084 + (depth-12)*106880 and c["checkpoint_selection"] == "raw_logical_accuracy"
        assert c["seed"] == int(code[2:])*100000 + 1000 + round(c["error_rate"]*1000)
        assert [t["epoch"] for t in h["train"]] == list(range(800))
        assert [e["epoch"] for e in h["eval"]] == list(range(4, 800, 5))
        assert len(h["phases"]) == 1
        phase = h["phases"][0]
        for key, val in dict(start_epoch=0, epochs=800, learning_rate=.0003,
                             batches=128, batch_size=64, eval_batches=64,
                             eval_every=5, final_eval_batches=1024).items():
            assert phase[key] == val
        best = max(h["eval"], key=lambda e: e["raw"]["logical_accuracy"])
        assert best["epoch"] == h["best_epoch"] == f["epoch"] and f["shots"] == 65536
        spec = next(s for s in specs if f"--seed={c['seed']}" in s["arguments"])
        index = int(spec["experiment_index"])
        args = dict(a[2:].split("=", 1) for a in shlex.split(spec["arguments"]) if a.startswith("--") and "=" in a)
        assert args["bb_cnn_depth"] == str(depth) and args["bb_cnn_osd"] == "residual"
        assert float(args["p"]) == c["error_rate"] and args["code"] == c["code"]
        assert (DEST / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
        for log_path in (DEST / f"log_exp_{index}.txt", path.with_name("training_log.txt")):
            log = log_path.read_text()
            assert "Traceback (most recent call last)" not in log
            logged = re.findall(r'^\[Train\] Epoch (\d+) \| (.+)$', log, re.M)
            assert [{"epoch": int(e), **json.loads(v)} for e, v in logged] == h["train"]
            final_line = next(line for line in log.splitlines() if line.startswith("[Selected best by raw LER]"))
            assert f"Epoch {f['epoch']} |" in final_line
            assert f"CNN LER {f['raw']['logical_error_rate']:.8f}" in final_line
            assert f"CNN+OSD-0 LER {f['osd']['logical_error_rate']:.8f}" in final_line
        meta = dict(job=job, experiment=index, code=c["code"], p=c["error_rate"], depth=depth,
                    seed=c["seed"], source=str(path.relative_to(ROOT)))
        r = dict(**meta, width=64, parameters=c["parameter_count"], epochs=800,
                 training_shots=800*8192, shots=f["shots"], best_epoch=f["epoch"]+1,
                 osd_convention=c["osd"], schedule="fresh 800 cosine",
                 audit_level="history_and_logs_only", checkpoint_verified=False, shot_bank_verified=False,
                 osd_call_fraction=f["osd_call_fraction"],
                 nn_us_per_shot_amortized=f["nn_batch_seconds"]*1e6/f["shots"],
                 osd_us_per_shot_amortized=f["osd_with_transfer_seconds"]*1e6/f["shots"],
                 best_validation_ties=sum(e["raw"]["logical_accuracy"] == best["raw"]["logical_accuracy"] for e in h["eval"]),
                 loss_last50=sum(t["total"] for t in h["train"][-50:])/50,
                 loss_previous50=sum(t["total"] for t in h["train"][-100:-50])/50)
        for evaluation in [*h["eval"], f]:
            n = evaluation["shots"]
            for method in ("raw", "osd"):
                m = evaluation[method]
                failures, _, _ = interval(m["logical_error_rate"], n)
                assert m["samples"] == n and abs(m["logical_accuracy"] + m["logical_error_rate"] - 1) < 1e-12
                assert abs(m["flagged_failure_rate"] + m["unflagged_logical_failure_rate"] - failures/n) < 1e-12
            assert evaluation["osd"]["syndrome_convergence"] == 1 and evaluation["harmed"] == 0
            assert round((evaluation["raw"]["logical_error_rate"]-evaluation["osd"]["logical_error_rate"])*n) == evaluation["rescued"]
        for method in ("raw", "osd"):
            m = f[method]
            count, lo, hi = interval(m["logical_error_rate"], f["shots"])
            r.update({f"{method}_failures": count, f"{method}_ler": m["logical_error_rate"],
                      f"{method}_ci95_low": lo, f"{method}_ci95_high": hi,
                      f"{method}_flagged": round(m["flagged_failure_rate"]*f["shots"]),
                      f"{method}_unflagged": round(m["unflagged_logical_failure_rate"]*f["shots"])})
        finals.append(r)
        training.extend({**meta, **t} for t in h["train"])
        validation.extend(dict(**meta, epoch=e["epoch"]+1, raw_ler=e["raw"]["logical_error_rate"],
                               osd_ler=e["osd"]["logical_error_rate"]) for e in h["eval"])
        histories[c["error_rate"]] = h
    assert set(histories) == {.02, .04, .06, .08}
    finals.sort(key=lambda r: r["p"])
    if write_results:
        write_csv("cnn_final", finals)
        write_csv("cnn_train", training)
        write_csv("cnn_validation", validation)
    return finals, histories, dict(completed_models=4, epochs_each=800,
        final_reported_shots=4*65536, source_snapshot_files_verified=len(snapshots),
        elapsed_job_log_hours=(finish-start).total_seconds()/3600,
        audit_level="history_and_logs_only", checkpoint_verified=False, shot_bank_verified=False,
        inference_steps_added=0, training_steps_added=0)


def plot(finals, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    old = list(csv.DictReader((OUT / "bb_update_2026_10_24_cnn_final.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, method in zip(axes, ("raw", "osd")):
        for depth, color in ((6, "#334790"), (7, "#d98116"), (8, "#12633c"), (12, "#c02c91")):
            rows = finals if depth == 12 else [r for r in old if r["code"] == "bb144" and int(r["depth"]) == depth]
            rows = sorted(rows, key=lambda r: float(r["p"]))
            label = f"Depth {depth}" + ((", residual" if depth == 12 else ", syndrome") if method == "osd" else "")
            ax.plot([float(r["p"])*100 for r in rows],
                    [float(r[f"{method}_ler"])*100 or float("nan") for r in rows],
                    color=color, ls="--" if method == "osd" and depth == 12 else "-", label=label)
            for r in rows:
                y, lo, hi = [float(r[f"{method}_{k}"])*100 for k in ("ler", "ci95_low", "ci95_high")]
                if y:
                    ax.errorbar(float(r["p"])*100, y, yerr=[[y-lo], [hi-y]],
                                fmt="s" if method == "osd" and depth == 12 else "o", color=color, capsize=3)
                else:
                    ax.plot(float(r["p"])*100, hi, "v", color=color, mfc="none", ms=8)
        ax.set(title="CNN alone" if method == "raw" else "CNN + OSD-0 (different conventions)",
               xlabel="Physical error rate p (%)", ylabel="Block LER (%)", yscale="log", xticks=[2,4,6,8])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("BB144: depths 6/7/8/12, width 64, fresh 800 epochs\n95% Wilson intervals; hollow triangles = zero-failure upper bounds")
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "depth12.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for column, p in enumerate(sorted(histories)):
        for depth, color in ((8, "#12633c"), (12, "#c02c91")):
            h = histories[p] if depth == 12 else read(ROOT / next(r["source"] for r in old if r["code"] == "bb144" and int(r["depth"]) == 8 and float(r["p"]) == p))
            ev = h["eval"]
            smooth = [sum(e["raw"]["logical_error_rate"] for e in ev[i-4:i+1])/5*100 for i in range(4,len(ev))]
            epochs = [e["epoch"]+1 for e in ev[4:]]
            axes[0, column].plot(epochs, smooth, color=color, label=f"Depth {depth}")
            tail = [(e,y) for e,y in zip(epochs, smooth) if e >= 200]
            axes[1, column].plot([e for e,y in tail], [y for e,y in tail], color=color, label=f"Depth {depth}")
        for row in (0, 1):
            ax = axes[row, column]
            ax.set(title=f"BB144 p={p:.0%}" + (" | epochs 200–800" if row else " | all epochs"),
                   xlabel="Epoch", ylabel="Raw validation LER (%)")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Trailing mean of 5 validations (25 epochs); linear axes include zero; panel scales differ")
    fig.savefig(PLOTS / "depth12_validation.png", dpi=150)
    plt.close(fig)


def report(finals, audit):
    old = list(csv.DictReader((OUT / "bb_update_2026_10_24_cnn_final.csv").open()))
    d5 = list(csv.DictReader((OUT / "bb_update_2026_10_24_depth5_references.csv").open()))
    comparisons = []
    for r in finals:
        for depth in (5, 8):
            prior = next(x for x in (d5 if depth == 5 else old)
                         if x["code"] == "bb144" and int(x["depth"]) == depth and float(x["p"]) == r["p"])
            for method in ("raw", "osd"):
                key = "recorded_osd_ler" if depth == 5 and method == "osd" else f"{method}_ler"
                before = float(prior[key])
                comparisons.append(dict(code="bb144", p=r["p"], from_depth=depth, to_depth=12,
                    method=method, old_ler=before, new_ler=r[f"{method}_ler"],
                    relative_reduction=1-r[f"{method}_ler"]/before if before else "",
                    same_osd_convention=(depth == 5) if method == "osd" else "not_applicable",
                    same_training_schedule=depth == 8, paired_shots_verified=False))
    write_csv("depth_comparisons", comparisons)
    lines = ["# 2026-10-26: BB144 depth 12 결과", "",
        "Job 60255255 / array task 3의 p=.02/.04/.06/.08 네 실험만 이번에 받았다. "
        "Depth 10과 BB72 depth 12, depth 6–8 residual 재평가 job 60255150은 이번 다운로드에 없다.", "",
        f"39개 파일 ({audit['bytes']:,} bytes)을 `800_epochs/residual_osd0/bb144/resdir_60255255/`로 이동했고 "
        "이동 전후 SHA-256을 검증했다. 원본 폴더 구조와 로그를 보존했다.", "",
        "모두 fresh 800 epochs, width 64, 6,553,600 training shots, final 65,536 shots. "
        "Raw validation LER로 best checkpoint를 선택했고 OSD는 residual OSD-0이다.", "",
        "## 검증 범위", "",
        "실험별 exit code 0, 완료 마커, 800개 학습 기록과 160개 validation 기록, "
        "history와 두 로그의 학습/최종 수치, 선택 epoch, 집계 failure count를 확인했다. "
        "소스 스냅샷 8개는 현재 코드와 일치한다. 이번 다운로드에는 `.pt`와 `.npz`가 없어 "
        "가중치 검증·샷별 재채점·depth 간 paired 검정은 수행하지 않았다. "
        "같은 seed라도 동일 샷임을 검증한 것으로 취급하지 않는다. 새 학습/추론은 실행하지 않았다.", "",
        "## 최종 성능", "", "LER 단위는 %. 괄호는 실패 수 / 65,536. Epoch는 1부터 센다.", "",
        "| p | CNN 단독 | CNN + residual OSD-0 | best epoch |",
        "|---|---:|---:|---:|"]
    for r in finals:
        lines.append(f"| {r['p']:.2f} | {r['raw_ler']*100:.6f} ({r['raw_failures']}) | "
                     f"{r['osd_ler']*100:.6f} ({r['osd_failures']}) | {r['best_epoch']} |")
    lines += ["", "## Depth 8과 비교", "",
        "| p | D8 CNN | D12 CNN | raw LER 상대 감소 | D8 + syndrome OSD | D12 + residual OSD |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in finals:
        prior = next(x for x in old if x["code"] == "bb144" and int(x["depth"]) == 8 and float(x["p"]) == r["p"])
        before = float(prior["raw_ler"])
        lines.append(f"| {r['p']:.2f} | {before*100:.6f} | {r['raw_ler']*100:.6f} | "
                     f"{(1-r['raw_ler']/before)*100:.1f}% | {float(prior['osd_ler'])*100:.6f} | {r['osd_ler']*100:.6f} |")
    lines += ["", "Raw는 p=.04/.06/.08에서 각각 약 22.4%/45.5%/41.5% 낮다. "
        "p=.02는 35→58 failures로 더 높다. 이는 관측된 차이이며 이번 분석에서 통계적 유의성을 검정하지 않았다. "
        "학습 seed도 설정별 하나뿐이다.", "",
        "OSD 후 p=.06/.08 수치는 약 12.4%/19.8% 낮지만 **OSD 방식도 바뀌므로 depth만의 효과가 아니다**. "
        "p=.04는 15→16 failures, p=.02는 0→2이다. 실패 0건은 실제 LER=0을 뜻하지 않으며 "
        "플롯의 빈 역삼각형은 95% Wilson 상한이다. D12의 p=.02 OSD LER 95% Wilson 구간은 "
        "약 0.000837–0.011128%로 작은 실패 수의 불확실성이 크다.", "",
        "기존 D5 residual과는 OSD 방식이 같고 p=.06에서 0.5447→0.2701%, "
        "p=.08에서 4.1718→2.2247%다. 하지만 D5는 추가 학습/restart schedule이어서 "
        "이 역시 depth만 바꾼 통제 실험은 아니다.", "",
        "## 학습과 비용", "",
        "p=.02는 validation 4,096샷에서 0 failures인 최상위 평가가 30회 동률이다. "
        "현 선택 규칙은 첫 동률인 epoch 185를 유지한다. 낮은 p에서 checkpoint를 구분할 "
        "평가 해상도가 부족할 수 있다는 단서이며, 마지막 가중치가 더 좋다는 증거는 아니다. "
        "p=.04/.06/.08의 선택 epoch는 610/770/775다.", "",
        "학습 초반 p=.02/.04의 D12는 LER가 높은 정체 구간을 거친 뒤 감소했다. "
        "학습 곡선은 전체 구간과 epoch 200 이후 확대 구간을 따로 보여준다. "
        "깊이가 늘면서 초반 최적화 양상이 바뀐 것으로 보이지만 원인은 추가 실험이 필요하다.", "",
        "파라미터 수는 D8 802,564 → D12 1,230,084 (+53.3%). D12 CNN 시간은 "
        "batch 총시간을 샷으로 나눈 약 466–499 μs/shot으로, D8의 약 309–329 μs/shot보다 크다. "
        "이는 단일 샷 latency나 통제된 하드웨어 benchmark가 아니다. 잡 내부 로그 기준 실행은 "
        f"{audit['elapsed_job_log_hours']:.3f}시간, 사용자 Slurm 알림의 wall time은 3:25:46이다.", "",
        "## 파일과 재생성", "",
        "[통합 플롯](../plots/bb/code_capacity/tanner_cnn/overview.png) · "
        "[D6/7/8/12 비교](../plots/bb/code_capacity/tanner_cnn/depth12.png) · "
        "[D8/12 학습 곡선](../plots/bb/code_capacity/tanner_cnn/depth12_validation.png)", "",
        f"[최종 수치]({STEM}_cnn_final.csv) · [비교표]({STEM}_depth_comparisons.csv) · "
        f"[검증]({STEM}_audit.json) · [이동 기록]({STEM}_relocation.csv)", "",
        "```bash", "python scripts/analyze_bb_october26.py", "python scripts/plot_bb_cnn_all.py", "```", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = archive(args.archive)
    finals, histories, checks = analyze()
    audit.update(checks)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    plot(finals, histories)
    report(finals, audit)
    for r in finals:
        print(f"p={r['p']} raw={r['raw_ler']:.6%} residual={r['osd_ler']:.6%} best_epoch={r['best_epoch']} ties={r['best_validation_ties']}")


if __name__ == "__main__":
    main()
