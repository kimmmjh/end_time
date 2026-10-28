#!/usr/bin/env python3
"""Archive and audit 24 reported standard-noise DEM CNN runs; no inference.

First import: --archive. Subsequent runs verify the archive and regenerate
tables, plots, and the report. Checkpoints and shot banks were not downloaded.
"""
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
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bb_oct26_evening_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/bb_oct26_evening_cache")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bb_cnn_plot_osd import interval

STEM = "bb_dem_2026_10_28"
OUT = ROOT / "results/analysis"
ARCHIVE = ROOT / "results/bb/circuit/standard/dem_cnn/800_epochs"
PLOTS = ROOT / "results/plots/bb/circuit/standard/dem_cnn"
JOBS = ((60310802, "bb72", 4, 0), (60310803, "bb144", 4, 1),
        (60316098, "bb72", 6, 2), (60316182, "bb144", 6, 3),
        (60318763, "bb72", 8, 4), (60330386, "bb72", 12, 6))
COLORS = {4: "#3478c6", 6: "#d98116", 8: "#12633c", 12: "#c02c91"}


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
    destinations = {job: ARCHIVE / code / f"resdir_{job}" for job, code, _, _ in JOBS}
    manifest = OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move, "Use --archive for the first import"
        rows = []
        for job, dest in destinations.items():
            src = ROOT / f"resdir_{job}"
            assert src.is_dir() and not dest.exists()
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    rows.append(dict(old_path=str(p.relative_to(ROOT)),
                                     new_path=str((dest / p.relative_to(src)).relative_to(ROOT)),
                                     bytes=p.stat().st_size, sha256=sha(p)))
        write_csv("relocation", rows)
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    for job, dest in destinations.items():
        src = ROOT / f"resdir_{job}"
        if move and src.is_dir() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
    actual = {p for dest in destinations.values() for p in dest.rglob("*") if p.is_file()}
    assert actual == {ROOT / r["new_path"] for r in rows}
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"], p
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), relocation_hashes_verified=True)


def audit_evaluation(e, shots):
    m = e["raw"]
    assert e["shots"] == m["shots"] == shots
    assert e["osd_shots"] == 0 and "osd" not in e and e["osd_with_transfer_seconds"] == 0
    count = interval(m["logical_error_rate"], shots)[0]
    assert count == m["failures"]
    assert m["logical_accuracy"] == 1 - m["logical_error_rate"]
    assert m["flagged_failure"] == 1 - m["syndrome_convergence"]
    assert m["flagged_failure"] + m["unflagged_failure"] == m["logical_error_rate"]
    assert 0 <= m["unflagged_failure"] <= m["observable_mismatch_rate"] <= m["logical_error_rate"]
    for key in ("flagged_failure", "unflagged_failure", "observable_mismatch_rate", "syndrome_convergence"):
        interval(m[key], shots)


def analyze():
    finals, training, validation, jobs, snapshots = [], [], [], [], []
    for job, code, depth, task in JOBS:
        dest = ARCHIVE / code / f"resdir_{job}"
        assert (dest / "completed.txt").is_file() and not (dest / "failed.txt").exists()
        metadata = (dest / "job_metadata.txt").read_text()
        assert f"Array job/task: 60310800/{task}" in metadata
        begin = datetime.fromisoformat(metadata.split("Started: ")[1].splitlines()[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat((dest / "completed.txt").read_text().split("Finished: ")[1].strip().replace("Z", "+00:00"))
        jobs.append(dict(job=job, code=code, depth=depth, array_task=task,
                         elapsed_job_log_hours=(end-begin).total_seconds()/3600, models=4))
        for p in sorted((dest / "source_snapshot").rglob("*.py")):
            relative = p.relative_to(dest / "source_snapshot")
            snapshots.append(dict(job=job, file=str(relative), sha256=sha(p),
                                  matches_current_source=sha(p) == sha(ROOT / relative)))
        with (dest / "experiments.tsv").open() as f:
            specs = list(csv.DictReader(f, delimiter="\t"))
        paths = sorted(dest.glob("outputs/*/*/history.json"))
        assert len(paths) == len(specs) == 4
        seen_p = set()
        for path in paths:
            h = read(path)
            c, f = h["config"], h["final"]
            p = c["noise_profile"]["base_error_rate"]
            seen_p.add(p)
            rounds = 6 if code == "bb72" else 12
            assert c["code"] == code and c["depth"] == depth and c["width"] == 64
            assert c["architecture"] == "bb_dem_cnn" and c["noise_model"] == "circuit"
            assert c["rounds"] == rounds and c["detector_frames"] == rounds+1
            assert c["num_detectors"] == int(code[2:])*(rounds+1)
            assert c["num_mechanisms"] == (17928 if code == "bb72" else 71280)
            assert c["num_observables"] == 24 and c["num_orbits"] == 1410
            assert c["parameter_count"] == 695233+(depth-4)*197248
            assert c["checkpoint_selection"] == "raw_syndrome_and_observable_accuracy"
            assert c["circuit_schema_version"] == 2
            assert c["seed"] == int(code[2:])*100000+2000+round(p*1000)
            assert c["noise_profile"] == dict(name="standard", base_error_rate=p, reset_error_rate=p,
                one_qubit_error_rate=0., two_qubit_error_rate=p, swap_error_rate=p,
                measurement_error_rate=p, gate_idle_error_rate=p, resonator_idle_error_rate=0., full_tick_idle=True)
            assert c["loss_weights"] == dict(syndrome_weight=1., logical_weight=1., mechanism_weight=.1, deep_supervision_weight=0.)
            assert len(h["phases"]) == 1
            phase = h["phases"][0]
            for key, value in dict(start_epoch=0, epochs=800, batches=128, batch_size=4,
                eval_batches=256, eval_every=5, final_eval_batches=2048, learning_rate=.0003,
                osd_eval_shots=0, schedule="cosine", resume_source=None, edge_chunk_size=4096,
                gradient_checkpoint=True).items():
                assert phase[key] == value, (path,key)
            assert [t["epoch"] for t in h["train"]] == list(range(800))
            assert [e["epoch"] for e in h["eval"]] == list(range(4,800,5))
            best = max(h["eval"], key=lambda e:e["raw"]["logical_accuracy"])
            assert h["best_epoch"] == f["epoch"] == best["epoch"]
            assert h["best_raw_accuracy"] == best["raw"]["logical_accuracy"]
            for e in h["eval"]:
                audit_evaluation(e, 1024)
            audit_evaluation(f, 8192)
            for t in h["train"]:
                assert all(math.isfinite(v) for v in t.values())
                assert abs(t["total"]-t["syndrome"]-t["logical"]-.1*t["mechanism"]) < 1e-6
            spec = next(s for s in specs if f"--seed={c['seed']}" in s["arguments"])
            index = int(spec["experiment_index"])
            args = dict(a[2:].split("=",1) for a in shlex.split(spec["arguments"]) if a.startswith("--") and "=" in a)
            assert args["code"] == code and int(args["bb_cnn_depth"]) == depth and float(args["p"]) == p
            assert args["bb_circuit_noise_model"] == "standard" and args["bb_osd_eval_shots"] == "0"
            assert (dest / f"exit_code_exp_{index}.txt").read_text().strip() == "0"
            for logpath in (dest / f"log_exp_{index}.txt", path.with_name("training_log.txt")):
                log = logpath.read_text()
                assert "Traceback (most recent call last)" not in log
                for label, name in (("Train", "train"), ("Validation", "eval")):
                    logged = re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$", log, re.M)
                    assert [dict(epoch=int(ep), **json.loads(value)) for ep,value in logged] == h[name]
                final_lines = re.findall(r"^\[Final selected best\] (.+)$",log,re.M)
                assert len(final_lines) == 1 and json.loads(final_lines[0]) == f
                logged_config = re.findall(r"^DEM CNN configuration: (.+)$",log,re.M)
                assert len(logged_config) == 1 and json.loads(logged_config[0]) == c
            meta = dict(job=job, code=code, depth=depth, p=p, seed=c["seed"], source=str(path.relative_to(ROOT)))
            row = dict(**meta, rounds=rounds, width=64, parameters=c["parameter_count"],
                detectors=c["num_detectors"], mechanisms=c["num_mechanisms"], graph_fingerprint=c["graph_fingerprint"],
                dem_fingerprint=c["dem_fingerprint"], epochs=800, train_shots=409600, shots=8192,
                best_epoch=f["epoch"]+1, best_validation_failures=best["raw"]["failures"],
                best_validation_ties=sum(e["raw"]["failures"] == best["raw"]["failures"] for e in h["eval"]),
                last_validation_failures=h["eval"][-1]["raw"]["failures"], osd_shots=0,
                nn_ms_per_shot_amortized=f["nn_seconds"]*1000/f["shots"],
                audit_level="history_and_logs_only", checkpoint_verified=False, shot_bank_verified=False)
            for short, key in (("block", "logical_error_rate"), ("observable", "observable_mismatch_rate"),
                               ("flagged", "flagged_failure"), ("unflagged", "unflagged_failure"),
                               ("convergence", "syndrome_convergence")):
                rate = f["raw"][key]
                count, low, high = interval(rate, f["shots"])
                row.update({f"{short}_rate":rate, f"{short}_count":count, f"{short}_low":low, f"{short}_high":high})
            for key in ("total", "syndrome", "logical", "mechanism"):
                for label, segment in (("first50",h["train"][:50]), ("previous50",h["train"][-100:-50]), ("last50",h["train"][-50:])):
                    row[f"{key}_{label}"] = statistics.mean(t[key] for t in segment)
            finals.append(row)
            training.extend(dict(**meta, **t) for t in h["train"])
            validation.extend(dict(**meta, epoch=e["epoch"]+1, **e["raw"]) for e in h["eval"])
        assert seen_p == {.001,.002,.004,.006}
    assert len(finals) == 24 and len(snapshots) == 66
    assert len({(r["code"],r["depth"],r["p"]) for r in finals}) == 24
    for code in ("bb72", "bb144"):
        for p in (.001,.002,.004,.006):
            matching = [r for r in finals if r["code"] == code and r["p"] == p]
            assert len({(r["graph_fingerprint"],r["dem_fingerprint"],r["seed"]) for r in matching}) == 1
    for file in {s["file"] for s in snapshots}:
        assert len({s["sha256"] for s in snapshots if s["file"] == file}) == 1
    finals.sort(key=lambda r:(r["code"],r["depth"],r["p"]))
    for suffix, rows in (("final",finals), ("train",training), ("validation",validation), ("jobs",jobs), ("snapshots",snapshots)):
        write_csv(suffix, rows)
    return finals, training, validation, jobs, snapshots


def plots(finals, training, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # Incomplete runs contribute training curves only, never final LER points.
    training, validation = list(training), list(validation)
    completed_keys = {(r["code"],r["depth"],r["p"]) for r in finals}
    partial = {}
    for path in sorted((ARCHIVE.parent / "interrupted").rglob("history.json")):
        h = read(path)
        c = h["config"]
        key = (c["code"],c["depth"],c["noise_profile"]["base_error_rate"])
        if c["architecture"] != "bb_dem_cnn" or "final" in h or key in completed_keys:
            continue
        if key not in partial or len(h["train"]) > len(partial[key]["train"]):
            partial[key] = h
    for (code,depth,p),h in partial.items():
        meta = dict(code=code,depth=depth,p=p)
        training.extend(dict(**meta,**t) for t in h["train"])
        validation.extend(dict(**meta,epoch=e["epoch"]+1,**e["raw"]) for e in h["eval"])
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,code in zip(axes,("bb72","bb144")):
        for depth,color in COLORS.items():
            group = sorted([r for r in finals if r["code"]==code and r["depth"]==depth],key=lambda r:r["p"])
            if not group:
                continue
            ys = np.array([r["block_rate"]*100 for r in group])
            lo = np.array([r["block_low"]*100 for r in group])
            hi = np.array([r["block_high"]*100 for r in group])
            ax.errorbar([r["p"]*100 for r in group],ys,yerr=[np.maximum(0,ys-lo),np.maximum(0,hi-ys)],
                        color=color,marker="o",capsize=3,label=f"Depth {depth}")
        ax.set(title=code.upper(),xlabel="Per-location error rate p (%)",ylabel="LER / block failure (%)",
               xticks=[.1,.2,.4,.6],ylim=(-2,102))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("DEM CNN: standard circuit noise, no OSD, 8,192 final shots\n"
                 "Completed runs only; LER includes syndrome failures; 95% Wilson intervals",fontsize=12)
    fig.savefig(PLOTS / "overview.png",dpi=170)
    plt.close(fig)
    for kind, source, key, ylabel, filename in (
        ("validation",validation,"logical_error_rate","Raw block failure (%)","validation.png"),
        ("loss",training,"logical","Logical parity loss","logical_loss.png")):
        fig, axes = plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
        for i,code in enumerate(("bb72","bb144")):
            for j,p in enumerate((.001,.002,.004,.006)):
                ax=axes[i,j]
                for depth,color in COLORS.items():
                    group=[r for r in source if r["code"]==code and r["depth"]==depth and r["p"]==p]
                    if not group:
                        continue
                    window=5 if kind=="validation" else 25
                    values=np.array([r[key] for r in group])*(100 if kind=="validation" else 1)
                    ys=np.convolve(values,np.ones(window)/window,mode="valid")
                    unfinished = (code,depth,p) in partial
                    ax.plot([r["epoch"]+(kind=="loss") for r in group[window-1:]],ys,color=color,
                            ls="--" if unfinished else "-", marker="x" if unfinished else None,
                            markevery=[-1] if unfinished else None,
                            label=f"D{depth}"+(" (partial)" if unfinished else ""))
                if kind=="loss":
                    ax.axhline(math.log(2),ls=":",color="#666666",lw=1,label="ln(2)")
                ax.set(title=f"{code.upper()} p={p:.1%}",xlabel="Epoch",ylabel=ylabel)
                if kind=="validation":
                    lower,upper=ax.get_ylim()
                    ax.set_ylim(max(0.,lower),min(100.1,upper))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8,ncol=2)
        fig.suptitle(("Raw validation failure" if kind=="validation" else "Training logical parity loss")+
                     " | trailing 25-epoch average; panel scales differ"+
                     ("\nDashed = unfinished run; x = last available smoothed point" if partial else ""),fontsize=13)
        fig.savefig(PLOTS / filename,dpi=160)
        plt.close(fig)


def report(finals,jobs,audit):
    lines=["# 2026-10-28: standard circuit-level DEM CNN 결과", "",
        "6개 job의 24개 모델 모두 800 epochs를 정상 완료했다. BB72 depth 4/6/8/12, "
        "BB144 depth 4/6 × p=.001/.002/.004/.006. BB144 depth 8/12의 final 평가는 없다. "
        "D12 중단 기록은 [별도 분석](bb_dem_2026_10_28_interrupted.md)을 참고한다.", "",
        f"{audit['files']}개 파일 ({audit['bytes']:,} bytes)을 `results/bb/circuit/standard/dem_cnn/800_epochs/` "
        "아래 코드별로 정리했다. 이동 전후 SHA-256을 검증하고 원본 폴더 구조를 보존했다.", "",
        "## 실험과 검증", "",
        "- Width 64, batch 4, 512 training shots/epoch: 모델당 총 409,600 training shots. "
        "Code-capacity 실험의 800 epochs (6,553,600 shots)보다 학습 샷 수가 16배 적다.",
        "- Validation 1,024 shots × 160회; 최종 best checkpoint를 8,192 circuit shots로 평가. "
        "학습은 DEM mechanism 샘플, 평가는 원래 Stim circuit 샘플이다.",
        "- Standard noise: reset/measurement flip p, CNOT 뒤 DEP2(p), 매 tick 비활성 qubit에 DEP1(p), "
        "H 자체는 ideal. BB72/BB144 noisy rounds=6/12, perfect reference 및 closing, "
        "detector frames=7/13; X/Z logical correlation sheets 총 24개를 평가한다.",
        "- 모든 평가에서 `osd_shots=0`. OSD 결과가 아니며 이 자료로 OSD 적용 후 성능을 추정할 수 없다.",
        "- Train/validation/final 로그와 history, 실행 인자, exit code를 대조했다. "
        f"66개 source snapshot은 job 간 동일하다. 현재 코드 전체 일치: {audit['all_snapshots_match_current']}. "
        "모든 loss는 유한하고 최종 샷 수·실패 집계·best 선택 규칙이 일치한다.",
        "- 다운로드에 `.pt`/`.npz`가 없어 가중치 추론 재현과 shot별 검증, depth 간 paired 검정은 하지 않았다. "
        "같은 code/p의 seed·graph fingerprint는 일치하지만 동일 샷임을 로컬에서 확인한 것은 아니다.", "",
        "## 최종 raw block failure (%)", "",
        "`logical_error_rate`라는 저장 키는 실제로 syndrome 불일치 또는 observable 불일치의 합집합이다. "
        "아래 수치는 round당 rate가 아니라 전체 memory experiment의 block failure다.", "",
        "| 코드 | p | D4 | D6 | D8 | D12 |", "|---|---:|---:|---:|---:|---:|"]
    for code in ("bb72","bb144"):
        for p in (.001,.002,.004,.006):
            vals=[]
            for depth in (4,6,8,12):
                r=next((r for r in finals if r["code"]==code and r["p"]==p and r["depth"]==depth),None)
                vals.append(f"{100*r['block_rate']:.3f}" if r else "미수신")
            lines.append(f"| {code.upper()} | {p:.1%} | "+" | ".join(vals)+" |")
    lines += ["", "## 해석", "",
        "**BB72의 D12는 낮은 p에서 개선되지만 깊이에 따른 단조 개선은 아니다.** "
        "p=.001 D4→D12는 16.211→7.947% (약 51.0% 상대 감소), "
        "p=.002는 53.650→28.918% (46.1% 감소). 반면 D8의 p=.002는 67.834%로 D4/D6보다 나쁘다. "
        "p=.004 D12도 81.445%, p=.006은 99.792%라 높은 p에서는 유효한 block correction이 드물다.", "",
        "**BB144는 D6가 p=.002에서 좋아졌지만 p=.001에서는 나빠졌다.** "
        "p=.002는 D4 95.886→D6 87.146%; p=.001은 51.123→57.361%. "
        "한 seed, 제한된 학습 예산에서 얻은 결과이므로 구조의 보편적 우열이나 threshold로 해석하지 않는다.", "",
        "**실패의 거의 전부는 detector syndrome을 정확히 재현하지 못하는 경우다.** "
        "예를 들어 BB72 D12 p=.002: block 28.918%, syndrome 불일치 28.845%, "
        "syndrome은 맞지만 logical이 틀린 실패 0.073%, observable 불일치 자체는 15.759%. "
        "Observable만 맞아도 syndrome이 틀리면 유효한 correction은 아니다. "
        "OSD로 consistency를 강제했을 때의 logical 성능은 별도 평가가 필요하다.", "",
        "**BB144 p=.004/.006의 D4/D6 네 모델은 800 epochs를 학습했지만 최종 평가는 epoch 5 가중치다.** "
        "160번 validation 모두 1,024/1,024 failures로 동률이어서 earliest-tie 규칙이 "
        "첫 평가를 유지했다. Final도 네 모델 모두 8,192/8,192 failures. "
        "이는 학습이 epoch 5에서 중단되었다는 뜻이 아니며, epoch 800 모델의 final 결과도 아니다. "
        "다만 마지막 validation에서도 성공은 0건이다.", "",
        "높은 p에서 logical parity loss가 ln(2)≈0.693 부근에 머무는 것은 학습 병목의 단서다. "
        "BB144 p=.004/.006의 마지막 50 epochs 평균은 약 0.685–0.697이다. "
        "Factorized parity 확률이 1/2 부근에서 구분되지 않는 상황과 부합하지만, "
        "gradient 소실·모델 collapse·noise mismatch 등의 원인을 이 로그만으로 확정할 수 없다. "
        "단순히 depth만 늘리면 해결된다는 근거도 없다.", "",
        "## 최종 상세 수치", "",
        "모두 전체 8,192샷 기준 %. Best epoch는 1-based. CI는 별도 CSV의 95% Wilson 구간이며, "
        "관측 실패율 100%를 실제 실패 확률이 정확히 1이라는 뜻으로 해석하지 않는다.", "",
        "| 코드 | D | p | 실패/8192 | syndrome 불일치 % | unflagged % | observable 불일치 % | best epoch | 동률 수 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in finals:
        lines.append(f"| {r['code']} | {r['depth']} | {r['p']:.1%} | {r['block_count']} | "
                     f"{100*r['flagged_rate']:.3f} | {100*r['unflagged_rate']:.3f} | "
                     f"{100*r['observable_rate']:.3f} | {r['best_epoch']} | {r['best_validation_ties']} |")
    lines += ["", "## 비용", "", "| 코드 | depth | 파라미터 | job 로그 경과 시간(h) | NN ms/shot 범위 |",
              "|---|---:|---:|---:|---:|"]
    for j in jobs:
        group=[r for r in finals if r["job"]==j["job"]]
        times=[r["nn_ms_per_shot_amortized"] for r in group]
        lines.append(f"| {j['code']} | {j['depth']} | {group[0]['parameters']:,} | {j['elapsed_job_log_hours']:.2f} | {min(times):.2f}–{max(times):.2f} |")
    lines += ["", "NN 시간은 batch4 forward 총시간을 샷 수로 나눈 값이며 end-to-end 단일 shot latency가 아니다. "
        "각 job은 네 p를 GPU 네 장에서 동시에 실행했다. BB144 D6는 로그 기준 약 23.65시간으로 "
        "24시간 제한에 가깝다. BB144 D8/D12가 제한 안에 끝날지는 이 자료로 확정할 수 없다.", "",
        "## 다음에 확인할 항목", "",
        "1. 같은 standard circuit과 평가 정의에서 classical baseline 및 CNN+residual OSD를 평가한다. "
        "현재 raw 실패의 대부분이 syndrome 불일치이므로 correction 완성 단계의 영향이 크다. "
        "기존 idle0 또는 다른 logical observable을 쓰는 baseline과 직접 비교하지 않는다.",
        "2. 높은 p에서 validation이 전부 동률인 네 모델은 best와 마지막 checkpoint를 별도로 평가하고 "
        "logical loss·posterior 분포·gradient를 확인한다. 선택 규칙을 바꾼 결과는 별도 실험으로 남긴다.",
        "3. BB144 D8/D12 결과를 확보한 후 깊이 효과를 판단한다. 미수신을 실패로 간주하지 않는다.", "",
        "## 파일", "",
        "[성능 그림](../plots/bb/circuit/standard/dem_cnn/overview.png) · "
        "[Validation 곡선](../plots/bb/circuit/standard/dem_cnn/validation.png) · "
        "[Logical parity loss](../plots/bb/circuit/standard/dem_cnn/logical_loss.png)", "",
        f"[최종 수치]({STEM}_final.csv) · [학습]({STEM}_train.csv) · [Validation]({STEM}_validation.csv) · "
        f"[검증]({STEM}_audit.json) · [이동 기록]({STEM}_relocation.csv)", "",
        "재생성: `python scripts/analyze_bb_dem_october28.py`. 학습·추론을 실행하지 않는다.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    audit=archive(args.archive)
    finals,training,validation,jobs,snapshots=analyze()
    audit.update(models=24,epochs_each=800,training_rows=len(training),validation_rows=len(validation),
        final_shots_total=sum(r["shots"] for r in finals), osd_shots_total=0,
        source_snapshot_files=len(snapshots),all_snapshots_match_current=all(s["matches_current_source"] for s in snapshots),
        checkpoint_verified=False,shot_bank_verified=False,paired_depth_tests=False,
        missing_models=[dict(code="bb144",depth=d,p=p) for d in (8,12) for p in (.001,.002,.004,.006)],
        training_steps_added=0,inference_steps_added=0)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    plots(finals,training,validation)
    report(finals,jobs,audit)
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
