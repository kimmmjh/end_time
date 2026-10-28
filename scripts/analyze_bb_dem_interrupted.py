#!/usr/bin/env python3
"""Audit/archive the timed-out BB144 D12 download; never invent final metrics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_bb_dem_october28 as base

OUT = base.OUT
STEM = "bb_dem_2026_10_28_interrupted"
SOURCE = ROOT / "resdir_60310800"
DEST = ROOT / "results/bb/circuit/standard/dem_cnn/interrupted/bb144/resdir_60310800"


def write_csv(suffix,rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def archive(move):
    manifest=OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move and SOURCE.is_dir() and not DEST.exists(), "First import requires --archive"
        rows=[dict(old_path=str(p.relative_to(ROOT)),new_path=str((DEST/p.relative_to(SOURCE)).relative_to(ROOT)),
                   bytes=p.stat().st_size,sha256=base.sha(p)) for p in sorted(SOURCE.rglob("*")) if p.is_file()]
        write_csv("relocation",rows)
    with manifest.open() as f:
        rows=list(csv.DictReader(f))
    if move and SOURCE.is_dir() and not DEST.exists():
        DEST.parent.mkdir(parents=True,exist_ok=True)
        SOURCE.rename(DEST)
    assert {ROOT/r["new_path"] for r in rows} == {p for p in DEST.rglob("*") if p.is_file()}
    for r in rows:
        p=ROOT/r["new_path"]
        assert p.stat().st_size==int(r["bytes"]) and base.sha(p)==r["sha256"]
    return dict(files=len(rows),bytes=sum(int(r["bytes"]) for r in rows),relocation_hashes_verified=True)


def analyze():
    assert (DEST/"interrupted.txt").is_file() and not (DEST/"completed.txt").exists()
    metadata=(DEST/"job_metadata.txt").read_text()
    assert "Array job/task: 60310800/7" in metadata
    start=datetime.fromisoformat(metadata.split("Started: ")[1].splitlines()[0].replace("Z","+00:00"))
    stop=datetime.fromisoformat((DEST/"interrupted.txt").read_text().split("Interrupted: ")[1].splitlines()[0].replace("Z","+00:00"))
    rows,train,validation,comparisons,snapshots=[],[],[],[],[]
    for path in sorted((DEST/"source_snapshot").rglob("*.py")):
        relative=path.relative_to(DEST/"source_snapshot")
        snapshots.append(dict(file=str(relative),sha256=base.sha(path),matches_current=base.sha(path)==base.sha(ROOT/relative)))
    assert len(snapshots)==11
    with (DEST/"experiments.tsv").open() as f:
        specs=list(csv.DictReader(f,delimiter="\t"))
    paths=sorted(DEST.glob("outputs/*/*/history.json"))
    assert len(paths)==len(specs)==4
    for path in paths:
        h=base.read(path); c=h["config"]; p=c["noise_profile"]["base_error_rate"]
        assert c["architecture"]=="bb_dem_cnn" and c["code"]=="bb144" and c["depth"]==12
        assert c["rounds"]==12 and c["width"]==64 and c["parameter_count"]==2273217
        assert "final" not in h and len(h["phases"])==1
        phase=h["phases"][0]
        assert phase["start_epoch"]==0 and phase["epochs"]==800 and phase["resume_source"] is None
        assert phase["batch_size"]==4 and phase["batches"]==128 and phase["osd_eval_shots"]==0
        completed=len(h["train"])
        assert 0<completed<800 and [r["epoch"] for r in h["train"]]==list(range(completed))
        assert [r["epoch"] for r in h["eval"]]==list(range(4,completed,5))
        for e in h["eval"]:
            base.audit_evaluation(e,1024)
        for t in h["train"]:
            assert all(math.isfinite(v) for v in t.values())
            assert abs(t["total"]-t["syndrome"]-t["logical"]-.1*t["mechanism"])<1e-6
        best=max(h["eval"],key=lambda e:e["raw"]["logical_accuracy"])
        assert best["epoch"]==h["best_epoch"] and best["raw"]["logical_accuracy"]==h["best_raw_accuracy"]
        index=int(next(s for s in specs if f"--seed={c['seed']}" in s["arguments"])["experiment_index"])
        worker=(DEST/f"log_exp_{index}.txt").read_text()
        assert "DUE TO TIME LIMIT" in worker
        for log in (worker,path.with_name("training_log.txt").read_text()):
            assert "[Final selected best]" not in log and "Traceback (most recent call last)" not in log
            for label,key in (("Train","train"),("Validation","eval")):
                found=re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$",log,re.M)
                assert [dict(epoch=int(e),**json.loads(v)) for e,v in found]==h[key]
            assert json.loads(re.search(r"^DEM CNN configuration: (.+)$",log,re.M)[1])==c
        meta=dict(job=60310800,code="bb144",depth=12,p=p,seed=c["seed"],source=str(path.relative_to(ROOT)))
        last=h["eval"][-1]
        row=dict(**meta,status="interrupted_time_limit",completed_epochs_in_history=completed,
                 remaining_to_800_inferred=800-completed,training_shots=completed*512,
                 validation_evaluations=len(h["eval"]),validation_shots_each=1024,
                 last_validation_epoch=last["epoch"]+1,last_validation_ler=last["raw"]["logical_error_rate"],
                 best_epoch=best["epoch"]+1,best_validation_ler=best["raw"]["logical_error_rate"],
                 best_validation_ties=sum(e["raw"]["failures"]==best["raw"]["failures"] for e in h["eval"]),
                 logical_loss_last50=statistics.mean(t["logical"] for t in h["train"][-50:]),
                 final_available=False,checkpoint_verified=False,shot_bank_verified=False,
                 audit_level="history_and_logs_only")
        rows.append(row)
        train.extend(dict(**meta,**t) for t in h["train"])
        validation.extend(dict(**meta,epoch=e["epoch"]+1,**e["raw"]) for e in h["eval"])
        for depth,job in ((4,60310803),(6,60316182),(12,60310800)):
            if depth==12:
                old=h
            else:
                source=next((base.ARCHIVE/"bb144"/f"resdir_{job}").glob(f"outputs/*/*p{p:g}_*/history.json"))
                old=base.read(source)
                normalized={k:v for k,v in c.items() if k not in ("depth","parameter_count")}
                assert normalized=={k:v for k,v in old["config"].items() if k not in ("depth","parameter_count")}
                assert old["phases"]==h["phases"]
            common=next(e for e in old["eval"] if e["epoch"]==389)
            comparisons.append(dict(code="bb144",p=p,depth=depth,epoch=390,shots=1024,
                failures=common["raw"]["failures"],ler=common["raw"]["logical_error_rate"],
                comparison="same_epoch_validation_not_final",paired_verified=False))
    rows.sort(key=lambda r:r["p"])
    assert {r["p"] for r in rows}=={.001,.002,.004,.006}
    for name,data in (("summary",rows),("train",train),("validation",validation),("epoch390",comparisons),("snapshots",snapshots)):
        write_csv(name,data)
    return rows,dict(elapsed_job_hours=(stop-start).total_seconds()/3600,
        models=4,final_results=0,train_rows=len(train),validation_rows=len(validation),
        timeout_confirmed_in_all_worker_logs=True,source_snapshots=11,
        all_snapshots_match_current=all(r["matches_current"] for r in snapshots),
        checkpoint_verified=False,shot_bank_verified=False,training_steps_added=0,inference_steps_added=0)


def report(rows,audit):
    lines=["# 2026-10-28: BB144 depth 12 중단 기록", "",
        "새 폴더 `resdir_60310800`은 원래 array 60310800/7의 BB144 D12 실험이다. "
        "재개 학습 결과가 아니며, 네 worker 로그 모두 `DUE TO TIME LIMIT`을 확인했다. "
        "로그 기준 24시간 후 중단됐고 800 epochs와 최종 8,192-shot 평가를 완료하지 못했다.","",
        f"{audit['files']}개 파일 ({audit['bytes']:,} bytes)을 "
        "`results/bb/circuit/standard/dem_cnn/interrupted/bb144/resdir_60310800/`에 보관했다. "
        "이동 전후 해시 및 train/validation 로그와 history를 대조했다. "
        "model.pt와 final_shots.npz는 로컬에 없어 checkpoint epoch/샷별 재현 검증은 할 수 없다.","",
        "## 진행 상황", "",
        "Epoch는 완료 횟수(1-based)다. 남은 횟수는 history 기준이며 실제 재개 스크립트는 model.pt를 읽어 계산한다.","",
        "| p | 완료 epochs | 800까지 남음 | 마지막 validation epoch | 마지막 validation LER | best epoch | best validation LER |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['p']:.3f} | {r['completed_epochs_in_history']} | {r['remaining_to_800_inferred']} | "
                     f"{r['last_validation_epoch']} | {r['last_validation_ler']:.2%} | {r['best_epoch']} | {r['best_validation_ler']:.2%} |")
    lines += ["", "각 validation은 1,024샷이다. Best는 여러 validation 중 선택된 값이므로 "
        "최종 독립 test LER로 해석하면 안 된다. 특히 p=.004/.006은 모든 validation에서 성공 0건으로 "
        "epoch 5가 best로 남아 있다.","", "## 같은 epoch에서의 비교", "",
        "서로 다른 학습 길이의 best 값을 비교하는 대신 모든 모델의 epoch 390 validation을 비교했다. "
        "Noise·seed·graph·training schedule은 일치하지만 샷 파일이 없어 paired 검정은 하지 않았다.","",
        "| p | D4 | D6 | D12 |", "|---:|---:|---:|---:|",
        "| .001 | 80.96% | 82.03% | 33.01% |",
        "| .002 | 99.02% | 96.39% | 77.64% |",
        "| .004 | 100% | 100% | 100% |",
        "| .006 | 100% | 100% | 100% |", "",
        "낮은 p에서 D12의 학습 진전은 D4/D6보다 빠르게 나타난다. "
        "그러나 아직 D12의 800-epoch final 결과는 없고, 더 학습했을 때 개선량은 미확정이다. "
        "높은 p에서는 여전히 유효한 correction이 관측되지 않는다. "
        "마지막 50 epochs logical parity loss 평균은 p=.001/.002/.004/.006에서 "
        + ", ".join(f"{r['logical_loss_last50']:.4f}" for r in rows)+"다. "
        "p=.006은 ln(2)≈0.693 부근으로 학습 병목을 보이지만 원인은 이 로그만으로 확정할 수 없다.", "",
        "## 플롯과 재개", "",
        "Overview는 완료된 24개 모델의 final LER만 유지한다. D12의 중간 validation을 final 점으로 넣지 않았다. "
        "Validation 및 logical-loss 그림에는 D12 곡선을 점선으로 추가하고 기록이 끝난 지점에서 멈췄다. "
        "BB144 D8 결과는 여전히 미수신이다.", "",
        "서버에 재개 스크립트를 업로드했다면 `sbatch --export=ALL,BB_DEM_DEPTH=12 run_bb_dem_resume.slurm`으로 "
        "원래 checkpoint에서 재개할 수 있다. 파일 이동은 로컬 archive만 변경했다. "
        "이미 재개 job을 제출했다면 중복 제출하지 않는다. "
        "현재 속도로는 남은 약 400 epochs도 다음 24시간 제한에 근접할 수 있다.", "",
        "[Overview](../plots/bb/circuit/standard/dem_cnn/overview.png) · "
        "[Validation](../plots/bb/circuit/standard/dem_cnn/validation.png) · "
        "[Logical loss](../plots/bb/circuit/standard/dem_cnn/logical_loss.png)","",
        f"[요약]({STEM}_summary.csv) · [동일 epoch 비교]({STEM}_epoch390.csv) · "
        f"[검증]({STEM}_audit.json) · [이동 기록]({STEM}_relocation.csv)","",
        "재생성: `python scripts/analyze_bb_dem_interrupted.py`",""]
    (OUT/f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",action="store_true")
    args=parser.parse_args()
    audit=archive(args.archive)
    rows,checks=analyze(); audit.update(checks)
    (OUT/f"{STEM}_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    report(rows,audit)
    finals,training,validation,_,_=base.analyze()
    base.plots(finals,training,validation)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
