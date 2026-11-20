#!/usr/bin/env python3
"""Import resumed DEM results and regenerate combined plots without inference."""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_dem_october28 as base

STEM = "bb_dem_2026_11_01"
JOBS = {60318790: (8, "interrupted"), 60428756: (8, "resumed_800_epochs"),
        60428753: (12, "interrupted")}


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (base.OUT / f"{STEM}_{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    destinations = {job: base.ARCHIVE.parent / category / "bb144" / f"resdir_{job}"
                    for job, (_, category) in JOBS.items()}
    manifest = base.OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert args.archive, "First import requires --archive"
        rows = []
        for job, dest in destinations.items():
            source = ROOT / f"resdir_{job}"
            assert source.is_dir() and not dest.exists()
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    rows.append(dict(old_path=str(p.relative_to(ROOT)),
                        new_path=str((dest / p.relative_to(source)).relative_to(ROOT)),
                        bytes=p.stat().st_size, sha256=base.sha(p)))
        write_csv("relocation", rows)
    rows = list(csv.DictReader(manifest.open()))
    for job, dest in destinations.items():
        source = ROOT / f"resdir_{job}"
        if args.archive and source.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.rename(dest)
    for r in rows:
        p = ROOT / r["new_path"]
        assert p.stat().st_size == int(r["bytes"]) and base.sha(p) == r["sha256"]
    assert {str(p.relative_to(ROOT)) for d in destinations.values() for p in d.rglob("*") if p.is_file()} == {r["new_path"] for r in rows}
    finals, training, validation, _, _ = base.analyze()
    summaries, snapshots = [], []
    for job, (depth, category) in JOBS.items():
        dest = destinations[job]
        complete = category == "resumed_800_epochs"
        assert (dest / ("completed.txt" if complete else "interrupted.txt")).exists()
        assert not (dest / "failed.txt").exists()
        for path in (dest / "source_snapshot").rglob("*.py"):
            relative = path.relative_to(dest / "source_snapshot")
            current = ROOT / relative
            snapshots.append(dict(job=job, file=str(relative), sha256=base.sha(path),
                matches_current=current.exists() and base.sha(path)==base.sha(current)))
        paths = sorted(dest.glob("outputs/*/*/history.json"))
        assert len(paths)==4
        seen = set()
        for path in paths:
            h = base.read(path)
            c = h["config"]
            p = c["noise_profile"]["base_error_rate"]
            seen.add(p)
            assert c["code"]=="bb144" and c["depth"]==depth and c["architecture"]=="bb_dem_cnn"
            reference = next(r for r in finals if r["code"]=="bb144" and r["depth"]==4 and r["p"]==p)
            rc = base.read(ROOT / reference["source"])["config"]
            assert {k:v for k,v in c.items() if k not in ("depth","parameter_count")} == {k:v for k,v in rc.items() if k not in ("depth","parameter_count")}
            assert [t["epoch"] for t in h["train"]]==list(range(len(h["train"])))
            assert all(math.isfinite(v) for t in h["train"] for v in t.values())
            for e in h["eval"]:
                base.audit_evaluation(e,1024)
            best=max(h["eval"],key=lambda e:e["raw"]["logical_accuracy"])
            assert h["best_epoch"]==best["epoch"]
            start=h["phases"][-1]["start_epoch"]
            index=(.001,.002,.004,.006).index(p)
            if job != 60318790:
                plan=base.read(dest / f"resume_exp_{index}.json")
                assert plan["completed_epochs"]==start and plan["depth"]==depth and plan["p"]==p
                assert start+h["phases"][-1]["epochs"]==800
                assert h["phases"][-1]["schedule"]=="cosine_restart"
                old_job=60318790 if depth==8 else 60310800
                old_dir=base.ARCHIVE.parent / "interrupted" / "bb144" / f"resdir_{old_job}"
                old=next(base.read(q) for q in old_dir.glob("outputs/*/*/history.json") if base.read(q)["config"]["noise_profile"]["base_error_rate"]==p)
                assert h["train"][:min(start,len(old["train"]))]==old["train"][:start]
            for logpath in (dest / f"log_exp_{index}.txt",path.with_name("training_log.txt")):
                log=logpath.read_text()
                assert "Traceback (most recent call last)" not in log
                for label,key in (("Train","train"),("Validation","eval")):
                    logged=[dict(epoch=int(ep),**json.loads(value)) for ep,value in re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$",log,re.M)]
                    expected=[r for r in h[key] if r["epoch"]>=start]
                    assert logged[:len(expected)]==expected, (logpath,label)
                found=re.findall(r"^\[Final selected best\] (.+)$",log,re.M)
                assert (len(found)==1 and json.loads(found[0])==h["final"]) if complete else not found
            meta=dict(job=job,code="bb144",depth=depth,p=p,source=str(path.relative_to(ROOT)))
            summaries.append(dict(**meta,epochs=len(h["train"]),remaining_epochs=800-len(h["train"]),
                resumed_from_epoch=start,final_available="final" in h,best_epoch=best["epoch"]+1,
                best_validation_ler=best["raw"]["logical_error_rate"],last_validation_epoch=h["eval"][-1]["epoch"]+1,
                last_validation_ler=h["eval"][-1]["raw"]["logical_error_rate"],
                final_ler=h.get("final",{}).get("raw",{}).get("logical_error_rate")))
            if complete:
                assert len(h["train"])==800
                assert (dest/f"exit_code_exp_{index}.txt").read_text().strip()=="0"
                f=h["final"]
                base.audit_evaluation(f,8192)
                assert f["epoch"]==best["epoch"]
                row=dict(**meta,epochs=800,shots=8192,best_epoch=f["epoch"]+1,resumed=True)
                for short,key in (("block","logical_error_rate"),("flagged","flagged_failure"),("unflagged","unflagged_failure"),("observable","observable_mismatch_rate")):
                    count,lo,hi=base.interval(f["raw"][key],8192)
                    row.update({f"{short}_rate":f["raw"][key],f"{short}_count":count,f"{short}_low":lo,f"{short}_high":hi})
                finals.append(row)
                training.extend(dict(**meta,**t) for t in h["train"])
                validation.extend(dict(**meta,epoch=e["epoch"]+1,**e["raw"]) for e in h["eval"])
            else:
                assert "final" not in h
                assert "DUE TO TIME LIMIT" in (dest/f"log_exp_{index}.txt").read_text()
        assert seen=={.001,.002,.004,.006}
    write_csv("summary",summaries)
    write_csv("final",finals)
    write_csv("snapshots",snapshots)
    base.plots(finals,training,validation)
    audit=dict(files=len(rows),sha256_verified=True,final_models=len(finals),new_final_models=4,
        incomplete_models=4,checkpoint_verified=False,shot_bank_verified=False,
        source_snapshot_files=len(snapshots),all_snapshots_match_current=all(s["matches_current"] for s in snapshots))
    (base.OUT/f"{STEM}_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    lines=["# 2026-11-01: BB144 DEM CNN 재개 결과", "",
        "D8은 네 p 모두 총 800 epochs와 final 8,192-shot 평가를 완료했다. D12는 재개 후 다시 24시간 제한으로 중단됐으며 final은 없다.", "",
        "원래 D8 중단 기록은 interrupted/bb144, 완료 D8은 resumed_800_epochs/bb144, 재중단 D12는 interrupted/bb144에 보관했다. 이동 SHA-256 및 로그/history/재개 계획을 검증했다.", "",
        "## 최종 LER / block failure (%)", "",
        "| p | D4 | D6 | D8 (resumed) |", "|---|---:|---:|---:|"]
    for p in (.001,.002,.004,.006):
        rates=[next(r["block_rate"] for r in finals if r["code"]=="bb144" and r["depth"]==d and r["p"]==p) for d in (4,6,8)]
        lines.append(f"| {p} | "+" | ".join(f"{100*r:.3f}" for r in rates)+" |")
    lines += ["", "D8은 p=.001/.002에서 D4/D6보다 개선됐다. p=.004/.006은 final 8,192샷 모두 실패했다. 이 두 모델의 best는 epoch 5이며, validation 성공률 동률로 초기 checkpoint가 유지됐다.", "",
        "## D12: 미완료 validation (1,024 shots)", "", "| p | 학습 epoch | 남은 epoch | 마지막 validation epoch | 마지막 LER % | best epoch | best LER % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in summaries if r["depth"]==12),key=lambda r:r["p"]):
        lines.append(f"| {r['p']} | {r['epochs']} | {r['remaining_epochs']} | {r['last_validation_epoch']} | {100*r['last_validation_ler']:.3f} | {r['best_epoch']} | {100*r['best_validation_ler']:.3f} |")
    lines += ["", "D12의 validation 개선은 유망하지만 final 결과와 직접 비교하지 않는다. Overview에는 완료된 28개 모델만 넣고, D12는 validation/loss 그림의 점선으로 표시했다.", "",
        "두 재개 실험은 저장된 LR에서 cosine을 다시 시작하고 독립 sampler phase를 사용한다. 중단 없는 800 epochs와 동일한 실험은 아니다. OSD는 전부 꺼져 있다.", "",
        "LER은 syndrome 불일치 또는 observable 불일치의 합집합인 전체 block failure이며 round당 LER이 아니다. .pt/.npz 제외 다운로드로 가중치 재평가·샷별 검증은 하지 않았다. 한 seed 결과이며 depth 간 학습 스케줄도 다르다.", "",
        "재생성: `python scripts/analyze_bb_dem_november1.py`", ""]
    (base.OUT/f"{STEM}.md").write_text("\n".join(lines))
    print(json.dumps(dict(audit=audit,summary=summaries),indent=2))


if __name__=="__main__":
    main()
