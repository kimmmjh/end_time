#!/usr/bin/env python3
"""Audit/archive the Nov 4 download and consolidate the Nov 2 follow-up campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_dem_november3 as prior

STEM = "bb_dem_2026_11_04"
OUT = prior.OUT
JOBS = {60688774: "train", 60688775: "evaluate", 60713197: "evaluate",
        60714731: "train", 60725684: "evaluate", 60739386: "evaluate"}
DESTINATIONS = {f"resdir_{j}_followup_{m}": prior.DEM / "followup_2026_11_02" / m / f"resdir_{j}_followup_{m}"
                for j, m in JOBS.items()}


def write_csv(suffix, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def archive(move):
    manifest = OUT / f"{STEM}_relocation.csv"
    if not manifest.exists():
        assert move, "First import requires --archive"
        rows = []
        # Audit every source before moving any directory.
        for name, dest in DESTINATIONS.items():
            source = ROOT / name
            assert source.is_dir() and not dest.exists()
            prior.audit_job(source, jobs=JOBS)
            for file in sorted(source.rglob("*")):
                if file.is_file():
                    rows.append(dict(old_path=str(file.relative_to(ROOT)),
                                     new_path=str((dest / file.relative_to(source)).relative_to(ROOT)),
                                     bytes=file.stat().st_size, sha256=prior.base.sha(file)))
        write_csv("relocation", rows)
    rows = list(csv.DictReader(manifest.open()))
    for name, dest in DESTINATIONS.items():
        source = ROOT / name
        if move and source.exists():
            assert not dest.exists(), "Refusing to merge or overwrite an archived snapshot"
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.rename(dest)
    files = {str(p.relative_to(ROOT)) for d in DESTINATIONS.values() for p in d.rglob("*") if p.is_file()}
    assert files == {r["new_path"] for r in rows}
    for r in rows:
        path = ROOT / r["new_path"]
        assert path.stat().st_size == int(r["bytes"]) and prior.base.sha(path) == r["sha256"]
    return dict(folders=len(DESTINATIONS), files=len(rows), bytes=sum(int(r["bytes"]) for r in rows),
                hashes_verified=True)


def table(lines, headers, rows):
    lines += ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    lines.append("")


def report(training, evaluations, comparisons, info):
    new_training = [r for r, _ in training if r["job"] in JOBS]
    new_points = {(r["code"], r["depth"], r["p"]) for r, _ in evaluations if r["job"] in JOBS}
    new_pairs = [r for r in comparisons if (r["code"], r["depth"], r["p"]) in new_points]
    lines = ["# 2026-11-04: 추가 결과 수신 및 누적 분석", "",
             "새로 받은 결과는 **기존 800→1,000 epoch continuation과 기존 800-epoch 모델의 best/latest 평가**다. "
             "최근 작성한 BB72 D8 / BB144 D6, p=0.002의 **200만 샘플 실험 결과는 포함되지 않았다**.", "",
             "## 핵심 결과", "",
             "- **BB144 p=0.4%에서 checkpoint 선택 효과가 크다.** D6 hierarchical 실패율은 best(epoch 5) "
             "94.727% → latest(epoch 800) 84.131%, D8은 94.360% → 68.835%다. "
             "이는 1,000 epoch 추가 학습 효과가 아니라 같은 800-epoch run 내 가중치 선택 차이다.",
             "- BB144 D12는 1,000 epoch까지 추가 학습한 뒤 raw 실패율이 p=0.1%에서 15.735% → 13.623%, "
             "p=0.2%에서 58.704% → 56.592%로 낮아졌다. Hierarchical 개선 여부는 아직 평가되지 않았다.",
             "- BB72 D12 p=0.4% raw는 81.445% → 80.139%다. p=0.6%는 99.792% → 99.829%로 개선이 보이지 않는다.",
             "- BB144 p=0.6% hierarchical은 새 D6/D8/D12 결과에서도 약 100%다. "
             "0개와 1개 성공의 차이는 실용적 성능 향상의 근거로 보기 어렵다.", "",
             "## 새 폴더 및 누적 수신 상태", "",
             "새 6개 폴더는 모두 완료 표시와 worker exit code 0을 확인했다. "
             "`results/bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`로 이동했고 "
             "원래 폴더 이름과 파일 내용은 보존했다.", ""]
    table(lines, ["Job", "종류", "새 완료 수"], [
        [j, m, sum(r["job"] == j for r, _ in (training if m == "train" else evaluations))]
        for j, m in JOBS.items()])
    lines += ["- 학습 21개: **17개 1,000 epochs 완료 + 4개 과거 중간 snapshot**, 미수신 worker 없음.",
              "- 평가 22개: **18개 완료 + 4개 과거 중간 snapshot**, 미수신 worker 없음.",
              "- 중간 snapshot의 서버 현재 상태는 알 수 없다. 이번 다운로드가 과거 미완료 폴더를 갱신하지 않았다.", "",
              "## 새 800→1,000 epoch 학습 결과", "",
              "Raw는 syndrome과 logical observable을 모두 맞춰야 성공인 block 실패율이다. "
              "각 final은 validation-selected best weights를 8,192 fresh circuit shots로 평가했다. "
              "전후 평가 bank는 다르므로 paired 비교가 아니다.", ""]
    headers = ["코드", "D", "p (%)", "800 예산 raw (%)", "1,000 예산 raw (%)", "차이 (%p)", "Best epoch"]
    def train_rows(rows):
        return [[r["code"], r["depth"], f"{100*r['p']:.1f}", f"{100*r['old_final_rate']:.3f}",
                 f"{100*r['block_rate']:.3f}", f"{r['delta_pp']:+.3f}",
                 f"{r['previous_best_epoch']}→{r['best_epoch']}"]
                for r in sorted(rows, key=lambda r:(r["code"],r["depth"],r["p"]))]
    table(lines, headers, train_rows(new_training))
    lines += ["BB144 D12 p=0.4%는 best가 epoch 768로 그대로다. "
              "99.976% → 99.988% 차이를 추가 학습에 의한 악화로 해석하지 않는다. "
              "같은 선택 가중치의 재평가 차이이며 `.pt`가 없어 가중치 해시는 확인하지 못했다.", "",
              "## 새 hierarchical best/latest 비교", "",
              "모두 기존 800-epoch run의 가중치이며, 각 8,192 shots / confidence 0.9 / residual DirectOSD0다. "
              "Best는 raw validation 기준이고, latest는 epoch 800이다. 아래 구간은 "
              "각 평가의 marginal Wilson 95% 구간이며 paired 차이의 신뢰구간이 아니다.", ""]
    table(lines, ["코드", "D", "p (%)", "Best epoch", "Best 실패/8192", "Best (%) [95% CI]",
                  "Latest 실패/8192", "Latest (%) [95% CI]", "차이 (%p)"], [
        [r["code"], r["depth"], f"{100*r['p']:.1f}", r["best_epoch"], r["best_failures"],
         f"{100*r['best_rate']:.3f} [{100*r['best_low']:.3f}, {100*r['best_high']:.3f}]",
         r["latest_failures"], f"{100*r['latest_rate']:.3f} [{100*r['latest_low']:.3f}, {100*r['latest_high']:.3f}]",
         f"{r['latest_minus_best_pp']:+.3f}"] for r in new_pairs])
    lines += ["해석:", "",
              "- BB144 D6/D8 p=0.4%의 큰 차이는 raw 성공률이 모두 0에 가까워 초기 epoch 5가 best로 남는 "
              "선택 방식이 hierarchical 성능을 잘 구분하지 못함을 보여준다. "
              "Latest의 mechanism score가 residual decoding에 더 유용해졌다는 해석과 일치하지만, "
              "LLR calibration이나 내부 표현의 개선을 직접 측정한 것은 아니다.",
              "- 같은 p=0.4% / latest epoch 800 조건에서 BB144 D6 84.131%, D8 68.835%다. "
              "이 지점에서는 깊은 모델의 이득이 관측된다. 모든 p에서 depth 효과가 없다고 결론 내리면 안 된다. "
              "다만 D12의 해당 hierarchical 평가는 없고 다중 seed 반복도 없어 일반적 단조 개선을 주장하지 않는다.",
              "- BB72 D6/D8 p=0.6%도 latest가 각각 2.197%p/1.855%p 낮지만 실패율은 여전히 94.116%/93.286%다.",
              "- 이전 BB72 D4 p=0.6%에서는 latest가 더 나빴다. Latest를 무조건 채택하는 대신 "
              "최종 hierarchical 지표로 checkpoint를 검증하는 것이 타당하다.",
              "- 새 결과의 hierarchical syndrome failure와 raw 대비 harmed는 모두 0이다. "
              "남은 실패는 syndrome-valid correction의 logical class 불일치다. Direct CNN+OSD 비교는 없어 "
              "계층형 local 단계 자체의 이득을 분리할 수 없다.", "",
              "## 다음 실험에 대한 판단", "",
              "BB72 D8 / BB144 D6, p=0.002의 200만 샘플 실험을 먼저 진행하는 계획은 유지할 만하다. "
              "이번 결과는 다른 p로 즉시 확대해야 한다는 근거라기보다 **hierarchical 기준 checkpoint 평가가 "
              "필수**라는 근거를 강화한다. 새 sample-scaling 스크립트는 baseline/세 milestone의 best/latest를 "
              "비교하고 독립 최종 bank를 사용하므로 이 문제를 다룬다. BB144 D12 p=0.001/0.002의 새 1,000-epoch "
              "모델도 추후 hierarchical 평가 후보지만, 현재 raw 결과만으로 우열을 정하지 않는다.", "",
              "## 아직 중간 snapshot만 있는 항목", ""]
    table(lines, ["Job", "Worker", "코드/D/p", "수신 epochs", "마지막 validation epoch"], [
        [r["job"], r["worker"], f"{r['code']}/D{r['depth']}/{r['p']:.1%}", r["epochs"], r["last_validation_epoch"]]
        for r, _ in training if r["status"] != "complete"])
    table(lines, ["Job", "Worker", "코드/D/p", "Weights", "수신 shots"], [
        [r["job"], r["worker"], f"{r['code']}/D{r['depth']}/{r['p']:.1%}", r["weights"], r["shots"]]
        for r, _ in evaluations if r["status"] != "complete"])
    lines += ["서버에서 `resdir_60701325_followup_train`과 `resdir_60701066_followup_evaluate`가 "
              "완료됐는지 확인하고 최신 파일을 추가로 받으면 누적 캠페인의 남은 상태를 확정할 수 있다. "
              "현재 자료만으로 완료나 실패를 추정하지 않는다.", "",
              "## 검증 및 통계적 한계", "",
              f"- 새 {info['folders']}개 폴더 / {info['files']}개 파일 / {info['bytes']:,} bytes를 이동 전후 SHA-256으로 검증했다.",
              "- 각 job의 array task, manifest, worker 명령, source config, exit code, 완료 표시를 검증했다.",
              "- 학습 이력의 기존 800-epoch prefix, LR 3e-5 continuation, loss 합, validation/final 수치, "
              "best 선택과 로그를 대조했다. 평가 누적 failure/shot 수, Wilson 구간, rescued/harmed 관계를 확인했다.",
              f"- 새 source snapshot {info['new_source_snapshot_files']}개 중 {info['new_matching_source_snapshots']}개가 "
              "현재 코드와 일치한다. 차이 파일은 snapshots CSV에 기록했다. 현재의 폴더 이름 규칙과 "
              "sample-scaling용 scheduler 저장 기능 등 이후 변경을 과거 실행에 소급하지 않는다.",
              f"- 새 `.pt` {info['checkpoint_files']}개 / `.npz` {info['shot_files']}개. "
              "Checkpoint 내용 및 실제 shot bank 일치는 확인할 수 없고, paired 유의성 검정도 하지 않았다. "
              "메타데이터의 sampler seed/batch/noise 설정은 pair 간 일치한다.",
              f"- 기존 baseline과 겹치는 새 best 평가 {info['new_repeated_baseline_points']}개는 raw/hierarchical "
              "실패율이 기존과 동일하다. 독립 반복처럼 합산하지 않았다.",
              "- Raw overview는 **1,000-epoch 예산 17개 + 800-epoch 예산 15개**다. 동일 학습 예산의 depth 비교가 아니다. "
              "Hierarchical 그림은 모두 800-epoch run에서 선택한 가중치이며 1,000-epoch 평가와 혼합하지 않았다.", "",
              "## 누적 완료 continuation", ""]
    table(lines, headers, train_rows([r for r, _ in training if r["status"] == "complete"]))
    lines += ["## 갱신한 플롯 및 데이터", "",
              "- [Raw + hierarchical 통합 그림](../plots/bb/circuit/standard/dem_cnn/overview.png)",
              "- 한 그림의 왼쪽은 raw, 오른쪽은 hierarchical이고 위/아래는 BB72/BB144다. "
              "Raw는 최신 완료 학습 run의 raw-validation best를 표시한다. "
              "Hierarchical은 완료 평가 중 각 code/depth/p의 최저 관측 LER checkpoint 하나를 표시한다. "
              "동일 평가 샷으로 선택했으므로 독립 최종 성능 추정치가 아니다. "
              "현재 hierarchical 가중치는 모두 800-epoch run에서 왔으며 1,000-epoch 평가는 대기 중이다.",
              "- 전후 비교·학습곡선·진단 그림은 제거했으며 수치와 이력은 표/CSV에 보존했다.", "",
              f"[학습 CSV]({STEM}_training.csv) · [평가 CSV]({STEM}_evaluations.csv) · "
              f"[Best/latest CSV]({STEM}_best_latest.csv) · [검증 JSON]({STEM}_audit.json)", "",
              "재생성: `python scripts/analyze_bb_dem_november4.py`. 최초 정리는 `--archive`. "
              "학습/추론을 실행하지 않는다. 공용 plot 경로는 최신 누적 그림이며 과거 날짜 보고서는 당시 기록이다.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    if (OUT / "bb_dem_2026_11_05_replacement.csv").exists():
        from scripts.analyze_bb_dem_november5 import main as latest_main
        print("Partial downloads were replaced; rebuilding the November 5 report and combined plot.")
        return latest_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    info = archive(args.archive)
    training, evaluations, snapshots = [], [], []
    for directory in [*prior.DESTINATIONS.values(), *DESTINATIONS.values()]:
        a, b, c = prior.audit_job(directory, jobs={**prior.JOBS, **JOBS})
        training.extend(a); evaluations.extend(b); snapshots.extend(c)
    assert len({r["worker"] for r, _ in training}) == len(training) == 21
    assert len({r["worker"] for r, _ in evaluations}) == len(evaluations) == 22
    comparisons = prior.common_comparisons(evaluations)
    old = list(csv.DictReader((OUT / "bb_dem_2026_11_02_hierarchical_summary.csv").open()))
    old_keys = {(r["code"], int(r["depth"]), float(r["p"])) for r in old}
    write_csv("training", [r for r, _ in training])
    write_csv("evaluations", [r for r, _ in evaluations])
    write_csv("best_latest", comparisons)
    write_csv("snapshots", snapshots)
    progress = [dict(job=row["job"], worker=row["worker"], code=row["code"], depth=row["depth"],
                     p=row["p"], weights=row["weights"], **prior.flatten_result(result))
                for row, logs in evaluations for result in logs.values()]
    write_csv("evaluation_progress", progress)
    info.update(training_received=len(training), training_complete=sum(r["status"] == "complete" for r, _ in training),
                training_partial=sum(r["status"] != "complete" for r, _ in training), training_missing=21-len(training),
                evaluation_received=len(evaluations), evaluation_complete=sum(r["status"] == "complete" for r, _ in evaluations),
                evaluation_partial=sum(r["status"] != "complete" for r, _ in evaluations), evaluation_missing=22-len(evaluations),
                new_training_complete=sum(r["job"] in JOBS for r, _ in training),
                new_evaluation_complete=sum(r["job"] in JOBS for r, _ in evaluations),
                evaluation_shots_received=sum(r["shots"] for r, _ in evaluations),
                new_evaluation_shots=sum(r["shots"] for r, _ in evaluations if r["job"] in JOBS),
                evaluation_progress_rows=len(progress),
                new_source_snapshot_files=sum(r["job"] in JOBS for r in snapshots),
                new_matching_source_snapshots=sum(r["matches_current"] and r["job"] in JOBS for r in snapshots),
                checkpoint_files=sum(len(list(d.rglob("*.pt"))) for d in DESTINATIONS.values()),
                shot_files=sum(len(list(d.rglob("*.npz"))) for d in DESTINATIONS.values()),
                new_repeated_baseline_points=sum(r["job"] in JOBS and r["weights"] == "best" and
                    (r["code"], r["depth"], r["p"]) in old_keys for r, _ in evaluations),
                independent_shot_audit=False, sample_scaling_results_received=False)
    from scripts.plot_bb_dem_latest import render
    render(prior.baseline(), training, evaluations)
    report(training, evaluations, comparisons, info)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
