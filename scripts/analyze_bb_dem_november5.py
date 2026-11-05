#!/usr/bin/env python3
"""Replace the two partial downloads with audited completed records and refresh plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_bb_dem_november3 as prior
from scripts import analyze_bb_dem_november4 as previous
from scripts.plot_bb_dem_latest import emit_csv, render

STEM = "bb_dem_2026_11_05"
REFRESH = {60701325: "train", 60701066: "evaluate"}
MANIFEST = prior.OUT / f"{STEM}_replacement.csv"


def files(directory):
    return {str(p.relative_to(directory)): p for p in directory.rglob("*") if p.is_file()}


def verify(directory, rows):
    actual = files(directory)
    assert set(actual) == {r["relative_path"] for r in rows}
    for row in rows:
        p = actual[row["relative_path"]]
        assert p.stat().st_size == int(row["bytes"])
        assert prior.base.sha(p) == row["sha256"]


def refresh(allow_replace):
    if not MANIFEST.exists():
        assert allow_replace, "First import requires --archive"
        rows = []
        # Validate both incoming folders before changing either archived copy.
        for job, mode in REFRESH.items():
            name = f"resdir_{job}_followup_{mode}"
            source, dest = ROOT / name, prior.DESTINATIONS[name]
            new_train, new_eval, _ = prior.audit_job(source)
            old_train, old_eval, _ = prior.audit_job(dest)
            assert all(r["status"] == "complete" for r, _ in new_train + new_eval)
            old_files, new_files = files(dest), files(source)
            assert old_files.keys() <= new_files.keys(), "Incoming folder omits archived files"
            for (old, oh), (new, nh) in zip(old_train, new_train):
                assert old["worker"] == new["worker"]
                for field in ("train", "eval"):
                    assert nh[field][:len(oh[field])] == oh[field]
            for (old, logs), (new, updated) in zip(old_eval, new_eval):
                assert old["worker"] == new["worker"] and old["shots"] < new["shots"]
                assert all(updated[n] == r for n, r in logs.items())
            for relative, p in sorted(new_files.items()):
                old_hash = prior.base.sha(old_files[relative]) if relative in old_files else ""
                digest = prior.base.sha(p)
                if old_hash and old_hash != digest:
                    assert relative.endswith(("history.json", "results.json", "training_log.txt")) or relative.startswith("log_worker_")
                    if p.suffix == ".txt":
                        assert p.read_bytes().startswith(old_files[relative].read_bytes())
                rows.append(dict(job=job, relative_path=relative,
                                 new_path=str((dest / relative).relative_to(ROOT)),
                                 bytes=p.stat().st_size, sha256=digest, previous_sha256=old_hash))
        emit_csv(MANIFEST, rows)
    rows = list(csv.DictReader(MANIFEST.open()))
    for job, mode in REFRESH.items():
        name = f"resdir_{job}_followup_{mode}"
        source, dest = ROOT / name, prior.DESTINATIONS[name]
        backup = dest.with_name(dest.name + ".superseded")
        group = [r for r in rows if int(r["job"]) == job]
        if source.exists():
            assert allow_replace, "Incoming folders still present; use --archive"
            verify(source, group)
            if dest.exists():
                assert not backup.exists()
                dest.rename(backup)
            source.rename(dest)
        verify(dest, group)
        if backup.exists():
            # Remove the obsolete copy only after all replacement bytes are verified.
            expected = {r["relative_path"]: r["previous_sha256"] for r in group if r["previous_sha256"]}
            assert set(files(backup)) == set(expected)
            assert all(prior.base.sha(p) == expected[k] for k, p in files(backup).items())
            shutil.rmtree(backup)
    return dict(folders=2, files=len(rows), bytes=sum(int(r["bytes"]) for r in rows),
                hashes_verified=True, previous_copies_removed=True)


def report(training, evaluations, comparisons, raw, hierarchical, info):
    lines = ["# 2026-11-05: 중간 사본을 완료 기록으로 교체", "",
             "`60701325_followup_train`과 `60701066_followup_evaluate`를 다시 받아 기존 중간 사본을 교체했다. "
             "두 폴더 모두 완료 표시와 worker exit code 0을 확인했다. "
             "이제 누적 **학습 21개 모두 1,000 epoch 완료, 평가 22개 모두 8,192 shots 완료**이며 미완료/미수신 worker가 없다. "
             "이는 기존 follow-up 캠페인의 완료 기록이고, 200만 샘플 실험 결과는 아니다.", "",
             "## 갱신된 추가 학습 결과", "",
             "Raw block failure는 syndrome과 logical observable을 모두 맞춰야 성공인 지표다. "
             "각 학습 예산의 raw-validation best를 별도 8,192-shot final bank로 평가했다.", ""]
    previous.table(lines, ["코드", "Depth", "p", "800 epoch raw (%)", "1,000 epoch raw (%)", "차이 (%p)", "Best epoch"], [
        [r["code"], r["depth"], r["p"], f"{100*r['old_final_rate']:.3f}",
         f"{100*r['block_rate']:.3f}", f"{r['delta_pp']:+.3f}",
         f"{r['previous_best_epoch']} → {r['best_epoch']}"]
        for r, _ in training if r["job"] == 60701325])
    lines += ["- **BB144 D8**: p=0.001은 35.950% → 34.106%, p=0.002는 78.894% → 74.329%다. "
              "Best epoch도 각각 950/985로 갱신되어 추가 학습의 이득이 관측된다. Hierarchical 개선 여부는 아직 평가되지 않았다.",
              "- **BB72 D12, p=0.001**: 7.947% → 7.300%지만 선택 epoch는 725로 동일하다. "
              "같은 선택 가중치의 새 샷 평가 차이로 보아야 하며 추가 학습 효과로 주장하지 않는다. 로컬 `.pt`는 없어 직접 가중치 대조는 못 했다.",
              "- **BB72 D12, p=0.002**: best는 760 → 885로 바뀌었지만 fresh-shot raw는 28.918% → 31.482%로 높아졌다. "
              "Validation 선택이 최종 bank의 개선을 보장하지 않는 사례다. 단일 seed 결과만으로 일반적인 학습 악화나 과적합을 확정하지 않는다.", "",
              "## 갱신된 hierarchical 평가", "",
              "아래는 **기존 800-epoch run**의 raw-validation best(epoch 5)와 latest(epoch 800) 비교다. "
              "추가 학습한 1,000-epoch 가중치 평가가 아니다.", ""]
    previous.table(lines, ["코드/Depth", "p", "Best 실패/8192", "Best LER (%)", "Latest 실패/8192", "Latest LER (%)", "차이 (%p)"], [
        [f"{r['code']}/D{r['depth']}", r["p"], r["best_failures"], f"{100*r['best_rate']:.3f}",
         r["latest_failures"], f"{100*r['latest_rate']:.3f}", f"{r['latest_minus_best_pp']:+.3f}"]
        for r in comparisons if r["code"] == "bb144" and r["depth"] == 4 and r["p"] in (.004,.006)])
    lines += ["- **BB144 D4, p=0.004**: 92.859% → 89.111%로 3.748%p 낮아졌다. "
              "기존 공통 6,016샷 중간 집계 대신 양쪽 모두 완료된 8,192샷을 사용했다.",
              "- **BB144 D4, p=0.006**: 양쪽 모두 8,192/8,192 실패다. 실용적인 개선은 없다.",
              "- p=0.004에서 latest epoch 800끼리 비교하면 BB144 D4/D6/D8은 각각 "
              "89.111% / 84.131% / 68.835%다. 이 조건에서는 depth 이득이 있지만 다른 p에 일반화하지 않는다.",
              "- 새 네 평가 모두 hierarchical syndrome failure와 harmed는 0이다. "
              "Best 평가 두 개는 기존 baseline과 실패 수가 같으며 독립 반복처럼 합산하지 않는다.", "",
              "## 플롯과 선택 기준", "",
              "[Raw + hierarchical 통합 그림](../plots/bb/circuit/standard/dem_cnn/overview.png)", "",
              f"- Raw {len(raw)}개 점: 최신 완료 run의 raw-validation best만 표시한다. "
              "이제 1,000-epoch 예산 21개 + 800-epoch 예산 11개다. 이번에 네 점을 갱신했다. "
              "BB72 D12 p=0.002도 더 낮았던 옛 final 값을 골라 남기지 않고 최신 완료 run의 값을 표시한다.",
              f"- Hierarchical {len(hierarchical)}개 점: 완료 평가 중 최저 관측 LER checkpoint를 표시한다. "
              "BB144 D4 p=0.004를 epoch 800의 89.111%로 갱신했고, p=0.006은 100%로 동일하다. "
              "표시된 평가 샷으로 선택했으므로 독립 최종 성능 추정치가 아니다.",
              "- 두 열의 가중치/평가 bank는 다를 수 있다. Hierarchical 가중치는 모두 800-epoch run에서 왔다. "
              "Error bar는 marginal Wilson 95% 구간이고 선택 편향을 보정한 구간은 아니다. "
              "실패 0회는 로그 축에서 95% 상한으로 표시한다.", "",
              "## 정리 및 검증", "",
              f"- 새 2개 폴더, {info['files']}개 파일, {info['bytes']:,} bytes를 SHA-256으로 검증하고 "
              "`results/bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`에 배치했다. "
              "기존 중간 사본과 루트 중복 폴더는 제거했다.",
              "- 기존 파일이 새 사본에 모두 포함되는지, 학습/평가 로그가 기존 기록을 보존하면서 늘어났는지 확인했다. "
              "변경 전후 해시는 replacement CSV에 남겼다. 11월 3일/4일 보고서와 원래 이동 manifest는 당시 수신 기록으로 보존한다.",
              "- 누적 12개 폴더의 config, manifest, 종료 코드, 학습 prefix, final, 샷 수 및 실패 수를 다시 검증했다. "
              "총 22개 평가/11쌍이 완료됐고 총 180,224 shots다. 이는 서로 다른 평가의 합계이며 독립 샷 수로 해석하지 않는다.",
              "- 새 `.pt`/`.npz`는 없어 실제 가중치와 paired shot bank를 직접 확인하지 못했고 paired 검정은 하지 않았다.", "",
              f"[누적 학습 CSV]({STEM}_training.csv) · [누적 평가 CSV]({STEM}_evaluations.csv) · "
              f"[Best/latest CSV]({STEM}_best_latest.csv) · [교체 manifest]({STEM}_replacement.csv) · "
              f"[검증 JSON]({STEM}_audit.json)", "",
              "재생성: `python scripts/analyze_bb_dem_november5.py`. 최초 교체는 `--archive`. 학습/추론은 실행하지 않는다.", ""]
    (prior.OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    info = refresh(args.archive)
    previous.archive(False)
    training, evaluations, snapshots = [], [], []
    for directory in [*prior.DESTINATIONS.values(), *previous.DESTINATIONS.values()]:
        a, b, c = prior.audit_job(directory, jobs={**prior.JOBS, **previous.JOBS})
        training.extend(a); evaluations.extend(b); snapshots.extend(c)
    assert len(training) == len({r["worker"] for r, _ in training}) == 21
    assert len(evaluations) == len({r["worker"] for r, _ in evaluations}) == 22
    assert all(r["status"] == "complete" for r, _ in training + evaluations)
    comparisons = prior.common_comparisons(evaluations)
    for suffix, rows in (("training", [r for r, _ in training]), ("evaluations", [r for r, _ in evaluations]),
                         ("best_latest", comparisons), ("snapshots", snapshots)):
        emit_csv(prior.OUT / f"{STEM}_{suffix}.csv", rows)
    raw, hierarchical = render(prior.baseline(), training, evaluations)
    info.update(training_complete=len(training), evaluation_complete=len(evaluations),
                training_partial=0, evaluation_partial=0, training_missing=0, evaluation_missing=0,
                evaluation_shots=sum(r["shots"] for r, _ in evaluations), completed_pairs=len(comparisons),
                raw_points=len(raw), hierarchical_points=len(hierarchical),
                sample_scaling_results_received=False, independent_shot_audit=False)
    report(training, evaluations, comparisons, raw, hierarchical, info)
    (prior.OUT / f"{STEM}_audit.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
