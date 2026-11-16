#!/usr/bin/env python3
"""Audit and archive the November 16 download; leave plots and Slurm stdout alone."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/bb/circuit/standard/dem_cnn/sample_scaling"
OUT = ROOT / "results/analysis"
STEM = "bb_dem_2026_11_16"
CAMPAIGN = "recovered_60828644_rhhDfV"
JOB = "resdir_61213230"
SNAPSHOT = ARCHIVE / "snapshots/2026-11-11" / CAMPAIGN


def read(path):
    return json.loads(path.read_text())


def files(root):
    paths = list(root.rglob("*"))
    assert not any(p.is_symlink() for p in paths), f"Unexpected symlink in {root}"
    return {p.relative_to(root): p for p in paths if p.is_file()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(campaign, job, previous):
    old, new = files(previous), files(campaign)
    assert old.keys() <= new.keys(), "New download is missing previously archived files"
    changed = {str(p) for p in old if sha(old[p]) != sha(new[p])}
    assert changed == {f"bb144_d6_p0.002/{name}" for name in
                       ("training/history.json", "training/training_log.txt", "comparison.json")}
    assert len(new) == 140 and len(files(job)) == 49
    assert not list(campaign.rglob("*.pt")) and not list(campaign.rglob("*.npz"))
    manifest = read(campaign / "source_manifest.json")
    assert manifest == read(job / "source_manifest.json")
    log = (job / "log_worker_1.txt").read_text()
    assert "Traceback (most recent call last)" not in log
    assert (job / "exit_code_worker_1.txt").read_text().strip() == "75"
    assert (job / "STOP_REQUESTED").exists() and (job / "interrupted.txt").exists()
    assert not (job / "failed.txt").exists()
    assert (job / "next_job_id.txt").read_text().strip() == "61393222"
    status, evaluations, histories = [], [], {}
    for i, name in enumerate(("bb72_d8_p0.002", "bb144_d6_p0.002")):
        folder = campaign / name
        base = read(folder / "training/history.json")
        assert len(base["train"]) == 3907 and (folder / "completed.json").exists()
        before = read(previous / name / "training/history.json")
        assert base["train"][:len(before["train"])] == before["train"]
        assert base["eval"][:len(before["eval"])] == before["eval"]
        history = read(folder / "continuous/training/history.json") if i else base
        if i:
            assert history["train"][:3907] == base["train"]
            assert history["eval"][:len(base["eval"])] == base["eval"]
            assert len(history["train"]) == 4870
            phase = history["phases"][-1]
            assert phase["schedule"] == "constant_restart" and phase["learning_rate"] == 1e-5
            assert phase["start_epoch"] == 3907
            assert not list((folder / "continuous").rglob("results.json"))
        histories[name] = history
        assert history["config"] == manifest["experiments"][i]["expected_config"]
        assert [r["epoch"] for r in history["train"]] == list(range(len(history["train"])))
        assert all(p["batch_size"] == 4 and p["batches"] == 128 for p in history["phases"])
        for row in history["train"]:
            assert all(math.isfinite(row[k]) for k in ("total", "logical", "syndrome", "mechanism"))
            assert abs(row["total"] - row["logical"] - row["syndrome"] - .1*row["mechanism"]) < 1e-6
        for row in history["eval"]:
            assert row["shots"] == row["raw"]["shots"] == 1024
            assert row["raw"]["logical_error_rate"] == row["raw"]["failures"] / 1024
        latest = history["eval"][-1]
        best = min(history["eval"], key=lambda r: r["raw"]["failures"])
        assert best["epoch"] == history["best_epoch"]
        status.append(dict(experiment=name, completed_epochs=len(history["train"]),
                           cumulative_training_samples=len(history["train"])*512,
                           original_campaign_complete=True, extension_epochs=len(history["train"])-3907,
                           latest_validation_epoch=latest["epoch"]+1,
                           latest_validation_failures=latest["raw"]["failures"],
                           latest_raw_LER=latest["raw"]["logical_error_rate"],
                           best_validation_epoch=best["epoch"]+1,
                           best_raw_LER=best["raw"]["logical_error_rate"]))
        for path in sorted(folder.rglob("results.json")):
            result, meta = read(path), read(path.with_name("metadata.json"))
            assert result["status"] == "complete"
            assert result["shots"] == meta["requested_shots"] == 8192
            assert meta["confidence_threshold"] == .9
            assert meta["config"] == history["config"]
            assert meta["sampling"] == "fresh_exact_circuit"
            assert meta["backend"] == "direct_hard_centered_osd0_v1"
            for metric in result["metrics"].values():
                n, f, z = metric["shots"], metric["failures"], 1.959963984540054
                assert n == 8192 and metric["block_failure_rate"] == f/n
                p = f/n
                center = (p+z*z/(2*n))/(1+z*z/n)
                half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                assert all(abs(a-b) < 1e-12 for a,b in zip(
                    metric["block_failure_wilson95"], (max(0.,center-half), min(1.,center+half))))
                assert abs(p-metric["syndrome_failure_rate"]-metric["unflagged_failure_rate"]) < 1e-12
            raw, hier = result["metrics"]["raw"], result["metrics"]["hierarchical"]
            assert hier["syndrome_failure_rate"] == 0
            pair = result["hierarchical_vs_raw"]
            assert raw["failures"]-hier["failures"] == pair["rescued"]-pair["harmed"]
            final = "final_evaluation" in path.parts
            assert meta["seed"] == (20261104 if final else 20261103)
            evaluations.append(dict(experiment=name, milestone="final" if final else path.parents[2].name,
                                    choice=meta["source"]["weights"], weight_epoch=meta["source"]["selected_epoch"],
                                    shots=8192, confidence=.9, raw_LER=raw["block_failure_rate"],
                                    hierarchical_failures=hier["failures"], hierarchical_LER=hier["block_failure_rate"],
                                    ci95_low=hier["block_failure_wilson95"][0], ci95_high=hier["block_failure_wilson95"][1],
                                    global_call_fraction=result["global_call_fraction"],
                                    hierarchical_seconds=result["hierarchical_seconds"], sampler_seed=meta["sampler_seed"],
                                    independent_final=final,
                                    result_path=str((ARCHIVE/CAMPAIGN/path.relative_to(campaign)).relative_to(ROOT))))
        selected = read(folder / "selected.json")
        comparison = read(folder / "comparison.json")
        winner = min(comparison, key=lambda r: (r["result"]["metrics"]["hierarchical"]["failures"],
                                               r["epoch"], r["weights"] != "best"))
        assert winner == selected == read(folder / "completed.json")["selected"]
        final = read(folder / "final_evaluation/attempt_001/metadata.json")
        chosen = next(p for p in folder.rglob("metadata.json") if
                      f"evaluations/epoch_{selected['epoch']}/{selected['weights']}/" in str(p))
        chosen_meta = read(chosen)
        assert final["source"]["sha256"] == chosen_meta["source"]["sha256"]
        assert final["source"]["weights"] == chosen_meta["source"]["weights"]
        assert final["sampler_seed"] != chosen_meta["sampler_seed"]
    assert len(evaluations) == 18
    train_rows = re.findall(r"^\[Train\] epoch (\d+) \| (.+)$", log, re.M)
    assert [int(e) for e,_ in train_rows] == list(range(3484, 4870))
    history = histories["bb144_d6_p0.002"]
    for epoch, data in train_rows:
        assert history["train"][int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
    vals = {r["epoch"]:r for r in history["eval"]}
    for epoch, data in re.findall(r"^\[Validation\] epoch (\d+) \| (.+)$", log, re.M):
        assert vals[int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
    return status, evaluations


def archive():
    manifest = OUT / f"{STEM}_relocation.csv"
    moves = [(ARCHIVE/CAMPAIGN, SNAPSHOT, "previous_snapshot"),
             (ROOT/CAMPAIGN, ARCHIVE/CAMPAIGN, "new_campaign"),
             (ROOT/JOB, ARCHIVE/JOB, "new_job")]
    if not manifest.exists():
        rows = []
        assert not SNAPSHOT.exists() and not (ARCHIVE/JOB).exists()
        for source, dest, role in moves:
            assert source.is_dir()
            for relative, path in sorted(files(source).items()):
                rows.append(dict(role=role, source=str(path.relative_to(ROOT)),
                                 destination=str((dest/relative).relative_to(ROOT)),
                                 bytes=path.stat().st_size, sha256=sha(path)))
        emit("relocation", rows)
    rows = list(csv.DictReader(manifest.open()))
    for source, dest, role in moves:
        # Supports re-running after any completed rename, without overwriting.
        if dest.exists():
            continue
        expected = [r for r in rows if r["role"] == role]
        for row in expected:
            assert sha(ROOT/row["source"]) == row["sha256"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
    for _, dest, role in moves:
        expected = {r["destination"]:r for r in rows if r["role"] == role}
        assert {str(p.relative_to(ROOT)) for p in files(dest).values()} == expected.keys()
        for rel, row in expected.items():
            path = ROOT/rel
            assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"]
    assert not (ROOT/CAMPAIGN).exists() and not (ROOT/JOB).exists()
    return rows


def report(status, evaluations):
    lines = ["# 2026-11-16: BB144 연속 학습 실행 61213230 정리", "",
             "**BB144는 3907 epoch 실험과 독립 final 평가를 완료했고, 추가 학습은 4870 epoch까지 진행됐다.**",
             "3907-latest의 hierarchical LER은 선택 bank에서 **0.598% (49/8192)**, 독립 final에서도 **0.598% (49/8192)**다.",
             "두 평가는 같은 가중치를 다른 sampler seed로 평가했다. 실패 횟수가 같은 것이며 동일 샷의 재사용은 아니다.", "",
             "## 실행 및 진행 상태", "",
             "- job 61213230: 2026-11-14 18:08:51 KST 시작 → 2026-11-16 17:39:36 KST 중단 (47시간 30분 45초).",
             "- worker 1은 epoch 3484까지 완료된 상태에서 이어서 1386 epoch를 학습했다: 기존 구간 423 + 추가 구간 963.",
             "- `STOP_REQUESTED`, `interrupted.txt`, worker exit 75가 있으며 traceback/failed.txt는 없다. 제출 스크립트는 정상 pause 후 성공 종료하도록 되어 있다.",
             "- `next_job_id.txt`에는 **61393222**가 기록됐다. 후속 job 제출 기록이며 실제 실행 여부는 이번 다운로드만으로 확인할 수 없다.",
             "- BB72는 3907 epoch/final 완료 상태 그대로이며 이번 job에서 학습하지 않았다.",
             "- BB144 원래 구간은 cosine LR, 3907 이후에는 고정 LR **1e-5**. Batch 4 × 128 batches = **512 samples/epoch**.",
             "- 새 평가를 포함한 모든 수신 평가의 threshold는 **0.9**, active detector 후보 제한 및 residual OSD-0을 사용했다.", "",
             "| 실험 | 완료 epoch | 누적 training samples | 최근 raw validation | raw validation best |",
             "|---|---:|---:|---|---|"]
    for row in status:
        lines.append(f"| {row['experiment']} | {row['completed_epochs']:,} | {row['cumulative_training_samples']:,} | "
                     f"epoch {row['latest_validation_epoch']}: {100*row['latest_raw_LER']:.3f}% | "
                     f"epoch {row['best_validation_epoch']}: {100*row['best_raw_LER']:.3f}% |")
    lines += ["", "Raw validation은 1024 shots다. BB144의 추가 학습 첫 hierarchical milestone은 **4907 epoch**이며 다운로드 시점에서 **37 epoch** 남았다.",
              "따라서 4870 epoch 또는 가중치 epoch 4712의 hierarchical LER은 아직 없다. 최근 raw validation과 8192-shot hierarchical 결과를 혼동하지 않는다.", "",
              "## BB144의 완료된 hierarchical 평가", "",
              "| Milestone | Choice | 가중치 epoch | Raw (%) | Hier (%) | 실패/8192 | Hier 95% Wilson (%) |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in evaluations:
        if row["experiment"] != "bb144_d6_p0.002":
            continue
        lines.append(f"| {row['milestone']} | {row['choice']} | {row['weight_epoch']} | {100*row['raw_LER']:.3f} | "
                     f"{100*row['hierarchical_LER']:.3f} | {row['hierarchical_failures']} | "
                     f"{100*row['ci95_low']:.3f}–{100*row['ci95_high']:.3f} |")
    lines += ["", "## 해석", "",
              "- 동일 선택 bank에서 BB144 baseline best 102회 → epoch 2930 60회 → epoch 3907 latest 49회로 감소했다.",
              "  baseline best 대비 약 **52.0%**, epoch 2930 대비 약 **18.3%** 상대 감소다. 후자의 차이는 8192 shots 중 11회로, paired shot 데이터 없이 통계적 유의성을 단정하지 않는다.",
              "- 3907 예산의 raw-validation best 가중치(epoch 3634)는 54회, latest(epoch 3907)는 49회다. Raw validation best가 hierarchical best와 같지는 않다.",
              "- 선택 bank에서 3907 latest를 고른 뒤 새 final bank로 평가했다. 독립 final LER **0.598%**, 95% Wilson 구간 **0.453–0.790%**다.",
              "- 모든 hierarchical 평가에서 syndrome failure는 0이다. 여기의 LER은 전체 memory experiment의 X/Z joint logical block failure이며 per-qubit/per-cycle 값이 아니다.",
              "- 3907 이후 963 epoch를 더 학습했지만 해당 구간의 hierarchical 평가는 아직 없다. 고정 LR 추가 학습이 hierarchical LER을 개선했는지는 다음 milestone 이후 판단해야 한다.",
              "- BB72의 독립 final 2.600% (213/8192)는 이전과 동일하다.", "",
              "## 실행 시간", "",
              "이번 BB144의 새 hierarchical 평가 세 번은 각각 약 2.38시간(best), 2.47시간(latest), 2.36시간(final)의 decoding 시간을 사용했다.",
              "합계 약 7.21시간이며, 전체 47.51시간에서 빼고 1386 epoch로 나누면 약 104.7초/epoch다.",
              "이는 validation·샘플 생성·저장·준비 비용을 포함한 거친 추정이며 순수 GPU 학습 속도나 profiler 결과가 아니다.", "",
              "## 보관 및 검증", "",
              f"- 최신 캠페인: `results/bb/circuit/standard/dem_cnn/sample_scaling/{CAMPAIGN}/`.",
              f"- 새 job 로그: `results/bb/circuit/standard/dem_cnn/sample_scaling/{JOB}/`.",
              f"- 이전 11월 11일 캠페인 123개 파일은 `results/bb/circuit/standard/dem_cnn/sample_scaling/snapshots/2026-11-11/{CAMPAIGN}/`에 보존했다.",
              "- 새 캠페인 140개 + 새 job 49개 파일을 프로젝트 루트에서 이동하고 SHA-256을 검증했다. 이전본에는 없던 17개 파일이 추가됐고 3개 파일이 갱신됐다.",
              "- BB144 이전 history가 새 history의 정확한 prefix임을 확인했다. 새 job의 Train 1386행 및 validation 행을 history와 대조했다.",
              "- 전체 평가 18개 complete/8192 shots, 실패율·Wilson 구간·syndrome 분해·선택 가중치·final seed를 검증했다.",
              "- `.pt`/`.npz`는 받지 않았으므로 checkpoint 내부 optimizer 상태 또는 per-shot paired 통계는 검증하지 않았다.",
              "- 복구 폴더 최상위의 오래된 failed.txt는 최초 실패 실행의 유산이다. 최신 실행 상태는 resdir_61213230으로 판단한다.",
              "- 기존 플롯과 루트 `.out`은 수정·이동하지 않았다. Downloads 사본과 서버 경로도 건드리지 않았다.", "",
              "[평가 CSV](bb_dem_2026_11_16_evaluations.csv) · [진행 CSV](bb_dem_2026_11_16_status.csv) · [이동 및 SHA-256 목록](bb_dem_2026_11_16_relocation.csv)",
              "", "재검증: `python3 scripts/analyze_bb_dem_november16.py`."]
    (OUT/f"{STEM}.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    protected = {p:sha(p) for p in [*ROOT.glob("*.out"), *files(ROOT/"results/plots").values()]}
    incoming = (ROOT/CAMPAIGN).is_dir()
    campaign = ROOT/CAMPAIGN if incoming else ARCHIVE/CAMPAIGN
    job = ROOT/JOB if (ROOT/JOB).is_dir() else ARCHIVE/JOB
    previous = SNAPSHOT if SNAPSHOT.exists() else ARCHIVE/CAMPAIGN
    status, evaluations = audit(campaign, job, previous)
    if incoming and not args.archive:
        print(json.dumps(dict(status=status, evaluations=len(evaluations), ready_to_archive=True), indent=2))
        return
    rows = archive()
    emit("status", status)
    emit("evaluations", evaluations)
    report(status, evaluations)
    assert all(p.exists() and sha(p) == checksum for p,checksum in protected.items())
    print(json.dumps(dict(verified_files=len(rows), evaluations=len(evaluations), status=status), indent=2))


if __name__ == "__main__":
    main()
