#!/usr/bin/env python3
"""Archive and audit the downloaded 60998882 sample-scaling continuation.

Standard library only. Never changes Slurm stdout files or plots. The archived
download intentionally excludes checkpoints and per-shot arrays.
"""
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
STEM = "bb_dem_2026_11_11"
CAMPAIGN = "recovered_60828644_rhhDfV"
JOB = "resdir_60998882"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(name):
    candidates = [p for p in (ROOT / name, ARCHIVE / name) if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one unambiguous copy of {name}: {candidates}")
    return candidates[0]


def emit(suffix, rows):
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(campaign, job):
    manifest = read(job / "source_manifest.json")
    assert manifest == read(campaign / "source_manifest.json")
    assert (job / "STOP_REQUESTED").exists() and (job / "interrupted.txt").exists()
    assert not (job / "failed.txt").exists()
    summaries, evaluations = [], []
    for worker, (name, expected_end) in enumerate([
        ("bb72_d8_p0.002", 3907), ("bb144_d6_p0.002", 3484)
    ]):
        folder = campaign / name
        history = read(folder / "training/history.json")
        train, vals = history["train"], history["eval"]
        assert [r["epoch"] for r in train] == list(range(expected_end))
        assert all(math.isfinite(r[k]) for r in train
                   for k in ("total", "logical", "syndrome", "mechanism"))
        assert all(abs(r["total"] - r["logical"] - r["syndrome"] - .1*r["mechanism"]) < 1e-6
                   for r in train)
        for row in vals:
            assert row["raw"]["shots"] == 1024
            assert row["raw"]["logical_error_rate"] == row["raw"]["failures"] / 1024
        log = (job / f"log_worker_{worker}.txt").read_text()
        header, _ = json.JSONDecoder().raw_decode(log)
        assert header["experiment"] == name
        assert Path(header["campaign_root"]).name == CAMPAIGN
        assert "Traceback (most recent call last)" not in log
        exit_code = int((job / f"exit_code_worker_{worker}.txt").read_text())
        assert exit_code == (0 if worker == 0 else 75)
        records = re.findall(r"^\[Train\] epoch (\d+) \| (.+)$", log, re.M)
        assert [int(e) for e, _ in records] == ([] if worker == 0 else list(range(2026, 3484)))
        for epoch, data in records:
            assert train[int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
        indexed_vals = {r["epoch"]: r for r in vals}
        for epoch, data in re.findall(r"^\[Validation\] epoch (\d+) \| (.+)$", log, re.M):
            assert indexed_vals[int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
        complete = folder / "completed.json"
        assert complete.exists() == (worker == 0)
        expected_config = manifest["experiments"][worker]["expected_config"]
        assert history["config"] == expected_config
        best = min(vals, key=lambda r: r["raw"]["logical_error_rate"])
        summaries.append(dict(
            experiment=name, completed_epochs=expected_end, target_epochs=3907,
            remaining_epochs=3907-expected_end, training_samples=expected_end*512,
            resumed_epochs=len(records), exit_code=exit_code, experiment_complete=complete.exists(),
            last_validation_epoch=vals[-1]["epoch"]+1,
            last_raw_validation=vals[-1]["raw"]["logical_error_rate"],
            best_raw_validation_epoch=best["epoch"]+1,
            best_raw_validation=best["raw"]["logical_error_rate"],
        ))
        paths = sorted(folder.rglob("results.json"))
        assert len(paths) == (9 if worker == 0 else 6)
        for path in paths:
            data, meta = read(path), read(path.with_name("metadata.json"))
            assert meta["config"] == expected_config
            assert meta["sampling"] == "fresh_exact_circuit"
            assert data["status"] == "complete" and data["shots"] == meta["requested_shots"] == 8192
            assert meta["confidence_threshold"] == .9
            assert meta["backend"] == "direct_hard_centered_osd0_v1"
            for metric in data["metrics"].values():
                assert metric["shots"] == 8192
                assert metric["block_failure_rate"] == metric["failures"] / 8192
                p, z, n = metric["block_failure_rate"], 1.959963984540054, 8192
                center = (p+z*z/(2*n))/(1+z*z/n)
                half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                assert all(abs(a-b) < 1e-12 for a,b in zip(
                    metric["block_failure_wilson95"], [max(0.,center-half), min(1.,center+half)]))
                assert abs(metric["block_failure_rate"] - metric["syndrome_failure_rate"]
                           - metric["unflagged_failure_rate"]) < 1e-12
            raw, hier = data["metrics"]["raw"], data["metrics"]["hierarchical"]
            assert hier["syndrome_failure_rate"] == 0
            pair = data["hierarchical_vs_raw"]
            assert raw["failures"] - hier["failures"] == pair["rescued"] - pair["harmed"]
            final = "final_evaluation" in path.parts
            assert meta["seed"] == (20261104 if final else 20261103)
            evaluations.append(dict(
                experiment=name, milestone="final" if final else path.parents[2].name,
                choice=meta["source"]["weights"], weight_epoch=meta["source"]["selected_epoch"],
                shots=8192, raw_failure_rate=raw["block_failure_rate"],
                raw_syndrome_failure_rate=raw["syndrome_failure_rate"],
                hierarchical_failures=hier["failures"], hierarchical_LER=hier["block_failure_rate"],
                ci95_low=hier["block_failure_wilson95"][0], ci95_high=hier["block_failure_wilson95"][1],
                global_call_fraction=data["global_call_fraction"], seconds=data["hierarchical_seconds"],
                sampler_seed=meta["sampler_seed"], independent_final=final,
                result_path=str((ARCHIVE / CAMPAIGN / path.relative_to(campaign)).relative_to(ROOT)),
            ))
        for milestone in (1953, 2930) if worker else ():
            best_result = read(folder / f"evaluations/epoch_{milestone}/best/attempt_001/results.json")
            latest_result = read(folder / f"evaluations/epoch_{milestone}/latest/attempt_001/results.json")
            assert best_result["metrics"] == latest_result["metrics"]
        if complete.exists():
            done = read(complete)
            assert done["selected"]["result"]["metrics"]["hierarchical"]["failures"] == 219
    return summaries, evaluations


def archive(move):
    record = OUT / f"{STEM}_relocation.csv"
    if not record.exists():
        if not move:
            raise RuntimeError("First archival requires --archive")
        rows = []
        for name in (CAMPAIGN, JOB):
            source = ROOT / name
            assert source.is_dir() and not (ARCHIVE / name).exists()
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    rows.append(dict(old_path=str(path.relative_to(ROOT)),
                                     new_path=str((ARCHIVE/name/path.relative_to(source)).relative_to(ROOT)),
                                     bytes=path.stat().st_size, sha256=sha(path)))
        emit("relocation", rows)
    rows = list(csv.DictReader(record.open()))
    for name in (CAMPAIGN, JOB):
        source, dest = ROOT/name, ARCHIVE/name
        if source.exists():
            assert move and not dest.exists()
            ARCHIVE.mkdir(parents=True, exist_ok=True)
            source.rename(dest)
    actual = {str(p.relative_to(ROOT)) for name in (CAMPAIGN,JOB)
              for p in (ARCHIVE/name).rglob("*") if p.is_file()}
    assert actual == {r["new_path"] for r in rows}
    for row in rows:
        path = ROOT / row["new_path"]
        assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"]
    return rows


def report(summaries, evaluations, archived):
    lines = ["# 2026-11-11: sample-scaling 재개 실행 60998882 분석", "",
        "**BB72는 3,907 epoch와 독립 final 평가까지 완료. BB144는 3,484/3,907 epoch이며 423 epoch가 남았다.**",
        "BB144의 최신 hierarchical 평가 가중치는 epoch 2,930이다. 3,484-epoch hierarchical LER은 아직 측정하지 않았다.", "",
        "## 실행 상태와 저장 위치", "",
        "- 실행 시작: 2026-11-09 03:16:06 KST. 중단 표시: 2026-11-11 02:45:38 KST (47시간 29분 32초).",
        "- `STOP_REQUESTED`, `interrupted.txt`, worker 종료 코드 0/75 및 제출본의 `--signal=B:USR1@1800`이 48시간 한도 약 30분 전 정상 저장·중단한 흐름과 일치한다. worker 로그에 traceback은 없다.",
        "- BB144 로그는 0-based epoch 2026에서 시작해 3483에서 끝난다. 복구된 2,026 epoch에서 재개하여 **1,458 epoch**를 추가했고, 처음부터 다시 학습하지 않았다.",
        "- BB72 worker에는 새 Train 행이 없다. 이미 완료된 결과를 재사용했다.",
        f"- 로컬 캠페인: `results/bb/circuit/standard/dem_cnn/sample_scaling/{CAMPAIGN}/`.",
        f"- 로컬 실행 로그: `results/bb/circuit/standard/dem_cnn/sample_scaling/{JOB}/`.",
        "- 서버의 실제 캠페인 경로는 `$HOME/end_time/recovered_60828644_rhhDfV` 그대로다. 로컬 정리는 서버 경로를 바꾸지 않는다.",
        "- 복구 폴더 루트의 오래된 `failed.txt`/worker 로그는 최초 실패 실행의 유산이다. 이번 실행 상태는 `resdir_60998882`로 판단한다.",
        "- 새 `slurm-bbdem_samples2m-60998882.out`은 프로젝트 루트에 없다. 기존 60852785 `.out`은 이동·수정하지 않았다.", "",
        "## 학습 진행", "",
        "| 실험 | 완료 epoch | 누적 training samples | 남은 epoch | 최근 raw validation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(f"| {row['experiment']} | {row['completed_epochs']:,} | {row['training_samples']:,} | {row['remaining_epochs']} | epoch {row['last_validation_epoch']}: {100*row['last_raw_validation']:.3f}% / 1024 shots |")
    lines += ["", "Batch size 4 × 128 batches = 512 samples/epoch. BB144의 남은 training samples는 216,576개다. "
              "validation best는 epoch 3,445, raw failure 68.066%이며 1,024 shots에서 선택된 값이다. "
              "이를 8,192-shot hierarchical 평가나 독립 test 값으로 해석하지 않는다.", "",
              "## 모든 완료 평가 (p=0.002)", "",
              "Raw는 syndrome OR logical mismatch인 correction failure. Hierarchical은 syndrome 일치 이후 X/Z joint logical block failure다. "
              "모두 전체 memory experiment 기준이고 per-qubit/per-cycle 정규화는 하지 않았다.", "",
              "| 실험 | Milestone | Choice | 가중치 epoch | Raw (%) | Hier (%) | 실패/8192 | Hier 95% Wilson (%) |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for row in evaluations:
        lines.append(f"| {row['experiment']} | {row['milestone']} | {row['choice']} | {row['weight_epoch']} | {100*row['raw_failure_rate']:.3f} | {100*row['hierarchical_LER']:.3f} | {row['hierarchical_failures']} | {100*row['ci95_low']:.3f}–{100*row['ci95_high']:.3f} |")
    lines += ["", "## 이전 결과 대비 해석", "",
        "**BB144 D6:** baseline best 102/8192 (1.245%) → epoch 1,953 94/8192 (1.147%) → epoch 2,930 60/8192 (0.732%). "
        "baseline best 대비 42회 감소, 0.513 percentage points / 약 41.2% 상대 감소다. "
        "baseline latest 101회를 기준으로 해도 약 40.6% 감소다. 1,953 대비로는 약 36.2% 감소다.", "",
        "이전 11월 10일 로컬 snapshot은 학습 2,026 epoch와 평가 1,953까지만 있었다. "
        "따라서 당시 작은 개선만 보였던 것과 달리, 이번 JSON에서는 약 150만 샘플 시점의 추가 개선이 확인된다. "
        "raw failure도 baseline best 84.900% → 77.332% → 72.461%로 내려갔다. "
        "하지만 3,484 시점의 hierarchical 성능이나 3,907 최종 성능은 아직 알 수 없다.", "",
        "BB144의 1,953 및 2,930에서 best/latest는 각각 같은 epoch의 가중치와 같은 샷을 평가하여 지표가 같다. "
        "이를 두 번의 독립 평가나 16,384 shots로 합산하면 안 된다. "
        "0.732%는 선택 bank의 값이고 95% Wilson 구간은 0.569–0.942%다. "
        "checkpoint 간 평가 샷을 공유하므로 독립 표본 검정을 적용하지 않았다. "
        "이번 다운로드에는 per-shot NPZ가 없어 paired 비교 검정은 수행하지 않았다.", "",
        "**BB72 D8:** 완료 상태와 결론은 유지된다. 선택 bank에서 epoch 1,810이 2.673%, "
        "그 가중치의 독립 final이 2.600% (213/8192, 95% CI 2.277–2.968%)다. "
        "약 100만 샘플 이후 hierarchical은 2.67–2.83% 범위로, 추가 학습의 이득이 작다. "
        "2,930-latest는 221회로 1,953-best 219회와 2회 차이뿐이므로 특정 epoch의 우위를 과장하지 않는다.", "",
        "## 남은 작업과 예상 시간", "",
        "BB144는 423 epoch 학습, epoch 3,907 best/latest 평가 2회, 선택 모델의 독립 final 평가 1회가 남았다. "
        "이번 epoch 2,930 평가 2회는 각각 약 2.61/2.62시간, 합계 5.23시간의 decoding 시간을 사용했다. "
        "47.49시간에서 이를 뺀 나머지를 1,458 epoch로 나누면 약 104초/epoch이며 "
        "validation·저장·준비 비용도 포함된 거친 추정이다. 남은 학습 약 12.3시간과 평가 약 8시간을 합쳐 "
        "**20–22시간 안팎**이 예상되나 실제 GPU/CPU 부하와 선택 checkpoint에 따라 달라진다. 48시간 요청을 유지하는 편이 여유가 있다.", "",
        "서버 checkpoint를 유지한 상태에서 다음 명령은 같은 캠페인을 이어간다. 완료된 BB72와 기존 평가는 재사용한다.", "",
        "```bash", "cd ~/end_time", "sbatch --export=ALL,BB_SCALE_RUN_DIR=$HOME/end_time/recovered_60828644_rhhDfV run_bb_dem_sample_scaling.slurm", "```", "",
        "## 정리 및 검증", "",
        f"- 두 로컬 폴더의 {len(archived)}개 파일, {sum(int(r['bytes']) for r in archived):,} bytes를 archive로 이동하고 모든 SHA-256을 이동 전후 대조했다.",
        "- BB144의 새 Train 1,458행과 validation 행을 history JSON과 대조했다. 두 실험의 epoch 연속성, loss 구성, manifest/config 일치를 확인했다.",
        "- 평가 JSON 15개 모두 complete/8192 shots이며 실패율, Wilson 구간, syndrome/unflagged 분해, rescued/harmed 산술을 확인했다.",
        "- `.pt`/`.npz`는 다운로드되지 않았다. 서버 checkpoint 자체의 무결성이나 추가 진행 여부를 로컬 자료만으로 확인한 것은 아니다.",
        "- BB144 완료 전이므로 기존 플롯은 갱신하지 않았다. 11월 10일 보고서는 당시 snapshot의 기록으로 보존했다.", "",
        f"[평가 CSV]({STEM}_evaluations.csv) · [진행 CSV]({STEM}_status.csv) · [이동/SHA-256 목록]({STEM}_relocation.csv)", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    protected = list((ROOT / "results/plots").rglob("*")) + list(ROOT.glob("*.out"))
    hashes = {p: sha(p) for p in protected if p.is_file()}
    summaries, evaluations = audit(locate(CAMPAIGN), locate(JOB))
    OUT.mkdir(parents=True, exist_ok=True)
    archived = archive(args.archive)
    emit("status", summaries)
    emit("evaluations", evaluations)
    report(summaries, evaluations, archived)
    assert all(p.is_file() and sha(p) == digest for p, digest in hashes.items())
    print(json.dumps(dict(status=summaries, evaluations=len(evaluations),
                          archived_files=len(archived), protected_files=len(hashes)), indent=2))


if __name__ == "__main__":
    main()
