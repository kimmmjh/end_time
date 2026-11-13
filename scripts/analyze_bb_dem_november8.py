#!/usr/bin/env python3
"""Archive job 60852785 and audit its logs; campaign results were not downloaded."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis"
STEM = "bb_dem_2026_11_08"
NAME = "resdir_60852785"
DEST = ROOT / "results/bb/circuit/standard/dem_cnn/sample_scaling" / NAME
STDOUT = ROOT / "slurm-bbdem_samples2m-60852785.out"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(suffix, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / f"{STEM}_{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def audit(directory):
    campaign = json.loads((directory / "source_manifest.json").read_text())
    assert campaign == json.loads((ROOT / "scripts/manifests/bb_dem_sample_scaling.json").read_text())
    metadata = (directory / "job_metadata.txt").read_text()
    assert "Persistent experiment directory: /global/homes/d/ds08152/end_time/resdir_60828644" in metadata
    assert (directory / "interrupted.txt").exists() and (directory / "STOP_REQUESTED").exists()
    assert not (directory / "failed.txt").exists() and not (directory / "completed.txt").exists()
    summary, training, validation, events = [], [], [], []
    for worker, entry in enumerate(campaign["experiments"]):
        log = (directory / f"log_worker_{worker}.txt").read_text()
        header, _ = json.JSONDecoder().raw_decode(log)
        assert header["experiment"] == entry["name"]
        assert Path(header["campaign_root"]).name == "resdir_60828644"
        assert "Traceback (most recent call last)" not in log
        exit_code = int((directory / f"exit_code_worker_{worker}.txt").read_text())
        assert exit_code == (0 if worker == 0 else 75)
        configs = re.findall(r"^DEM CNN configuration: (.+)$", log, re.M)
        assert configs and all(json.loads(c) == entry["expected_config"] for c in configs)
        records = {}
        for label in ("Train", "Validation"):
            records[label] = [dict(epoch=int(epoch), **json.loads(value)) for epoch, value in
                              re.findall(r"^\["+label+r"\] epoch (\d+) \| (.+)$", log, re.M)]
        train, vals = records["Train"], records["Validation"]
        last = train[-1]["epoch"] + 1
        assert [r["epoch"] for r in train] == list(range(1000, last))
        assert last == (3907 if worker == 0 else 2216)
        starts = [1000, 1953, 2930]
        expected_validation = []
        for start, end in zip(starts, [1953, 2930, 3907]):
            expected_validation.extend(e for e in range(start, min(end, last))
                                       if (e-start+1) % 5 == 0 or e == end-1)
        assert [r["epoch"] for r in vals] == expected_validation
        for row in train:
            assert all(math.isfinite(v) for v in row.values())
            assert abs(row["total"]-row["syndrome"]-row["logical"]-.1*row["mechanism"]) < 1e-6
            training.append(dict(experiment=entry["name"], **row, completed_epochs=row["epoch"]+1))
        for row in vals:
            raw = row["raw"]
            assert row["shots"] == raw["shots"] == 1024
            assert raw["logical_error_rate"] == raw["failures"] / 1024
            assert raw["logical_accuracy"] == 1-raw["logical_error_rate"]
            validation.append(dict(experiment=entry["name"], epoch_index=row["epoch"],
                                   completed_epochs=row["epoch"]+1, **raw))
        baseline = json.loads((ROOT / entry["imported_run"] / "history.json").read_text())
        assert len(baseline["train"]) == 1000 and baseline["config"] == entry["expected_config"]
        best = min(baseline["eval"] + vals, key=lambda r:r["raw"]["logical_error_rate"])
        evaluation_calls = re.findall(r"^Evaluating (best|latest) (\S+) -> (\S+)$", log, re.M)
        assert len(evaluation_calls) == (9 if worker == 0 else 4)
        for weights, checkpoint, output in evaluation_calls:
            assert "resdir_60828644/" in checkpoint and "resdir_60828644/" in output
            milestone = int(Path(checkpoint).stem.split("_")[1])
            events.append(dict(experiment=entry["name"], milestone=milestone, weights=weights,
                               final_bank="/final_evaluation/" in output, output=output,
                               metrics_available=False))
        if worker == 0:
            assert evaluation_calls[-1][0] == "best" and evaluation_calls[-1][1].endswith("epoch_1953.pt")
            assert "/final_evaluation/" in evaluation_calls[-1][2]
        first_mean = statistics.mean(r["raw"]["logical_error_rate"] for r in vals[:20])
        last_mean = statistics.mean(r["raw"]["logical_error_rate"] for r in vals[-20:])
        summary.append(dict(experiment=entry["name"], worker=worker, exit_code=exit_code,
                            completed_epochs_from_log=last, added_epochs=len(train), samples=last*512,
                            remaining_epochs=3907-last, last_validation_epoch=vals[-1]["epoch"]+1,
                            last_validation_failures=vals[-1]["raw"]["failures"], validation_shots=1024,
                            last_validation_LER=vals[-1]["raw"]["logical_error_rate"],
                            validation_best_epoch=best["epoch"]+1, validation_best_LER=best["raw"]["logical_error_rate"],
                            first20_validation_mean=first_mean, last20_validation_mean=last_mean,
                            first50_logical_loss=statistics.mean(r["logical"] for r in train[:50]),
                            last50_logical_loss=statistics.mean(r["logical"] for r in train[-50:]),
                            hierarchical_LER_available=False, checkpoint_files_available=False))
    assert not list(directory.rglob("results.json")) and not list(directory.rglob("history.json"))
    assert not list(directory.rglob("*.pt")) and not list(directory.rglob("*.npz"))
    return summary, training, validation, events


def archive(move):
    source = ROOT / NAME
    path = OUT / f"{STEM}_relocation.csv"
    if not path.exists():
        assert move and source.is_dir() and not DEST.exists()
        audit(source)
        emit("relocation", [dict(old_path=str(p.relative_to(ROOT)),
                                 new_path=str((DEST / p.relative_to(source)).relative_to(ROOT)),
                                 bytes=p.stat().st_size, sha256=sha(p))
                            for p in sorted(source.rglob("*")) if p.is_file()])
    rows = list(csv.DictReader(path.open()))
    if source.exists():
        assert move and not DEST.exists()
        DEST.parent.mkdir(parents=True, exist_ok=True)
        source.rename(DEST)
    assert {str(p.relative_to(ROOT)) for p in DEST.rglob("*") if p.is_file()} == {r["new_path"] for r in rows}
    for row in rows:
        p = ROOT / row["new_path"]
        assert p.stat().st_size == int(row["bytes"]) and sha(p) == row["sha256"]
    return dict(files=len(rows), bytes=sum(int(r["bytes"]) for r in rows), hashes_verified=True)


def report(summary, info):
    lines = ["# 2026-11-08: sample-scaling 실행 로그 분석", "",
        "**실제 결과 저장 폴더는 `resdir_60828644`이고, 이번에 받은 `resdir_60852785`는 실행 로그 폴더다.** "
        "`job_metadata.txt`, 두 worker 로그, Slurm stdout이 모두 이 경로를 가리킨다. "
        "사용자는 서버의 `resdir_60828644`를 삭제했다고 확인했다. "
        "이는 실제 저장 경로를 확인하지 않은 어시스턴트의 잘못된 삭제 안내 이후 발생했다. "
        "아래 수치는 남은 worker 로그에서 검증한 값이며 checkpoint나 상세 hierarchical 평가 JSON을 복원한 것은 아니다.", "",
        "## 진행 상태", "",
        "실행 시작 2026-11-06 12:02:21 UTC, 중단 표시 2026-11-08 11:32:33 UTC: "
        "47시간 30분 12초. 48시간 한도 30분 전 stop 요청과 일치한다. "
        "`STOP_REQUESTED`/`interrupted.txt`가 있고 worker 종료 코드는 각각 0/75다. "
        "로그상 BB72는 전체 실험을 완료했고 BB144는 epoch 경계에서 중단했다. "
        "Python traceback은 없으며 `srun ... exit code 75`는 이 스크립트의 의도된 중단 상태다.", "",
        "| 모델 (p=0.002) | 완료 epoch | 누적 샘플 | 최근 raw validation | Validation 최저 관측값 |",
        "|---|---:|---:|---|---|"]
    for row in summary:
        lines.append(f"| {row['experiment']} | {row['completed_epochs_from_log']:,}/3,907 | {row['samples']:,} | "
                     f"{100*row['last_validation_LER']:.3f}% (epoch {row['last_validation_epoch']}, "
                     f"{row['last_validation_failures']}/1024) | {100*row['validation_best_LER']:.3f}% "
                     f"(epoch {row['validation_best_epoch']}) |")
    lines += ["", "로그 epoch는 0부터 시작하므로 표에서는 1을 더한 완료 epoch 수를 썼다. "
                 "Raw LER은 syndrome과 logical observable을 모두 맞춰야 성공인 block failure다. "
                 "1,024-shot validation이며 8,192-shot final 또는 hierarchical LER과 다르다. "
                 "Validation 최저값은 여러 평가에서 선택한 값이므로 독립 테스트 성능이 아니다.", "",
                 "## 학습 경향", ""]
    for row in summary:
        lines.append(f"- {row['experiment']}: 추가 학습 시작 20회/마지막 20회 validation의 평균 raw 실패율은 "
                     f"{100*row['first20_validation_mean']:.3f}% → {100*row['last20_validation_mean']:.3f}%. "
                     f"처음/마지막 50 epoch의 평균 logical loss는 {row['first50_logical_loss']:.5f} → "
                     f"{row['last50_logical_loss']:.5f}.")
    lines += ["", "두 모델 모두 raw validation과 학습 logical loss가 낮아지는 경향이다. "
              "다른 가중치에서 측정한 validation의 평균은 추세 요약일 뿐 샷을 합친 단일 모델의 LER이 아니다. "
              "BB144는 목표까지 1,691 epoch가 남아 있었다. 원래 계획의 continuation은 복구 가능한 checkpoint가 있어야 가능하다.", "",
              "## Hierarchical 결과의 한계", "",
              "- BB72 로그에는 baseline 및 세 milestone의 best/latest 평가 8회와 독립 final 평가 1회가 있다. "
              "종료 코드 0과 실행 코드상 모두 완료된 것으로 판단된다. 마지막 final 호출은 "
              "`epoch_1953.pt`의 `best`를 사용한다. 이는 1,953 epoch 시점에 저장한 checkpoint 안의 raw-best 가중치가 "
              "hierarchical 선택에서 이겼다는 뜻이며 실제 선택 가중치 epoch가 1,953이라는 뜻은 아니다.",
              "- BB144는 baseline 및 epoch 1,953의 best/latest 평가를 마치고 학습을 계속했다. "
              "epoch 2,930/3,907 및 final 평가에는 도달하지 않았다.",
              "- 평가 수치는 별도 `attempt_*.log`와 `results.json`에 쓰이므로 현재 worker 로그에는 없다. "
              "따라서 **hierarchical LER은 현재 자료로 알 수 없다**. Raw 개선만으로 hierarchical 개선을 단정하지 않는다.", "",
              "## 파일 정리 및 기존 플롯", "",
              f"- `{NAME}`의 {info['files']}개 파일 ({info['bytes']:,} bytes)을 "
              "`results/bb/circuit/standard/dem_cnn/sample_scaling/` 아래로 이동하고 전체 SHA-256을 확인했다.",
              "- 루트의 `.out`은 읽기만 했다. 이동·수정·삭제·아카이브 복사를 하지 않았다. "
              "같은 캠페인 경로와 중단 후 재개 명령을 확인했다. 재생성은 `.out` 없이도 가능하다.",
              "- 기존 raw/hierarchical 통합 플롯은 유지했다. 이번에는 새 final/hierarchical 수치가 없으므로 "
              "1,024-shot validation을 기존 8,192-shot final 대신 넣지 않는다.", "",
              "## 삭제된 캠페인 복구 확인", "",
              "NERSC 문서에 따르면 Global Home은 최근 7일 스냅샷을 제공하며 로그인 노드에서 접근할 수 있다. "
              "삭제 전 스냅샷에 해당 파일이 포함돼 있다면 일부 또는 전체 진행을 복구할 수 있다. "
              "실제 가용 여부와 저장 시점은 서버 확인이 필요하다. "
              "[공식 스냅샷 안내](https://docs.nersc.gov/filesystems/backups/#snapshots)", "",
              "```bash", 'ls -F "$HOME/.snapshots"', "```", "",
              "스냅샷의 `end_time/resdir_60828644`를 확인한 뒤 새 별도 폴더에 복사하고, "
              "JSON/checkpoint의 완료 epoch 및 일관성을 검증해야 한다. 실행 중 촬영된 스냅샷은 파일 간 시점이 다를 수 있다. "
              "복구 없이 epoch 2,216부터 이어갈 수 있다고 가정하지 않는다.", "",
              f"[요약 CSV]({STEM}_summary.csv) · [학습 로그 CSV]({STEM}_training.csv) · "
              f"[Validation CSV]({STEM}_validation.csv) · [평가 호출 목록]({STEM}_evaluation_calls.csv) · "
              f"[이동 manifest]({STEM}_relocation.csv) · [검증 JSON]({STEM}_audit.json)", "",
              "재생성: `python scripts/analyze_bb_dem_november8.py`. 최초 이동은 `--archive`.", ""]
    (OUT / f"{STEM}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    out_hash = sha(STDOUT) if STDOUT.exists() else None
    info = archive(args.archive)
    summary, training, validation, events = audit(DEST)
    for suffix, rows in (("summary", summary), ("training", training),
                         ("validation", validation), ("evaluation_calls", events)):
        emit(suffix, rows)
    info.update(job=60852785, campaign_directory="resdir_60828644", campaign_deleted_reported_by_user=True,
                training_rows=len(training), validation_rows=len(validation), evaluation_calls=len(events),
                hierarchical_metrics_available=False, checkpoints_available=False,
                stdout_handling="read_only_left_in_original_location", overview_updated=False)
    if out_hash:
        assert sha(STDOUT) == out_hash
    report(summary, info)
    (OUT / f"{STEM}_audit.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
