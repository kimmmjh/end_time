#!/usr/bin/env python3
"""Audit/archive job 61629228 without modifying training results or plots."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resdir_61629228'
DEST = ROOT / 'results/bb/circuit/standard/dem_cnn/osd_benchmark' / NAME
OUT = ROOT / 'results/analysis'
STEM = 'bb_dem_osd_2026_11_20'
BACKENDS = ('reference_cpu', 'packed_cpu', 'packed_cuda')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(folder):
    paths = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in paths), folder
    return {p.relative_to(folder): p for p in paths if p.is_file()}


def emit(kind, rows):
    with (OUT / f'{STEM}_{kind}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(folder):
    assert len(files(folder)) == 94
    assert (folder/'completed.txt').is_file() and not (folder/'failed.txt').exists()
    assert not list(folder.rglob('failure.json')) and not list(folder.rglob('mismatch.npz'))
    assert not list(folder.rglob('*.pt')) and not list(folder.rglob('*.npz'))
    summaries = []
    for worker, code, epoch, calls in ((0,'bb72',8459,66), (1,'bb144',5478,106)):
        assert (folder/f'exit_code_worker_{worker}.txt').read_text().strip() == '0'
        base = folder/f'worker_{worker}'
        d, m = read(base/'results.json'), read(base/'metadata.json')
        assert d['status'] == 'complete' and d['shots'] == m['requested_shots'] == 128
        assert d['cuda_tested'] and d['real_shot_osd_exercised'] and d['osd_shots'] == calls
        assert m['config']['code'] == code and m['source']['selected_epoch'] == epoch
        assert m['source']['weights'] == 'latest' and m['source']['checkpoint_epoch'] == epoch
        assert m['source']['sha256'] == m['original_checkpoint']['sha256']
        assert m['device'] == 'cuda' and m['gpu'] == 'NVIDIA A100-SXM4-40GB'
        assert m['device_self_check_shots'] == 24 and m['batch_size'] == 4 and m['repeats'] == 2
        assert m['seed'] == 20261120 and m['sampling'] == 'fresh_exact_circuit_shared_across_backends'
        assert d['confidence'] == m['confidence'] == .9 and m['comparison'] == 'ge'
        assert m['active_candidates_only'] and m['torch_threads'] == 1
        log = (folder/f'log_worker_{worker}.txt').read_text()
        assert 'Traceback' not in log
        header, _ = json.JSONDecoder().raw_decode(log)
        assert header['training'] is False and header['automatic_resubmission'] is False
        progress = re.findall(r'^Shots (\d+)/128; OSD calls (\d+); seconds (.+)$', log, re.M)
        assert [int(n) for n,_,_ in progress] == list(range(4,129,4))
        assert int(progress[-1][1]) == calls and ast.literal_eval(progress[-1][2]) == d['backend_seconds']
        rows = list(csv.DictReader((base/'timings.csv').open()))
        assert len(rows) == 192
        keys = [(int(r['batch_start']),int(r['repeat']),r['backend']) for r in rows]
        assert len(set(keys)) == 192
        assert set(keys) == {(i,j,b) for i in range(0,128,4) for j in range(2) for b in BACKENDS}
        grouped = {}
        for r in rows:
            value = float(r['seconds'])
            assert math.isfinite(value) and value > 0
            grouped.setdefault((int(r['batch_start']),r['backend']),[]).append(value)
        seen_calls = []
        for start in range(0,128,4):
            ns = {int(r['osd_shots']) for r in rows if int(r['batch_start']) == start}
            assert len(ns) == 1 and 0 <= next(iter(ns)) <= 4
            seen_calls.append(next(iter(ns)))
        assert sum(seen_calls) == calls
        common = d['common_cnn_local_transfer_seconds']
        for b in BACKENDS:
            total = sum(statistics.median(grouped[(i,b)]) for i in range(0,128,4))
            assert math.isclose(total,d['backend_seconds'][b],rel_tol=1e-12)
            assert math.isclose(total+common,d['pipeline_seconds_estimate'][b],rel_tol=1e-12)
            assert math.isclose(d['backend_seconds']['reference_cpu']/total,
                                d['backend_speedup_vs_reference'][b],rel_tol=1e-12)
            assert d['correction_mismatch_shots'][b] == d['logical_mismatch_shots'][b] == 0
            metric = d['metrics'][b]
            assert metric == d['metrics']['reference_cpu']
            n,f,z = metric['shots'],metric['failures'],1.959963984540054
            assert n == 128 and f == (8 if worker == 0 else 1)
            assert metric['syndrome_failure_rate'] == 0 and metric['block_failure_rate'] == f/n
            assert metric['unflagged_failure_rate'] == f/n
            p = f/n
            center = (p+z*z/(2*n))/(1+z*z/n)
            half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            assert all(abs(a-b)<1e-12 for a,b in zip(metric['block_failure_wilson95'],
                       (max(0.,center-half),min(1.,center+half))))
            summaries.append(dict(code=code,depth=m['config']['depth'],epoch=epoch,backend=b,
                shots=128,osd_shots=calls,confidence=.9,backend_seconds=total,
                common_cnn_local_transfer_seconds=common,pipeline_seconds_estimate=total+common,
                backend_speedup=d['backend_speedup_vs_reference'][b],
                pipeline_speedup=d['pipeline_seconds_estimate']['reference_cpu']/(total+common),
                repeat0_seconds=sum(float(r['seconds']) for r in rows if r['backend']==b and r['repeat']=='0'),
                repeat1_seconds=sum(float(r['seconds']) for r in rows if r['backend']==b and r['repeat']=='1'),
                correction_mismatches=0,logical_mismatches=0,hierarchical_failures=f,
                hierarchical_LER=p,ci95_low=metric['block_failure_wilson95'][0],
                ci95_high=metric['block_failure_wilson95'][1],checkpoint_sha256=m['source']['sha256'],
                result_path=str((DEST/f'worker_{worker}/results.json').relative_to(ROOT))))
    return summaries


def archive(folder):
    manifest = OUT/f'{STEM}_relocation.csv'
    if not manifest.exists():
        assert folder == ROOT/NAME and not DEST.exists()
        emit('relocation',[dict(source=str(p.relative_to(ROOT)),
            destination=str((DEST/rel).relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
            for rel,p in sorted(files(folder).items())])
    rows = list(csv.DictReader(manifest.open()))
    if folder != DEST:
        assert not DEST.exists()
        for row in rows:
            assert sha(ROOT/row['source']) == row['sha256']
        DEST.parent.mkdir(parents=True,exist_ok=True)
        folder.rename(DEST)
    assert {str(p.relative_to(ROOT)) for p in files(DEST).values()} == {r['destination'] for r in rows}
    for r in rows:
        p=ROOT/r['destination']
        assert p.stat().st_size == int(r['bytes']) and sha(p) == r['sha256']
    return rows


def report(rows):
    lines = ['# 2026-11-20: CPU/GPU OSD 비교 — job 61629228', '',
        '**검증 샷에서 CPU/GPU의 최종 보정 결과는 일치했다. 현재 GPU 후보는 BB72에서 느리고, BB144에서 소폭 빨랐다.**', '',
        '## 실행과 정확성', '',
        '- 2026-11-20 **15:20:30–15:36:55 KST**, 전체 job 16분 25초. 두 worker exit 0, completed.txt 확인; traceback/failure 기록 없음.',
        '- BB72 D8 **epoch 8459 latest**, BB144 D6 **epoch 5478 latest**의 기존 CNN을 고정했다. 학습이나 다음 job 자동 제출은 수행하지 않았다.',
        '- NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, CUDA 13.0, CPU Torch thread 1. 각 모델 128 shots, batch 4, 반복 2회, threshold **0.9**, active detector 후보 필터 사용.',
        '- 각 모델에서 같은 CNN 출력·residual syndrome·확률을 reference CPU, packed CPU, packed CUDA에 전달했다. 최종 correction bit와 logical prediction의 불일치는 모두 **0**이며 syndrome failure도 0이다.',
        '- 실제 residual OSD 호출 샷은 BB72 **66/128**, BB144 **106/128**. 호출되지 않은 샷은 사전 보정 후 residual syndrome이 0이었다. 호출된 샷 중에도 OSD 내부 hard-decision 검사를 통과하면 소거가 생략될 수 있다.',
        '- CUDA 장치 toy self-check 24 shots/model도 완료했다. 이 결과는 받은 유한 샷에 대한 일치 검증이며 가능한 모든 입력에 대한 증명은 아니다.', '',
        '## 처리 시간', '',
        '아래는 128 shots에 대한 **batch별 반복 시간 중앙값의 합**이다. OSD backend 시간에는 CPU 정렬·전송·syndrome 검사가 포함된다.',
        'Pipeline 추정은 공통 CNN/사전 보정/전송 시간 + 해당 backend 시간이다. Setup/warmup·샘플 생성·scoring·파일 IO는 제외되어 job 벽시계 시간과 다르다.', '',
        '| 모델 | Backend | OSD backend (s) | Pipeline 추정 (s) | OSD 속도비 CPU/해당 | Pipeline 속도비 CPU/해당 |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['code'].upper()} | {r['backend']} | {r['backend_seconds']:.3f} | "
                     f"{r['pipeline_seconds_estimate']:.3f} | {r['backend_speedup']:.3f}× | {r['pipeline_speedup']:.3f}× |")
    lines += ['',
        '- **BB72:** GPU backend 13.597초 vs 기존 CPU 4.511초 → **3.01배 느림**. Pipeline 추정도 15.013초 vs 5.927초로 **2.53배 느림**.',
        '- **BB144:** GPU backend 114.634초 vs CPU 138.077초 → **1.20배 빠름**, backend 시간 **16.98% 감소**. Pipeline 추정은 117.626초 vs 141.068초로 **16.62% 감소**.',
        '- Packed CPU는 두 모델 모두 기존 NumPy CPU보다 느렸다. 비트 압축만 적용한 현재 Torch 구현도 자동으로 빨라지지 않았다.',
        '- BB144 pipeline 추정에서 기존 CPU backend 비중은 약 97.9%다. 이 검증 조건에서는 OSD backend가 평가 시간 대부분을 차지하지만, 학습 전체 시간 중 비중을 측정한 것은 아니다.',
        '- 반복별 총 시간은 CSV에 보존했다. 두 반복의 같은 샷을 다시 사용한 것이므로 256개의 독립 평가 샷으로 세면 안 된다. 반복 2회·소규모 bank·한 job 결과로 다른 checkpoint나 시스템 부하까지 일반화하지 않는다.', '',
        '## LER은 기능 검증용 참고값', '',
        '| 모델 | 세 backend 공통 실패 수 | Hier LER | 95% Wilson |',
        '|---|---:|---:|---:|',
        '| BB72 | 8/128 | 6.250% | 3.201–11.849% |',
        '| BB144 | 1/128 | 0.781% | 0.138–4.293% |', '',
        '이는 최신 가중치·새 seed 20261120·128 shots의 결과다. 이전 milestone의 가중치/8192-shot bank와 다르다.',
        'BB144의 직전 연속 milestone 평가는 threshold 0.5였고 이번 검증은 0.9다. 이 표로 학습의 개선/악화나 GPU의 LER 변화를 단정하지 않는다.',
        '동일한 이번 샷에서는 CPU/GPU 결과가 정확히 같았으므로 GPU로 계산한 것 때문에 8/128 또는 1/128의 실패가 추가된 것은 아니다. Raw CNN LER은 이 benchmark가 별도로 기록하지 않았다.', '',
        '## 해석과 다음 판단', '',
        '현재 후보를 두 코드에 일괄 적용할 근거는 없다. **BB72는 기존 CPU 유지**, BB144는 측정된 소폭 이득을 별도로 판단하는 것이 맞다. 이번 정리에서 production backend는 변경하지 않았다.',
        '소스상 GPU 후보는 샷을 순차 처리하고 pivot마다 `.item()`으로 CPU와 동기화하며 여러 Torch 연산을 실행한다. 정렬·parity 검사는 CPU에 남아 있다.',
        '따라서 작은 BB72에서는 launch/동기화 비용이 계산 절약보다 클 가능성이 있다. 이는 코드와 측정 결과에 따른 설명이며 GPU profiler로 확인된 원인 분해는 아니다.',
        '추가 속도 개선에는 pivot 동기화·커널 호출을 줄이거나 샷 병렬화를 구현하고 다시 비교해야 한다. 단순히 GPU 사용으로 바꾸는 것만으로 큰 가속은 얻지 못했다.',
        '이 변경은 평가의 OSD 계산에만 해당한다. CNN 학습 step을 빠르게 하거나 hard OSD를 미분 가능하게 만들지는 않는다.', '',
        '## 보관 및 검증', '',
        f'- [실행 결과 폴더](../bb/circuit/standard/dem_cnn/osd_benchmark/{NAME}/)로 94개 파일을 이동하고 전후 SHA-256/크기를 대조했다.',
        '- worker별 timings.csv 192행(32 batches × 2 repeats × 3 backends), 총 384행을 집계해 results.json의 시간/속도비와 일치함을 확인했다. Worker 로그의 최종 샷 수·OSD 호출 수·누적 시간도 일치한다.',
        '- 메타데이터의 원본/복사 checkpoint SHA 값은 일치한다. 실제 `.pt`와 per-shot `.npz`는 다운로드에서 제외되어 로컬 재실행 및 보정 배열 재검산은 할 수 없다. 정확한 보정 일치는 실행 코드의 검사와 수신 로그/집계 결과로 확인했다.',
        '- 제출 Slurm 및 source_snapshot을 함께 보존했다. 기존 학습 결과·LER 플롯·루트 `.out`은 변경하지 않았다.', '',
        '[요약 CSV](bb_dem_osd_2026_11_20_summary.csv) · [이동/해시 목록](bb_dem_osd_2026_11_20_relocation.csv)', '',
        '재검증: `python3 scripts/analyze_bb_dem_osd_november20.py`.']
    (OUT/f'{STEM}.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    protected={p:sha(p) for p in [*ROOT.glob('*.out'), *files(ROOT/'results/plots').values(),
                                *files(ROOT/'results/bb/circuit/standard/dem_cnn/sample_scaling').values()]}
    incoming=ROOT/NAME
    folder=incoming if incoming.exists() else DEST
    rows=audit(folder)
    if incoming.exists() and not args.archive:
        print(json.dumps(dict(ready_to_archive=True,files=len(files(folder)),summary=rows),indent=2))
        return
    relocated=archive(folder)
    emit('summary',rows)
    report(rows)
    assert all(p.exists() and sha(p)==value for p,value in protected.items())
    print(json.dumps(dict(verified_files=len(relocated),workers=2,timing_rows=384,
                         correction_mismatches=0,report=str(OUT/f'{STEM}.md')),indent=2))


if __name__=='__main__':
    main()
