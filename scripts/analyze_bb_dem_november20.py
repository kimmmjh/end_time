#!/usr/bin/env python3
"""Validate and archive the November 20 campaign download, preserving older data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/bb/circuit/standard/dem_cnn/sample_scaling'
NAME = 'recovered_60828644_rhhDfV'
DEST = BASE / NAME
PREVIOUS = BASE / 'snapshots/2026-11-16' / NAME
OUT = ROOT / 'results/analysis'
STEM = 'bb_dem_2026_11_20'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files(folder):
    paths = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in paths), folder
    return {p.relative_to(folder): p for p in paths if p.is_file()}


def emit(kind, rows):
    with (OUT / f'{STEM}_{kind}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(current, previous):
    new, old = files(current), files(previous)
    assert len(new) == 228 and len(old) == 140
    assert old.keys() <= new.keys(), 'Previously received files are missing'
    changed = {str(p) for p in old if sha(old[p]) != sha(new[p])}
    assert changed == {f'bb144_d6_p0.002/continuous/training/{n}'
                       for n in ('history.json', 'training_log.txt')}
    assert not list(current.rglob('*.pt')) and not list(current.rglob('*.npz'))
    for p in current.rglob('*.json'):
        read(p)
    manifest = read(current / 'source_manifest.json')
    statuses, evaluations = [], []
    for i, end in enumerate((8459, 5478)):
        name = manifest['experiments'][i]['name']
        folder = current / name
        base = read(folder / 'training/history.json')
        h = read(folder / 'continuous/training/history.json')
        prior = previous / name / 'continuous/training/history.json'
        if not prior.exists():
            prior = previous / name / 'training/history.json'
        before = read(prior)
        assert len(base['train']) == 3907 and (folder / 'completed.json').exists()
        assert h['train'][:len(before['train'])] == before['train']
        assert h['eval'][:len(before['eval'])] == before['eval']
        assert h['train'][:3907] == base['train']
        assert [v['epoch'] for v in h['train']] == list(range(end))
        assert h['config'] == manifest['experiments'][i]['expected_config']
        for row in h['train']:
            assert all(math.isfinite(row[k]) for k in ('total','logical','syndrome','mechanism'))
            assert abs(row['total']-row['logical']-row['syndrome']-.1*row['mechanism']) < 1e-6
        for row in h['eval']:
            assert row['shots'] == row['raw']['shots'] == 1024
            assert row['raw']['logical_error_rate'] == row['raw']['failures']/1024
        assert all(p['batch_size'] == 4 and p['batches'] == 128 for p in h['phases'])
        assert all(p['learning_rate'] == 1e-5 and p['schedule'].startswith('constant')
                   for p in h['phases'] if p['start_epoch'] >= 3907)
        log = (folder / 'continuous/training/training_log.txt').read_text()
        train_log = re.findall(r'^\[Train\] epoch (\d+) \| (.+)$', log, re.M)
        assert [int(e) for e,_ in train_log] == list(range(3907,end))
        for epoch, data in train_log:
            assert h['train'][int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
        vals = {v['epoch']:v for v in h['eval']}
        for epoch, data in re.findall(r'^\[Validation\] epoch (\d+) \| (.+)$', log, re.M):
            assert vals[int(epoch)] == dict(epoch=int(epoch), **json.loads(data))
        last, best = h['eval'][-1], min(h['eval'], key=lambda v:v['raw']['failures'])
        assert best['epoch'] == h['best_epoch']
        next_epoch = 3907 + ((end-3907)//1000+1)*1000
        statuses.append(dict(experiment=name, completed_epochs=end,
            previous_epochs=len(before['train']), added_epochs=end-len(before['train']),
            training_samples=end*512, latest_validation_epoch=last['epoch']+1,
            latest_raw_LER=last['raw']['logical_error_rate'], latest_raw_failures=last['raw']['failures'],
            best_validation_epoch=best['epoch']+1, best_raw_LER=best['raw']['logical_error_rate'],
            next_milestone=next_epoch, epochs_to_next_milestone=next_epoch-end))
        for meta_path in folder.rglob('metadata.json'):
            if meta_path.parent.name.startswith('attempt_'):
                assert meta_path.with_name('results.json').exists(), meta_path
        for path in sorted(folder.rglob('results.json')):
            d, m = read(path), read(path.with_name('metadata.json'))
            assert d['status'] == 'complete' and d['shots'] == m['requested_shots'] == 8192
            assert m['config'] == h['config'] and m['sampling'] == 'fresh_exact_circuit'
            assert m['backend'] == 'direct_hard_centered_osd0_v1' and m['active_candidates_only']
            confidence = .5 if 'evaluation_policy_q_gt_0p5' in path.parts else .9
            assert m['confidence_threshold'] == confidence
            if confidence == .5:
                assert m['confidence_comparison'] == 'gt'
            final = 'final_evaluation' in path.parts
            assert m['seed'] == (20261104 if final else 20261103)
            for metric in d['metrics'].values():
                n, f, z = metric['shots'], metric['failures'], 1.959963984540054
                assert n == 8192 and metric['block_failure_rate'] == f/n
                p = f/n
                center = (p+z*z/(2*n))/(1+z*z/n)
                half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                assert all(abs(a-b)<1e-12 for a,b in zip(metric['block_failure_wilson95'],
                           (max(0.,center-half),min(1.,center+half))))
                assert abs(p-metric['syndrome_failure_rate']-metric['unflagged_failure_rate'])<1e-12
            raw, hier = d['metrics']['raw'], d['metrics']['hierarchical']
            assert hier['syndrome_failure_rate'] == 0
            pair = d['hierarchical_vs_raw']
            assert raw['failures']-hier['failures'] == pair['rescued']-pair['harmed']
            milestone = 0 if final else int(next(s[6:] for s in path.parts if s.startswith('epoch_')))
            evaluations.append(dict(experiment=name, stage='continuous' if 'continuous' in path.parts else 'original',
                milestone=milestone, independent_final=final, choice=m['source']['weights'],
                weight_epoch=m['source']['selected_epoch'], confidence=confidence, shots=8192,
                raw_LER=raw['block_failure_rate'], hierarchical_LER=hier['block_failure_rate'],
                hierarchical_failures=hier['failures'], ci95_low=hier['block_failure_wilson95'][0],
                ci95_high=hier['block_failure_wilson95'][1], sampler_seed=m['sampler_seed'],
                checkpoint_sha256=m['source']['sha256'], hierarchical_seconds=d['hierarchical_seconds'],
                result_path=str((DEST/path.relative_to(current)).relative_to(ROOT))))
        for p in folder.rglob('selected.json'):
            selected, comparison = read(p), read(p.with_name('comparison.json'))
            assert selected == min(comparison, key=lambda v:(v['result']['metrics']['hierarchical']['failures'],
                                                            v['epoch'], v['weights'] != 'best'))
    assert len(evaluations) == 37
    # Historical BB144 threshold reruns use the same checkpoint and sample bank.
    for e in evaluations:
        if e['experiment'].startswith('bb144') and e['stage'] == 'original' and e['confidence'] == .5:
            other = next(v for v in evaluations if v['experiment']==e['experiment']
                         and v['stage']=='original' and v['confidence']==.9
                         and all(v[k]==e[k] for k in ('milestone','choice','independent_final')))
            assert all(e[k]==other[k] for k in ('weight_epoch','checkpoint_sha256','sampler_seed',
                                               'hierarchical_failures','raw_LER'))
    assert 'Job ID: 60828644' in (current/'job_metadata.txt').read_text()
    # These inherited failure markers must not be attributed to the latest run.
    assert sha(current/'failed.txt') == sha(previous/'failed.txt')
    return statuses, evaluations


def archive():
    manifest = OUT/f'{STEM}_relocation.csv'
    moves = [(DEST, PREVIOUS, 'previous_snapshot'), (ROOT/NAME, DEST, 'new_campaign')]
    if not manifest.exists():
        assert not PREVIOUS.exists()
        rows = []
        for source, target, role in moves:
            assert source.is_dir()
            for relative, path in sorted(files(source).items()):
                rows.append(dict(role=role, source=str(path.relative_to(ROOT)),
                    destination=str((target/relative).relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path)))
        emit('relocation', rows)
    rows = list(csv.DictReader(manifest.open()))
    for source, target, role in moves:
        # Resume safely after either rename without overwriting an existing directory.
        if role == 'previous_snapshot' and target.exists():
            continue
        if role == 'new_campaign' and not source.exists():
            continue
        expected = [r for r in rows if r['role'] == role]
        assert expected and not target.exists()
        for row in expected:
            assert sha(ROOT/row['source']) == row['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for _, target, role in moves:
        expected = {r['destination']:r for r in rows if r['role'] == role}
        assert {str(p.relative_to(ROOT)) for p in files(target).values()} == expected.keys()
        for p, row in expected.items():
            assert (ROOT/p).stat().st_size == int(row['bytes']) and sha(ROOT/p) == row['sha256']
    return rows


def report(status, evaluations):
    lines = ['# 2026-11-20: BB72·BB144 추가 학습 수신 결과', '',
        '**BB72는 8,459 epoch, BB144는 5,478 epoch까지 학습 기록이 있다.**',
        '이번 수신본의 평가 37개는 모두 8192 shots/complete이며 hierarchical syndrome failure는 모두 0이다.',
        '11월 16일 수신본보다 BB72는 4552 epoch, BB144는 608 epoch 늘었다.', '',
        '## 학습 상태', '',
        '| 모델 | 완료 epoch | 누적 학습 samples | 최근 raw validation (1024 shots) | 다음 milestone |',
        '|---|---:|---:|---|---:|']
    for s in status:
        lines.append(f"| {s['experiment']} | {s['completed_epochs']:,} | {s['training_samples']:,} | "
                     f"epoch {s['latest_validation_epoch']}: {100*s['latest_raw_LER']:.3f}% "
                     f"({s['latest_raw_failures']}/1024) | {s['next_milestone']} ({s['epochs_to_next_milestone']} 남음) |")
    lines += ['', '3907 epoch 이후 LR은 1e-5 고정, batch 4 × 128 batches = 512 samples/epoch다.',
        '최신 가중치 8459/5478의 hierarchical 평가는 아직 없다. 원래 3907-epoch 캠페인의 completed.json은 연속 학습 종료 표시가 아니다.',
        'Raw-validation best는 BB72 epoch 8032 (29.688%), BB144 epoch 5287 (65.820%)다. 이는 1024-shot raw 지표로 선택된 가중치이며 hierarchical-best와 다르다.', '',
        '## 새 연속 학습 평가', '',
        '`best`는 해당 milestone까지 raw validation으로 고른 모델, `latest`는 milestone 끝 모델이다.',
        '아래 모든 행은 8192-shot 선택 bank 평가다. 표의 최소값을 고르는 것은 이 평가 샷에 대한 선택이며 독립 최종 성능이 아니다.', '',
        '| 모델 | Milestone | Choice | 가중치 epoch | Threshold | Raw (%) | Hier (%) | 실패/8192 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for e in sorted((e for e in evaluations if e['stage']=='continuous'),key=lambda e:(e['experiment'],e['milestone'],e['choice'])):
        lines.append(f"| {e['experiment']} | {e['milestone']} | {e['choice']} | {e['weight_epoch']} | "
                     f"{e['confidence']} | {100*e['raw_LER']:.3f} | {100*e['hierarchical_LER']:.3f} | {e['hierarchical_failures']} |")
    lines += ['', '## 이전 결과와 비교', '',
        '- **BB72:** threshold 0.9에서 기존 선택 bank 최소 2.673% (epoch 1810, 219/8192) → 새 최소 **2.344%** (epoch 7542, 192/8192). 0.330%p, 약 12.3% 상대 감소다. 새 값의 95% Wilson 구간은 2.038–2.694%이며, 여러 checkpoint 중 선택했으므로 독립 검증이 필요하다.',
        '- 7907-latest는 2.759%로 같은 milestone의 best 2.344%보다 높다. 학습 epoch가 늘어날수록 hierarchical LER이 단조 감소하는 것은 아니다. 기존 독립 final 2.600%와 새 선택 bank 최솟값도 구별한다.',
        '- **BB144:** 새 4907-best는 **0.586% (48/8192)**, latest는 **0.854% (70/8192)**다. Threshold는 **0.5 (strict >)**다.',
        '- 이번 수신본에는 과거 milestone의 threshold 0.5 재평가도 있다. 같은 0.5 정책의 3907-latest 0.598% (49/8192)와 비교하면 best는 실패 한 건 감소에 그친다. 개선이라고 단정하기 어렵다. 4907-best의 95% Wilson 구간은 0.442–0.776%, latest는 0.677–1.078%다.',
        '- BB144 기존 1000/1953/2930/3907 best/latest 및 final의 0.5·0.9 평가에서 hierarchical 실패 횟수는 각각 같았다. 같은 가중치·sampler seed 기록을 확인했다. 개별 샷의 보정 결과가 같다는 뜻은 아니며 모든 가중치에서 threshold 영향이 없다는 결론도 아니다.',
        '- 연속 학습에 대한 별도의 독립 final 평가는 없다. 집계 파일만으로 checkpoint 간 paired 유의성 검정이나 X/Z별 재집계는 할 수 없다.', '',
        '## 파일 검증 및 정리', '',
        '- 최신 캠페인 228개 파일을 `results/bb/circuit/standard/dem_cnn/sample_scaling/recovered_60828644_rhhDfV/`에 배치했다.',
        '- 이전 140개 파일은 `sample_scaling/snapshots/2026-11-16/recovered_60828644_rhhDfV/`에 보존했다. 이동 전후 총 368개 파일을 SHA-256 및 크기로 검증했다.',
        '- 이전 파일 누락 없음; 기존 140개 중 BB144 continuous history/training log 두 개만 갱신됐고 88개 파일이 추가됐다. 이전 학습 이력은 새 이력의 prefix와 일치한다.',
        '- JSON 파싱, epoch 연속성, loss 구성·유한값, history와 training log, shot 수·실패율·Wilson 구간·선택 기록을 검사했다. 받은 evaluation attempt에 누락된 results.json은 없다.',
        '- `.pt`와 `.npz`는 다운로드에서 제외되어 로컬에 없다. 따라서 실제 checkpoint 로딩/optimizer 복구 가능 여부나 서버의 모든 파일 수신 여부는 이번 검증 범위 밖이다.',
        '- 루트 `failed.txt` 및 log_worker 파일은 **60828644 / 2026-11-05 초기 flock 실패**의 잔존 기록이다. 최신 job 실패로 해석하지 않는다. 이번 추가분에 최신 `resdir_JOBID` 실행 로그가 없어 정상 종료·timeout·취소 여부 및 후속 제출 상태는 확인 불가다.',
        '- 기존 플롯 및 프로젝트 루트 `.out`은 변경하지 않았다. 서버 경로와 Downloads 원본도 변경하지 않았다.', '',
        '## 다음 CPU/GPU OSD 검증', '',
        '기존 CNN 가중치를 고정하고 CNN → local correction → residual OSD-0 → 최종 correction을 실행한다.',
        'OSD 부분만 기존 CPU/비트 압축 CPU/GPU 후보로 바꾸어 동일 샷의 정확한 보정 결과와 시간을 비교한다. CNN 재학습은 없다.',
        '기본 latest 설정이면 서버가 이 수신 상태와 같을 때 BB72 epoch 8459, BB144 epoch 5478 모델을 복사해 사용한다. 서버 학습이 더 진행됐다면 서버 최신 모델이 선택된다.',
        '기본 threshold 0.9, 모델당 128 shots로 기능·시간을 검증한다. 이 작은 bank의 LER을 위 8192-shot 결과와 성능 순위 비교에 쓰지 않는다.',
        '현재 연속 평가의 hierarchical_seconds는 BB72 약 316–338초, BB144 약 8805–8871초/8192 shots다. CNN·전송·OSD 합계이므로 이를 순수 OSD 시간으로 해석하지 않는다.', '',
        '[전체 평가 CSV](bb_dem_2026_11_20_evaluations.csv) · [학습 상태 CSV](bb_dem_2026_11_20_status.csv) · [이동/해시 목록](bb_dem_2026_11_20_relocation.csv)', '',
        '재검증: `python3 scripts/analyze_bb_dem_november20.py`.']
    (OUT/f'{STEM}.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    protected = {p:sha(p) for p in [*ROOT.glob('*.out'), *files(ROOT/'results/plots').values()]}
    incoming = ROOT/NAME
    current = incoming if incoming.exists() else DEST
    previous = PREVIOUS if PREVIOUS.exists() else DEST
    statuses, evaluations = audit(current, previous)
    if incoming.exists() and not args.archive:
        print(json.dumps(dict(statuses=statuses, evaluations=len(evaluations), ready_to_archive=True),indent=2))
        return
    rows = archive()
    emit('status', statuses)
    emit('evaluations', evaluations)
    report(statuses, evaluations)
    assert all(p.exists() and sha(p)==value for p,value in protected.items())
    print(json.dumps(dict(verified_files=len(rows), evaluations=len(evaluations), statuses=statuses),indent=2))


if __name__ == '__main__':
    main()
