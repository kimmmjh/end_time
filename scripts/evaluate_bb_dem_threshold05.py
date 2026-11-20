#!/usr/bin/env python3
"""Evaluate BB72 latest/best and BB144 latest at q>0.5, without training."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_bb_hierarchical_dem_cnn import evaluate, paired, write_json
from scripts.run_bb_dem_sample_scaling import MANIFEST, atomic_copy, digest, worker_lock


def compare_shots(best, latest, shots):
    import numpy as np

    left = sorted((best/'shots').glob('batch_*.npz'))
    right = sorted((latest/'shots').glob('batch_*.npz'))
    if not left or [p.name for p in left] != [p.name for p in right]:
        raise RuntimeError('Checkpoint evaluations do not have matching shot batches')
    best_success, latest_success = [], []
    for a,b in zip(left,right):
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            for key in ('detectors','observables'):
                if not np.array_equal(x[key],y[key]):
                    raise RuntimeError(f'Checkpoint comparison changed {key} in {a.name}')
            best_success.append(x['hierarchical_success'])
            latest_success.append(y['hierarchical_success'])
    best_success, latest_success = np.concatenate(best_success), np.concatenate(latest_success)
    if len(best_success) != shots or len(latest_success) != shots:
        raise RuntimeError('Incomplete checkpoint comparison')
    return paired(latest_success, best_success)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, default=REPO/'recovered_60828644_rhhDfV')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(0,1), default=0)
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.shots,args.batch_size) < 1 or args.seed < 0:
        parser.error('Counts must be positive and seed nonnegative')
    entry = json.loads(MANIFEST.read_text())['experiments'][args.worker]
    if entry['name'] != ('bb72_d8_p0.002','bb144_d6_p0.002')[args.worker]:
        raise ValueError('Unexpected source manifest entry')
    choices = ('latest','best') if args.worker == 0 else ('latest',)
    experiment = args.campaign_root.resolve()/entry['name']
    source = experiment/'continuous/training/model.pt'
    plan = dict(experiment=entry['name'], source=str(source), weights=list(choices),
                confidence=.5, comparison='gt', shots=args.shots, seed=args.seed,
                training=False, automatic_resubmission=False, backend='direct_hard_centered_osd0_v1')
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return plan
    import torch
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        snapshot = args.output/'checkpoint.pt'
        # Never copy an actively changing campaign or silently fall back to old weights.
        with worker_lock(experiment):
            checksum = digest(source)
            atomic_copy(source,snapshot)
            if digest(snapshot) != checksum:
                raise RuntimeError('Checkpoint snapshot hash mismatch')
        write_json(args.output/'source.json',dict(**plan,sha256=checksum,
                    expected_config=entry['expected_config']))
        results, metadata = {}, {}
        for choice in choices:
            config = argparse.Namespace(weights=choice, device=args.device,
                seed=args.seed, batch_size=args.batch_size, shots=args.shots,
                edge_chunk_size=4096, confidence=.5, compare_direct_osd=False,
                save_corrections=False, output=args.output/choice)
            results[choice] = evaluate(config,snapshot,entry['expected_config'])
            metadata[choice] = json.loads((config.output/'metadata.json').read_text())
            if results[choice]['status'] != 'complete' or results[choice]['shots'] != args.shots:
                raise RuntimeError('Threshold evaluation incomplete')
        latest = metadata['latest']
        paired_result = None
        if 'best' in metadata:
            best = metadata['best']
            if (best['source']['sha256'] != latest['source']['sha256'] or
                    best['sampler_seed'] != latest['sampler_seed']):
                raise RuntimeError('Evaluations used different checkpoint snapshots or seeds')
            paired_result = compare_shots(args.output/'best',args.output/'latest',args.shots)
        if digest(snapshot) != checksum:
            raise RuntimeError('Checkpoint snapshot changed during evaluation')
        summary = dict(status='complete',experiment=entry['name'],confidence=.5,comparison='gt',
            sources={choice:m['source'] for choice,m in metadata.items()},
            seed=args.seed,sampler_seed=latest['sampler_seed'],shots=args.shots,
            evaluations=results,best_vs_latest=paired_result,
            note='Best means raw-validation-selected weights embedded in the current checkpoint. '
                 'For BB72, rescued means latest failed and best succeeded on the same shots. '
                 'This is a selection bank, not an independent final test.')
        write_json(args.output/'comparison.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        return summary
    except Exception as exc:
        write_json(args.output/'failure.json',dict(status='failed',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
