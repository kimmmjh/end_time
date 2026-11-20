# BB72 latest/best and BB144 latest at threshold 0.5

`run_bb_dem_threshold05_eval.slurm` replaces the BB72-only threshold comparison.
It runs these evaluations at p=0.002 using strict **q>0.5**:

| Worker | Model | Weights | Last downloaded epoch |
|---|---|---|---:|
| 0 | BB72 D8 | latest | 8459 |
| 0 | BB72 D8 | raw-validation best | 8032 |
| 1 | BB144 D6 | latest | 5478 |

The actual epochs come from each server checkpoint when the job starts. The
earlier BB144 0.5 result of 0.586% used epoch 4712; it is not the current
raw-validation best (5287 in the latest download). This job intentionally skips
BB144 best as requested. BB72 best means the raw-validation selection embedded
in `model.pt`, not the historical hierarchical-selected epoch 7542.

Each worker copies its model's `continuous/training/model.pt` while holding the
campaign worker lock, verifies the hash and releases the lock. All evaluation
uses the fixed copy. Active training or a stale lock causes failure rather than
lock removal or fallback to old weights. Stop training before submitting.

## Synchronizing scripts through Git

`scripts/` is tracked again. These Python helpers and manifests are required by
the root Slurm launchers. Commit/push the restored files and `.gitignore` change
on the Mac, then `git pull --ff-only` in `~/end_time` on Perlmutter. Future updates
use Git for both the launchers and their scripts; no separate rsync is needed.

If the server previously pulled the untracking commit and received scripts via
rsync, its untracked copies may block the first pull that restores tracking.
Only if Git reports that these untracked script files would be overwritten,
move the existing directory aside and retry:

```bash
cd ~/end_time
scripts_backup=$(mktemp -d "$HOME/end_time_scripts_before_retrack.XXXXXX")
mv scripts "$scripts_backup/"
git pull --ff-only
```

The restored tracked files then come from Git. The old server copy remains in
the printed variable's directory for comparison; do not copy it over the new
files. If the pull fails for another reason, address that error before launching.

## Launch and read results

On Perlmutter, after syncing and stopping training:

```bash
cd ~/end_time
export BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV"
sbatch --export=ALL run_bb_dem_threshold05_eval.slurm
```

Defaults: one node, two workers with one GPU each, 8 hours, **8192 shots per
evaluation**, batch size 4, seed 20261103, END/FAIL email enabled. GPU CNN
inference and existing NumPy CPU OSD-0 are used. There is no training, threshold
0.9 evaluation, BB144-best evaluation or automatic successor submission.
`BB_SCALE_CONFIDENCE`/`BB_OSD_CONFIDENCE` do not change this fixed 0.5 experiment.

The two BB72 choices use identical samples, verified from saved detector and
observable arrays. Their raw predictions may differ because weights differ.
The output paired counts `best_vs_latest.rescued` mean latest failed and best
succeeded; `harmed` means the reverse. These are selection-bank results, not
independent final tests.

Outputs under `resdir_JOBID/`:

- `worker_0/latest/`, `worker_0/best/`: BB72 evaluation metadata and results.
- `worker_1/latest/`: BB144 evaluation metadata and results.
- Each worker has `comparison.json` with selected epochs and metrics,
  `source.json` and a private `checkpoint.pt`.
- Root worker logs, exit codes and completed/failed marker report job status.

Download JSON/log files as usual; excluding `.pt` and `.npz` retains metrics
and the paired summary. Keep per-shot arrays on the server for later analysis.
Optional overrides: `BB_THRESHOLD_SHOTS`, `BB_THRESHOLD_SEED`.

```bash
BB_DRY_RUN=1 bash run_bb_dem_threshold05_eval.slurm
python -m pytest -q tests/test_bb_dem_threshold05.py
```
