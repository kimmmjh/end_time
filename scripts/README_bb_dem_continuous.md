# Continuous BB144 training on Perlmutter

`run_bb_dem_sample_scaling.slurm` now runs **BB144 D6, p=0.002 only**, with one
GPU. It first finishes the original 3907-epoch campaign, including its milestone
and independent final evaluations, then continues without a terminal epoch.
BB72 is not launched by default.

Sync these three changed runtime files before submitting:

- `run_bb_dem_sample_scaling.slurm`
- `scripts/run_bb_dem_sample_scaling.py`
- `src/_bb_dem_cnn_experiment.py`

On Perlmutter, with the earlier job stopped:

```bash
cd ~/end_time
sbatch --export=ALL,BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV" \
  run_bb_dem_sample_scaling.slurm
```

The first job resumes the last saved epoch (3484 in the last downloaded BB144
snapshot). Existing original-campaign settings and immutable checkpoints are
preserved. Do not edit the original source manifest to increase its target.

## Training and checkpoint state

Through epoch 3907, the saved cosine schedule continues unchanged. After that,
the extension starts from **latest epoch-3907 weights and optimizer state**, not
the validation-selected best weights. Its LR is set once to **1e-5**, then stays
constant across job boundaries. This is an initial continuation setting, not a
claim that this LR is optimal. A fixed LR avoids tying an indefinite run to an
arbitrary cosine endpoint.

Each completed epoch atomically saves model weights, AdamW first and second
moments and step counters, optimizer parameter groups, scheduler state, epoch,
training history, raw-validation best weights, and PyTorch CPU/CUDA RNG states.
Loading the checkpoint restores these values; restarting a job does not reset
AdamW or the extension LR. A hard kill during an epoch can lose that epoch, but
leaves the previous completed checkpoint available.

Stim samplers do not expose serializable RNG state. As before, restarting a
training segment derives new independent training/validation seeds from the
base seed and completed epoch. This preserves fresh sampling, but does not
promise a bitwise-identical sample stream to an uninterrupted allocation.

The batch size remains 4, with 128 batches (512 training samples) per epoch.
Raw validation uses 1024 shots every 5 epochs within a resumed segment, plus
the segment endpoint. Extension milestones are **4907, 5907, 6907, ...**. Each
evaluates raw-validation best and latest weights with 8192 hierarchical shots
on the existing selection bank. Interrupted evaluation attempts are retained
and retried; completed matching evaluations are reused. The original final
held-out bank is not used to choose the extension's best model. The indefinite
extension has no automatic final held-out evaluation; run a fresh held-out
evaluation once you decide to finish model selection.

## Time limits, automatic submission, and stopping

Each allocation requests 48 hours. Slurm sends USR1 about 30 minutes before
the time limit; the worker finishes its current epoch and saves, or interrupts
the current hierarchical evaluation. After the worker exits cleanly with its
pause code, the launcher submits one successor with an `afterok` dependency
and exits successfully. The successor can start when resources become
available; queue wait time is not avoided. The next job ID is recorded in
`resdir_JOBID/next_job_id.txt`.

Python errors, cancellation, hard timeout before a graceful exit, or failed
submission do not trigger an automatic retry loop. Read that job's logs and
resume manually after resolving the problem. A stale worker lock after a hard
kill must be inspected and its owning job confirmed stopped before removal.

To stop gracefully and prevent further jobs:

```bash
touch ~/end_time/recovered_60828644_rhhDfV/STOP_CONTINUOUS
```

An already queued successor also checks this file and exits without training.
Keep the file until the active/queued jobs have stopped. To intentionally start
again later, remove only this stop marker, then use the submission command above:

```bash
rm ~/end_time/recovered_60828644_rhhDfV/STOP_CONTINUOUS
```

To run one 48-hour allocation with manual resubmission instead of a chain:

```bash
sbatch --export=ALL,BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV",BB_SCALE_AUTO_RESUBMIT=0 \
  run_bb_dem_sample_scaling.slurm
```

`BB_SCALE_CONTINUATION_LR` and `BB_SCALE_CONTINUATION_EVERY` can change the
initial extension recipe before it starts. Once `continuous/campaign.json`
exists, a conflicting recipe is rejected rather than silently changing a run.
`BB_SCALE_CONTINUOUS=0` restores the original finite 3907-epoch behavior.
`BB_SCALE_WORKERS=0` selects BB72; `0,1` selects both and also requires
`sbatch --ntasks-per-node=2`. The default is `1` (BB144).

## Output locations

Job logs and submitted scripts go to **`resdir_JOBID`**, without descriptive
suffixes. Model data continues in the original persistent recovery directory:

```text
recovered_60828644_rhhDfV/bb144_d6_p0.002/
  training/                  original training through 3907
  checkpoints/               original immutable milestone checkpoints
  final_evaluation/          original held-out evaluation
  continuous/
    campaign.json            frozen extension settings
    training/model.pt        latest checkpoint to resume
    training/best_model.pt   best raw-validation checkpoint
    training/history.json    full cumulative history
    checkpoints/             immutable extension milestones
    evaluations/             hierarchical best/latest evaluations
    comparison.json          completed extension evaluation pairs
    selected.json            best on the selection bank, including old milestones
```

Do not delete the persistent recovery directory when cleaning up job logs.
All runs still serialize access using the same per-experiment worker lock.

Slurm reference: [signals and dependencies](https://slurm.schedmd.com/sbatch.html).
