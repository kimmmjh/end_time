# Continuous BB72 and BB144 training on Perlmutter

`run_bb_dem_sample_scaling.slurm` now runs **BB72 D8 and BB144 D6, p=0.002**,
with two independent workers on one node, one GPU and 16 CPU cores per worker.
Each worker first finishes its original 3907-epoch campaign, including its milestone
and independent final evaluations, then continues without a terminal epoch.
The models have independent checkpoints, optimizers, and evaluation progress;
this is not distributed training of a single model. Their epoch counts need not match.

Sync these runtime files before submitting:

- `run_bb_dem_sample_scaling.slurm`
- `scripts/run_bb_dem_sample_scaling.py`
- `src/_bb_dem_cnn_experiment.py`
- `models/_bb_hierarchical_dem_cnn.py`
- `scripts/evaluate_bb_hierarchical_dem_cnn.py`

On Perlmutter, with the earlier job stopped:

```bash
cd ~/end_time
sbatch --export=ALL,BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV" \
  run_bb_dem_sample_scaling.slurm
```

The first job resumes the last saved epoch (3484 in the last downloaded BB144
snapshot before continuous training was added). The November 16 download has
BB72 at 3907 and BB144 at 4870 epochs; newer server checkpoints take precedence.
Existing original-campaign settings and immutable checkpoints are
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

## Hierarchical pre-correction threshold

The default pre-correction rule is now **q > 0.5**, where q is the CNN's
probability for each DEM error mechanism. Exactly q=0.5 abstains. The active
detector candidate filter remains enabled: a mechanism must also touch at least
one active detector. Selected mechanisms can still be undone by residual OSD-0;
they are not permanently frozen. This changes inference only, not training loss
or checkpoint compatibility.

The original manifest and existing `campaign.json` files retain their historical
0.9 setting so that training can resume without changing the frozen recipe.
The effective evaluation threshold is passed separately as `--confidence 0.5`
(Slurm variable `BB_SCALE_CONFIDENCE`, default `0.5`). The model and standalone
evaluator also default to this strict half-probability rule. Explicit higher
thresholds retain the historical inclusive comparison (q >= threshold).

New evaluation results, comparisons, selected weights, and final evaluations
are saved under **`evaluation_policy_q_gt_0p5/`** within the experiment directory
and, for extension milestones, within `continuous/`. Each evaluation metadata
file records the threshold and comparison operator. Training and immutable
checkpoint paths stay the same.

On the first run with the new threshold, existing milestone checkpoints are
re-evaluated at q>0.5 before the runner proceeds; this can take hours. Existing
0.9 results are not reused as 0.5 results or mixed into model selection. This
does not repeat training. Completed evaluations under the new rule are reused
on subsequent jobs. To reproduce the historical policy explicitly, set
`BB_SCALE_CONFIDENCE=0.9`.

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
two tasks, which is now the script default. For BB144 only, set
`BB_SCALE_WORKERS=1` and submit with `--ntasks-per-node=1`.
The default is `0,1` (both). Set the comma-containing value with shell `export`,
not inside Slurm's comma-separated `--export=ALL,...` list.

## Switching an existing BB144-only automatic chain to both models

An already submitted job uses its saved script and worker selection; changing
the repository script does not expand that job's allocation or change its successors.
After syncing the updated launcher, stop the old chain gracefully:

```bash
touch ~/end_time/recovered_60828644_rhhDfV/STOP_CONTINUOUS
squeue -u "$USER" -o "%.18i %.24j %.10T %.20N"
```

Wait until the old campaign's running and queued jobs have stopped. Then start
the two-worker chain, keeping the current evaluation threshold at 0.9:

```bash
cd ~/end_time
rm -f recovered_60828644_rhhDfV/STOP_CONTINUOUS
export BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV"
export BB_SCALE_WORKERS=0,1
export BB_SCALE_CONFIDENCE=0.9
sbatch --export=ALL --ntasks-per-node=2 run_bb_dem_sample_scaling.slurm
```

Both workers must finish saving before the launcher submits one successor.
An error from either worker prevents automatic submission. Original completed
evaluations are reused at confidence 0.9; BB72 starts its extension after 3907,
while BB144 continues its own extension from its last saved epoch. When one
model is doing CPU OSD evaluation, the other worker can continue GPU training.

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

With the default q>0.5 policy, the `evaluations/`, `comparison.json`,
`selected.json`, and original-campaign `final_evaluation/` and `completed.json`
live in `evaluation_policy_q_gt_0p5/` under their respective parent instead of
overwriting the historical files shown above.

Do not delete the persistent recovery directory when cleaning up job logs.
All runs still serialize access using the same per-experiment worker lock.

Slurm reference: [signals and dependencies](https://slurm.schedmd.com/sbatch.html).
