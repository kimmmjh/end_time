# BB DEM CNN: continuation to two million samples

Run BB72 D8 and BB144 D6 at p=0.002 from the completed epoch-1,000 latest
checkpoints. Keep architecture, loss, batch size 4, 128 batches/epoch, weight
decay, clipping, and gradient checkpointing unchanged. Continue AdamW state.
Start one cosine schedule at 3e-5 over 2,907 additional epochs (372,096 updates).
The total is 3,907 epochs, or **2,000,384 training samples per model**.

## Submit and preview

On Perlmutter, from `~/end_time`:

```bash
BB_DRY_RUN=1 bash run_bb_dem_sample_scaling.slurm
sbatch run_bb_dem_sample_scaling.slurm
```

The metadata-only preview works without downloaded `.pt` files. Real execution
requires the original checkpoints on the server, verifies their configurations,
and checks that the source job has stopped before copying weights.

The frozen source manifest is `scripts/manifests/bb_dem_sample_scaling.json`.
Both sources live directly under the repository's server root:

| Experiment | Source job | Source worker | Resume epoch | Existing best epoch |
|---|---|---:|---:|---:|
| BB72 D8, p=0.002 | resdir_60694884_followup_train | 10 | 1000 | 985 |
| BB144 D6, p=0.002 | resdir_60694884_followup_train | 8 | 1000 | 985 |

`BB_REPO_ROOT` overrides the repository location; `BB_DEM_SOURCE_ROOT` overrides
the parent of the source `resdir_*` folders. Sources are never modified.

The launcher uses account `m5328_g`, regular GPU QoS, one node, 24 hours,
two workers with one GPU and 16 CPUs each, and the existing
`$PSCRATCH/envs/nde` environment. It does not use the other two GPUs on a
four-GPU node. Email is enabled for END/FAIL to `ds08152@gmail.com`.

## Evaluation and checkpoint selection

| Epochs completed | Cumulative training samples | Action |
|---:|---:|---|
| 1000 | 512,000 | Baseline best/latest hierarchical evaluation |
| 1953 | 999,936 | Save immutable checkpoint; evaluate best/latest |
| 2930 | 1,500,160 | Save immutable checkpoint; evaluate best/latest |
| 3907 | 2,000,384 | Save immutable checkpoint; evaluate best/latest |

Raw validation remains 1,024 circuit shots every five epochs within a training
segment, plus its endpoint. Checkpoints are saved after every completed epoch.
Raw best retains the earliest validation tie, including the inherited best.
There is no full raw final-shot bank at each segment: the hierarchical evaluator
already reports paired raw/local/hierarchical outcomes.

All eight selection evaluations use 8,192 shots, batch size 4, seed 20261103,
confidence 0.9, and the existing residual DirectOSD0 decoder. Equal seed,
batch size, circuit, and simulator version give the same selection bank for a
given code. Results retain per-batch observations for paired analysis.

Select the checkpoint/weight choice with fewest hierarchical failures. Ties
prefer the earlier milestone, then raw-best over latest. Re-evaluate that winner
on 8,192 independent shots with seed 20261104. This final bank is not used for
selection. The baseline can win. These shot counts are an initial budget;
very low LER or small differences may require more evaluation shots.

## Wall-time recovery

Training plus dense OSD evaluation can take several allocations. Thirty minutes
before wall time, Slurm requests a stop. Training saves and stops at an epoch
boundary. Evaluations stop promptly; incomplete attempts are retained and
restarted on a fresh copy of the same bank next time. Completed evaluations are
reused, never pooled with their interrupted attempts. Hard termination can lose
the current epoch; the last atomically saved checkpoint remains recoverable.

The first job writes `resdir_<JOBID>/`. To continue it after that job stops:

```bash
sbatch --export=ALL,BB_SCALE_RUN_DIR=$HOME/end_time/resdir_<FIRST_JOBID> \
  run_bb_dem_sample_scaling.slurm
```

Replace `<FIRST_JOBID>` with the numeric ID. Continue pointing to this **same
first campaign directory** on every resubmission. Each new job writes its own
`resdir_<NEW_JOBID>/` logs and source snapshot; model and evaluation progress
remain under the original campaign directory. Never submit overlapping resumes.
Per-worker filesystem locks reject concurrent modification. This script does
not automatically submit or requeue jobs.

The saved scheduler state is restored, so neither milestones nor resubmissions
restart the cosine. AdamW state and PyTorch RNG state are restored. Stim gets a
new independent stream at each training segment/recovery, so this is **not a
bitwise reproduction of an uninterrupted sample stream**. Sample totals count
completed, checkpointed training epochs only.

Exit 75 / `interrupted.txt` means more work remains and resubmission is needed;
it can trigger a FAIL email. `completed.txt` means both models, including final
held-out evaluation, finished. A second worker may finish before the first; its
completed results are reused on resubmission.

## Files to inspect

Under each `resdir_<FIRST_JOBID>/{bb72_d8_p0.002,bb144_d6_p0.002}/`:

- `training/model.pt`: latest resumable weights, AdamW, scheduler, and history.
- `training/history.json`: training losses and raw validation metrics.
- `checkpoints/epoch_*.pt`: immutable milestone checkpoints including raw-best weights.
- `evaluations/epoch_*/{best,latest}/attempt_*/`: metrics and paired shot batches.
- `comparison.json`: completed baseline/milestone selection results.
- `selected.json`: winner selected by hierarchical failure count.
- `final_evaluation/`: independent final evaluation, including partial attempts.
- `completed.json`: winner and final result once the experiment finishes.

For local plotting/analysis, download JSON/log files from the original campaign
and subsequent job folders; retain `.pt` on the server for recovery. Download
`.npz` files as well if paired shot-level statistical comparisons are wanted.
