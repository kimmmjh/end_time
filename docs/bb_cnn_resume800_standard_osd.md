# BB144 depth 5 p=.04 to 800 epochs, then standard OSD-0

The active root launchers are `run_bb_0.slurm` and `run_bb_1.slurm`.
The completed three-job continuation campaign's scripts 0–2 are preserved in
[`scripts/slurm_archive/bb_depth45_resume800/`](../scripts/slurm_archive/bb_depth45_resume800/).
There is no active root script 2 for that old campaign.

| Script | Action | GPUs |
| --- | --- | ---: |
| `run_bb_0.slurm` | Resume BB144 depth 5 p=.04 from job 59622483, 400→800 total epochs; evaluate standard OSD afterward | 1 |
| `run_bb_1.slurm` | Evaluate the latest completed model for each configuration/seed, currently 44 models, with raw/residual/standard OSD-0 | 4 |

## Submit on Perlmutter

Copy the updated project Python sources and both Slurm scripts to `~/end_time`.
In particular, the launchers require the current `scripts/evaluate_bb_cnn_osd.py`,
`scripts/continue_bb_tanner_cnn.py`, and `src/_direct_osd.py`; copying only the
Slurm files is insufficient. The environment remains `$PSCRATCH/envs/nde`.

```bash
cd ~/end_time
train_job=$(sbatch --parsable run_bb_0.slurm)
sbatch --dependency=afterok:"${train_job%%;*}" run_bb_1.slurm
```

Script 1 starts only after script 0 succeeds, so it includes the new BB144
p=.04 800-epoch result instead of the current 400-epoch result. Script 0 also
saves its own standard OSD evaluation; the full sweep evaluates that one model
again alongside the other 43, preserving a self-contained complete sweep.
Running script 1 on its own uses the latest **currently completed** models.

Local command previews do not train or evaluate:

```bash
BB_DRY_RUN=1 bash run_bb_0.slurm
BB_DRY_RUN=1 bash run_bb_1.slurm
python scripts/evaluate_bb_cnn_osd.py --auto-roots --latest-only \
  --require-complete --expected-runs 44 --output-root /tmp/bb_osd_preview --dry-run
```

## Required existing files

**Yes, the server needs existing model results.** Preserve the original
`resdir_JOB/outputs/date/run/` structure. Results may remain directly under
`~/end_time/resdir_JOB`, or in the organized local-style
`results/bb/code_capacity/depolarizing/tanner_cnn/` archive.

| Purpose | Files needed per source run |
| --- | --- |
| Script 0: continue training | `model.pt` and `history.json` from BB144 d5 p=.04 in `resdir_59622483` |
| Script 1: saved-bank OSD evaluation | `model.pt`, `history.json`, **`final_shots.npz`** for each latest model |

`model.pt` contains both latest and validation-selected best weights, optimizer,
RNG and sampler state. `best_model.pt` is not a substitute for this workflow.
Training chunks, `chunk_*.npz`, `reference_final_shots.npz`, and old log files are
not needed by the post-hoc evaluator. Do **not** exclude all `.npz`: the saved
`final_shots.npz` is required. No CNN retraining is needed to change OSD.

The current 44 selected models come from these jobs (only the required run
subdirectories and files need to be present):

```text
58793764 58793765
58962932 58962935 58962937
59122447 59122448 59122449 59122450 59122451 59122452
59622483 59622488 59622490
```

After script 0, its new job contributes the BB144 d5 p=.04 source. Keep job
59622483 too, because it supplies three other latest models. The evaluator
checks for 44 latest models before starting; missing selected files cause an
error rather than silently evaluating older weights. Incomplete historical
runs are reported as skipped. Identical archive/root copies are deduplicated;
different results tied at the same completed epoch count are rejected.

To evaluate only the three recently added jobs, use explicit input roots and
change the expected count to 12:

```bash
BB_OSD_EXPECTED_RUNS=12 sbatch run_bb_1.slurm \
  resdir_59622483 resdir_59622488 resdir_59622490
```

This subset command uses their current results, including the 400-epoch
BB144 d5 p=.04 result; it does not discover script 0's new job automatically.

## Training and decoder behavior

Script 0 adds 400 epochs with a cosine LR restart at 1e-4, preserving optimizer,
RNG, samplers, and prior best selection. Width 64, 128×64 training shots/epoch,
4,096-shot validation, and 65,536 final shots remain unchanged.

Training retains raw-validation checkpoint selection and the original residual
OSD diagnostics. **After training, standard syndrome OSD-0 is evaluated on the
new saved final bank.** Script 1 runs inference and OSD only, with no optimizer.
Raw CNN, old residual OSD-0, and new syndrome OSD-0 use identical best weights
and input shots. The new algorithm solves the original syndrome with zero
nonbasis bits and columns ordered by decreasing error probability. Both OSD
arms repair invalid CSS sectors only. This is OSD-0, not higher-order OSD or a
complete ASTRA reproduction. See [decoder details](bb_cnn_posthoc_osd.md).

## Results

Both scripts create a fresh `resdir_JOB` and preserve source results.

- Script 0: training `outputs/...`; new decoder `standard_osd0/.../report.json`
  and `paired_shots.npz`; `log_standard_osd0.txt`. `training_completed.txt`
  records training success; `completed.txt` requires OSD evaluation success too.
- Script 1: `standard_osd0/resdir_SOURCE/date/run/report.json` and
  `paired_shots.npz`; four `standard_osd0/summary_shard_N.json` indexes,
  `log_shard_N.txt`, and shard exit codes. `completed.txt` requires all four
  shards to succeed. Each shard uses one GPU and evaluates 11 of the 44 models.

Reports record final-bank/checkpoint hashes, raw/old/new LER, paired
rescues/harms, and differences from historical saved corrections. Standard OSD
is not assumed to outperform the existing residual version.
