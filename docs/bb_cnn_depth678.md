# Depth 6–8 with syndrome OSD-0

`run_bb_depth678.slurm` trains 24 new code-capacity Tanner CNNs. All validation
and final `osd` metrics use `SyndromeOSD0`, selected explicitly with
`--bb_cnn_osd=syndrome`. This is the same decoder as the new post-hoc evaluator:
descending error-probability column order, zero nonbasis bits, and repair of
invalid CSS sectors only. No BP or higher-order OSD runs.

| Array task | Code | Depth | Noise rates |
| --- | --- | ---: | --- |
| 0 | BB72 | 6 | .02, .04, .06, .08 |
| 1 | BB144 | 6 | .02, .04, .06, .08 |
| 2 | BB72 | 7 | .02, .04, .06, .08 |
| 3 | BB144 | 7 | .02, .04, .06, .08 |
| 4 | BB72 | 8 | .02, .04, .06, .08 |
| 5 | BB144 | 8 | .02, .04, .06, .08 |

Each array task uses one node with four GPU steps, one per noise rate. Each
run uses width 64, 800 fresh epochs, 128 batches × 64 shots per epoch
(6,553,600 training shots), validation every five epochs on 4,096 shots, and
65,536 final shots. AdamW uses a single cosine schedule from LR 3e-4,
weight decay 1e-4, and gradient clipping 1.0. Raw validation LER selects the
best checkpoint; OSD does not enter the loss or selection criterion.

Seeds match the earlier grid: BB72 7201020/7201040/7201060/7201080 and BB144
14401020/14401040/14401060/14401080. For a given code/rate, the new depths
share sampler streams and evaluation schedule. Existing depth-4/5 800-epoch
runs used a 400+400 schedule with an LR restart, so comparisons against those
runs are not a controlled depth-only ablation. Each setting has one seed.

## Submit

Copy the updated Python sources and launcher to `~/end_time` on Perlmutter.
In particular, `main.py`, `src/_bb_tanner_cnn_experiment.py`, and
`src/_direct_osd.py` must be current; copying only the Slurm script will not
enable the new OSD option. The environment is `$PSCRATCH/envs/nde`.

```bash
cd ~/end_time
sbatch run_bb_depth678.slurm
```

Submit a subset instead of the full array, if desired:

```bash
# Depth 6 only, both codes (8 models).
sbatch --array=0-1 run_bb_depth678.slurm
# BB144 only, all three depths (12 models).
sbatch --array=1,3,5 run_bb_depth678.slurm
# Full campaign with at most two array tasks active (8 GPUs).
sbatch --array=0-5%2 run_bb_depth678.slurm
```

Command previews do not load modules, create results, or start training:

```bash
BB_DRY_RUN=1 bash run_bb_depth678.slurm
BB_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash run_bb_depth678.slurm
```

## Results and resuming

Each array task creates its own `resdir_JOB`, with four training logs,
`experiments.tsv`, command files, source snapshots, and per-step exit codes.
Each run writes `outputs/DATE/RUN/history.json`, `model.pt`, `best_model.pt`,
and `final_shots.npz`. The checkpoint and bank metadata record
`osd="syndrome_osd0_v1"`; their `osd` results already use the new decoder.
`completed.txt` is written only if all four processes finish successfully.

The 24-hour allocation is a limit, not a measured runtime guarantee for these
depths. Checkpoints are saved every epoch. Resume with the same model/noise
arguments, `--bb_cnn_osd=syndrome`, `--load_model=/path/to/model.pt`, and
`--epochs=ADDITIONAL_EPOCHS`. Resumption starts a new cosine schedule.
The continuation helper also preserves the checkpoint's OSD convention.
Switching OSD while resuming is rejected to avoid mixing evaluation histories.

The default CLI remains `--bb_cnn_osd=residual` for existing launchers.
The post-hoc evaluator supports both conventions and identifies which saved
OSD branch it reproduces. Existing `run_bb_0.slurm` and `run_bb_1.slurm` remain
available for the previous campaign. Script 1's default expected count of 44
applies to the old result set; once new results are present, use explicit input
roots and the corresponding `BB_OSD_EXPECTED_RUNS` count.
