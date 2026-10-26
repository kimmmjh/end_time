# Tanner CNN depth 10/12

Submit from `~/end_time` on Perlmutter after uploading
`run_bb_depth1012.slurm`:

```bash
sbatch run_bb_depth1012.slurm
```

The four array tasks are 0=BB72 depth10, 1=BB144 depth10,
2=BB72 depth12, 3=BB144 depth12. Each task allocates one node with
four GPUs and runs p=.02/.04/.06/.08 concurrently, one experiment per GPU.
There are 16 experiments in total; each array task has a 24-hour limit.

Training follows the depth 6/7/8 campaign: fresh 800 epochs, width 64,
batch size 64, 128 batches per epoch, cosine LR 3e-4, matching code/p seeds,
and residual (hard-centered) OSD-0 evaluation. Validation uses 4,096 shots every five epochs;
raw validation LER selects the best checkpoint. Final evaluation uses 65,536
shots. Both latest and best weights and the final shot bank are retained for
later syndrome OSD comparison. This launcher performs training; it does not
run the separate three-arm posthoc evaluator.

The script contains `--mail-user=ds08152@gmail.com` and `--mail-type=END,FAIL`.
Notifications apply to the array as a whole, not every experiment, following
the [Slurm mail-type rules](https://slurm.schedmd.com/sbatch.html#OPT_mail-type).

Preview all 16 commands without submitting or training:

```bash
BB_DRY_RUN=1 bash run_bb_depth1012.slurm
```

To submit BB144 only, use `sbatch --array=1,3 run_bb_depth1012.slurm`.
Outputs follow the existing layout: `resdir_<task job ID>/outputs/`, with
per-experiment logs, exit codes, source snapshots and completion markers.
The Slurm log filename includes both the array job ID and task index.

## Received results (2026-10-26)

All 16 models are archived and plotted: 60255257=BB72 D10 (task 0),
60261485=BB144 D10 (task 1), 60261486=BB72 D12 (task 2),
60255255=BB144 D12 (task 3). All completed 800 epochs.
See [analysis](../results/analysis/bb_update_2026_10_26_evening.md) and
[same-residual comparison](../results/plots/bb/code_capacity/tanner_cnn/depth_comparison.png).
These downloads contain history/logs only; weights and shot banks were excluded.
The depth 6/7/8 residual reevaluation (60255150) is also imported, with original
input hashes verified; its posthoc paired shot files were not downloaded.
