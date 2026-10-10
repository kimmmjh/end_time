# Previous code-capacity campaign launchers

Exact prior scripts 0–8: depths 4/5/3 at 400 epochs and depth-2 continuation
to 400. Active root scripts 0–2 now run selected depth-4/5 continuations and recovery.

For a historical dry run from the repository root, set the repository explicitly:

```bash
BB_REPO_ROOT="$PWD" BB_DRY_RUN=1 bash scripts/slurm_archive/bb_capacity400/run_bb_0.slurm
```

Cluster execution still defaults to `$HOME/end_time` as in the original scripts.
