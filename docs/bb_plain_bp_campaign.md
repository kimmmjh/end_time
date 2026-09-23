# Ordinary BP circuit-level campaign

As of September 23, `run_bb_0.slurm` through `run_bb_4.slurm` run **evaluation
only** via `scripts/evaluate_bb_circuit_bp.py`. They replace the corresponding
Neural Relay training jobs. As of September 24, jobs 5–9 run the separate
[Tanner CNN code-capacity campaign](bb_tanner_cnn.md#slurm-campaign-jobs-59);
they are not part of this ordinary-BP sweep.

## Experiments

| Script | Code | p values / seeds |
| --- | --- | --- |
| `run_bb_0.slurm` | BB72 | .001, .002, .003, .004 |
| `run_bb_1.slurm` | BB72 | .005, .006, .008, .010 |
| `run_bb_2.slurm` | BB144 | .001, .002, .003, .004 |
| `run_bb_3.slurm` | BB144 | .005, .006, .008, .010 |
| `run_bb_4.slurm` | Both | Two extra seeds per code at .004 |

Each experiment samples **4096 fresh exact circuit shots once**, then compares
ordinary normalized min-sum BP with maximum **12 and 1000 iterations** on that
bank. Each shot stops at the first syndrome-valid correction. A single trajectory
records both caps; a valid correction is retained at later caps even if logically
wrong. Shots still invalid at the cap count as flagged failures. No training,
neural update, Relay memory, or OSD is run.

Settings match the earlier circuit profile: BB72 has 6 noisy cycles and a batch
size of 16; BB144 has 12 cycles and a batch size of 8. Both have a perfect closing
frame, legacy noise with measurement rate `q=p` and idle rate 0, min-sum scaling
0.625, message clipping at 30, and parallel updates. The seeds match the earlier
sweep labels (BB72 `7201001` etc., BB144 `14401001` etc.), but this standalone
sampler does not recreate the earlier training validation/final streams. Only
the caps within one saved shot bank are paired. These settings are not a full
reproduction of a paper's circuit and decoder.

Reported LER is block failure over the full experiment: an invalid syndrome or
any of the 24 logical-observable mismatches. It is not LER per cycle. A 4096-shot
evaluation cannot establish extremely small LERs; the saved Wilson 95% intervals
remain nonzero above even when no failures are observed.

## Run

Preview without creating results or invoking Slurm:

```bash
BB_DRY_RUN=1 bash run_bb_0.slurm
```

Submit the full p sweep; job 4 is optional additional sampling at p=.004:

```bash
for i in 0 1 2 3; do sbatch "run_bb_${i}.slurm"; done
sbatch run_bb_4.slurm
```

The existing allocation settings remain: account `m5328_g`, four tasks, one GPU
per task, 16 CPUs per task, and 24 hours. Each decoder uses one Torch CPU thread
and GPU tensor updates. Jobs load `python/3.10`, activate `$PSCRATCH/envs/nde`, and
use `$HOME/end_time` (override with `BB_REPO_ROOT`). Deploy the updated scripts,
model, and source files together. No GPU throughput estimate has been measured.
More shots or a longer cap can be requested through inherited environment values:

```bash
BB_BP_SHOTS=16384 BB_BP_REFERENCE_ITERATIONS=10000 sbatch run_bb_0.slurm
```

Local CPU smoke test (new output directory required):

```bash
python scripts/evaluate_bb_circuit_bp.py --code=bb72 --p=0.003 \
  --rounds=1 --shots=8 --batch-size=4 --iteration-caps 12 1000 \
  --seed=923 --device=cpu --out=/tmp/bb_plain_bp_smoke
```

## Outputs

Each allocation writes `resdir_<jobid>`, with `exp_0` through `exp_3` containing:

- `results.json`: configuration, versions/source hashes, completion status,
  LER/confidence intervals, failure counts, mean iterations, and paired gains
  relative to the shortest cap. Positive gain means the longer cap helped.
- `summary.csv`: one aggregate row per cap.
- `shots.npz`: detectors, logical observables, and circuit configuration for
  later evaluation of other decoders on exactly these shots.
- `outcomes.npz`: per-shot success, convergence, and actual iterations per cap.

Partial aggregates are updated every 16 batches and marked `status=running`.
Only `status=complete` has the full requested shot count and saved outcomes.
The script refuses to overwrite an existing experiment directory. Shared
evaluation wall time includes scoring; it is not separate latency for each cap.
The launcher retains commands, logs, source snapshots, and per-experiment exit
codes. No model checkpoints or figures are created.
