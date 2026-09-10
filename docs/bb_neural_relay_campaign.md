# September 2026 Neural Relay BP Slurm campaign

`run_bb_0.slurm` through `run_bb_9.slurm` now launch this campaign in place of
the previous raw-BP campaign. Ten allocations run four independent experiments
each: **40 trainings**, all with OSD disabled. The circuit and data budgets are
matched to the archived circuit-schema-v2 experiments.

## Common settings

| Setting | BB72 | BB144 |
| --- | --- | --- |
| Code | `[[72,12,6]]` | `[[144,12,12]]` |
| Noisy cycles / detector frames | 6 / 7 | 12 / 13 |
| Noise profile | legacy, `q=p`, idle=0 | legacy, `q=p`, idle=0 |
| Batch size | 16 | 8 |
| Epochs / batches per epoch | 100 / 128 | 100 / 128 |
| Training samples per epoch | 2,048 | 1,024 |
| Validation | 1,024 samples every 10 epochs | 1,024 samples every 10 epochs |
| Fresh selected-best evaluation | 4,096 samples | 4,096 samples |

All models use hidden width 32, orbit embedding width 8, min-sum scaling 0.625,
residual scale 2, relaxation delta 0.5, deep-supervision weight 0.2, syndrome and
logical loss weights 1, mechanism BCE weight 0.1, gradient clipping 1, AdamW
weight decay 1e-4, learning rate 3e-4 and `amp_dtype=none`. Training starts fresh.

The reference Relay configuration is **R=4 legs, T=12 iterations per leg,
S=2 successful legs sought**. First-leg memory is 0.125; subsequent memory is
independent `Uniform(-0.24,0.66)` per shot and mechanism. Training unrolls all
48 iterations. Evaluation stops each shot adaptively with a cap of 48.
Successful legs need not produce distinct corrections.

## Experiment matrix

| Script | Four experiments | Main comparison |
| --- | --- | --- |
| `run_bb_0.slurm` | BB72 Relay, p=.001/.002/.003/.004 | Completed historical raw and OSD low-p points |
| `run_bb_1.slurm` | BB72 Relay, p=.005/.006/.008/.010 | Completed historical OSD high-p points |
| `run_bb_2.slurm` | BB144 Relay, p=.001/.002/.003/.004 | Completed historical raw and OSD low-p points |
| `run_bb_3.slurm` | BB144 Relay, p=.005/.006/.008/.010 | New high-p coverage; historical OSD runs are partial |
| `run_bb_4.slurm` | Two additional p=.004 seeds for each code | Three Relay training seeds per code with jobs 0/2 |
| `run_bb_5.slurm` | Legacy T=12, p=.003/.004 for each code | Fresh anchors for the archived no-OSD decoder |
| `run_bb_6.slurm` | Legacy T=48, p=.003/.004 for each code | Same training depth and inference iteration cap as Relay |
| `run_bb_7.slurm` | Zero memory and constant later memory, p=.004 for each code | Contribution of variable bias and memory disorder |
| `run_bb_8.slurm` | R=1,T=48,S=1 and R=12,T=4,S=2, p=.004 for each code | Alternative allocation of a 48-step budget |
| `run_bb_9.slurm` | S=1 and global-sharing Relay, p=.004 for each code | Candidate stopping and parameter sharing |

Seed A is `7201000 + 1000*p` for BB72 and `14401000 + 1000*p` for BB144,
where the listed p grid makes the offset an integer. These are the exact
seeds of the previous reference sweeps. The p=.004 replicate seeds are
`7202004`, `7203004`, `14402004`, and `14403004`, matching the historical OSD
replicate campaign. Controls use the reference seed for their code and p.

The zero-memory control sets all gamma values to zero while retaining the
Relay execution and candidate selection. The neural MLP still sees the carried
posterior, and Relay uses corrected extrinsic clipping, so this is not identical
to the legacy decoder. The constant-memory control uses first gamma=0.125 and
later gamma=0.21, matching the reference distribution's mean without its
per-variable disorder. The one-leg control necessarily uses S=1 and first-leg
constant memory; job 8 compares complete budget allocations rather than
isolating only the leg count. Global sharing uses fewer trainable parameters;
it is not a parameter-count-matched equivariance ablation.

## What can be compared

1. **Historical no-OSD results:** compare absolute selected-best LER and flagged /
   unflagged failures against
   `results/analysis/bb_circuit_no_osd_2026_09.csv`. The eight low-p reference
   rows match code, noise, rounds, seeds, optimizer settings and sample counts.
   Job 5 repeats four of these legacy points with the current code/environment.
2. **Historical OSD results:** use completed rows in
   `results/analysis/bb_circuit_campaign_2026_08.csv`, comparing their
   `neural_osd_logical_error_rate` with the new raw Relay LER. These are different
   pipelines with separately selected checkpoints. This is not an OSD ablation
   on one frozen model; high-p BB144 partial rows are not final baselines.
3. **Neural versus non-neural within a Relay run:** the paired baseline is Relay
   min-sum with the same memory draws and budget. `vanilla_*` history fields
   refer to this baseline. Legacy runs instead compare against legacy min-sum.
   Do not subtract or pool the paired gains of these two different baselines.
4. **Additional compute versus Relay:** compare jobs 0/2 with job 6 at p=.003/.004.
   All have 48 training steps per sample and the same inference cap. Early
   stopping, candidate retention, resets and clipping differ. Use recorded mean
   iterations/legs and wall times; equal iteration caps do not imply equal
   GPU latency or memory usage.

All runs select checkpoints by their own raw paired gain. S=1 also changes
validation/selection, so its separately trained run is not a clean inference-only
candidate ablation. For causal decoder-only comparisons, freeze one checkpoint
and evaluate all variants on one saved test bank. Matching seeds/configurations
across archived jobs alone does not establish paired statistical significance.
The 4,096-shot final evaluations preserve comparability but cannot resolve very
small LERs or support a threshold claim by themselves.

New Relay histories have `architecture=bb_neural_relay_bp_circuit` and
`baseline_decoder=relay_min_sum`. The previous summary scripts filter for the
legacy architecture and must not be used to aggregate the new Relay campaign.
Read each run's `history.json` and its recorded config; `experiments.tsv` maps
the launch index to the variant and exact arguments. Existing archived result
files are not overwritten by this campaign.

## Preview, submit and archive

Preview any complete job without loading modules, activating an environment,
creating results, or invoking Slurm:

```bash
BB_DRY_RUN=1 bash run_bb_0.slurm
```

The real jobs retain account `m5328_g`, four GPUs per allocation, one GPU and
16 CPU cores per experiment, and a 24-hour limit. They load `python/3.10`,
activate `$PSCRATCH/envs/nde`, and run in `$HOME/end_time`. `BB_REPO_ROOT` can
explicitly override the repository location. Deploy the updated model, main,
trainer and helper files along with the `.slurm` files.

Submit the overlapping low-p references and fresh controls first:

```bash
for i in 0 2 5 6; do sbatch "run_bb_${i}.slurm"; done
```

Or submit all ten jobs:

```bash
for i in {0..9}; do sbatch "run_bb_${i}.slurm"; done
```

Results use `resdir_<SLURM_JOB_ID>` and preserve command files, variant labels,
the submitted script, defaults, decoder/trainer source snapshots, git identity,
per-experiment exit codes, and training outputs. Each training output contains
resumable `model.pt`, selected `best_model.pt`, and history. A time-limited run
must be reported as partial until a fresh final evaluation is present.

The 48-step runs cost more than the archived T=12 trainings. No GPU memory or
wall-time guarantee has been measured for this campaign; verify the first jobs'
resource use before committing the rest of the allocation budget.
