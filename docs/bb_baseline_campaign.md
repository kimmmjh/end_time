# Circuit BB: library BP and BP+OSD baselines

`run_bb_0.slurm`–`run_bb_4.slurm` now call
`scripts/evaluate_bb_circuit_baselines.py`. Each experiment evaluates six rows
on **one exact Stim circuit shot bank**:

| Detector input | Methods |
| --- | --- |
| `x`: X-check detectors only | plain BP, BP+OSD-0, BP+OSD-CS3 |
| `xz`: joint X/Z detector matrix | plain BP, BP+OSD-0, BP+OSD-CS3 |

`xz` is a joint DEM, not two independent CSS decoders. All rows predict the
same 12 logical-X measurement flips, i.e. logical-Z errors. This is not the
old experiment's 24-observable joint-Pauli correction task.

## Source of the settings

- [Blue et al., arXiv:2504.13043v2, §1.3 and §2.7](https://arxiv.org/html/2504.13043v2):
  logical-plus input, ideal reference and closing rounds, d noisy cycles,
  noisy idle locations, logical-Z Block LER over the whole memory experiment.
  Its BP-OSD LER baseline uses X checks and OSD order 3; timing uses OSD-0.
  It does not report a separate plain-BP LER curve or specify a BP iteration cap.
- [Diffusion, arXiv:2509.22347v1, §IV.6](https://arxiv.org/html/2509.22347v1):
  min-sum, maximum 1000 iterations, scaling 1.0, combination-sweep OSD order 3.
  That paper uses CUDA-Q; this campaign uses `ldpc==2.4.1`, so this is a match
  of these decoder settings, not a bitwise reproduction of CUDA-Q.
- [Bravyi et al. author implementation](https://github.com/sbravyi/BivariateBicycleCodes/blob/fa77e3333d3ec44c79d8f914dd24c040d1da471b/decoder_setup.py):
  eight-tick extraction, with seven CNOT layers. The X neighbor order is
  `[idle,1,4,3,5,0,2]`; the Z order is `[3,5,0,1,2,4,idle]`.

The circuit is implemented independently in `src/bb_paper_baselines.py`,
using native RX/MX and R/M (no extra Hadamard ticks). Preparation and readout
flip with probability p; CNOT uses DEP2(p); every inactive qubit at each tick
uses DEP1(p). The first and closing extraction cycles are perfect. Logical
observables come from the final noiseless physical-X measurements. Gate supports
are checked against Hx/Hz and Stim checks detector/observable determinism.
These are documented literature conditions; matching published numerical curves
still requires the new statistics. Unspecified choices in Blue cannot be claimed
as an exact author-code reproduction.

## Decoder and metrics

A single `ldpc.BpOsdDecoder` call runs BP on the **original DEM priors**, stops
when its syndrome matches, and otherwise runs OSD-CS3 on that BP posterior.
`bp_decoding`, `osd0_decoding`, and the returned CS3 correction supply the three
paired outcomes. Converged BP corrections are retained for all three methods.
There is no posterior reseeding, extra BP pass, neural update, Relay memory,
custom clipping, or 0.625 normalisation. The backend receives the fixed identity
scale 1.0. Library scheduling is parallel; OpenMP is one thread per worker.

Two error metrics are saved explicitly:

- `logical_z_mismatch_rate`: any of the 12 logical-X predictions disagrees
  with the observed value. Use this for Blue's logical-prediction task.
- `block_failure_rate`: invalid syndrome OR a logical mismatch. Use this when
  evaluating a decoder required to return a valid correction.

Also saved: syndrome convergence, flagged/unflagged failures, mean BP iterations,
paired gains for both metrics, and Wilson 95% intervals. For OSD the two error
rates coincide because all corrections satisfy the syndrome. Zero observed
failures are reported with a nonzero upper confidence bound, not proof of zero
LER. Wall time is for the combined pipeline, not a per-method latency benchmark.

## Sweep

| Script | Code | p | Shots per experiment |
| --- | --- | --- | ---: |
| 0 | BB72 | .001, .002, .003, .004 | 100,000 |
| 1 | BB72 | .005, .006, .007, .008 | 100,000 |
| 2 | BB144 | .001, .002, .003, .004 | 100,000 |
| 3 | BB144 | .005, .006, .007, .008 | 100,000 |
| 4 | BB72 and BB144 | .001 and .002, new seeds | 1,000,000 |

Noisy cycles are 6 for BB72 and 12 for BB144. Every experiment produces all six
method/input rows. Job 4 is the extra low-error statistics campaign; its samples
are independent of jobs 0/2. Very small LER may still require more shots; assess
failure counts and confidence intervals before comparing curves.

Jobs 0–4 use **Perlmutter CPU nodes**, account `m5328`, 4 tasks/node, 32 physical
CPUs/task, 32 single-thread processes/experiment, and a 24-hour limit. Each step
requests 48 GiB. Native `ldpc` BP/OSD is CPU code. Jobs 5–9 remain the separate
GPU Tanner CNN campaign. The allocation has not been submitted from this workspace.

```bash
BB_DRY_RUN=1 bash run_bb_0.slurm
for i in 0 1 2 3 4; do sbatch "run_bb_${i}.slurm"; done
```

All launch logic is embedded in each `.slurm`, with no repository `.sh` helpers.
The environment is `$PSCRATCH/envs/nde`, repository `$HOME/end_time` or
`BB_REPO_ROOT`. CPU allocation account availability depends on the project;
`sbatch --account=YOUR_CPU_ACCOUNT run_bb_0.slurm` overrides the header if needed.
The CPU/GPU account naming follows the
[NERSC examples](https://docs.nersc.gov/jobs/interactive/).

Overrides, applied before submission:

```bash
BB_BASELINE_SHOTS=200000 BB_BASELINE_WORKERS=16 sbatch run_bb_0.slurm
```

`BB_BASELINE_ITERATIONS` can change the cap for an explicitly labelled ablation;
its default is 1000. Do not pool runs with different caps/circuits/input policies.
Old queued scripts are not replaced by a git pull: submit updated files as new jobs.

## Artifacts and resumption

Each job writes `resdir_JOB/exp_0`–`exp_3`. Each experiment saves:

- `circuit.stim`, `detector_error_model.dem` and checksums;
- per-input sparse check/logical matrices, prior probabilities and selected detector rows;
- `shots.npz`: the entire circuit bank, bit-packed with little-endian bit order;
- `chunks/chunk_START_STOP.npz`: packed corrections for all six methods,
  logical predictions, success/convergence flags and actual BP iterations;
- `results.json`, `summary.csv`: continuously updated aggregates, versions,
  exact settings, source hashes and invocation history.

Chunks are atomically saved and can be independently rescored. Completed chunks
are reused on resume; the bank is never resampled. Physics, decoder, shot count,
seed and source hashes must match. Workers/chunk size may change. A killed or
wall-time-limited job may have `status=running`; inspect `shots_completed` and
resume it rather than treating it as a complete sweep.

To continue an entire previous allocation, use its absolute result directory and
**the same numbered script and overrides**:

```bash
BB_BASELINE_RESUME_ROOT="$HOME/end_time/resdir_OLDJOB" sbatch run_bb_0.slurm
```

The new allocation saves its own launcher logs; data continues in the old
experiment directories. Already completed experiments return without decoding
again. To resume just one experiment, use the same Python command with `--resume`.

Local smoke test:

```bash
python scripts/evaluate_bb_circuit_baselines.py --code=bb72 --p=.003 \
  --shots=8 --seed=927 --workers=2 --chunk-size=2 --out=/tmp/bb_library_smoke
```

## Removed baseline data

The five September 27 imports using scale 0.625 were removed at the user's
request: `58793753`, `58793754`, `58793756`, `58793757`, `58793759`, together
with their plain-BP aggregate CSVs and PNG. The Tanner CNN data and previous
Neural/Relay experiments are retained. No old numerical result is relabelled as
this new baseline.
