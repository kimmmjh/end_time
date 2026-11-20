# CPU/CUDA OSD-0 comparison on Perlmutter

This experiment compares the existing NumPy `DirectOSD0` with an experimental
bit-packed Torch implementation on CPU and CUDA. It uses the same CNN outputs,
residual syndromes, probabilities and shots for all three backends. Exact
corrections must agree, as well as syndrome satisfaction and logical predictions.
The production training and evaluation backend is unchanged.

The candidate packs eight matrix entries into each byte and moves GF(2) row
elimination to CUDA. Sorting and sparse parity checks remain on CPU. Samples
are decoded sequentially; pivot selection synchronizes with the host. This is
an initial correctness/timing prototype, not an optimized CUDA kernel. It may
be slower than the CPU reference. Packed CPU measurements help distinguish the
benefit of packing from the benefit of using a GPU.

## Stop the successor and synchronize

Let the running training allocation finish and save normally. Once its
successor is PENDING, inspect the queue and cancel that specific pending job:

```bash
squeue -u "$USER" -o "%.18i %.24j %.10T %.12M %.20N"
next_job=REPLACE_WITH_PENDING_JOB_ID
scancel --state=PENDING "$next_job"
```

Repeat for another training chain if one exists. The state filter restricts
cancellation to PENDING jobs ([Slurm documentation](https://slurm.schedmd.com/scancel.html)).
If the job has already started, the command will not stop it. Inspect its state
before deciding when to run the benchmark. Do not remove an active worker lock.

Commit/push the local changes and pull them on Perlmutter using your usual Git
workflow. New runtime files to synchronize together:

- `src/_packed_osd.py`
- `scripts/benchmark_bb_dem_osd.py`
- `run_bb_dem_osd_benchmark.slurm`

Keep the existing evaluator, sample-scaling helper and source manifest in sync
as well: the benchmark imports their checkpoint and locking utilities.

## Submit after training has stopped

```bash
cd ~/end_time
export BB_SCALE_RUN_DIR="$HOME/end_time/recovered_60828644_rhhDfV"
export BB_OSD_WORKERS=0,1
export BB_OSD_CONFIDENCE=0.9
sbatch --export=ALL run_bb_dem_osd_benchmark.slurm
```

Defaults are BB72 D8 and BB144 D6, one GPU each on one node, 4 hours,
128 shots per model, batch size 4, two timing repetitions and latest weights.
There is no training or automatic successor submission. END/FAIL email is enabled.
To check the other inference policy, submit a separate job with
`BB_OSD_CONFIDENCE=0.5`; the rule is strict q>0.5 at this threshold.
Use `BB_OSD_WEIGHTS=best` to select raw-validation best weights instead of latest.

Each worker selects `continuous/training/model.pt` if present, otherwise
`training/model.pt`. It acquires the training worker lock while copying the
checkpoint, verifies its hash and evaluates its private copy. An existing lock
causes failure, not automatic removal. Existing campaign weights and results
are preserved. CUDA absence causes failure rather than a silent CPU fallback.

Preview without loading weights or writing results:

```bash
BB_DRY_RUN=1 bash run_bb_dem_osd_benchmark.slurm
```

## Read the result

Outputs use `resdir_JOBID`, without a descriptive suffix:

- `log_worker_0.txt`, `log_worker_1.txt`: progress and errors.
- `worker_0/results.json`, `worker_1/results.json`: correctness, timings and metrics.
- Each worker also stores `metadata.json`, `timings.csv` and its `checkpoint.pt` copy.
- A disagreement stops that worker and saves `mismatch.npz` and `failure.json`.

Require `status=complete`, `cuda_tested=true`, `real_shot_osd_exercised=true`,
zero correction/logical mismatch counts and full syndrome convergence before
interpreting CUDA performance. Toy self-checks also exercise ties, saturated
probabilities, dependent rows and inconsistent syndromes on the selected device.

`backend_seconds` sums per-batch median times across repetitions. It includes
backend sorting, transfers and parity checks, with CUDA synchronization at
timing boundaries; setup and warmup are excluded and setup is reported separately.
`pipeline_seconds_estimate` adds the shared CNN/local-correction/transfer time.
Sampling, scoring and file IO are excluded. These are serial timing estimates,
not whole-job elapsed times. `backend_speedup_vs_reference` above 1 means faster
than the existing reference for these shots. If no real shot calls OSD, the
timing comparison is uninformative despite passing toy checks.

The small sample count is for correctness and timing, **not** a precise LER
measurement or checkpoint selection. Keep production CPU OSD until CUDA has
passed on both codes and shown a useful measured speedup. GPU OSD is still a
discrete operation; moving it to CUDA does not make hierarchical LER differentiable.

Local tests:

```bash
python -m pytest -q tests/test_packed_osd.py tests/test_bb_dem_osd_benchmark.py
```

CPU tests can run without CUDA; the CUDA test is skipped when no CUDA device
is available. Passing CPU tests alone does not validate GPU execution.
