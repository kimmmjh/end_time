# Circuit-level detector/mechanism CNN

Use `--architecture=bb_dem_cnn --noise_model=circuit` for BB72 or BB144.
`bb_tanner_cnn` with circuit noise is an alias. Capacity Tanner CNN checkpoints
and neural BP checkpoints are different architectures and cannot initialize or
resume this model. The dedicated sweep launcher is
[`run_bb_dem_depth46812.slurm`](../run_bb_dem_depth46812.slurm).

## Depth 4/6/8/12 campaign (800 epochs)

```bash
cd ~/end_time
sbatch run_bb_dem_depth46812.slurm
```

Upload the launcher and current model/source files to the server before
submitting. The eight array tasks are:

| Array task | Code | Depth | Noisy rounds |
|---|---|---:|---:|
| 0 / 1 | BB72 / BB144 | 4 | 6 / 12 |
| 2 / 3 | BB72 / BB144 | 6 | 6 / 12 |
| 4 / 5 | BB72 / BB144 | 8 | 6 / 12 |
| 6 / 7 | BB72 / BB144 | 12 | 6 / 12 |

Each task uses four GPUs for p=.001/.002/.004/.006, one p per GPU: **32 models**.
All train from scratch for 800 epochs with width 64, AdamW, cosine LR 3e-4,
weight decay 1e-4, gradient clipping 1, float32, and activation checkpointing.
This uses the existing `standard` circuit noise profile, including idle noise,
with ideal reference and closing cycles. It is not the capacity-noise p sweep.

The initial memory-conscious batch size is **4**. Each epoch has 128 batches
(512 training shots), totaling **409,600 training shots per model**. Validation
uses 1,024 independent circuit shots every five epochs; the selected best is
evaluated on 8,192 fresh circuit shots. These shot budgets are smaller than the
capacity campaign's 6,553,600 training / 65,536 final shots: 800 epochs does not
mean the same amount of data. They are initial settings, not tuned benchmarks.

Each array task has a 24-hour wall-time limit; full GPU runtime/memory at these
settings has not been measured, so 800-epoch completion within that limit is
not guaranteed. Checkpoints are saved every epoch. Resume support is described
below; restarting the launcher itself starts fresh experiments.

Email is configured directly as `ds08152@gmail.com`, `END,FAIL`, for completion
or failure of the array as a whole. The script retains the existing per-job
`resdir_JOB_ID` structure, per-experiment logs/exit codes, source snapshots,
weights, history and final shot bank.

Preview without submitting:

```bash
BB_DRY_RUN=1 bash run_bb_dem_depth46812.slurm
```

Runtime controls (exported environment variables):

- `BB_DEM_BATCH_SIZE=4`: a positive divisor of 512; batch counts adjust to keep
  training, validation and final shot budgets fixed.
- `BB_DEM_EDGE_CHUNK_SIZE=4096`: sparse message chunk size.
- `BB_DEM_OSD_SHOTS=0`: default raw-only scoring. A positive value enables the
  existing **residual OSD-0** on the first min(requested, available) shots of
  **every validation and final evaluation**, using the same paired raw subset.
  This dense CPU solver is costly for large DEMs and is not a full-bank result
  unless the requested subset covers that evaluation. `--bb_cnn_osd` is not
  the control for this architecture.

For example, `sbatch --export=ALL,BB_DEM_BATCH_SIZE=2,BB_DEM_OSD_SHOTS=8
run_bb_dem_depth46812.slurm` reduces batch size and enables an eight-shot OSD
diagnostic at each evaluation. Eight shots are a diagnostic, not an LER benchmark.

Measured graph sizes for the standard profile at p=.002:

| Code | Detectors | Mechanisms | Edges | Edge orbits |
|---|---:|---:|---:|---:|
| BB72, rounds 6 | 504 | 17,928 | 105,408 | 1,410 |
| BB144, rounds 12 | 1,872 | 71,280 | 419,904 | 1,410 |

At width 64 these two graphs share the parameter counts below because the
weights depend on edge orbits, not the number of nodes.

| Depth | Initial D→M | M→D→M blocks | Total directed convolutions | Parameters |
|---|---:|---:|---:|---:|
| 4 | 1 | 3 | 7 | 695,233 |
| 6 | 1 | 5 | 11 | 1,089,729 |
| 8 | 1 | 7 | 15 | 1,484,225 |
| 12 | 1 | 11 | 23 | 2,273,217 |

Each model also has detector and mechanism stems and a two-linear-layer output
head. These depths count DEM graph propagation blocks and are not identical to
the code-capacity model's architecture or computational cost.

Launcher validation: all 32 commands pass the real CLI routing/configuration
checks; all 14 DEM CNN tests pass. A BB144 rounds12/width64/depth12 model also
completed a finite forward/backward check on CPU with batch size 1. This does
not establish GPU batch4 memory use, training convergence, or full-run timing.

## Graph and model

The existing BB memory circuit supplies an undecomposed Stim DEM
(`decompose_errors=False`, no disjoint-error approximation enabled). Every
merged DEM error instruction is a binary mechanism node; each detector target
becomes an incident edge. A mechanism can touch several X and Z detectors at
different times. These joint hyperedges are retained. Detector/observable
signatures must both agree before independent duplicate mechanisms are XOR
merged; `fold_errors` converts original sampler bits into this merged indexing.
Mechanisms are DEM latent variables, not necessarily individual physical gates.

The circuit/DEM generator remains the existing BB implementation: `rounds`
counts noisy rounds and there is one perfect closing detector frame. Noise
profiles `legacy`, `standard`, and `si1000` use the existing rate semantics.
Omitting `--rounds` selects the code distance, 6 or 12. This does not add a
different preparation/readout experiment or support arbitrary external circuits.

```text
detector event, X/Z type, start/end flags, log(1 + degree)
    -> pointwise detector embedding
mechanism prior LLR, log(1 + degree)
    -> pointwise mechanism embedding
    -> detector -> mechanism convolution                  depth 1
    -> [mechanism -> detector -> mechanism residual block] depth - 1 times
    -> pointwise head + prior LLR
    -> [batch, num_mechanisms] log(P(no fault) / P(fault))
```

Each directed convolution is

```text
y_v = W_out [ sum_(u,v) diag(g_orbit(u,v)) W_in h_u / sqrt(degree(v)) ] + b
```

This is the sparse gather/scatter equivalent of a shift-and-channel-mixing
convolution. Kernels are **factorized**, not arbitrary independent W-by-W
matrices per orbit. Each layer and direction has independent parameters.
`edge_orbit` retains the translated detector signature, the edge's role in that
signature, and temporal boundary category. Thus spatial translations share
parameters, as do repeated bulk-time patterns. Temporal boundaries are never
wrapped. No absolute spatial ID or logical-observable label enters the network.
LayerNorm is over channels at a single node, so it does not enlarge the spatial
receptive field. Depth 1 sees adjacent detectors; each further block adds two
graph hops. This is a fixed-depth feature network, without BP or Relay updates.

Gathered messages are processed in `--edge_chunk_size` chunks (CLI default 1024).
Activation checkpointing is on during training; disable it with
`--bb_dem_cnn_no_checkpoint`. Width must be at least 2 for channel LayerNorm.
The full graph's node features and parity loss still consume memory; chunking
does not make memory independent of rounds or batch size. Start with a small
batch and profile before launching large BB144 experiments. Float32 is required
(`--amp_dtype=none`).

## Training and scoring

Training samples the DEM with latent error labels and uses the existing
`CircuitDegeneracyAwareLoss`: detector parity + logical parity + 0.1 mechanism
BCE by default. `--bb_pauli_loss_weight` controls the mechanism auxiliary term
here; `--bb_syndrome_loss_weight` and `--bb_logical_loss_weight` control the
parities. These are factorized soft surrogates, not exact coset likelihoods or
hard parity guarantees. There is no intermediate supervision in this version.

Validation and final evaluation use the independent **circuit** detector
sampler. Hard decisions use `LLR < 0` (ties choose no fault). With DEM matrices
H (detectors) and O (observables), a successful mechanism correction satisfies
both H e_hat = observed detectors and O e_hat = observed observable flips over
GF(2). Metrics separate flagged syndrome failures and unflagged logical failures.
`logical_error_rate` includes both; `observable_mismatch_rate` is also reported
separately to avoid equating an inconsistent correction with a valid decoder.

The earliest checkpoint with highest raw validation correction accuracy is
selected. Final test shots never select a model. Optional OSD results never
select a model either. These tests establish execution and correctness, not a
trained performance advantage over existing decoders.

## Optional direct OSD-0

Default `--bb_osd_eval_shots=0` evaluates raw CNN only. A positive value scores
an additional direct OSD-0 branch on the first min(requested, available) shots
of each evaluation. Only inconsistent corrections in this subset are repaired.
Syndrome-valid raw corrections are preserved, including wrong logical classes.
The postprocessor is the existing `DirectOSD0`, which changes least-reliable
basis bits around the hard decision and never calls BP. It is the explicit
`direct_hard_centered_osd0_v1` convention, not published zero-free-variable
syndrome OSD or an ldpc OSD-CS decoder.

`--bb_osd_method` and `--bb_osd_order` configure only the neural-BP experiment;
they do not change this fixed direct OSD-0 policy. OSD is a dense CPU reference
and can be expensive on large DEMs. Report the scored subset size: full raw
accuracy must not be subtracted from subset OSD accuracy. `paired_raw_accuracy`,
`osd_minus_raw_gain`, `rescued`, and `harmed` use exactly the same subset.
The final bank records `osd_indices` to make this pairing independently auditable.

## Run

Small smoke run, including two paired OSD shots per evaluation:

```bash
python main.py --code=bb72 --architecture=bb_dem_cnn \
  --noise_model=circuit --bb_circuit_noise_model=standard --p=.003 \
  --rounds=1 --bb_cnn_width=8 --bb_cnn_depth=2 \
  --epochs=1 --batch_size=2 --batches=2 --eval_batches=1 \
  --eval_every=1 --final_eval_batches=2 --bb_osd_eval_shots=2 \
  --amp_dtype=none --seed=42 --save_model
```

An initial training configuration to tune (not benchmarked hyperparameters):

```bash
python main.py --code=bb72 --architecture=bb_dem_cnn \
  --noise_model=circuit --bb_circuit_noise_model=standard --p=.003 \
  --rounds=6 --bb_cnn_width=32 --bb_cnn_depth=3 \
  --epochs=100 --batch_size=8 --batches=128 --eval_batches=32 \
  --eval_every=5 --final_eval_batches=256 --lr=.0003 \
  --amp_dtype=none --seed=42 --save_model
```

Outputs under `outputs/<date>/<timestamp>_circuit_..._bb_dem_cnn_.../`:

- `history.json`: graph/noise/model config, phases, losses, validation, selected
  epoch, fresh final metrics and cumulative forward/OSD timers (not single-shot latency).
- `final_shots.npz`: detectors, true observables, raw mechanism corrections,
  predicted observables, success and convergence; optional paired OSD subset.
- `training_log.txt`: configuration and progress.
- `model.pt`, `best_model.pt` with `--save_model`: latest and selected model,
  optimizer, history, selection state, and Torch RNG states.

Resume with the same code/noise/rounds/width/depth/loss/seed options and add
`--load_model=/absolute/path/model.pt --epochs=ADDITIONAL_EPOCHS`.
Graph fingerprints include DEM priors, topology and orbit ordering. Mismatches
are rejected. Adam moments persist and each phase starts a new cosine schedule.
The selected best remains eligible even if subsequent validation ties it.
Stim has no serializable sampler state: resumed phases create independent
train/evaluation streams from (base seed, completed epochs, circuit schema),
so resume does not replay the initial stream or reproduce an uninterrupted run
bit for bit. `best_model.pt` resumes from its own selected epoch. The capacity
`--bb_cnn_compare_resume` feature is not supported here.
