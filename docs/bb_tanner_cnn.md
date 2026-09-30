# Joint Tanner CNN: raw and direct OSD-0

The 20-run depth/seed campaign has completed. See the
[September 27 result analysis](../results/analysis/bb_update_2026_09_27.md)
and [consolidated comparison](../results/plots/bb/code_capacity/tanner_cnn/overview.png).
Depth 2 consistently improves on depth 1, but raw syndrome consistency and the
gap to previous Neural BP4 remain unresolved. These are code-capacity results.

`--architecture=bb_tanner_cnn` trains a feed-forward CNN on the BB joint
`X checks — data qubits — Z checks` graph. This first implementation supports
**code-capacity noise with one perfect syndrome**. Circuit and phenomenological
noise are rejected explicitly. Its LER must be compared with code-capacity
baselines, not with the ongoing circuit-level BP sweep.

## Convolutions and receptive field

BB72 uses a 6×6 periodic grid and BB144 uses 12×6. Checks have two feature maps
(X/Z), and qubits have two feature maps (left/right block). For an edge with
`qubit_position = check_position + displacement`, each learned kernel tap
applies a 1×1 channel-mixing convolution to a cyclically shifted source map.
There are 12 edge roles, retaining check type, qubit block and polynomial term.
Summing these taps is a sparse periodic spatial convolution whose support is
exactly the Tanner connectivity. It is not an ordinary dense 3×3 image kernel.

The default width is 64 and depth is 2:

```text
X/Z syndrome maps + check-type indicators → 1×1 embedding
four log channel priors + qubit-block indicators → 1×1 embedding
    ↓
check → qubit Tanner convolution + SiLU
    ↓                          depth=1 ends spatial processing here
qubit → check Tanner convolution + pointwise check update + SiLU
check → qubit Tanner convolution + pointwise qubit update + SiLU
    ↓                          each extra depth adds one such Q→C→Q block
1×1 Conv(width,width) → SiLU → 1×1 Conv(width,4)
    ↓
add log channel prior → Pauli logits [batch,n,4], order I,X,Y,Z
```

The prior offset is a classifier initialization aid; no BP messages, min-sum,
relaxation, or Relay mechanism is used. Parameters are shared over cyclic
translations and distinguish edge roles. Weights are distinct across spatial
blocks. Node features are exchanged as in a typed graph CNN, so this is also a
graph-convolution architecture; calling it a CNN does not mean it is unrelated
to message passing. It has fixed feed-forward depth, not adaptive BP iterations.

Depth 1 sees only the three X and three Z checks touching each qubit. Depth 2
sees checks up to three Tanner edges away. Unlike stacking pointwise MLP layers
on the same six-bit patch, increasing depth expands spatial context. No pooling
or spatial normalization can silently introduce a global receptive field.
`encode()` and the Pauli output head are separate for future temporal features
and a DEM-mechanism head; circuit-level decoding is not implemented yet.

## Shared output and exact scoring

Both evaluation branches use identical logits and identical circuit-free shots.
From softmax probabilities, `qx=P(X)+P(Y)` and `qz=P(Z)+P(Y)`. The raw decision is
`x=(qx>0.5), z=(qz>0.5)`, with ties choosing zero. This is a component-marginal
decision, not four-state Pauli argmax. Success requires the correction to match
the measured syndrome and leave no logical residual. Stabilizer-equivalent
corrections count as successful even when they differ from the sampled error.

**Raw CNN** returns that hard correction. **CNN+OSD-0** preserves every
syndrome-valid sector and repairs only invalid sectors. The X component uses
`Hz` and the Z-check syndrome; the Z component uses `Hx` and the X-check syndrome.
The OSD path uses exactly the same initial hard correction as the raw path.

The new `DirectOSD0` backend does not call BP or `ldpc.BpOsdDecoder`. For each
failed sector it solves the residual syndrome `s XOR H@hard`, picking linearly
independent columns in ascending `abs(log((1-q)/q))` order and changing only the
selected basis bits. Nonbasis bits retain the hard decision. Dependent check
rows are supported; an inconsistent input syndrome raises an error. Ties use
stable column order, so OSD itself need not preserve translation equivariance.

This is **hard-decision-centred, order-zero reliability reprocessing**. It is
recorded as `direct_hard_centered_osd0_v1`; it is not claimed to reproduce ldpc's
zero-free-variable syndrome OSD bit for bit. There is no higher-order search.
X/Z probabilities are predicted jointly but the OSD sectors are separate, so
post-processing does not fully exploit the four-state joint probability.
The NumPy backend prioritizes a transparent comparison, not optimized latency.

## Training and outputs

Training reuses the degeneracy-aware capacity objective:
`syndrome_loss + logical_loss + 0.1 * Pauli_cross_entropy` by default. Existing
`--bb_*_loss_weight` options adjust it. There is no intermediate/deep supervision
or differentiation through OSD. AdamW uses a cosine LR schedule, float32, and
gradient clipping. The encoder starts near the supplied channel prior.

One checkpoint is selected by **raw CNN validation logical accuracy** (earliest
on ties). Both branches then evaluate that same checkpoint on fresh held-out
shots. Test results never select checkpoints. Training and evaluation samplers
have separate random streams. A resumed invocation restores model, optimizer,
samplers, Torch RNG, best selection, and history; it starts a new cosine schedule.
Resume requires matching architecture, noise, graph and scoring settings.

Each run creates `outputs/<date>/<timestamp>_capacity_..._bb_tanner_cnn_.../`:

- `history.json`: config, train/validation records, selected epoch and fresh
  `final` evaluation. Each evaluation has `raw`, `osd`, paired OSD-minus-raw gain,
  rescued/harmed counts, OSD call fraction, flagged/unflagged rates and timings.
- `final_shots.npz`: selected-best test syndromes, true Pauli labels, both hard
  corrections and per-shot success, plus config metadata.
- `model.pt` / `best_model.pt` with `--save_model`: latest and selected states,
  including optimizer and sampler states from the corresponding epochs.
- `training_log.txt`: configuration and readable raw/OSD results.

`nn_batch_seconds` measures forward/decision time, with CUDA synchronization.
`osd_with_transfer_seconds` includes transfers and the CPU OSD stage. They exclude
sampling/scoring and are batch timings, not single-shot latency benchmarks.
The existing `--bb_osd_*` circuit wrapper settings do not configure this backend;
CNN evaluation always runs the paired direct OSD-0 branch on all evaluation shots.

## Run

Small CPU/GPU auto-selected smoke run:

```bash
python main.py --code=bb72 --architecture=bb_tanner_cnn \
  --noise_model=capacity --p=0.06 --bb_channel=depolarizing \
  --bb_cnn_width=8 --bb_cnn_depth=2 \
  --epochs=1 --batch_size=4 --batches=2 --eval_batches=1 \
  --eval_every=1 --final_eval_batches=2 --amp_dtype=none --seed=24 --save_model
```

An initial BB72 training configuration (a starting point, not tuned settings):

```bash
python main.py --code=bb72 --architecture=bb_tanner_cnn \
  --noise_model=capacity --p=0.06 --bb_channel=depolarizing \
  --bb_cnn_width=64 --bb_cnn_depth=2 \
  --epochs=100 --batch_size=64 --batches=128 --eval_batches=16 \
  --eval_every=5 --final_eval_batches=256 \
  --lr=0.0003 --amp_dtype=none --seed=24 --save_model
```

Use `--code=bb144` for the other graph, `--bb_cnn_depth=1` for the single-patch
control, or `--load_model=/path/to/model.pt` with matching options to continue.
An independent-X/Z capacity channel is also supported through the existing
`--bb_channel=independent_xz --x_error_rate=... --z_error_rate=...` options.

## Slurm campaign: depths 4, 5, 3 at 400 epochs

The numbered scripts start with a new depth sweep, in requested order **4, 5, 3**.
Each script trains four fresh models, one per p value, on one GPU each.

| Script | Code | Depth | Epochs | p values |
| --- | --- | ---: | ---: | --- |
| `run_bb_0.slurm` | BB72 | 4 | 400 | .02, .04, .06, .08 |
| `run_bb_1.slurm` | BB144 | 4 | 400 | .02, .04, .06, .08 |
| `run_bb_2.slurm` | BB72 | 5 | 400 | .02, .04, .06, .08 |
| `run_bb_3.slurm` | BB144 | 5 | 400 | .02, .04, .06, .08 |
| `run_bb_4.slurm` | BB72 | 3 | 400 | .02, .04, .06, .08 |
| `run_bb_5.slurm` | BB144 | 3 | 400 | .02, .04, .06, .08 |

All use width 64, depolarizing code-capacity noise and the original code/p/seed
matrix: BB72 seeds 7201020/7201040/7201060/7201080 and BB144 seeds
14401020/14401040/14401060/14401080. Each epoch draws 128 batches of 64 shots,
for **3,276,800 training shots per model**, four times the initial 100-epoch
budget. AdamW starts at LR 3e-4 and uses a single 400-epoch cosine decay;
weight decay 1e-4, gradient clip 1 and loss weights 1/1/0.1 stay unchanged.

Validation uses 4,096 shots every five epochs. Raw validation LER selects the
checkpoint; final evaluation uses 65,536 fresh shots for paired raw CNN and
CNN+direct OSD-0. Latest and best checkpoints are saved. Depth-2 jobs below
also reach 400 total epochs, but use a 100+300 schedule with an LR restart;
this is a matched training-shot budget, not an identical LR schedule.

Preview or submit the new 24-model sweep:

```bash
BB_DRY_RUN=1 bash run_bb_0.slurm
for i in 0 1 2 3 4 5; do sbatch "run_bb_${i}.slurm"; done
```

The numbering groups depths in submission order; independent Slurm jobs may
start in a different order. Each allocation retains the 24-hour limit and
four tasks with one GPU and 16 CPUs each. Completion within that limit has not been timed
for the deeper models. Checkpoints are saved every epoch; an interrupted
fresh run can be continued with matching model options and
`--load_model=/absolute/path/to/model.pt --epochs=REMAINING_EPOCHS` through
`main.py` (resumption starts a new cosine schedule).

The retained depth-1 controls are now `run_bb_9.slurm` (BB72) and
`run_bb_10.slurm` (BB144), still at their original 100-epoch budget.

## Slurm campaign: depth-2 continuation to 400 epochs

Jobs **6, 7 and 8** now continue the 12 existing depth-2 CNN models from their
latest 100-epoch checkpoints to **400 total epochs**. Jobs 9/10 retain the initial
100-epoch depth-1 control commands; they are not part of this continuation.
The circuit-level library BP/OSD baseline scripts are preserved as
`run_bb_baseline_0.slurm`–`run_bb_baseline_4.slurm`.

| Script | Code | p values | Seeds in experiment order | Source job |
| --- | --- | --- | --- | --- |
| `run_bb_6.slurm` | BB72 | .02, .04, .06, .08 | 7201020, 7201040, 7201060, 7201080 | 58793761 |
| `run_bb_7.slurm` | BB144 | .02, .04, .06, .08 | 14401020, 14401040, 14401060, 14401080 | 58793763 |
| `run_bb_8.slurm` | BB72, BB72, BB144, BB144 | .06 for all | 7202060, 7203060, 14402060, 14403060 | 58793766 |

All use **depolarizing code capacity**: `p` is the total probability of a
non-identity data-qubit error, `P(I,X,Y,Z)=(1-p,p/3,p/3,p/3)`, with one perfect
syndrome. The model remains width 64, depth 2, with the same loss, optimizer,
OSD-0 policy and raw-validation-LER checkpoint selection.

- Continue `model.pt` (latest epoch 100), retaining optimizer moments, sampler
  states, RNG states and the previously selected best checkpoint. Do not resume
  `best_model.pt`, which can be from an earlier epoch.
- Add **300 epochs**, 128 batches of 64 shots per epoch: 2,457,600 additional
  training shots, **3,276,800 total** (4× the initial CNN budget). This is still
  one third of the earlier Neural BP4 budget of 9,830,400 training shots.
- The initial schedule reached LR=0. Resume with **3e-4 and a new 300-epoch
  cosine decay**, retaining AdamW moments. This is additional training with an
  LR restart, not an uninterrupted 400-epoch cosine run.
- Keep float32, weight decay 1e-4, gradient clip 1, and loss weights
  syndrome=1, logical=1, Pauli=0.1.
- Validate every 5 epochs on 4,096 fresh shots. The old best remains eligible;
  the final selected epoch need not be epoch 400.
- Evaluate the selected model on **65,536 fresh final shots**, then evaluate
  the frozen pre-resume best on **the exact same new shots**. Both report raw
  CNN and CNN+direct OSD-0. No final test results select checkpoints.

The original 100-epoch results remain intact. Each new result directory stores:

- `history.json`: all 400 training epochs, both training phases, final selected
  model metrics and `resume_comparison`. The latter contains old-model metrics,
  source checkpoint path/SHA-256, selected epochs, and paired raw/OSD LER
  reductions, standard errors and rescued/harmed counts. Positive
  `ler_reduction` means improvement after additional training.
- `final_shots.npz`: new selected model's final test bank and corrections.
- `reference_final_shots.npz`: frozen old best on that same bank.
- Latest `model.pt` and selected `best_model.pt`, plus the training log.

Use the new paired comparison to judge improvement; the published initial
100-epoch final scores used a different test bank. A gain is possible but not
assumed. Low-p precision is still limited by the number of observed failures.
These remain capacity results, separate from the circuit-level baseline plots.
When importing new jobs, keep them separate from the original 20-run archive;
`summarize_bb_september27.py` audits that fixed 100-epoch campaign.

### Source checkpoint lookup and submission

The Python launcher `scripts/continue_bb_tanner_cnn.py` resolves the exact
code/p/seed match inside the specified source job. It checks, in order-independent
fashion, these locations under `BB_REPO_ROOT` (normally `$HOME/end_time`):

- `resdir_<source_job>/outputs/.../model.pt` (original server results).
- `results/bb/code_capacity/depolarizing/tanner_cnn/<bb72|bb144|replicates>/resdir_<source_job>/outputs/.../model.pt`
  (organized local archive).

Missing, ambiguous, mismatched or incomplete checkpoints stop the job; there
is no fallback to fresh training. If results were moved elsewhere, set
`BB_CNN_SOURCE_ROOT` to the directory containing the source `resdir_*` folders,
or to one source result directory. The three source jobs above are exceptions
to `.gitignore` only at the repository root.
Once committed and pushed, they can be transferred to the server with
`git pull`. Archived copies and other experiment results remain ignored.

Preview the four job commands without importing Python or invoking Slurm:

```bash
BB_DRY_RUN=1 bash run_bb_6.slurm
```

Validate and expand one real checkpoint without training (use the training
Python environment):

```bash
python scripts/continue_bb_tanner_cnn.py \
  --source-job=58793761 --code=bb72 --p=.06 --seed=7201060 --dry-run
```

After updating the Python sources and Slurm files on Perlmutter, submit only
the continuation jobs:

```bash
for i in 6 7 8; do sbatch "run_bb_${i}.slurm"; done
```

Each script embeds the complete four-task launcher, with no repository `.sh`
helpers. It uses account `m5328_g`, 24 hours, one node, four concurrent `srun`
tasks with one GPU and 16 CPUs each, `python/3.10`, and `$PSCRATCH/envs/nde`.
All four source checkpoints are validated before any training steps start.
New outputs go under `resdir_<new_job>/outputs/<date>/<timestamp>_capacity_.../`;
allocation logs also retain exact commands, exit codes, completion markers,
source snapshots and the submitted script. No source checkpoint is overwritten.

Python sources are read from the checkout at startup. Previously queued jobs
retain Slurm's saved batch scripts; submit the updated scripts for this campaign.
