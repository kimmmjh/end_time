# Joint Tanner CNN: raw and direct OSD-0

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

## Slurm campaign: jobs 5–9

`run_bb_5.slurm` through `run_bb_9.slurm` now launch 20 Tanner CNN experiments,
replacing the earlier Relay ablation scripts. Jobs 0–4 still launch the ordinary
circuit BP campaign. The CNN campaign uses **depolarizing code capacity**:
`p` is the total probability of a non-identity data-qubit error,
`P(I,X,Y,Z)=(1-p,p/3,p/3,p/3)`, with one perfect syndrome. These LERs must not
replace or be overlaid as directly comparable points on the circuit-level BP
curves. This initial sweep compares CNN depth and the effect of OSD; it does
not yet establish superiority over a matched capacity BP+OSD baseline.

| Script | Code | Depth | p values | Seeds in experiment order |
| --- | --- | --- | --- | --- |
| `run_bb_5.slurm` | BB72 | 2 | .02, .04, .06, .08 | 7201020, 7201040, 7201060, 7201080 |
| `run_bb_6.slurm` | BB144 | 2 | .02, .04, .06, .08 | 14401020, 14401040, 14401060, 14401080 |
| `run_bb_7.slurm` | BB72 | 1 | .02, .04, .06, .08 | same as job 5 |
| `run_bb_8.slurm` | BB144 | 1 | .02, .04, .06, .08 | same as job 6 |
| `run_bb_9.slurm` | BB72, BB72, BB144, BB144 | 2 | .06 for all | 7202060, 7203060, 14402060, 14403060 |

Depth 1 is the six-neighbor local-filter control; depth 2 expands the receptive
field with another qubit-to-check-to-qubit block. The width is fixed at 64, so
depth 2 also has more parameters: this is a depth comparison, not a
parameter-count-matched comparison. Jobs 5/7 and 6/8 share the sampler seeds,
batch sizes and evaluation schedule, giving the same train/validation/test
shots for each matching code/p pair. Each depth still selects its own
checkpoint using its raw validation LER. Job 9 gives depth 2 three independent
training/test seeds per code at p=.06 when combined with jobs 5/6; the other
points have one seed, and depth 1 has no extra seed replication yet.

All 20 runs use the following settings, embedded directly in each of the five
`.slurm` files. They also include the complete four-task launcher and do not
load any repository `.sh` helpers:

- 100 epochs, 128 batches of 64 shots per epoch: 819,200 training shots.
- Validation every 5 epochs, 64 batches: 4,096 fresh shots per validation.
- Final selected-checkpoint evaluation: 1,024 batches, **65,536 fresh shots**.
- AdamW, initial LR 3e-4 with cosine decay, weight decay 1e-4, gradient clip 1,
  float32; loss weights syndrome=1, logical=1, Pauli=0.1.
- Every evaluation reports both CNN and CNN+direct OSD-0 on the same shots.
  There are no separate OSD training jobs or extra BP iterations.

The final shot count improves resolution but is not a guarantee of precise
very-low-LER estimates; zero observed failures do not establish zero true LER.
The p grid and training budget are initial settings, not tuned thresholds.

Preview expanded commands locally without writing results or invoking Slurm:

```bash
BB_DRY_RUN=1 bash run_bb_5.slurm
```

Submit the base models first, then controls and seed repeats as desired:

```bash
sbatch run_bb_5.slurm
sbatch run_bb_6.slurm
# Depth controls and seed repeats:
for i in 7 8 9; do sbatch "run_bb_${i}.slurm"; done
# Or submit all CNN jobs together:
# for i in 5 6 7 8 9; do sbatch "run_bb_${i}.slurm"; done
```

Each script retains account `m5328_g`, 24 hours, one node, and four concurrent
`srun` tasks with one GPU and 16 CPUs each. It loads `python/3.10`, activates
`$PSCRATCH/envs/nde`, and uses `$HOME/end_time` (override with `BB_REPO_ROOT`).
Update the repository's model, trainer and `.slurm` scripts on the server
together. Python sources are still read from that checkout at startup; they are
not frozen when the job is submitted. GPU runtime for the full campaign has not
been measured.

If a job was already queued with the previous script, pulling this version does
not replace Slurm's saved batch script. Cancel the intended pending job and
submit the updated `.slurm` to remove its old shell-helper dependencies.

Results are under `resdir_<SLURM_JOB_ID>/outputs/<date>/<timestamp>_capacity_.../`
with the files described above. The allocation directory additionally contains
`experiments.tsv`, exact commands, per-experiment logs/exit codes, completion or
failure markers, and decoder-source snapshots. `submitted_script.slurm`
includes the CNN defaults and launcher; separate `.sh` snapshots are not used.
Each label and model directory identifies the code, p, depth and seed.
