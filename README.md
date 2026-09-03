# TheEND time: equivariant neural QEC decoders

This repository contains translation-equivariant toric-code decoders and an
orbit-equivariant neural belief-propagation decoder for bivariate-bicycle (BB)
codes. The toric path supports code-capacity, phenomenological, and Stim-based
circuit noise. The BB path supports code capacity, which decodes four-state
Pauli beliefs on the code Tanner graph, and Stim-based circuit noise, which
decodes binary beliefs on the detector error model. BB phenomenological noise
is not implemented.

## Install

```bash
pip install -r requirements.txt
```

Circuit-level generation requires Stim 1.15 or newer.

## BB code-capacity neural BP

The BB path currently provides the published `[[72,12,6]]` and
`[[144,12,12]]` constructions. It has one perfect syndrome measurement, so
there is no physical-time axis, CNN, ConvGRU, pooling, or MWPM:

```text
independent Pauli errors on n data qubits
        |
one perfect syndrome [Hx z | Hz x]       [B, 2*n/2]
        |
exact four-state Tanner-graph BP4 update
        + orbit-shared neural residual and relaxation
        |  (12 unrolled algorithmic iterations by default)
        v
per-qubit log P(I,X,Y,Z | syndrome)       [B, n, 4]
        |
hard Pauli correction; score residual modulo stabilizers
```

For BB72 the data shapes are syndrome `[B,72]`, Pauli target `[B,72]`, and
output `[B,72,4]`. For BB144 they are `[B,144]`, `[B,144]`, and
`[B,144,4]`. The output is not a `2^(2k)` logical-class prediction.

Run one depolarizing point with:

```bash
python main.py --code=bb72 --architecture=bb_neural_bp \
  --noise_model=capacity --loss_fn=bb_coset --bb_channel=depolarizing \
  --measurement_error_rate=0 --p=0.08 \
  --bp_iterations=12 --bp_residual_hidden_dim=64 \
  --bp_parameter_sharing=orbit --epochs=300 --batch_size=64 \
  --batches=512 --eval_batches=256 --eval_every=5 \
  --final_eval_batches=2048 --lr=0.0003 --amp_dtype=none --save_model
```

`p` means total non-identity probability for the depolarizing channel:
`P(I,X,Y,Z)=(1-p,p/3,p/3,p/3)`. To study independent X and Z components:

```bash
python main.py --code=bb72 --architecture=bb_neural_bp \
  --noise_model=capacity --loss_fn=bb_coset --bb_channel=independent_xz \
  --x_error_rate=0.04 --z_error_rate=0.04 --p=0.04 --amp_dtype=none
```

For independent X/Z noise, the total non-identity probability is
`px+pz-px*pz`, so equal numeric `p` values do not describe the same physical
channel as depolarizing noise.

### Where the equivariance is implemented

The BB checks are cyclic polynomial matrices. A simultaneous cyclic shift of
check and qubit cell indices maps every Tanner edge into an edge of the same
type. The implementation partitions the graph into 12 edge orbits:

```text
check type (X or Z) x qubit block (left or right) x polynomial term (3)
```

Every edge in one orbit uses exactly the same residual MLP and learned
relaxation coefficient. Therefore, shifting the syndrome cyclically shifts the
qubit posterior in the same way; it cannot introduce an absolute-site
dependence. `--bp_parameter_sharing=global` is a generic shared-neural-BP
ablation with one shared update for each X/Z check sector, while `edge` gives
every edge separate parameters and intentionally breaks the symmetry.

This is more than adding an equivariant pooling layer to the old model. The
research idea being tested is equivariant parameter tying, but the decoder and
targets also change from a toric-grid logical classifier to four-state message
passing on a BB Tanner graph. Repeated BP steps are algorithm unrolling, not
measurement-round recurrence. Vanilla BP itself is graph-equivariant, so the
training log always evaluates `neural=False` vanilla BP4 on the exact same
shots; `orbit`, `global`, and `edge` runs should be reported as ablations.

The loss is a degeneracy-aware factorized surrogate:

```text
L = L_syndrome + L_logical-parity-surrogate + 0.1 L_Pauli-auxiliary
```

The first term asks the correction to reproduce the measured syndrome. The
second asks the true-error/correction residual to have trivial logical parity.
Consequently, corrections differing from the sampled error by a stabilizer are
valid. Exact per-qubit Pauli cross entropy is only a small optimization aid.
At soft-probability level this is not an exact coset negative log-likelihood
and its value can depend on the chosen logical basis; the final hard success
test itself is exact and basis-independent.
Reported `Accuracy` is block logical success, not qubit accuracy. Logs also
separate syndrome-nonconverged (flagged) failures from syndrome-converged but
logical (unflagged) failures.

The current ten Perlmutter jobs run the corrected **circuit-level no-OSD** BB
neural-BP campaign described below. Every job requests four GPUs and launches
four independent experiments concurrently.

| Script | Four concurrent experiments |
| --- | --- |
| `run_bb_0.slurm` | BB72 threshold sweep, p=0.001/0.002/0.003/0.004 |
| `run_bb_1.slurm` | BB72 threshold sweep, p=0.005/0.006/0.008/0.010 |
| `run_bb_2.slurm` | BB144 on the same low-p grid as job 0 |
| `run_bb_3.slurm` | BB144 on the same high-p grid as job 1 |
| `run_bb_4.slurm` | two extra training seeds per code at p=0.004 |
| `run_bb_5.slurm` | global-sharing controls at p=0.003/0.005, both codes |
| `run_bb_6.slurm` | T=6 and T=24 controls at p=0.004, both codes |
| `run_bb_7.slurm` | residual-only and learned-relaxation-only controls |
| `run_bb_8.slurm` | no mechanism auxiliary BCE and no deep supervision |
| `run_bb_9.slurm` | idle-noise and doubled-readout-noise controls at p=0.003 |

Submit selected jobs, or submit the full suite with:

```bash
for i in {0..9}; do sbatch "run_bb_${i}.slurm"; done
```

The first phase is 100 epochs with 128 online-data batches per epoch and can be
continued only from a compatible no-OSD `model.pt`. BB72 uses six noisy cycles
and batch size 16; BB144 uses twelve noisy cycles and batch size 8. The default
model is T=12, hidden-width 32, orbit-embedding width 8, and normalized min-sum
scale 0.625. `--bb_osd_eval_shots=0` prevents OSD construction and execution.
Validation runs every ten epochs, checkpoints are selected by raw Neural-BP2
paired gain against raw vanilla BP2, and the final raw evaluation uses 4,096
Stim shots.

The threshold sweeps use `q=p` and zero idle error. Job 9 separately tests
`idle=p` and `q=2p`. `global` sharing remains translation equivariant but has
far fewer trainable group embeddings than `orbit`, so job 5 is a useful
parameter-sharing control rather than a parameter-count-matched comparison.

All jobs use account `m5328_g`, `$PSCRATCH/envs/nde`, and `$HOME/end_time`.
Each allocation launches four `srun --exclusive` steps with one GPU and 16 CPU
cores each. Results are stored under
`$HOME/end_time/resdir_<SLURM_JOB_ID>` with `log_exp_0.txt`, ...,
`log_exp_3.txt`, per-experiment exit codes, and a completed/failed marker.
The timestamped model directories are under each result directory's
`outputs/YYYY-MM-DD/`. There is no `exp_<index>` layer in the Slurm layout.
The shared launch logic and canonical settings are in
`scripts/run_bb_slurm_batch.sh`.

For the same no-OSD comparison outside Slurm, use the direct-GPU runner:

```bash
# Split the BB72 sweep across two physical GPUs.
bash run_bb_circuit_no_osd.sh 0 bb72 low
bash run_bb_circuit_no_osd.sh 1 bb72 high

# The same interface also supports BB144.
bash run_bb_circuit_no_osd.sh 0 bb144 low
bash run_bb_circuit_no_osd.sh 1 bb144 high
```

`low` uses base `p=0.001,0.002,0.003,0.004`; `high` uses
`p=0.005,0.006,0.008,0.010`; `all` runs all eight points sequentially. The
optional fourth argument selects `legacy`, `standard`, or `si1000` (default:
`legacy`). The
runner keeps the circuit, model, optimizer, and seeds matched to the OSD
campaign but sets `--bb_osd_eval_shots=0`. Consequently no OSD object or CPU
post-processing is used during validation/final evaluation, and
`best_model.pt` is selected by raw Neural BP2 paired gain against vanilla BP2.
Each invocation creates `resdir_<script-pid>/exp_<index>` and stops after the
first failed experiment.

The completed BB72 and BB144 depolarizing sweeps are summarized in
[`results/analysis/bb_neural_bp_depolarizing_orbit.md`](results/analysis/bb_neural_bp_depolarizing_orbit.md),
with a machine-readable CSV and a Neural BP versus vanilla BP4 comparison
plot in `results/analysis/` and `results/plots/`.

Before submitting all ten jobs, time one BB144 T=24 circuit batch on a GPU:

```bash
python main.py --code=bb144 --architecture=bb_neural_bp \
  --noise_model=circuit --rounds=12 --p=0.004 --measurement_error_rate=0.004 \
  --loss_fn=bb_coset --bp_iterations=24 --bp_residual_hidden_dim=32 \
  --bp_orbit_embedding_dim=8 --bp_parameter_sharing=orbit \
  --epochs=1 --batches=2 --batch_size=8 \
  --eval_batches=1 --eval_every=1 --final_eval_batches=1 \
  --bb_osd_eval_shots=0 --lr=0.0003 --amp_dtype=none --seed=14401004
```

Each neural model starts exactly as its vanilla BP stage because its final
residual layer is zero initialized and relaxation starts at one: BP4 for code
capacity and normalized min-sum BP2 for circuit noise. `model.pt` is the latest
resumable checkpoint. `best_model.pt` uses the configured paired selection
metric: Neural+OSD gain when OSD evaluation is enabled, or raw Neural BP gain
when it is disabled. Generator RNG state, optimizer state, and plot history
are saved, and an incompatible BB graph/model checkpoint is rejected.

## BB circuit-level neural BP

The code-capacity path above assumes one perfect syndrome.  `--noise_model=circuit`
instead decodes a full Stim memory experiment, and that changes the decoder
rather than merely adding a time axis:

| | `--noise_model=capacity` | `--noise_model=circuit` |
| --- | --- | --- |
| Variable node | data qubit, four-state `I,X,Y,Z` | DEM fault mechanism, binary |
| Check node | stabilizer row of `Hx`/`Hz` | detector |
| Graph | code Tanner graph | Stim detector error model |
| Base algorithm | exact sum-product BP4 | normalised min-sum BP2 |

The quaternary structure disappears because Stim has already factorised each
circuit fault into independent mechanisms, so `EquivariantNeuralBP4` is not
reused; the circuit decoder is `models/_equivariant_neural_bp2.py`.

```bash
python main.py --code=bb144 --architecture=bb_neural_bp --noise_model=circuit \
  --bb_circuit_noise_model=legacy \
  --p=0.001 --measurement_error_rate=0.001 --bb_idle_error_rate=0 --rounds=12 \
  --bp_iterations=12 --bp_residual_hidden_dim=32 --bp_orbit_embedding_dim=8 \
  --bp_parameter_sharing=orbit --loss_fn=bb_coset \
  --epochs=300 --batch_size=8 --batches=512 --eval_batches=32 --eval_every=5 \
  --final_eval_batches=128 --bb_osd_eval_shots=1024 \
  --lr=0.0003 --amp_dtype=none --save_model
```

`--p` is the base physical error rate, and `--rounds` is the number of **noisy**
extraction cycles and defaults to the code distance. The circuit adds a perfect
reference cycle before them and a separate perfect closing cycle after them, so
a run with `--rounds=R` returns `R+1` detector frames and `(R+1)*n` detector bits
for these BB constructions. The closing frame is required to expose faults late
in the last noisy cycle.

### Selectable BB circuit noise profiles

`--bb_circuit_noise_model` selects one of three profiles:

| Profile | Reset | H / 1Q | CNOT | Measurement | Idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `legacy` | `X_ERROR(p)` | `DEP1(p)` | `DEP2(p)` | configurable `q` | configurable data-only CNOT-layer idle, default 0 |
| `standard` | basis flip `p` | ideal | `DEP2(p)` | basis flip `p` | `DEP1(p)` on every inactive qubit every tick |
| `si1000` | `X_ERROR(2p)` | `DEP1(p/10)` | `DEP2(p)` | `X_ERROR(5p)` | gate idle `DEP1(p/10)` every tick plus resonator idle `DEP1(2p)` on M/R ticks |

`standard` and `si1000` implement Tables II and III of
[arXiv:2607.05897](https://arxiv.org/abs/2607.05897) on this repository's
periodic-BB `R-H-CX-H-M` schedule. The two SI1000 idle channels are emitted as
independent Stim instructions so they stack as specified. Table III also gives
native SWAP an error rate `1.5p`; that rate is recorded in metadata, but this
non-routed BB circuit contains no SWAP gates. Reproducing the paper's tile-code
thresholds additionally requires its open-boundary code families and routed or
unrouted schedules; selecting the channel alone does not reproduce those
circuits.

The paper profiles have only one free noise parameter, so their other rates are
derived from `--p`. Incompatible explicit `--measurement_error_rate` or
`--bb_idle_error_rate` values are rejected. Use `legacy` when intentionally
varying those rates independently.

```bash
# Table II standard model
python main.py --code=bb72 --architecture=bb_neural_bp --noise_model=circuit \
  --bb_circuit_noise_model=standard --p=0.003 --rounds=6 \
  --loss_fn=bb_coset --epochs=100 --batch_size=16 --batches=128 \
  --eval_batches=64 --eval_every=10 --final_eval_batches=256 \
  --bb_osd_eval_shots=0 --amp_dtype=none --save_model

# Table III modified SI1000
python main.py --code=bb72 --architecture=bb_neural_bp --noise_model=circuit \
  --bb_circuit_noise_model=si1000 --p=0.003 --rounds=6 \
  --loss_fn=bb_coset --epochs=100 --batch_size=16 --batches=128 \
  --eval_batches=64 --eval_every=10 --final_eval_batches=256 \
  --bb_osd_eval_shots=0 --amp_dtype=none --save_model
```

The sequential GPU runner accepts the same selection as its fourth argument:

```bash
bash run_bb_circuit_no_osd.sh 0 bb72 low standard
bash run_bb_circuit_no_osd.sh 1 bb72 high si1000
```

This corrected convention is circuit schema version 2.  Checkpoints and
reported points made by the earlier implementation, where `R` contained only
`R-1` noisy cycles, are intentionally rejected rather than silently mixed with
new experiments.

### Syndrome extraction circuit

A BB check has weight six and is not nearest-neighbour in any planar embedding,
so `src/stim_utils.py` cannot express it and `src/bb_stim_utils.py` builds its
own cycle.  Layers are assigned per Tanner-edge *orbit*, which makes the
schedule automatically invariant under the code's cyclic translation group.
A schedule is legal when each ancilla and each data qubit sees six distinct
layers, and when, for every `k` and `j`,

```text
[t(X,L,a_k) < t(Z,L,b_j)] == [t(X,R,b_j) < t(Z,R,a_k)]
```

The second condition is what keeps detectors deterministic: an X ancilla acts
on a data qubit while a Z ancilla reads it, so an odd number of shared qubits
with X-before-Z leaves the two ancillas entangled.  Exhaustive search shows
depth six admits no legal schedule and depth seven admits 8,496, matching the
depth-7 circuit of Bravyi et al.  `search_schedules` reproduces that count and
`assert_detectors_deterministic` checks the result against Stim itself.

### Equivariance and orbits

A mechanism's orbit is the canonical form of its detector signature under a
simultaneous space and time translation, with distance to the first and last
frame retained so that boundary mechanisms are not tied to bulk ones.  Only the
detector signature enters the key: belief propagation reads the check matrix
alone, so mechanisms that share a detector pattern but differ in observables
may be tied exactly.

Both codes and every noisy-round count from three upwards give **1,410
orbits** with the default boundary width.  The
orbit structure describes one extraction cycle, so it does not grow with the
lattice or with the experiment length, and the decoder's parameter count stays
fixed as the code and the circuit grow.  Saturation requires
`rounds + 1 >= 2*boundary_width + 2` detector frames; below that every mechanism
touches a time boundary.

Parameters are shared through one residual MLP conditioned on a learned orbit
embedding, rather than one MLP per orbit.  That decouples how finely orbits are
resolved from how many parameters exist -- BB72 at 12 rounds has 1,410 orbits
and 13,299 trainable parameters -- and makes `--bp_parameter_sharing=edge` a
change of index tensor rather than hundreds of separate networks, so unlike the
code-capacity path it is actually runnable.

### Belief propagation is not the baseline here

Plain BP is a weak decoder on a quantum LDPC detector error model, so a raw
Neural-BP-versus-BP comparison is not enough.  Use `--bb_osd_eval_shots` to
also report Neural-BP+OSD against BP+OSD on identical shots.  The current `ldpc`
interface performs one BP update after receiving each posterior as a new
reliability prior, so this is precisely a paired *posterior-seeded BP(1)+OSD*
comparison, not an OSD-only claim.  When this evaluation is enabled,
`best_model.pt` is selected by paired OSD gain (then Neural+OSD accuracy as a
tie-break); without it, selection uses paired Neural-BP gain.

Reported block success uses strict recovery semantics.  If `c` is the predicted
mechanism correction, `H_dem c` must equal the measured detector vector **and**
`O_dem c` must equal the sampled logical-observable vector.  A prediction whose
logical bits happen to match while leaving an unsatisfied detector is a flagged
failure, not a successful decode.

Because the residual head is zero-initialised and the relaxation starts at one,
an untrained model reproduces normalised min-sum bitwise, and the paired gain
starts at exactly zero.

`model.pt` is the latest resumable optimisation state.  Passing it back via
`--load_model` restores the model, AdamW moments, epoch numbering, history and
previous best state; `--epochs` then means additional epochs and starts a fresh
OneCycle phase at the requested learning rate.  Stim does not expose a compiled
sampler RNG state, so the base seed and resumed epoch deterministically derive a
new, non-replayed sampler stream rather than pretending to provide bit-for-bit
sampler continuation.

### Validation status

`tests/test_bb_circuit_level.py` covers the schedule conditions, Stim's
deterministic stabilizer-flow check, exact noisy-round/closing-frame count, DEM
label algebra, graph compatibility, orbit invariance, finite zero-LLR
gradients, strict correction scoring, checkpoint resume, the initialised
neural/vanilla identity, and OSD scoring/selection.  The circuit distance is
**not** yet certified: the DEM rejects an undetectable single-fault logical
mechanism, but that is not a proof that the full fault distance equals the code
distance.  Threshold or distance-scaling claims need an exact or independently
validated circuit-distance analysis first.

## Toric-code input and labels

Every toric noise model is converted to the model input

```text
(batch, 2, rounds, L^2)
```

Channel 0 contains vertex-check detection events and channel 1 contains
face-check detection events. The last dimension uses PanQEC's stabilizer
index order and reshapes directly to `(L, L)` inside `TransformedEND3D`.
`rounds` defaults to `L`, but can be changed with `--rounds`.

## Temporal architectures

The default model remains the original space-time CNN:

```bash
python main.py --architecture=cnn3d --noise_model=phenomenological \
  --L=7 --rounds=7 --p=0.01 --measurement_error_rate=0.01
```

The recurrent decoder processes each measurement round with the same circular
2D CNN and carries a spatial hidden state between rounds with ConvGRU:

```text
(B, 2, T, L^2)
      |
shared equivariant 2D CNN, independently for each round
      |
(B, T, C, L, L)
      |
stacked ConvGRU with circular convolutions
      |
final spatial hidden state -> existing equivariant logical pooling
```

Enable it with:

```bash
python main.py --architecture=convgru --noise_model=phenomenological \
  --L=7 --rounds=7 --p=0.01 --measurement_error_rate=0.01 \
  --channels 64 64 64 --depths 3 3 3 \
  --gru_channels 64 --gru_layers 2 --gru_kernel_size 3
```

`--channels` and `--depths` configure the per-round 2D encoder.
`--gru_channels` defaults to the encoder's last channel width, and
`--gru_layers` defaults to one. The same option works with circuit-level data:

```bash
python main.py --architecture=convgru --noise_model=circuit \
  --L=5 --rounds=5 --p=0.004 --measurement_error_rate=0.004
```

For the toric code, each sample has four logical commutation bits in
`[logical X_0, logical X_1, logical Z_0, logical Z_1]` order. Training
converts these bits to one of 16 classes.

## Equivariant ConvGRU-weighted MWPM

`convgru_weighted_mwpm` places the neural model before MWPM. It preserves every
ConvGRU time output and predicts a conditional probability for every local edge
in the sparse Stim detector-error-model graph:

```text
Stim detection events (B,2,T,L^2)
        |
shared circular 2D CNN for every round
        |
forward + reverse ConvGRU -> h(x,y,t)
        |
symmetric endpoint edge head + relative (dx,dy,dt) + DEM prior
        |
q_e(syndrome) and w_e = log((1-q_e)/q_e)
        |
shot-specific standard MWPM
        |
logical prediction
```

Absolute spatial coordinates and site-specific embeddings are not used. The
endpoint scorer sees only shared detector features and modular relative
geometry, so translating a syndrome and all its toric edges translates the
intermediate representation without changing the edge scores. Time is not
circular. The default model is bidirectional because threshold evaluation is
offline; add `--causal_edge_gru` for an online forward-only model.

The final edge layer is initialized to zero. Therefore the initial conditional
logits equal the static DEM priors exactly, and evaluation initially reproduces
ordinary standard MWPM. Training uses Stim's decomposed DEM sampler with
`return_errors=True`: correlated `^` components are split, parallel mechanisms
are XOR-merged onto the same matching edge, and ordinary unweighted edge BCE is
applied. The optional entropy term does not use inverse-frequency weighting, as
the learned logits are later interpreted as matching probabilities. Start with
`--edge_entropy_weight=0`; treat a nonzero value as a separate ablation.

Run a single point with:

```bash
python main.py --architecture=convgru_weighted_mwpm --noise_model=circuit \
  --L=5 --rounds=5 --p=0.010 --measurement_error_rate=0.010 \
  --channels 96 96 96 --depths 4 4 4 \
  --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 \
  --edge_hidden_channels=192 --edge_chunk_size=1024 \
  --edge_delta_scale=6 --edge_entropy_weight=0 \
  --loss_fn=edge_bce --lr=0.0003 --save_model
```

PyMatching 2.3 has no batched API for shot-specific weights. Training therefore
does not invoke MWPM; validation rebuilds a matcher per shot from the fixed
check/fault matrices and the new weight vector. This is intentionally standard
MWPM: `--matching_correlations` is rejected because rebuilding a check-matrix
matcher cannot retain the original DEM correlation metadata. The training plot
still overlays raw static MWPM on exactly the same evaluation shots.

This implementation predicts conditional weights on the sparse physical DEM
graph. It follows the neural-before-MWPM design, but is not an exact reproduction
of NMWPM's complete active-defect graph and Transformer.

### Legacy logical-residual hybrid

`convgru_mwpm` remains available for old checkpoints. It runs static MWPM and a
parallel ConvGRU 16-class residual classifier, then applies a calibrated logical
XOR correction. It does not change any matching edge. Use `--loss_fn=ce` and
optionally `--matching_correlations` with this legacy architecture.

## Noise models

Code capacity:

```bash
python main.py --noise_model=capacity --L=5 --p=0.1 --epochs=100
```

Phenomenological noise:

```bash
python main.py --noise_model=phenomenological --L=7 --p=0.01 \
  --measurement_error_rate=0.01 --epochs=250
```

Circuit-level noise:

```bash
python main.py --noise_model=circuit --L=5 --rounds=5 --p=0.004 \
  --measurement_error_rate=0.004 --epochs=100 --save_model
```

The circuit uses one ancilla per stabilizer, four collision-free CNOT layers,
depolarizing faults after one- and two-qubit gates, reset faults, and
measurement flips. A noiseless reference cycle is followed by `rounds - 1`
noisy cycles and one final perfect closing cycle. Thus `rounds` is the number
of returned detector frames. The perfect final frame is necessary to detect
data faults from the last noisy cycle; omitting it creates undetectable
single-fault logical errors. Stim samples all detectors and all four logical
correlation sheets from the same shot, preserving X/Z/Y and hook-error
correlations.

Training and evaluation deliberately use two views of that same circuit. For
edge-weight training, Stim samples the circuit's decomposed detector error
model (DEM) with `return_errors=True`; this supplies detector events, logical
observables, and the hidden fault mechanisms that fired, which are the targets
for edge BCE. Validation and final accuracy instead use the compiled circuit
detector sampler, so reported decoding accuracy is measured on fresh exact
circuit shots rather than on those latent DEM labels. Flat detectors are
reshaped to `(batch, 2, rounds, L^2)` before entering the network.

## Pure ConvGRU threshold-gap scripts

Edit `GPU_ID` near the top of each file to select the physical GPU directly.
The two independent runners train pure ConvGRU decoders without MWPM. They
fill the unresolved transition intervals in the current threshold plot:

```text
run_0.sh: (L=13,p=.016), (L=15,p=.011), (L=13,p=.018), (L=15,p=.013)
run_1.sh: (L=13,p=.017), (L=15,p=.012), (L=13,p=.019), (L=15,p=.014)
```

Each point is a fresh run, rather than a resume from a model trained at a
different physical error rate. This keeps all threshold points comparable.
The common model is ConvGRU-96x2 with a 96-96-96, depth-4-4-4 circular CNN,
ordinary cross entropy, 300 epochs, and maximum learning rate `3e-4`:

```bash
# Terminal 1; uses the GPU_ID written in run_0.sh.
bash run_0.sh

# Terminal 2; uses the GPU_ID written in run_1.sh.
bash run_1.sh
```

Each runner creates `resdir_<script-pid>/exp_<index>`, runs its four experiments
sequentially, forwards termination signals, and stops after the first failure.
Both L values use 32,768 generated training shots per epoch: L13 uses
`batch_size=32,batches=1024`, while L15 uses `batch_size=16,batches=2048`.
Validation runs every five epochs and the final evaluation uses 65,536 samples
for L13 or 32,768 for L15.

## Circuit-level matching baseline

To generate the matching-only circuit baseline:

```bash
python scripts/circuit_pymatching_threshold.py \
  --L 5 7 9 --p 0.008 0.009 0.010 0.011 0.012 \
  --shots 262144 --batch_size 2048 \
  --output circuit_pymatching_threshold.csv
```

Add `--enable_correlations` for the correlated-matching baseline.

After the neural and matching runs finish, pass their result directories
together to the threshold plotter. Its default `--metric=final` uses the fresh
`[Selected Best]` evaluation when that line is present, rather than the noisier
last-epoch validation value:

```bash
python scripts/plot_threshold.py \
  /path/to/hybrid_L5_resdir \
  /path/to/hybrid_L7_resdir \
  /path/to/hybrid_L9_resdir \
  /path/to/pymatching_resdir \
  --out circuit_hybrid_vs_mwpm.png \
  --csv circuit_hybrid_vs_mwpm.csv
```

## Offline circuit data

Training samples data on the fly. To save a reusable compressed dataset:

```bash
python scripts/prepare_circuit_data.py --output data/circuit_L5.npz \
  --L=5 --rounds=5 --p=0.004 --measurement_error_rate=0.004 \
  --samples=100000 --batch_size=4096 --seed=1234
```

The archive contains:

- `syndromes`: `uint8 [samples, 2, rounds, L^2]`
- `logical_bits`: `uint8 [samples, 4]`
- `classes`: `uint8 [samples]`
- `metadata`: JSON describing the generation parameters

## Resume training

`--epochs` is interpreted as the number of additional epochs. Resume restores
the model weights, optimizer moments, completed epoch, and loss/accuracy
history. Because a completed OneCycleLR cannot be extended consistently, the
additional phase starts a new OneCycleLR cycle using the requested `--lr`.
The new output directory contains plots spanning both the original and resumed
epochs, with the resume boundary marked by a dashed line.
Because Stim samplers do not expose RNG state, a resumed run deterministically
derives a new sampler seed from the completed epoch count instead of replaying
the beginning of the original training stream.

```bash
python main.py --architecture=convgru --noise_model=phenomenological \
  --L=11 --p=0.03 --measurement_error_rate=0.03 --epochs=300 \
  --channels 96 96 96 --depths 4 4 4 \
  --gru_channels=96 --gru_layers=2 --lr=0.0003 --save_model \
  --load_model=/absolute/path/to/model.pt
```
