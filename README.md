# TheEND time: equivariant neural QEC decoders

This repository contains translation-equivariant toric-code decoders and an
orbit-equivariant neural belief-propagation decoder for bivariate-bicycle (BB)
codes. The toric path supports code-capacity, phenomenological, and Stim-based
circuit noise. The first BB implementation is deliberately code capacity only.

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

Ten Perlmutter jobs provide the next BB experiment suite. Jobs 0--7 are GPU
neural-BP training jobs; jobs 8--9 are CPU-only classical-decoder benchmarks.
The completed orbit/depolarizing seed-A runs remain the full-model reference.

| Script | Four concurrent experiments |
| --- | --- |
| `run_bb_0.slurm` | orbit model, seed B, BB72/BB144 at p=0.08/0.10 |
| `run_bb_1.slurm` | orbit model, seed C, the same four points |
| `run_bb_2.slurm` | global h64 and parameter-matched global h393, both codes at p=0.08 |
| `run_bb_3.slurm` | 6 and 24 BP iterations, both codes at p=0.08 (T=12 is archived) |
| `run_bb_4.slurm` | orbit model, marginal-matched independent X/Z at p=0.08/0.10 |
| `run_bb_5.slurm` | residual-only and learned-relaxation-only, both codes at p=0.08 |
| `run_bb_6.slurm` | no Pauli auxiliary and no deep supervision, both codes at p=0.08 |
| `run_bb_7.slurm` | no logical surrogate and no syndrome loss, both codes at p=0.08 |
| `run_bb_8.slurm` | BB72 CSS-split BP+OSD-0/CS-7/LSD-0, p=0.04/0.06/0.08/0.10 |
| `run_bb_9.slurm` | BB144 versions of the same three classical baselines |

Submit selected jobs, or submit the full suite with:

```bash
for i in {0..9}; do sbatch "run_bb_${i}.slurm"; done
```

For the independent-X/Z runs, `p` labels the reference depolarizing point and
the actual component rates are `p_x=p_z=2p/3`. This preserves the X and Z
component marginals while removing their on-qubit correlation. It does not
preserve the total non-identity rate: that rate is
`1-(1-2p/3)^2`, so independent-X/Z results must be plotted against their
component rates or labeled as matched-marginal points. `global` sharing is
still equivariant; comparing it with `orbit` tests the granularity of
equivariant parameter sharing, not equivariance versus a non-equivariant
network. `global h393` has 14,210 trainable parameters versus 14,196 for
`orbit h64`, removing nearly all of the parameter-count confound. The current
per-edge implementation is intentionally omitted because its hundreds of
separately called MLPs make a 300-epoch job impractical.

Jobs 0--7 request four tasks and four GPUs, then launch four independent
`srun --exclusive` steps with one GPU each. Jobs 8--9 request four CPU tasks
because `ldpc` BP+OSD/LSD does not use CUDA. They use the presumed CPU account
`m5328`; verify that allocation or edit the two `#SBATCH --account` lines
before submission. All scripts use `$PSCRATCH/envs/nde` and
`$HOME/end_time`. Results are stored under
`$HOME/end_time/resdir_<SLURM_JOB_ID>` with `log_exp_0.txt`, ...,
`log_exp_3.txt`, per-experiment exit codes, and a completed/failed marker.
For neural jobs, the timestamped model directories share
`outputs/YYYY-MM-DD/`; classical jobs instead write their CSV/NPZ files
directly into the result directory. There is no `exp_<index>` layer in the
Slurm layout. The shared launch logic is in
`scripts/run_bb_slurm_batch.sh`.

The classical jobs require exactly `ldpc==2.4.1` and preflight that version
before creating a result directory. At each `(code,p)` point all three methods
decode one shared 131,072-shot bank using binary component prior `2p/3`,
minimum-sum scale 0.625, parallel scheduling, and `n` BP iterations. These are
CSS-separated, correlation-unaware baselines: their X and Z sectors are
decoded separately, so they discard the Y-induced correlation retained by
joint BP4. CSV files contain exact block logical error, Wilson intervals,
flagged/unflagged failures, and latency. NPZ files preserve the sampled errors
and per-shot outcomes. Those classical methods are paired with each other;
they are not paired with the archived neural results unless a neural
checkpoint is evaluated later on the saved NPZ test bank.

The completed BB72 and BB144 depolarizing sweeps are summarized in
[`results/analysis/bb_neural_bp_depolarizing_orbit.md`](results/analysis/bb_neural_bp_depolarizing_orbit.md),
with a machine-readable CSV and a Neural BP versus vanilla BP4 comparison
plot in `results/analysis/` and `results/plots/`.

Before launching the training suite, run a one-epoch timing/sanity check on
the heaviest unrolled configuration:

```bash
python main.py --code=bb144 --architecture=bb_neural_bp \
  --noise_model=capacity --rounds=1 --measurement_error_rate=0 \
  --loss_fn=bb_coset --bb_channel=depolarizing --p=0.08 \
  --bp_iterations=24 --bp_residual_hidden_dim=64 \
  --bp_parameter_sharing=orbit --epochs=1 --batches=2 --batch_size=8 \
  --eval_batches=1 --eval_every=1 --final_eval_batches=1 \
  --lr=0.0003 --amp_dtype=none --seed=14408
```

Each neural model starts exactly as vanilla BP4 because its final residual
layer is zero initialized and relaxation starts at one. `model.pt` is the
latest resumable checkpoint; `best_model.pt` is selected by held-out block
logical accuracy. Generator RNG state, optimizer state, and plot history are
saved, and an incompatible BB graph/model checkpoint is rejected.

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
