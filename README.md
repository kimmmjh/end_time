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

Two GPU runners sweep the same depolarizing points for both code sizes. Edit
`GPU_ID` at the top if needed:

```bash
bash run_bb_0.sh  # GPU 0, BB72, p=0.04,0.06,0.08,0.10
bash run_bb_1.sh  # GPU 1, BB144, the same p values
```

On Perlmutter, submit the Slurm versions instead:

```bash
sbatch run_bb_0.slurm  # one GPU node: four concurrent BB72 p points
sbatch run_bb_1.slurm  # one GPU node: four concurrent BB144 p points
```

Each Slurm job requests four tasks and four GPUs, then launches four
independent `srun` steps with one GPU each. The scripts use the
existing NERSC account `m5328_g`, environment
`$PSCRATCH/envs/nde`, and repository `$HOME/end_time`; edit those values if
the server paths change. Results are stored under
`$HOME/end_time/resdir_<SLURM_JOB_ID>` with `log_exp_0.txt`, ..., and one
`exp_<index>/outputs` tree per p value.

Before launching all eight long jobs, run a one-epoch timing/sanity check on
the server:

```bash
python main.py --code=bb72 --architecture=bb_neural_bp \
  --noise_model=capacity --measurement_error_rate=0 --loss_fn=bb_coset \
  --p=0.08 --epochs=1 --batches=2 --batch_size=8 \
  --eval_batches=1 --final_eval_batches=1 --amp_dtype=none
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

## Circuit-level experiment scripts

Edit `GPU_ID` near the top of each file to select the physical GPU directly.
The two independent runners use `convgru_weighted_mwpm` and jointly cover
L=5,7,9 and p=q=0.008,...,0.012 with 300 epochs per point:

```bash
# Terminal 1; uses the GPU_ID written in run_0.sh.
bash run_0.sh

# Terminal 2; uses the GPU_ID written in run_1.sh.
bash run_1.sh
```

The final `[Selected Best]` line is measured on fresh held-out circuit samples
and recommends `neural_weighted_mwpm` only when the paired 95% lower bound over
raw MWPM is positive. `accuracy_curve.png` overlays raw MWPM and
`hybrid_net_gain_curve.png` reports the paired difference. Each runner creates
its own `resdir_<script-pid>/exp_<index>`, runs one experiment at a time on its
configured GPU, forwards termination signals, and stops after the first failed
experiment. The two point lists are disjoint and together contain all 15 grid
points. Dynamic PyMatching evaluation is CPU work, so these scripts use fewer
evaluation batches than the old logical-residual runs.

Each runner uses `batch_size=32` and `batches=2048`, i.e. 65,536 supervised
DEM shots per epoch. Every five epochs, validation uses 4,096 exact circuit
shots; the final epoch is always evaluated. The selected best checkpoint is
finally evaluated on 16,384 fresh circuit shots. Set `--eval_every=1` to restore
per-epoch validation.

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
