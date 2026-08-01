# TheEND time: toric-code neural decoder

This repository trains a translation-equivariant neural decoder for the 2D
toric code. It supports code-capacity, phenomenological, and Stim-based
circuit-level noise.

## Install

```bash
pip install -r requirements.txt
```

Circuit-level generation requires Stim 1.15 or newer.

## Input and labels

Every noise model is converted to the model input

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

## Equivariant ConvGRU + MWPM hybrid

For circuit-level noise, `convgru_mwpm` lets PyMatching handle global
space-time pairing and trains the equivariant ConvGRU to predict only the
logical residual left by that matching correction:

```text
Stim detection events
        |----------------------|
        v                      v
DEM-based PyMatching    circular CNN + ConvGRU
        |                      |
MWPM logical class      residual logical logits
        |----------------------|
                    XOR
                     |
             final logical class
```

The circular CNN and ConvGRU produce a translation-equivariant spatial map.
Because the residual after matching is a closed-cycle homology class, spatial
mean pooling gives the appropriate translation-invariant 16-class readout.
The model therefore retains exact toric translation symmetry while
PyMatching supplies the non-local pairing operation.

Run a single point with:

```bash
python main.py --architecture=convgru_mwpm --noise_model=circuit \
  --L=5 --rounds=5 --p=0.010 --measurement_error_rate=0.010 \
  --channels 96 96 96 --depths 4 4 4 \
  --gru_channels=96 --gru_layers=2 --gru_kernel_size=3 \
  --loss_fn=ce --hybrid_calibration_batches=256 \
  --lr=0.0003 --save_model
```

Use ordinary cross entropy for the residual task. Residual class zero means
that MWPM succeeded and is intentionally much more common; inverse-frequency
class weighting overemphasizes rare overrides and can make the hybrid worse
than MWPM. After training, `--hybrid_calibration_batches` uses fresh samples to
calibrate a selective override threshold. If no neural override improves the
calibration accuracy, the gate falls back to the exact MWPM prediction. Add
`--matching_correlations` to use PyMatching's correlated two-pass decoder;
without it, the neural residual model is trained against ordinary DEM-based
MWPM.

This implementation is a neural **residual/postdecoder**, not per-shot edge
reweighting. PyMatching 2.3 uses static graph weights and does not accept
batched shot-dependent weights. A true local neural predecoder would require
explicit Stim fault-mechanism targets, hard local corrections, and a second
matching pass on the residual syndrome.

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

## Circuit-level experiment scripts

The two-GPU runner covers L=5,7,9 and p=q=0.008,...,0.012 with 300 epochs per
point:

```bash
GPU_IDS="0 1" bash run_convgru_mwpm_full.sh
```

The final `[Selected Best]` line is measured on fresh held-out shots and reports
`Recommended: hybrid` only when the paired 95% lower bound is positive;
otherwise it reports `Recommended: mwpm`. Each hybrid run also overlays MWPM
in `accuracy_curve.png` and writes `hybrid_net_gain_curve.png` with paired 95%
error bars. The runner creates `resdir_<script-pid>/exp_<index>`, keeps at most
one process on each GPU, forwards termination signals, and stops the remaining
schedule after the first failed experiment. `GPU_IDS` must contain exactly two
distinct physical GPU indices.

The existing single-GPU runner uses the same safe CE loss and calibrated MWPM
fallback, and scans all five `p` values sequentially for one lattice size:

```bash
GPU_ID=1 bash run_circuit_hybrid.sh 7
```

To generate the matching-only circuit baseline:

```bash
python scripts/circuit_pymatching_threshold.py \
  --L 5 7 9 --p 0.008 0.009 0.010 0.011 0.012 \
  --shots 262144 --batch_size 2048 \
  --output circuit_pymatching_threshold.csv
```

Add `--enable_correlations` for the correlated-matching baseline.

The same baseline can be launched with a shell runner matching the hybrid
experiment grid:

```bash
# Ordinary MWPM for L=5,7,9.
bash run_circuit_pymatching.sh all

# Optional correlated PyMatching comparison.
bash run_circuit_pymatching.sh all correlated
```

To run only one lattice size, replace `all` with `5`, `7`, or `9`. The default
is 262,144 shots per point. It can be overridden without editing the script:

```bash
SHOTS=1000000 BATCH_SIZE=4096 bash run_circuit_pymatching.sh 7
```

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

```bash
python main.py --architecture=convgru --noise_model=phenomenological \
  --L=11 --p=0.03 --measurement_error_rate=0.03 --epochs=300 \
  --channels 96 96 96 --depths 4 4 4 \
  --gru_channels=96 --gru_layers=2 --lr=0.0003 --save_model \
  --load_model=/absolute/path/to/model.pt
```
