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
measurement flips. A noiseless reference cycle is followed by the requested
number of noisy cycles. Stim samples all detectors and all four logical
correlation sheets from the same shot, preserving X/Z/Y and hook-error
correlations.

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

```bash
python main.py --noise_model=circuit --L=5 --epochs=100 \
  --load_model=outputs/path/to/model.pt
```
