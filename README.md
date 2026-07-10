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

## Larger-architecture threshold scans

The SLURM files `run_3.slurm` through `run_6.slurm` keep the existing code
distances `L=9,11,13,15` and enlarge the neural network from
`channels=[64,64,64], depths=[3,3,3]` (2.11M parameters) to
`channels=[96,96,96], depths=[4,4,4]` (6.15M parameters). Each script scans
`p=q` densely at `0.01,0.0125,0.015,0.0175,0.02`. These five points run as a
Slurm array with at most four concurrent tasks, matching a four-GPU node.

The number of generated samples is:

```text
training samples per epoch = batch_size * batches
evaluation samples          = batch_size * eval_batches
final evaluation samples    = batch_size * final_eval_batches
```

Accordingly, the large-model scripts use 131,072 samples for the final
reported threshold point. `shots=batch_size` inside the Stim data generator
is only the size of one generated batch; the trainer requests one batch on
every iteration. The separate final evaluation size is available from the
CLI:

```bash
python main.py --noise_model=phenomenological --L=9 --p=0.025 \
  --measurement_error_rate=0.025 --channels 96 96 96 --depths 4 4 4 \
  --batch_size=32 --eval_batches=512 --final_eval_batches=4096
```

`run_7.slurm` generates PyMatching baselines for `L=9,11,13,15`. Each `(L,p)`
point uses 262,144 shots. The baseline implements
the same phenomenological convention as the training data: depolarizing data
errors every round, noisy measurements with `q=p`, and a perfect last
measurement. Standard CSS MWPM decodes the X and Z components independently.

After the jobs finish, combine neural logs and PyMatching CSV files directly:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/plot_threshold.py \
  threshold_L9_L11_L13_L15.csv \
  resdir_RUN3 resdir_RUN4 resdir_RUN5 resdir_RUN6 resdir_RUN7 \
  --group L_arch_decoder \
  --out threshold_nn_vs_pymatching.png \
  --csv threshold_nn_vs_pymatching.csv \
  --title "Phenomenological threshold: NN vs PyMatching"
```

The plotter automatically discovers `log_exp_*.txt` and
`pymatching*.csv` inside each supplied result directory. The
`L_arch_decoder` grouping keeps the original NN, enlarged NN, and PyMatching
curves separate. An existing aggregated threshold CSV can also be used as an
input path, so the old plot can be extended without reconstructing its
original log-file list.
