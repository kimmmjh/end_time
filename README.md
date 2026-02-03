# TheEND 2D: Toric Code Neural Decoder (Clean Version)

This directory contains the essential implementation of the 2D Toric Code Equivariant Neural Decoder, supporting both Phenomenological and Circuit-Level Noise (via Stim).

## Essentials

*   **`main.py`**: The entry point for training and evaluation.
*   **`models/`**: Contains the `TransformedEND2D` architecture.
*   **`src/`**: Logic for training (`Trainer`), data generation (`DataGenerator`), and Stim integration (`stim_utils`).
*   **`config/`**: Configuration files (Hydra).

## How to Run

### 1. Circuit-Level Noise (Stim)
Train the decoder on realistic circuit noise (Z-basis and X-basis):

```bash
python main.py --config-name 2d \
    default.L=6 \
    default.p=0.01 \
    default.circuit_noise=True \
    default.measurement_error_rate=0.01 \
    ++net.channels=[64,64,64] \
    ++net.depths=[2,2,2]
```

### 2. Code Capacity Noise (Phenomenological)
Train on simple bit-flip/phase-flip noise:

```bash
python main.py --config-name 2d \
    default.L=8 \
    default.p=0.05
```

## Requirements
*   `torch`
*   `panqec`
*   `stim`
*   `hydra-core`
*   `numpy`
*   `scikit-learn`
*   `wandb` (optional)
