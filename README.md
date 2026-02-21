# TheEND time: Circuit level noise Toric Code Neural Decoder

This directory contains the implementation of the 2D Toric Code Equivariant Neural Decoder, supporting both Phenomenological and Circuit-Level Noise (via Stim). Implemented based on oliverweissl/NeuralDecoderToric3D

## Essentials

*   **`main.py`**: The entry point for training and evaluation.
*   **`models/`**: Contains the `TransformedEND2D` architecture.
*   **`src/`**: Logic for training (`Trainer`), data generation (`DataGenerator`), and Stim integration (`stim_utils`).
*   **`config/`**: Configuration files (Hydra).

### Noise Models

**1. Code Capacity Noise** (Perfect measurements, only data qubit errors)
```bash
python main.py net=ch64_64_64 default.noise_model=capacity default.L=5 default.p=0.01 default.epochs=100 +save_model=True
```

**2. Phenomenological Noise** (Noisy measurements, perfect instantaneous gates)
```bash
python main.py net=ch64_64_64 +default.noise_model=phenomenological default.L=5 default.p=0.01 default.measurement_error_rate=0.01 default.epochs=100 +save_model=True
```

**3. Circuit-level Noise** (Full realistic quantum circuit simulation with CNOT hook errors)
```bash
python main.py net=ch64_64_64 default.circuit_noise=True default.L=5 default.p=0.004 default.measurement_error_rate=0.004 default.epochs=100 +save_model=True
```