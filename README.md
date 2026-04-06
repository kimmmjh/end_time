# TheEND time: Circuit level noise Toric Code Neural Decoder

This directory contains the implementation of the 2D Toric Code Equivariant Neural Decoder, supporting both Phenomenological and Circuit-Level Noise (via Stim). Implemented based on oliverweissl/NeuralDecoderToric3D

## Essentials

*   **`main.py`**: The entry point for training and evaluation.
*   **`models/`**: Contains the `TransformedEND2D` architecture.
*   **`src/`**: Logic for training (`Trainer`), data generation (`DataGenerator`), and Stim integration (`stim_utils`).
*   **`src/`**: Logic for training (`Trainer`), data generation (`DataGenerator`), and Stim integration (`stim_utils`).

### Noise Models and Execution

**1. Code Capacity Noise** (Perfect measurements, only data qubit errors)
```bash
python main.py --noise_model=capacity --L=5 --p=0.1 --epochs=100 --save_model
```

**2. Phenomenological Noise** (Noisy measurements, perfect instantaneous gates)
```bash
python main.py --noise_model=phenomenological --L=7 --p=0.01 --measurement_error_rate=0.01 --epochs=250 --loss_fn=ce --save_model
```

**3. Circuit-level Noise** (Full realistic quantum circuit simulation with CNOT hook errors)
```bash
python main.py --noise_model=circuit --L=5 --p=0.004 --measurement_error_rate=0.004 --epochs=100 --save_model
```

### Load Model
python main.py --noise_model=phenomenological --L=5 --epochs=100 --channels 64 64 64 --depths 3 3 3 --loss_fn=ce --p=0.01 --load_model=outputs/"output_path"/model.pt
