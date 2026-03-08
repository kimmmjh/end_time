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
python main.py --noise_model=capacity --L=5 --p=0.01 --epochs=100 --save_model
```

**2. Phenomenological Noise** (Noisy measurements, perfect instantaneous gates)
```bash
python main.py --noise_model=phenomenological --L=5 --p=0.01 --measurement_error_rate=0.01 --epochs=100 --loss_fn=ce --save_model
```

**3. Circuit-level Noise** (Full realistic quantum circuit simulation with CNOT hook errors)
```bash
python main.py --noise_model=circuit --L=5 --p=0.004 --measurement_error_rate=0.004 --epochs=100 --save_model
```

### Advanced Architecture Configuration
You can also boost the intelligence of the network by widening the convolution channels or deepening the ResNet blocks:
```bash
# Deep and Wide network for hard phenomenological problems
python main.py --noise_model=phenomenological --channels 128 128 128 --depths 5 5 5 --epochs=300
```