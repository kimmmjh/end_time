# TheEND time: Circuit level noise Toric Code Neural Decoder

This directory contains the implementation of the 2D Toric Code Equivariant Neural Decoder, supporting both Phenomenological and Circuit-Level Noise (via Stim). Implemented based on oliverweissl/NeuralDecoderToric3D

## Essentials

*   **`main.py`**: The entry point for training and evaluation.
*   **`models/`**: Contains the `TransformedEND2D` architecture.
*   **`src/`**: Logic for training (`Trainer`), data generation (`DataGenerator`), and Stim integration (`stim_utils`).
*   **`config/`**: Configuration files (Hydra).

## How to Run

### 1. Circuit-Level Noise (Stim)
Train the decoder on realistic circuit noise (Z-basis and X-basis):

``` bash
python main.py net=ch64_64_64 default.circuit_noise=True default.L=7 default.p=0.01 default.measurement_error_rate=0.01 default.epochs=100 default.batches=128 +save_model=True
```

### 2. Code Level Noise (Phenomenological)
Train on simple bit-flip/phase-flip noise:
Probability of No Error (I): 1 - p, Probability of X,Y,Z: p / 3 each

``` bash
python main.py net=ch64_64_64 default.circuit_noise=False default.L=17 default.p=0.10 default.epochs=100 default.batches=128 +save_model=True
```

For resume learning:
``` bash
+load_model=outputs/"date&time"/model.pt
```


python main.py net=ch64_64_64 default.circuit_noise=True default.L=7 default.p=0.004 default.measurement_error_rate=0.004 default.epochs=100 +save_model=True