import torch
import numpy as np
import hydra
from hydra.utils import instantiate
from src._data_generator import DataGenerator

@hydra.main(config_path="config", config_name="2d", version_base="1.2")
def check_distribution(args):
    L = 7
    p = 0.004  # The value user used
    
    print(f"Checking distribution for L={L}, p={p}")
    code = instantiate(args.default.code, L)
    
    gen = DataGenerator(
        code=code,
        error_rate=p,
        batch_size=1000,
        circuit_noise=True,
        measurement_error_rate=p,
        verbose=False
    )
    
    X, y = gen.generate_batch(use_qmc=False, device=torch.device("cpu"))
    
    # y is the class index (0..15)
    counts = np.bincount(y.numpy(), minlength=16)
    print("Class distribution:")
    for i, c in enumerate(counts):
        print(f"Class {i}: {c} ({c/1000:.2%})")

    most_common = np.argmax(counts)
    baseline_acc = counts[most_common] / 1000.0
    print(f"Baseline Accuracy (predict most common class {most_common}): {baseline_acc:.4f}")

if __name__ == "__main__":
    check_distribution()
