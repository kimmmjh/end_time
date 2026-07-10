from panqec.codes import StabilizerCode
from .metrics import categorical_accuracy
import torch
from ._data_generator import (
    CapacityDataGenerator,
    CircuitLevelDataGenerator,
    PhenomenologicalDataGenerator,
)
from torch import nn
import numpy as np
from time import time


def evaluate_decoder(
    model: nn.Module,
    code: StabilizerCode,
    error_rate: float,
    trials: int = 100_000,
    batch_size: int = 512,
    noise_model: str = "capacity",
    measurement_error_rate: float = 0.0,
    rounds: int | None = None,
    seed: int | None = None,
) -> tuple[float, float, list[float]]:
    """
    Evaluate the Neural Decoders performance.

    :param model: The model.
    :param code: The stabilizer code.
    :param error_rate: The error rate.
    :param trials: The amount of trials.
    :param batch_size: The batch size.
    :param noise_model: One of capacity, phenomenological, or circuit.
    :param measurement_error_rate: Measurement flip probability.
    :param rounds: Number of syndrome rounds; defaults to the lattice size.
    :param seed: Optional sampler seed.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator_classes = {
        "capacity": CapacityDataGenerator,
        "phenomenological": PhenomenologicalDataGenerator,
        "circuit": CircuitLevelDataGenerator,
    }
    if noise_model not in generator_classes:
        raise ValueError(f"Unknown noise model: {noise_model!r}.")
    data_generator = generator_classes[noise_model](
        code=code,
        verbose=False,
        error_rate=error_rate,
        batch_size=batch_size,
        measurement_error_rate=measurement_error_rate,
        rounds=rounds,
        seed=seed,
    )

    runtimes = []
    accuracies = []
    for _ in range(trials // batch_size):
        X, y = data_generator.generate_batch(device=device)
        start = time()
        with torch.no_grad():
            y_pred = model(X)
        runtimes.append((time() - start)/batch_size)
        acc, _ = categorical_accuracy(y_pred, y)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies.mean(), accuracies.std(), runtimes
