import torch
from torch import Tensor
from panqec.codes import StabilizerCode
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from scipy.sparse import csr_matrix, vstack
from .stim_utils import generate_toric_memory_circuit


class DataGenerator:
    """Data generator object."""

    noise_model: str
    """Stabilizer code specific attributes."""
    logicals: csr_matrix
    stabilizers: csr_matrix
    n: int  # number of physical qubits
    d: int

    """Generation attributes."""
    batch_size: int
    error_rate: float

    """Some private attributes."""
    _verbose_print: Callable[[str], None]
    _categorical_dict: dict[tuple[int, ...], int]
    _categorical_classification: bool

    def __init__(
        self,
        code: StabilizerCode,
        error_rate: float,
        batch_size: int,
        categorical_classification: bool = True,
        one_hot: bool = False,
        verbose: bool = True,
        measurement_error_rate: float = 0.0,
        rounds: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the Dataset.

        :param code: The stabilizer code associated.
        :param error_rate: The error rate.
        :param batch_size: The batch size.
        :param categorical_classification: Whether the task is to do categorical classification or multi label.
        :param one_hot: If classes should be returned one-hot encoded (Only has affect when using categorical classification).
        :param verbose: If messages should be printed.
        :param noise_model: Noise model to use: "capacity", "phenomenological", or "circuit".
        :param measurement_error_rate: The error rate of the measurement step.
        """
        self._verbose_print: Callable[[str], None] = (
            print if verbose else lambda x: None
        )
        self._categorical_classification = categorical_classification
        self._one_hot = one_hot
        self._measurement_error_rate = measurement_error_rate
        self.rounds = code.size[0] if rounds is None else rounds
        self.seed = seed
        if self.rounds < 1:
            raise ValueError(f"rounds must be positive, got {self.rounds}.")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {error_rate}.")
        if not 0.0 <= measurement_error_rate <= 1.0:
            raise ValueError(
                "measurement_error_rate must be in [0, 1], got "
                f"{measurement_error_rate}."
            )

        self.d = len(code.size)
        self.L = code.size[0]

        self.error_rate = error_rate
        self.batch_size = batch_size

        """Get X and Z logicals from lattice and combine them."""
        x_logical, z_logical = csr_matrix(code.logicals_x), csr_matrix(code.logicals_z)
        self.logicals = vstack((x_logical, z_logical))

        """Transpose the stabilizers."""
        block_size = code.size[0] ** self.d
        x, y = code.stabilizer_matrix.shape

        original = np.array(code.stabilizer_matrix.todense())
        matrix = np.zeros_like(original)
        for i in range(x // block_size):
            for j in range(y // block_size):
                matrix[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = original[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ].T

        self.stabilizers = csr_matrix(matrix)
        self.n = code.n

    def _check_class(self, logical_error: NDArray) -> int:
        """
        Get the class corresponding to a logical error.

        :param logical_error: The logical error.
        :returns: The class as int [0: n].
        """
        power = 2 ** (np.array(range(self.d * 2))[::-1])
        return np.inner(logical_error, power)

    def _generate_sample(self) -> tuple[NDArray, NDArray, csr_matrix]:
        """
        Generate a sample of the dataset.

        :returns: The syndrome and logical error. If used for ldpc library additionally return errors.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_batch(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Generate the dataset.

        :param device: The device that uses the data.
        :returns: The syndrome and logical error.
        """

        # MAKE THE RETURNING SHAPE TO (b, 2=X/Z, r=L, L, L)
        self._verbose_print("\tGenerating Errors")
        syndrome_matrices, logical_errors, errors = self._generate_sample()

        """ Transform to indices if we use categorical classification."""
        if self._categorical_classification:
            # CrossEntropyLoss requires indices as y_true.
            logical_errors = np.apply_along_axis(self._check_class, 1, logical_errors)

            # Transform to one-hot encoded classes if needed.
            if self._one_hot:
                encoded_arr = np.zeros(
                    (logical_errors.size, 2 ** (2 * self.d)), dtype=int
                )
                encoded_arr[np.arange(logical_errors.size), logical_errors] = 1
                logical_errors = encoded_arr

        """Convert to tensors."""
        syndrome_matrices = torch.tensor(
            data=syndrome_matrices, dtype=torch.float, device=device
        )
        logical_errors = torch.tensor(
            data=logical_errors,
            dtype=torch.long if not self._one_hot else torch.float,
            device=device,
        )

        return syndrome_matrices, logical_errors


class CapacityDataGenerator(DataGenerator):
    def __init__(
        self,
        code,
        error_rate,
        batch_size,
        categorical_classification=True,
        one_hot=False,
        verbose=True,
        noise_model="capacity",
        measurement_error_rate=0,
        rounds=None,
        seed=None,
    ):
        super().__init__(
            code=code,
            error_rate=error_rate,
            batch_size=batch_size,
            categorical_classification=categorical_classification,
            one_hot=one_hot,
            verbose=verbose,
            measurement_error_rate=measurement_error_rate,
            rounds=rounds,
            seed=seed,
        )

    def _generate_sample(self):
        num_qubits = self.n
        repetitions = (
            1  # Capacity noise has 0 time steps (just 1 perfectly measured final frame)
        )
        p = self.error_rate
        H = self.stabilizers
        num_stabilisers = H.shape[0]

        # Vectorized batch generation (much faster than a for-loop!)
        errors = np.random.choice(
            ["I", "X", "Y", "Z"],
            size=(self.batch_size, repetitions, num_qubits),
            p=[1 - p, p / 3, p / 3, p / 3],
        )

        errors_x = np.isin(errors, ["X", "Y"]).astype(np.uint8)
        errors_z = np.isin(errors, ["Z", "Y"]).astype(np.uint8)

        # PanQEC matrices use BSF [X|Z], so commutation is evaluated by
        # multiplying the symplectic dual [error_Z|error_X].
        noise_new = np.concatenate(
            (errors_z, errors_x), axis=2
        )  # Shape: (b, 1, 2*num_qubits)
        noise_total = noise_new[:, 0, :]  # Shape: (b, 2*num_qubits)

        self._verbose_print("\tConstructing Syndrome Matrices")
        # Direct parity check matrix calculation: H * Error = Syndrome
        syndrome = (noise_total @ H.T) % 2

        self._verbose_print("\tMeasuring Logicals")
        # Direct logical operator matrix calculation
        logical_errors = (noise_total @ self.logicals.T) % 2

        # Reshape to the same 4D format as the Phenomenological generator
        syndrome_matrices = syndrome.reshape(
            self.batch_size, repetitions, 2, num_stabilisers // 2
        ).transpose(
            0, 2, 1, 3
        )  # Final Shape: (b, 2=X/Z, r=1, num_stabilisers//2)

        return np.array(syndrome_matrices), np.array(logical_errors), None


class PhenomenologicalDataGenerator(DataGenerator):
    """Data generator for phenomenological noise model."""

    def __init__(
        self,
        code: StabilizerCode,
        error_rate: float,
        batch_size: int,
        categorical_classification: bool = True,
        one_hot: bool = False,
        verbose: bool = True,
        measurement_error_rate: float = 0.0,
        rounds: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.noise_model = "phenomenological"

        super().__init__(
            code=code,
            error_rate=error_rate,
            batch_size=batch_size,
            categorical_classification=categorical_classification,
            one_hot=one_hot,
            verbose=verbose,
            measurement_error_rate=measurement_error_rate,
            rounds=rounds,
            seed=seed,
        )

    def _generate_sample(self):
        num_qubits = self.n
        repetitions = self.rounds
        p = self.error_rate
        q = self._measurement_error_rate
        H = self.stabilizers
        num_stabilisers = H.shape[0]
        detectors = []
        observables = []

        for i in range(self.batch_size):
            errors = np.random.choice(
                ["I", "X", "Y", "Z"],
                size=(repetitions, num_qubits),
                p=[1 - p, p / 3, p / 3, p / 3],
            )
            errors_x = (np.isin(errors, ["X", "Y"])).astype(np.uint8)
            errors_z = (np.isin(errors, ["Z", "Y"])).astype(np.uint8)

            # Symplectic dual of the Pauli error in PanQEC BSF convention.
            noise_new = np.concatenate((errors_z, errors_x), axis=1)
            noise_cumulative = (np.cumsum(noise_new, 0) % 2).astype(np.uint8)
            noise_total = noise_cumulative[-1, :]
            syndrome = (H @ noise_cumulative.T).T % 2
            syndrome_error = (np.random.rand(repetitions, num_stabilisers) < q).astype(
                np.uint8
            )
            syndrome_error[-1, :] = 0  # Perfect measurement in the final round.
            noisy_syndrome = (syndrome + syndrome_error) % 2

            detection_events = np.empty_like(noisy_syndrome)
            detection_events[0] = noisy_syndrome[0]
            detection_events[1:] = noisy_syndrome[1:] ^ noisy_syndrome[:-1]

            logical_error = (self.logicals @ noise_total.T).T % 2
            detectors.append(detection_events)
            observables.append(logical_error)

        detectors = np.array(detectors)
        detectors = detectors.reshape(
            self.batch_size, repetitions, 2, num_stabilisers // 2
        ).transpose(
            0, 2, 1, 3
        )  # Reshape to (b, 2=X/Z, r=L, num_stabilisers//2)

        return np.array(detectors), np.array(observables), None


class CircuitLevelDataGenerator(DataGenerator):
    """Batch sampler for ancilla-based circuit-level toric-code noise."""

    def __init__(
        self,
        code: StabilizerCode,
        error_rate: float,
        batch_size: int,
        categorical_classification: bool = True,
        one_hot: bool = False,
        verbose: bool = True,
        measurement_error_rate: float = 0.0,
        rounds: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.noise_model = "circuit"
        super().__init__(
            code=code,
            error_rate=error_rate,
            batch_size=batch_size,
            categorical_classification=categorical_classification,
            one_hot=one_hot,
            verbose=verbose,
            measurement_error_rate=measurement_error_rate,
            rounds=rounds,
            seed=seed,
        )
        self.num_stabilizers = self.stabilizers.shape[0]
        self.circuit = generate_toric_memory_circuit(
            code,
            rounds=self.rounds,
            gate_error_rate=self.error_rate,
            measurement_error_rate=self._measurement_error_rate,
        )
        self.sampler = self.circuit.compile_detector_sampler(seed=self.seed)

    def _generate_sample(self):
        self._verbose_print("\tSampling Stim circuit")
        # This is the number of shots in one generated batch. Trainer calls
        # generate_batch() `batches` times per training epoch, so the total is
        # batch_size * batches (and batch_size * eval_batches for evaluation).
        detectors, logical_errors = self.sampler.sample(
            shots=self.batch_size,
            separate_observables=True,
        )
        expected_detectors = self.rounds * self.num_stabilizers
        expected_observables = self.logicals.shape[0]
        if detectors.shape != (self.batch_size, expected_detectors):
            raise RuntimeError(
                "Unexpected Stim detector shape: "
                f"{detectors.shape}, expected "
                f"{(self.batch_size, expected_detectors)}."
            )
        if logical_errors.shape != (self.batch_size, expected_observables):
            raise RuntimeError(
                "Unexpected Stim observable shape: "
                f"{logical_errors.shape}, expected "
                f"{(self.batch_size, expected_observables)}."
            )

        syndrome_matrices = detectors.reshape(
            self.batch_size,
            self.rounds,
            2,
            self.num_stabilizers // 2,
        ).transpose(0, 2, 1, 3)

        return (
            syndrome_matrices.astype(np.uint8, copy=False),
            logical_errors.astype(np.uint8, copy=False),
            None,
        )
