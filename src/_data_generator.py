import code

import torch
from torch import Tensor
from panqec.codes import StabilizerCode
import numpy as np
from numpy.typing import NDArray
from ._auxiliary_functions import generate_syndrome, sample_errors, get_logical_errors
from typing import Callable
from scipy.sparse import csr_matrix, vstack
from .stim_utils import generate_stim_circuit, generate_phenomenological_circuit


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

        self._initialize_circuit()

    def _initialize_circuit(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
        pass

    def _check_class(self, logical_error: NDArray) -> int:
        """
        Get the class corresponding to a logical error.

        :param logical_error: The logical error.
        :returns: The class as int [0: n].
        """
        power = 2 ** (np.array(range(self.d * 2))[::-1])
        return np.inner(logical_error, power)

    def _generate_sample(self, use_qmc: bool) -> tuple[NDArray, NDArray, csr_matrix]:
        """
        Generate a sample of the dataset.

        :param use_qmc: Whether quasi-monte carlo sampling is used.
        :returns: The syndrome and logical error. If used for ldpc library additionally return errors.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_batch(
        self, use_qmc: bool, device: torch.device
    ) -> tuple[Tensor, Tensor, csr_matrix] | tuple[Tensor, Tensor]:
        """
        Generate the dataset.

        :param use_qmc: Whether quasi-monte carlo sampling is used.
        :param device: The device that uses the data.
        :returns: The syndrome and logical error. If used for ldpc library additionally return errors.
        """

        # MAKE THE RETURNING SHAPE TO (b, 2=X/Z, r=L, L, L)
        self._verbose_print("\tGenerating Errors")
        syndrome_matrices, logical_errors, errors = self._generate_sample(use_qmc)

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
        for_ldpc=False,
        noise_model="capacity",
        measurement_error_rate=0,
    ):
        super().__init__(
            code,
            error_rate,
            batch_size,
            categorical_classification,
            one_hot,
            verbose,
            for_ldpc,
            noise_model,
            measurement_error_rate,
        )

    def _generate_sample(self, use_qmc: bool) -> tuple[NDArray, NDArray, csr_matrix]:
        errors = sample_errors(self.error_rate, self.n, use_qmc, self.batch_size)
        self._verbose_print("\tConstructing Syndrome Matrices")
        syndrome_matrices = generate_syndrome(self.stabilizers, errors)
        self._verbose_print("\tMeasuring Logicals")
        logical_errors = get_logical_errors(self.logicals, errors)
        return syndrome_matrices, logical_errors, errors


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
        )

    def _initialize_circuit(self) -> None:
        return None

    def _generate_sample(self, use_qmc):
        num_qubits = self.n
        repetitions = self.L
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

            noise_new = np.concat((errors_x, errors_z), axis=1)
            noise_cumulative = (np.cumsum(noise_new, 0) % 2).astype(np.uint8)
            noise_total = noise_cumulative[-1, :]
            syndrome = (H @ noise_cumulative.T).T % 2
            syndrome_error = (np.random.rand(repetitions, num_stabilisers) < q).astype(
                np.uint8
            )
            syndrome_error[:, -1] = (
                0  # Perfect measurements in last round to ensure even parity
            )
            noisy_syndrome = (syndrome + syndrome_error) % 2
            logical_error = (self.logicals @ noise_total.T).T % 2
            detectors.append(noisy_syndrome)
            observables.append(logical_error)

        detectors = np.array(detectors)
        detectors = detectors.reshape(
            self.batch_size, repetitions, 2, num_stabilisers // 2
        ).transpose(
            0, 2, 1, 3
        )  # Reshape to (b, 2=X/Z, r=L, num_stabilisers//2)

        return np.array(detectors), np.array(observables), None


"""

        if self.noise_model in ["circuit", "phenomenological"]:
            # Initialize Stim Circuits and Samplers
            generator = (
                generate_stim_circuit
                if self.noise_model == "circuit"
                else generate_phenomenological_circuit
            )

            self.circuit_z = generator(
                code,
                rounds=self.L,
                p=self.error_rate,
                q=self._measurement_error_rate,
                basis="Z",
            )
            self.sampler_z = self.circuit_z.compile_detector_sampler()

            self.circuit_x = generator(
                code,
                rounds=self.L,
                p=self.error_rate,
                q=self._measurement_error_rate,
                basis="X",
            )
            self.sampler_x = self.circuit_x.compile_detector_sampler()


            # Stim-based generation with mixed basis
            # Z-basis samples (Observes Logical Z, Errors X)
            dets_z, obs_z = self.sampler_z.sample(
                shots=self.batch_size, separate_observables=True
            )

            # X-basis samples (Observes Logical X, Errors Z)
            dets_x, obs_x = self.sampler_x.sample(
                shots=self.batch_size, separate_observables=True
            )

            # Pad to match total logicals (usually 4)
            num_logicals = self.logicals.shape[0]

            if obs_z.shape[1] < num_logicals:
                obs_z = np.pad(
                    obs_z, ((0, 0), (0, num_logicals - obs_z.shape[1])), "constant"
                )

            if obs_x.shape[1] < num_logicals:
                obs_x = np.pad(
                    obs_x, ((0, 0), (0, num_logicals - obs_x.shape[1])), "constant"
                )

                # Combine
                detectors.append(np.stack((dets_z, dets_x), axis=0))
                observables.append(np.stack((obs_z, obs_x), axis=0))

            # Format syndromes
            detectors = np.array(detectors)
            syndrome_matrices = detectors

            # Format logical errors
            logical_errors = np.array(observables).astype(np.uint8)

"""
