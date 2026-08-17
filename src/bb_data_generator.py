"""Code-capacity Pauli samplers for bivariate-bicycle codes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .bb_code import BBCodeSpec

# Public and deliberately fixed: it makes class logits and labels unambiguous.
PAULI_TO_INDEX = {"I": 0, "X": 1, "Y": 2, "Z": 3}
INDEX_TO_PAULI = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class BBCodeCapacityBatch:
    """A neural-BP batch.

    ``syndrome`` is ordered as all X-check outcomes followed by all Z-check
    outcomes.  Consequently,

    * ``syndrome[:, :num_x_checks] = Hx @ z_error``; and
    * ``syndrome[:, num_x_checks:] = Hz @ x_error``.

    ``logical`` stores the residual labels in ``[logical X | logical Z]``
    order.  All binary neural features/targets are float tensors, while
    ``pauli`` is a long tensor with ``I=0, X=1, Y=2, Z=3``.
    """

    syndrome: Tensor
    pauli: Tensor
    x_error: Tensor
    z_error: Tensor
    logical: Tensor
    channel_probabilities: Tensor
    num_x_checks: int
    k: int

    @property
    def syndrome_x_checks(self) -> Tensor:
        return self.syndrome[:, : self.num_x_checks]

    @property
    def syndrome_z_checks(self) -> Tensor:
        return self.syndrome[:, self.num_x_checks :]

    @property
    def logical_x(self) -> Tensor:
        return self.logical[:, : self.k]

    @property
    def logical_z(self) -> Tensor:
        return self.logical[:, self.k :]

    def __iter__(self) -> Iterator[Tensor]:
        """Allow the common ``syndrome, pauli = batch`` shorthand."""

        yield self.syndrome
        yield self.pauli


class BBCodeCapacityGenerator:
    """Generate exact code-capacity samples for a BB CSS code.

    ``depolarizing`` samples ``I/X/Y/Z`` with probabilities
    ``(1-p, p/3, p/3, p/3)``.  Locations are independent, but the X and Z
    components on one qubit are coupled by the O(p) probability of Y.

    ``independent_xz`` instead samples X and Z components as independent
    Bernoulli variables.  Its probability of any non-identity Pauli is
    ``p_x + p_z - p_x*p_z`` rather than the constructor's scalar
    ``error_rate``.  This distinction is intentional for correlation studies.
    """

    def __init__(
        self,
        code: BBCodeSpec,
        error_rate: float,
        batch_size: int | None = None,
        *,
        noise_model: str = "depolarizing",
        x_error_rate: float | None = None,
        z_error_rate: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.code = code
        self.error_rate = self._validate_probability("error_rate", error_rate)
        self.batch_size = batch_size
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        normalized = noise_model.lower().replace("-", "_")
        if normalized == "independent":
            normalized = "independent_xz"
        if normalized not in {"depolarizing", "independent_xz"}:
            raise ValueError(
                "noise_model must be 'depolarizing' or 'independent_xz', "
                f"got {noise_model!r}."
            )
        self.noise_model = normalized
        self.x_error_rate = self._validate_probability(
            "x_error_rate",
            error_rate if x_error_rate is None else x_error_rate,
        )
        self.z_error_rate = self._validate_probability(
            "z_error_rate",
            error_rate if z_error_rate is None else z_error_rate,
        )
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_probability(name: str, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return value

    @property
    def channel_probabilities(self) -> NDArray[np.float64]:
        """Return channel probabilities in ``I, X, Y, Z`` order."""

        if self.noise_model == "depolarizing":
            p = self.error_rate
            return np.asarray((1.0 - p, p / 3.0, p / 3.0, p / 3.0))

        px = self.x_error_rate
        pz = self.z_error_rate
        return np.asarray(
            (
                (1.0 - px) * (1.0 - pz),
                px * (1.0 - pz),
                px * pz,
                (1.0 - px) * pz,
            )
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize sampler state so resumed training does not replay shots."""

        return {
            "version": 1,
            "code_name": self.code.name,
            "code_n": self.code.n,
            "hx_shape": tuple(self.code.hx.shape),
            "hz_shape": tuple(self.code.hz.shape),
            "noise_model": self.noise_model,
            "channel_probabilities": self.channel_probabilities.copy(),
            "bit_generator": type(self._rng.bit_generator).__name__,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict` with validation."""

        required = {
            "version",
            "code_name",
            "code_n",
            "hx_shape",
            "hz_shape",
            "noise_model",
            "channel_probabilities",
            "bit_generator",
            "rng_state",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "Generator checkpoint is missing keys: " + ", ".join(sorted(missing))
            )
        if state["version"] != 1:
            raise ValueError(
                f"Unsupported generator checkpoint version {state['version']!r}."
            )

        checkpoint_signature = (
            int(state["code_n"]),
            tuple(state["hx_shape"]),
            tuple(state["hz_shape"]),
        )
        current_signature = (
            self.code.n,
            tuple(self.code.hx.shape),
            tuple(self.code.hz.shape),
        )
        if (
            state["code_name"] != self.code.name
            or checkpoint_signature != current_signature
        ):
            raise ValueError(
                "Generator checkpoint code identity/shape does not match this "
                f"code: checkpoint={(state['code_name'], checkpoint_signature)}, "
                f"current={(self.code.name, current_signature)}."
            )
        if state["noise_model"] != self.noise_model or not np.array_equal(
            np.asarray(state["channel_probabilities"], dtype=np.float64),
            self.channel_probabilities,
        ):
            raise ValueError(
                "Generator checkpoint channel does not match the configured "
                "noise model/probabilities."
            )

        current_bit_generator = type(self._rng.bit_generator).__name__
        if state["bit_generator"] != current_bit_generator:
            raise ValueError(
                "Generator checkpoint uses a different NumPy bit generator: "
                f"checkpoint={state['bit_generator']!r}, "
                f"current={current_bit_generator!r}."
            )
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def sample(
        self, batch_size: int | None = None, *, device: torch.device | str = "cpu"
    ) -> BBCodeCapacityBatch:
        size = self.batch_size if batch_size is None else batch_size
        if size is None:
            raise ValueError(
                "batch_size must be passed to sample() or set in the constructor."
            )
        if size < 1:
            raise ValueError(f"batch_size must be positive, got {size}.")

        if self.noise_model == "depolarizing":
            pauli = self._rng.choice(
                4, size=(size, self.code.n), p=self.channel_probabilities
            ).astype(np.int64)
        else:
            x_error = self._rng.random((size, self.code.n)) < self.x_error_rate
            z_error = self._rng.random((size, self.code.n)) < self.z_error_rate
            pauli = np.zeros((size, self.code.n), dtype=np.int64)
            pauli[x_error & ~z_error] = PAULI_TO_INDEX["X"]
            pauli[x_error & z_error] = PAULI_TO_INDEX["Y"]
            pauli[~x_error & z_error] = PAULI_TO_INDEX["Z"]

        return self.batch_from_pauli(pauli, device=device)

    def generate_batch(self, device: torch.device | str) -> BBCodeCapacityBatch:
        """Generate the constructor-configured batch size."""

        return self.sample(device=device)

    def batch_from_pauli(
        self,
        pauli: NDArray[np.generic],
        *,
        device: torch.device | str = "cpu",
    ) -> BBCodeCapacityBatch:
        """Construct exact syndrome/logical targets for supplied Pauli labels."""

        pauli_array = np.asarray(pauli, dtype=np.int64)
        if pauli_array.ndim != 2 or pauli_array.shape[1] != self.code.n:
            raise ValueError(
                f"pauli must have shape (batch, {self.code.n}), got "
                f"{pauli_array.shape}."
            )
        if np.any((pauli_array < 0) | (pauli_array > 3)):
            raise ValueError("pauli entries must use I=0, X=1, Y=2, Z=3.")

        x_error = np.isin(
            pauli_array, (PAULI_TO_INDEX["X"], PAULI_TO_INDEX["Y"])
        ).astype(np.uint8)
        z_error = np.isin(
            pauli_array, (PAULI_TO_INDEX["Z"], PAULI_TO_INDEX["Y"])
        ).astype(np.uint8)

        # X checks anti-commute with Z components; Z checks anti-commute with X.
        syndrome_x = (z_error @ self.code.hx.T) % 2
        syndrome_z = (x_error @ self.code.hz.T) % 2
        syndrome = np.concatenate((syndrome_x, syndrome_z), axis=1)

        # A physical X residual anti-commutes with logical Z, and vice versa.
        logical_x = (x_error @ self.code.logicals_z.T) % 2
        logical_z = (z_error @ self.code.logicals_x.T) % 2
        logical = np.concatenate((logical_x, logical_z), axis=1)

        return BBCodeCapacityBatch(
            syndrome=torch.as_tensor(syndrome, dtype=torch.float32, device=device),
            pauli=torch.as_tensor(pauli_array, dtype=torch.long, device=device),
            x_error=torch.as_tensor(x_error, dtype=torch.float32, device=device),
            z_error=torch.as_tensor(z_error, dtype=torch.float32, device=device),
            logical=torch.as_tensor(logical, dtype=torch.float32, device=device),
            channel_probabilities=torch.as_tensor(
                self.channel_probabilities, dtype=torch.float32, device=device
            ),
            num_x_checks=self.code.num_x_checks,
            k=self.code.k,
        )
