"""Circuit-level BB sampling for neural detector-error-model decoding.

Two Stim samplers are used deliberately, mirroring the toric circuit path.
Training targets come from the detector error model sampled with
``return_errors=True``, which reveals which fault mechanisms actually fired.
Validation and final accuracy instead use the compiled *circuit* detector
sampler, so reported decoding accuracy is measured on fresh exact circuit
shots rather than on latent DEM labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .bb_code import BBCodeSpec
from .bb_dem import (
    BBDemGraph,
    build_bb_dem_graph,
    detector_error_model_fingerprint,
)
from .bb_stim_utils import CIRCUIT_SCHEMA_VERSION, generate_bb_memory_circuit

try:
    import stim
except ImportError:  # Keep code-capacity noise usable without Stim.
    stim = None


@dataclass(frozen=True)
class BBCircuitBatch:
    """One batch of circuit-level detector frames.

    ``mechanisms`` is ``None`` for batches drawn from the circuit sampler,
    because the exact circuit shot does not expose which DEM mechanism fired.
    """

    detectors: Tensor
    observables: Tensor
    mechanisms: Tensor | None

    def __iter__(self) -> Iterator[Tensor]:
        yield self.detectors
        yield self.observables


class BBCircuitGenerator:
    """Sample circuit-level BB shots for training and evaluation.

    Parameters
    ----------
    code:
        A :class:`BBCodeSpec` or its name.
    rounds:
        Number of noisy syndrome-extraction cycles.  Defaults to the code
        distance.  The generator adds one perfect closing detector frame.
    gate_error_rate, measurement_error_rate, idle_error_rate:
        Circuit noise strengths, forwarded to the circuit generator.
    batch_size:
        Shots per generated batch.
    seed:
        Base seed.  Distinct streams are derived for the DEM and circuit
        samplers so training and evaluation never share shots.
    """

    def __init__(
        self,
        code: BBCodeSpec | str,
        *,
        rounds: int | None = None,
        gate_error_rate: float,
        measurement_error_rate: float,
        idle_error_rate: float = 0.0,
        batch_size: int = 32,
        seed: int | None = None,
        graph: BBDemGraph | None = None,
    ) -> None:
        if stim is None:
            raise ImportError(
                "Circuit-level BB sampling requires Stim. Install the project "
                "dependencies with `pip install -r requirements.txt`."
            )
        if isinstance(code, str):
            code = BBCodeSpec.from_name(code)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        resolved_rounds = int(code.d if rounds is None else rounds)
        if resolved_rounds < 1:
            raise ValueError("rounds must be positive.")

        self.code = code
        self.rounds = resolved_rounds
        self.batch_size = int(batch_size)
        self.gate_error_rate = float(gate_error_rate)
        self.measurement_error_rate = float(measurement_error_rate)
        self.idle_error_rate = float(idle_error_rate)

        self.circuit = generate_bb_memory_circuit(
            code,
            rounds=resolved_rounds,
            gate_error_rate=gate_error_rate,
            measurement_error_rate=measurement_error_rate,
            idle_error_rate=idle_error_rate,
        )
        self.detector_error_model = self.circuit.detector_error_model(
            decompose_errors=False, allow_gauge_detectors=False
        )
        if graph is None:
            self.graph = build_bb_dem_graph(
                code,
                rounds=resolved_rounds,
                gate_error_rate=gate_error_rate,
                measurement_error_rate=measurement_error_rate,
                idle_error_rate=idle_error_rate,
                circuit=self.circuit,
            )
        else:
            self._validate_supplied_graph(graph)
            self.graph = graph

        sequence = np.random.SeedSequence(seed)
        dem_seed, circuit_seed = sequence.spawn(2)
        self._dem_sampler = self.detector_error_model.compile_sampler(
            seed=int(dem_seed.generate_state(1, dtype=np.uint32)[0])
        )
        self._circuit_sampler = self.circuit.compile_detector_sampler(
            seed=int(circuit_seed.generate_state(1, dtype=np.uint32)[0])
        )

    @property
    def num_detectors(self) -> int:
        return self.graph.num_detectors

    @property
    def detector_frames(self) -> int:
        return self.graph.detector_frames

    @property
    def num_mechanisms(self) -> int:
        return self.graph.num_mechanisms

    @property
    def num_observables(self) -> int:
        return self.graph.num_observables

    def _validate_supplied_graph(self, graph: BBDemGraph) -> None:
        """Reject graph reuse across a different code, circuit or noise rate."""

        mismatches: list[str] = []
        expected = {
            "code_name": self.code.name,
            "circuit_schema_version": CIRCUIT_SCHEMA_VERSION,
            "rounds": self.rounds,
            "detector_frames": self.rounds + 1,
            "num_detectors": int(self.detector_error_model.num_detectors),
            "num_observables": int(self.detector_error_model.num_observables),
            "dem_fingerprint": detector_error_model_fingerprint(
                self.detector_error_model
            ),
        }
        for key, value in expected.items():
            if getattr(graph, key, None) != value:
                mismatches.append(
                    f"{key}: graph={getattr(graph, key, None)!r}, circuit={value!r}"
                )
        if mismatches:
            raise ValueError(
                "Supplied BB DEM graph is incompatible with this circuit:\n  "
                + "\n  ".join(mismatches)
            )

    def _to_tensor(
        self, array: NDArray[Any], device: Any, dtype: Any = torch.float32
    ) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)

    def sample_dem(
        self, batch_size: int | None = None, *, device: Any = None
    ) -> BBCircuitBatch:
        """Sample the DEM, exposing the fault mechanisms that fired."""

        shots = self.batch_size if batch_size is None else int(batch_size)
        if shots < 1:
            raise ValueError("batch_size must be positive.")
        detectors, observables, errors = self._dem_sampler.sample(
            shots=shots, return_errors=True
        )
        folded = self.graph.fold_errors(np.asarray(errors, dtype=np.uint8))
        return BBCircuitBatch(
            detectors=self._to_tensor(np.asarray(detectors, dtype=np.uint8), device),
            observables=self._to_tensor(
                np.asarray(observables, dtype=np.uint8), device
            ),
            mechanisms=self._to_tensor(folded, device),
        )

    def sample_circuit(
        self, batch_size: int | None = None, *, device: Any = None
    ) -> BBCircuitBatch:
        """Sample fresh exact circuit shots for honest evaluation."""

        shots = self.batch_size if batch_size is None else int(batch_size)
        if shots < 1:
            raise ValueError("batch_size must be positive.")
        detectors, observables = self._circuit_sampler.sample(
            shots=shots, separate_observables=True
        )
        return BBCircuitBatch(
            detectors=self._to_tensor(np.asarray(detectors, dtype=np.uint8), device),
            observables=self._to_tensor(
                np.asarray(observables, dtype=np.uint8), device
            ),
            mechanisms=None,
        )

    def generate_batches(
        self, batches: int, *, source: str = "dem", device: Any = None
    ) -> Iterator[BBCircuitBatch]:
        if source not in {"dem", "circuit"}:
            raise ValueError(f"source must be 'dem' or 'circuit', got {source!r}.")
        sampler = self.sample_dem if source == "dem" else self.sample_circuit
        for _ in range(int(batches)):
            yield sampler(device=device)


__all__ = ["BBCircuitBatch", "BBCircuitGenerator"]
