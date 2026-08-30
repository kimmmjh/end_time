"""Orbit-equivariant binary neural belief propagation on a circuit-level DEM.

This module is the circuit-level counterpart of
:class:`models._equivariant_neural_bp.EquivariantNeuralBP4`.  Three things
change relative to the code-capacity decoder, and each is forced by the
detector error model rather than chosen:

``binary messages``
    A DEM mechanism either fires or it does not, so every message is a scalar
    log-likelihood ratio instead of a four-state Pauli belief.  The quaternary
    structure that let BP4 retain the X/Z correlation of a ``Y`` error is
    already accounted for by Stim when it factorises circuit faults into
    mechanisms.

``irregular degree``
    A BB detector is touched by between roughly ninety and two hundred and
    fifty mechanisms.  The code-capacity decoder handled irregular graphs with
    a Python loop over checks; at circuit scale that loop dominates runtime, so
    the check update here is fully segmented with scatter reductions.

``shared network plus orbit embedding``
    One residual MLP is evaluated over every edge at once and is conditioned
    on a learned embedding of the edge's orbit, instead of instantiating one
    MLP per orbit.  This keeps the parameter count independent of how finely
    the orbits are resolved, and makes the ``global``/``orbit``/``edge``
    sharing ablation a change of index tensor rather than of architecture.

The base algorithm is normalised min-sum, matching the ``bp_method="ms"``,
``ms_scaling_factor=0.625`` configuration used by the classical ``ldpc``
baselines, so a paired ``neural=False`` run is exactly the classical decoder's
belief-propagation stage.  The residual head is zero-initialised and the
relaxation starts at one, so an untrained model reproduces that baseline
bitwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

# Number of hand-built per-edge features fed to the residual network.
EDGE_FEATURE_DIM = 8


class EquivariantNeuralBP2(nn.Module):
    """Neural min-sum BP over a circuit-level detector error model.

    Parameters
    ----------
    graph:
        A :class:`src.bb_dem.BBDemGraph` describing detectors, mechanisms,
        Tanner edges, per-edge orbit ids and mechanism priors.
    iterations:
        Number of unrolled BP iterations.  This is algorithmic iteration, not
        a measurement round: the rounds are already inside the graph.
    hidden_dim:
        Width of the shared residual MLP.
    orbit_embedding_dim:
        Width of the learned per-orbit embedding.  Zero removes the embedding
        entirely, which makes every edge share one update.
    sharing:
        ``orbit`` keys the embedding by the DEM orbit id and is the equivariant
        default.  ``global`` ties every edge together.  ``edge`` gives each
        Tanner edge its own embedding row and intentionally breaks
        equivariance.  Unlike the code-capacity decoder, ``edge`` costs one
        embedding row rather than one MLP, so it is actually runnable.
    normalisation:
        Fixed min-sum scaling factor applied before any learned term.
    residual_scale:
        Maximum magnitude of the learned additive residual after ``tanh``.
    max_relaxation_delta:
        The update coefficient is ``1 + max_relaxation_delta * tanh(raw)``.
    message_clip:
        Messages and posteriors are clamped to this magnitude each iteration.
    gradient_checkpoint:
        Recompute each BP iteration during the backward pass.  Circuit-level
        graphs have hundreds of thousands of edges, so storing activations for
        every unrolled iteration is usually infeasible.
    """

    def __init__(
        self,
        graph: Any,
        *,
        iterations: int = 12,
        hidden_dim: int = 32,
        orbit_embedding_dim: int = 8,
        sharing: str = "orbit",
        normalisation: float = 0.625,
        residual_scale: float = 2.0,
        max_relaxation_delta: float = 0.5,
        message_clip: float = 30.0,
        gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if orbit_embedding_dim < 0:
            raise ValueError("orbit_embedding_dim must be non-negative.")
        if sharing not in {"orbit", "global", "edge"}:
            raise ValueError(
                f"sharing must be orbit, global or edge, got {sharing!r}."
            )
        if not 0.0 < normalisation <= 1.0:
            raise ValueError("normalisation must lie in (0, 1].")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if not 0.0 <= max_relaxation_delta < 1.0:
            raise ValueError("max_relaxation_delta must lie in [0, 1).")
        if message_clip <= 0.0:
            raise ValueError("message_clip must be positive.")

        edge_detector = torch.as_tensor(
            np.asarray(graph.edge_detector), dtype=torch.long
        )
        edge_mechanism = torch.as_tensor(
            np.asarray(graph.edge_mechanism), dtype=torch.long
        )
        edge_orbit = torch.as_tensor(np.asarray(graph.edge_orbit), dtype=torch.long)
        if not (edge_detector.numel() == edge_mechanism.numel() == edge_orbit.numel()):
            raise ValueError("Edge arrays must all have the same length.")

        self.num_detectors = int(graph.num_detectors)
        self.num_mechanisms = int(graph.num_mechanisms)
        self.num_observables = int(graph.num_observables)
        self.num_edges = int(edge_detector.numel())
        self.iterations = int(iterations)
        self.sharing = sharing
        self.normalisation = float(normalisation)
        self.residual_scale = float(residual_scale)
        self.max_relaxation_delta = float(max_relaxation_delta)
        self.message_clip = float(message_clip)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        if sharing == "orbit":
            sharing_index = edge_orbit
            num_groups = int(graph.num_orbits)
        elif sharing == "global":
            sharing_index = torch.zeros_like(edge_orbit)
            num_groups = 1
        else:
            sharing_index = torch.arange(self.num_edges, dtype=torch.long)
            num_groups = self.num_edges
        self.num_groups = num_groups
        self.orbit_embedding_dim = int(orbit_embedding_dim)

        self.register_buffer("edge_detector", edge_detector)
        self.register_buffer("edge_mechanism", edge_mechanism)
        self.register_buffer("edge_orbit", edge_orbit)
        self.register_buffer("sharing_index", sharing_index)

        prior_log_odds = torch.as_tensor(
            np.asarray(graph.prior_log_odds), dtype=torch.float32
        )
        if prior_log_odds.numel() != self.num_mechanisms:
            raise ValueError("prior_log_odds must have one entry per mechanism.")
        self.register_buffer("prior_log_odds", prior_log_odds)

        degrees = torch.zeros(self.num_detectors, dtype=torch.float32).index_add(
            0, edge_detector, torch.ones(self.num_edges, dtype=torch.float32)
        )
        if torch.any(degrees == 0):
            raise ValueError("The DEM contains a detector with no mechanism.")
        # Scaled to O(1) so the feature does not dominate LayerNorm statistics.
        self.register_buffer("detector_degree", degrees / degrees.mean())

        feature_dim = EDGE_FEATURE_DIM + self.orbit_embedding_dim
        if self.orbit_embedding_dim:
            self.orbit_embedding = nn.Embedding(num_groups, self.orbit_embedding_dim)
            nn.init.normal_(self.orbit_embedding.weight, std=0.1)
        else:
            self.orbit_embedding = None
        self.residual_network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero output makes an untrained model exactly normalised min-sum.
        final = self.residual_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.relaxation_raw = nn.Parameter(torch.zeros(num_groups))

    # ------------------------------------------------------------------
    # Segmented primitives
    # ------------------------------------------------------------------
    def _expand_index(self, index: Tensor, batch_size: int) -> Tensor:
        return index.unsqueeze(0).expand(batch_size, -1)

    def _segment_min_two(
        self, magnitudes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Smallest and second smallest magnitude incident on every detector.

        Returns per-edge tensors ``(min_excluding_edge, smallest, is_argmin)``.
        Ties are broken by edge position so that exactly one edge per detector
        is treated as the argument of the minimum.
        """

        batch_size = magnitudes.shape[0]
        index = self._expand_index(self.edge_detector, batch_size)
        infinity = torch.full(
            (batch_size, self.num_detectors),
            float("inf"),
            device=magnitudes.device,
            dtype=magnitudes.dtype,
        )
        smallest = infinity.scatter_reduce(
            1, index, magnitudes, reduce="amin", include_self=True
        )
        smallest_per_edge = smallest.gather(1, index)

        positions = torch.arange(
            self.num_edges, device=magnitudes.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        candidate = torch.where(
            magnitudes == smallest_per_edge,
            positions,
            torch.full_like(positions, self.num_edges),
        )
        first = torch.full(
            (batch_size, self.num_detectors),
            self.num_edges,
            device=magnitudes.device,
            dtype=torch.long,
        ).scatter_reduce(1, index, candidate, reduce="amin", include_self=True)
        is_argmin = positions == first.gather(1, index)

        masked = torch.where(is_argmin, torch.full_like(magnitudes, float("inf")), magnitudes)
        second = infinity.scatter_reduce(
            1, index, masked, reduce="amin", include_self=True
        )
        second_per_edge = second.gather(1, index)
        # A degree-one detector has no second minimum; an infinite magnitude
        # there correctly means "no constraint from the other edges".
        excluding = torch.where(is_argmin, second_per_edge, smallest_per_edge)
        return excluding, smallest_per_edge, is_argmin

    def _segment_sign_parity(self, messages: Tensor) -> Tensor:
        batch_size = messages.shape[0]
        negatives = (messages < 0).to(messages.dtype)
        parity = torch.zeros(
            (batch_size, self.num_detectors),
            device=messages.device,
            dtype=messages.dtype,
        ).index_add(1, self.edge_detector, negatives)
        return 1.0 - 2.0 * torch.remainder(parity, 2.0)

    # ------------------------------------------------------------------
    # Belief propagation
    # ------------------------------------------------------------------
    def _check_update(self, variable_messages: Tensor, syndrome: Tensor) -> Tensor:
        """Normalised min-sum detector-to-mechanism messages."""

        magnitudes = variable_messages.abs()
        excluding, smallest, _ = self._segment_min_two(magnitudes)
        # An isolated infinite second minimum would poison the arithmetic.
        excluding = torch.clamp(excluding, max=self.message_clip)

        parity = self._segment_sign_parity(variable_messages)
        edge_parity = parity.gather(1, self._expand_index(self.edge_detector, parity.shape[0]))
        edge_sign = torch.where(
            variable_messages >= 0,
            torch.ones_like(variable_messages),
            -torch.ones_like(variable_messages),
        )
        syndrome_sign = 1.0 - 2.0 * syndrome
        edge_syndrome_sign = syndrome_sign.gather(
            1, self._expand_index(self.edge_detector, syndrome.shape[0])
        )
        sign = edge_parity * edge_sign * edge_syndrome_sign
        del smallest
        return self.normalisation * sign * excluding

    def _variable_update(self, check_messages: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = check_messages.shape[0]
        aggregate = torch.zeros(
            (batch_size, self.num_mechanisms),
            device=check_messages.device,
            dtype=check_messages.dtype,
        ).index_add(1, self.edge_mechanism, check_messages)
        posterior = self.prior_log_odds.unsqueeze(0) + aggregate
        posterior = posterior.clamp(-self.message_clip, self.message_clip)
        index = self._expand_index(self.edge_mechanism, batch_size)
        variable_messages = posterior.gather(1, index) - check_messages
        variable_messages = variable_messages.clamp(
            -self.message_clip, self.message_clip
        )
        return posterior, variable_messages

    def _residual(
        self,
        exact: Tensor,
        variable_messages: Tensor,
        previous: Tensor,
        syndrome: Tensor,
        posterior: Tensor,
    ) -> Tensor:
        batch_size = exact.shape[0]
        detector_index = self._expand_index(self.edge_detector, batch_size)
        mechanism_index = self._expand_index(self.edge_mechanism, batch_size)

        features = torch.stack(
            (
                exact,
                variable_messages,
                previous,
                syndrome.gather(1, detector_index),
                self.prior_log_odds.unsqueeze(0).expand(batch_size, -1).gather(
                    1, mechanism_index
                ),
                posterior.gather(1, mechanism_index),
                self.detector_degree.unsqueeze(0)
                .expand(batch_size, -1)
                .gather(1, detector_index),
                variable_messages.abs(),
            ),
            dim=-1,
        )
        if self.orbit_embedding is not None:
            embedding = self.orbit_embedding(self.sharing_index)
            features = torch.cat(
                (features, embedding.unsqueeze(0).expand(batch_size, -1, -1)), dim=-1
            )
        return self.residual_scale * torch.tanh(
            self.residual_network(features).squeeze(-1)
        )

    def _iteration(
        self,
        variable_messages: Tensor,
        check_messages: Tensor,
        posterior: Tensor,
        syndrome: Tensor,
        neural: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        exact = self._check_update(variable_messages, syndrome)
        if bool(neural.item()):
            residual = self._residual(
                exact, variable_messages, check_messages, syndrome, posterior
            )
            relaxation = 1.0 + self.max_relaxation_delta * torch.tanh(
                self.relaxation_raw
            )
            coefficient = relaxation[self.sharing_index].unsqueeze(0)
            # Written around coefficient one so the initialised path equals the
            # exact min-sum message bitwise.
            updated = exact + (coefficient - 1.0) * (exact - check_messages) + residual
        else:
            updated = exact
        updated = updated.clamp(-self.message_clip, self.message_clip)
        posterior, variable_messages = self._variable_update(updated)
        return variable_messages, updated, posterior

    def forward(
        self,
        syndrome: Tensor,
        *,
        neural: bool = True,
        return_all: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Decode a batch of detector frames.

        Parameters
        ----------
        syndrome:
            Binary detector events with shape ``[batch, num_detectors]``.
        neural:
            If false, bypass the learned residual and relaxation without
            mutating the model, giving normalised min-sum BP on the same shots.
        return_all:
            Also return every iteration's posterior, shaped
            ``[batch, iterations, num_mechanisms]``, for deep supervision.

        Returns
        -------
        Tensor
            Posterior log-likelihood ratios per mechanism.  A positive value
            means the mechanism most likely did **not** fire.
        """

        if syndrome.ndim != 2 or syndrome.shape[1] != self.num_detectors:
            raise ValueError(
                "syndrome must have shape "
                f"(batch, {self.num_detectors}), got {tuple(syndrome.shape)}."
            )
        if syndrome.device != self.edge_detector.device:
            raise ValueError(
                "syndrome and model must be on the same device; move one "
                "explicitly before decoding."
            )
        dtype = self.relaxation_raw.dtype
        syndrome_float = syndrome.to(dtype=dtype)
        if not torch.all((syndrome_float == 0) | (syndrome_float == 1)):
            raise ValueError("syndrome must contain only binary values.")

        batch_size = syndrome_float.shape[0]
        posterior = self.prior_log_odds.unsqueeze(0).expand(batch_size, -1)
        variable_messages = posterior.gather(
            1, self._expand_index(self.edge_mechanism, batch_size)
        )
        check_messages = torch.zeros_like(variable_messages)
        neural_flag = torch.tensor(bool(neural))

        history: list[Tensor] = []
        for _ in range(self.iterations):
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                variable_messages, check_messages, posterior = checkpoint(
                    self._iteration,
                    variable_messages,
                    check_messages,
                    posterior,
                    syndrome_float,
                    neural_flag,
                    use_reentrant=False,
                )
            else:
                variable_messages, check_messages, posterior = self._iteration(
                    variable_messages,
                    check_messages,
                    posterior,
                    syndrome_float,
                    neural_flag,
                )
            if return_all:
                history.append(posterior)

        if return_all:
            return posterior, torch.stack(history, dim=1)
        return posterior

    @staticmethod
    def hard_decision(posterior: Tensor) -> Tensor:
        """Mechanisms whose posterior LLR is negative are declared fired."""

        return (posterior < 0).to(torch.uint8)


__all__ = ["EquivariantNeuralBP2", "EDGE_FEATURE_DIM"]
