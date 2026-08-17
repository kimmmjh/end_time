"""Orbit-equivariant quaternary neural belief propagation for CSS codes.

The decoder in this module is deliberately independent of the toric-code CNN
stack.  It operates directly on the Tanner graph of ``Hx`` and ``Hz`` and
keeps a four-state ``I, X, Y, Z`` distribution on every variable-to-check
message.  Consequently, a depolarizing prior retains the correlation between
the X and Z components of a Y error.

The exact sum-product BP4 update is always computed first.  A small residual
network and a learnable relaxation coefficient may then modify each
check-to-variable message.  Their parameters are shared by Tanner-edge orbit
and check type, so graph automorphisms which preserve those types commute with
the decoder.  Both neural modifications are initialized to the identity; a
new model therefore starts as vanilla BP4 exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

# Public Pauli ordering used by every input and output of the model.
PAULI_ORDER = ("I", "X", "Y", "Z")


def _binary_matrix(matrix: Any, *, name: str) -> Tensor:
    """Convert a dense or SciPy-sparse binary matrix to a CPU tensor."""

    if isinstance(matrix, Tensor):
        dense = matrix.detach().cpu()
    elif hasattr(matrix, "toarray"):
        dense = torch.as_tensor(np.asarray(matrix.toarray()))
    else:
        dense = torch.as_tensor(np.asarray(matrix))

    if dense.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix, got {tuple(dense.shape)}.")
    if dense.is_floating_point() and not torch.all(dense == dense.round()):
        raise ValueError(f"{name} must contain only binary entries.")
    dense = dense.to(torch.int64)
    if not torch.all((dense == 0) | (dense == 1)):
        raise ValueError(f"{name} must contain only 0 and 1.")
    return dense


class _OrbitResidual(nn.Module):
    """One shared residual message function for an edge orbit."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Features are exact C2V, current V2C, previous C2V (four each), and
        # the measured syndrome bit of the incident check.
        feature_dim = 4 + 4 + 4 + 1
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        # Zero output makes an untrained model *exactly* vanilla BP4 rather
        # than merely close to it.
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class EquivariantNeuralBP4(nn.Module):
    """Quaternary neural BP with automorphism-orbit parameter sharing.

    Parameters
    ----------
    hx, hz:
        CSS X- and Z-stabilizer parity-check matrices.  ``Hx`` rows detect the
        Z component of an error and ``Hz`` rows detect its X component.  Both
        matrices must have the same number of columns and satisfy
        ``Hx @ Hz.T == 0 (mod 2)``.
    edge_orbits:
        Optional one-dimensional orbit id for every Tanner edge.  The edge
        order is deterministic: row-major nonzeros of ``Hx`` followed by
        row-major nonzeros of ``Hz``.  Sharing is keyed by the pair
        ``(check_type, edge_orbit)``, so X- and Z-check edges are not silently
        tied.  With ``None``, all edges of a given check type share one orbit.
    iterations:
        Number of unrolled BP iterations.  This is algorithmic iteration, not
        a measurement-time/recurrent dimension.
    residual_hidden_dim:
        Width of each orbit-shared residual MLP.
    residual_scale:
        Maximum magnitude of a learned residual log-message after ``tanh``.
    max_relaxation_delta:
        The update coefficient is
        ``1 + max_relaxation_delta * tanh(raw)``.  It starts at one (no
        damping), can learn under-relaxation, and can also learn mild
        over-relaxation.

    Notes
    -----
    Pauli logits always use the order ``(I, X, Y, Z)``.  There is no CNN,
    temporal GRU, MWPM, or logical-class head in this module.
    """

    def __init__(
        self,
        hx: Any,
        hz: Any,
        *,
        edge_orbits: Sequence[int] | Tensor | np.ndarray | None = None,
        iterations: int = 12,
        residual_hidden_dim: int = 64,
        residual_scale: float = 2.0,
        max_relaxation_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if residual_hidden_dim < 1:
            raise ValueError("residual_hidden_dim must be positive.")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if not 0.0 <= max_relaxation_delta < 1.0:
            raise ValueError("max_relaxation_delta must lie in [0, 1).")

        hx_tensor = _binary_matrix(hx, name="hx")
        hz_tensor = _binary_matrix(hz, name="hz")
        if hx_tensor.shape[1] != hz_tensor.shape[1]:
            raise ValueError(
                "hx and hz must have the same number of qubit columns, got "
                f"{hx_tensor.shape[1]} and {hz_tensor.shape[1]}."
            )
        if hx_tensor.shape[1] < 1:
            raise ValueError("The code must contain at least one qubit.")
        if hx_tensor.shape[0] + hz_tensor.shape[0] < 1:
            raise ValueError("The code must contain at least one stabilizer check.")
        if hx_tensor.shape[0] and torch.any(hx_tensor.sum(dim=1) == 0):
            raise ValueError("hx contains an empty stabilizer row.")
        if hz_tensor.shape[0] and torch.any(hz_tensor.sum(dim=1) == 0):
            raise ValueError("hz contains an empty stabilizer row.")
        if hx_tensor.shape[0] and hz_tensor.shape[0]:
            commutator = torch.remainder(hx_tensor @ hz_tensor.T, 2)
            if torch.any(commutator):
                raise ValueError("hx and hz do not define commuting CSS checks.")

        x_nonzero = torch.nonzero(hx_tensor, as_tuple=False)
        z_nonzero = torch.nonzero(hz_tensor, as_tuple=False)
        num_x_checks = int(hx_tensor.shape[0])
        num_z_checks = int(hz_tensor.shape[0])

        edge_check = torch.cat(
            (
                x_nonzero[:, 0],
                z_nonzero[:, 0] + num_x_checks,
            )
        )
        edge_qubit = torch.cat((x_nonzero[:, 1], z_nonzero[:, 1]))
        edge_check_type = torch.cat(
            (
                torch.zeros(x_nonzero.shape[0], dtype=torch.long),
                torch.ones(z_nonzero.shape[0], dtype=torch.long),
            )
        )
        check_node_type = torch.cat(
            (
                torch.zeros(num_x_checks, dtype=torch.long),
                torch.ones(num_z_checks, dtype=torch.long),
            )
        )
        num_edges = int(edge_check.numel())
        if num_edges < 1:
            raise ValueError("The Tanner graph contains no edges.")

        raw_orbits = self._validate_edge_orbits(edge_orbits, num_edges=num_edges)
        edge_type, edge_type_keys = self._compact_edge_types(
            edge_check_type, raw_orbits
        )
        edge_type_order = torch.argsort(edge_type, stable=True)
        edge_type_counts = torch.bincount(
            edge_type, minlength=len(edge_type_keys)
        ).tolist()
        edge_type_offsets = [0]
        for count in edge_type_counts:
            edge_type_offsets.append(edge_type_offsets[-1] + int(count))

        check_degrees = torch.bincount(
            edge_check, minlength=num_x_checks + num_z_checks
        )
        check_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.long), check_degrees.cumsum(dim=0))
        )
        degree_values = [int(degree) for degree in check_degrees.tolist()]
        offsets = [0]
        for degree in degree_values:
            offsets.append(offsets[-1] + degree)

        self.iterations = int(iterations)
        self.residual_scale = float(residual_scale)
        self.max_relaxation_delta = float(max_relaxation_delta)
        self.num_qubits = int(hx_tensor.shape[1])
        self.num_x_checks = num_x_checks
        self.num_z_checks = num_z_checks
        self.num_checks = num_x_checks + num_z_checks
        self.num_edges = num_edges
        self.num_edge_types = len(edge_type_keys)
        self.edge_type_keys = tuple(edge_type_keys)
        # BB checks normally all have degree six.  That permits one batched
        # GPU update over [batch, checks, degree, Pauli] rather than a Python
        # loop and thousands of small kernel launches.  The precomputed CPU
        # slices keep the generic irregular-code fallback free of GPU .item()
        # synchronizations.
        self.uniform_check_degree = (
            degree_values[0] if len(set(degree_values)) == 1 else None
        )
        self.check_slices = tuple(
            (
                offsets[check],
                offsets[check + 1],
                0 if check < num_x_checks else 1,
            )
            for check in range(num_x_checks + num_z_checks)
        )
        self.edge_type_slices = tuple(
            (edge_type_offsets[index], edge_type_offsets[index + 1])
            for index in range(len(edge_type_keys))
        )

        self.register_buffer("hx", hx_tensor.to(torch.uint8))
        self.register_buffer("hz", hz_tensor.to(torch.uint8))
        self.register_buffer("edge_check", edge_check)
        self.register_buffer("edge_qubit", edge_qubit)
        self.register_buffer("edge_check_type", edge_check_type)
        self.register_buffer("check_node_type", check_node_type)
        self.register_buffer("edge_orbit", raw_orbits)
        self.register_buffer("edge_type", edge_type)
        self.register_buffer("edge_type_order", edge_type_order)
        self.register_buffer("check_ptr", check_ptr)
        # A check anticommutes with these Pauli states.  Rows are X-check and
        # Z-check; columns follow I, X, Y, Z.
        self.register_buffer(
            "anticommutes",
            torch.tensor(
                [[False, False, True, True], [False, True, True, False]],
                dtype=torch.bool,
            ),
        )

        self.residual_mlps = nn.ModuleList(
            _OrbitResidual(residual_hidden_dim) for _ in range(self.num_edge_types)
        )
        # raw=0 gives update coefficient exactly one, hence no damping at
        # initialization while retaining a non-zero optimization derivative.
        self.relaxation_raw = nn.Parameter(torch.zeros(self.num_edge_types))

    @staticmethod
    def _validate_edge_orbits(
        edge_orbits: Sequence[int] | Tensor | np.ndarray | None,
        *,
        num_edges: int,
    ) -> Tensor:
        if edge_orbits is None:
            return torch.zeros(num_edges, dtype=torch.long)
        raw = torch.as_tensor(edge_orbits).detach().cpu()
        if raw.ndim != 1 or raw.numel() != num_edges:
            raise ValueError(
                "edge_orbits must have one entry per row-major Tanner edge: "
                f"expected ({num_edges},), got {tuple(raw.shape)}."
            )
        if raw.is_floating_point() and not torch.all(raw == raw.round()):
            raise ValueError("edge_orbits must contain integer ids.")
        raw = raw.to(torch.long)
        if torch.any(raw < 0):
            raise ValueError("edge_orbits must be non-negative.")
        return raw

    @staticmethod
    def _compact_edge_types(
        check_types: Tensor, raw_orbits: Tensor
    ) -> tuple[Tensor, list[tuple[int, int]]]:
        edge_keys = [
            (int(check_type), int(orbit))
            for check_type, orbit in zip(check_types.tolist(), raw_orbits.tolist())
        ]
        # Stable sorting (rather than first occurrence in a wrapped row) keeps
        # ModuleList indices compatible when the same BB orbit set is reused
        # at another code size.
        keys = sorted(set(edge_keys))
        pair_to_type = {key: index for index, key in enumerate(keys)}
        edge_types = [pair_to_type[key] for key in edge_keys]
        return torch.tensor(edge_types, dtype=torch.long), keys

    def _parse_syndrome(self, syndrome: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(syndrome, tuple):
            if len(syndrome) != 2:
                raise ValueError("A syndrome tuple must be (x_check, z_check).")
            x_syndrome, z_syndrome = syndrome
            if x_syndrome.ndim != 2 or z_syndrome.ndim != 2:
                raise ValueError(
                    "Both syndrome sectors must have shape (batch, checks)."
                )
            if x_syndrome.shape[0] != z_syndrome.shape[0]:
                raise ValueError("Both syndrome sectors must have the same batch size.")
            if x_syndrome.shape[1] != self.num_x_checks:
                raise ValueError(
                    f"Expected {self.num_x_checks} X-check bits, got "
                    f"{x_syndrome.shape[1]}."
                )
            if z_syndrome.shape[1] != self.num_z_checks:
                raise ValueError(
                    f"Expected {self.num_z_checks} Z-check bits, got "
                    f"{z_syndrome.shape[1]}."
                )
            syndrome_tensor = torch.cat((x_syndrome, z_syndrome), dim=1)
        else:
            syndrome_tensor = syndrome
            if syndrome_tensor.ndim != 2 or syndrome_tensor.shape[1] != self.num_checks:
                raise ValueError(
                    "syndrome must have shape "
                    f"(batch, {self.num_checks}), got {tuple(syndrome_tensor.shape)}."
                )

        if syndrome_tensor.device != self.edge_check.device:
            raise ValueError(
                "syndrome and model must be on the same device; move the model or "
                "input explicitly before decoding."
            )
        if not torch.all((syndrome_tensor == 0) | (syndrome_tensor == 1)):
            raise ValueError("syndrome must contain only binary values.")
        return syndrome_tensor

    def _prior(
        self,
        *,
        batch_size: int,
        p: float | Tensor | None,
        prior_logits: Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if (p is None) == (prior_logits is None):
            raise ValueError("Provide exactly one of p or prior_logits.")

        if prior_logits is not None:
            prior = prior_logits.to(device=device, dtype=dtype)
            if prior.ndim == 2 and prior.shape == (self.num_qubits, 4):
                prior = prior.unsqueeze(0)
            if prior.ndim != 3 or prior.shape[1:] != (self.num_qubits, 4):
                raise ValueError(
                    "prior_logits must have shape "
                    f"(batch, {self.num_qubits}, 4) or ({self.num_qubits}, 4), "
                    f"got {tuple(prior.shape)}."
                )
            if prior.shape[0] not in (1, batch_size):
                raise ValueError(
                    "prior_logits batch dimension must be one or match syndrome."
                )
            return torch.log_softmax(prior.expand(batch_size, -1, -1), dim=-1)

        probability = torch.as_tensor(p, device=device, dtype=dtype)
        if probability.ndim == 0:
            probability = probability.expand(batch_size)
        elif probability.ndim == 2 and probability.shape == (batch_size, 1):
            probability = probability[:, 0]
        elif probability.ndim != 1 or probability.shape[0] not in (1, batch_size):
            raise ValueError("p must be a scalar or contain one value per batch item.")
        probability = probability.expand(batch_size)
        if not torch.all((probability >= 0.0) & (probability <= 1.0)):
            raise ValueError("Depolarizing probability p must lie in [0, 1].")

        probabilities = torch.stack(
            (
                1.0 - probability,
                probability / 3.0,
                probability / 3.0,
                probability / 3.0,
            ),
            dim=-1,
        )
        tiny = max(float(torch.finfo(dtype).tiny), 1e-12)
        log_prior = probabilities.clamp_min(tiny).log()
        log_prior = torch.log_softmax(log_prior, dim=-1)
        return log_prior.unsqueeze(1).expand(-1, self.num_qubits, -1)

    def _exact_check_update(
        self, variable_messages: Tensor, syndrome: Tensor
    ) -> Tensor:
        """Return exact normalized sum-product check-to-variable messages."""

        if self.uniform_check_degree is not None:
            return self._exact_check_update_uniform(variable_messages, syndrome)
        return self._exact_check_update_general(variable_messages, syndrome)

    @staticmethod
    def _product_excluding_each(parity_bias: Tensor, *, dim: int) -> Tensor:
        """Multiply all entries except the one at each position along ``dim``."""

        prefix_shape = list(parity_bias.shape)
        prefix_shape[dim] = 1
        ones = torch.ones(
            prefix_shape, device=parity_bias.device, dtype=parity_bias.dtype
        )
        prefix = torch.cat((ones, torch.cumprod(parity_bias, dim=dim)), dim=dim)
        suffix = torch.cat(
            (
                torch.flip(
                    torch.cumprod(torch.flip(parity_bias, dims=(dim,)), dim=dim),
                    dims=(dim,),
                ),
                ones,
            ),
            dim=dim,
        )
        leading = [slice(None)] * parity_bias.ndim
        trailing = [slice(None)] * parity_bias.ndim
        leading[dim] = slice(None, -1)
        trailing[dim] = slice(1, None)
        return prefix[tuple(leading)] * suffix[tuple(trailing)]

    def _exact_check_update_uniform(
        self, variable_messages: Tensor, syndrome: Tensor
    ) -> Tensor:
        """Vectorized BP4 check update for a Tanner graph of uniform degree."""

        degree = self.uniform_check_degree
        assert degree is not None
        batch_size = variable_messages.shape[0]
        dtype = variable_messages.dtype
        tiny = max(float(torch.finfo(dtype).tiny), 1e-12)

        incoming = variable_messages.reshape(
            batch_size, self.num_checks, degree, 4
        ).exp()
        anti_mask = self.anticommutes[self.check_node_type]
        p_anti = (incoming * anti_mask[None, :, None, :].to(dtype=dtype)).sum(dim=-1)
        parity_bias = 1.0 - 2.0 * p_anti
        product_excluding_edge = self._product_excluding_each(parity_bias, dim=2)

        syndrome_sign = 1.0 - 2.0 * syndrome
        state_sign = 1.0 - 2.0 * anti_mask.to(dtype=dtype)
        compatible_probability = 0.5 * (
            1.0
            + syndrome_sign[:, :, None, None]
            * state_sign[None, :, None, :]
            * product_excluding_edge[:, :, :, None]
        )
        log_message = compatible_probability.clamp_min(tiny).log()
        return torch.log_softmax(log_message, dim=-1).reshape(
            batch_size, self.num_edges, 4
        )

    def _exact_check_update_general(
        self, variable_messages: Tensor, syndrome: Tensor
    ) -> Tensor:
        """Exact fallback for CSS Tanner graphs with irregular check degrees."""

        outputs: list[Tensor] = []
        dtype = variable_messages.dtype
        tiny = max(float(torch.finfo(dtype).tiny), 1e-12)

        for check, (start, stop, check_type) in enumerate(self.check_slices):
            incoming = variable_messages[:, start:stop, :].exp()
            anti_mask = self.anticommutes[check_type]
            p_anti = incoming[..., anti_mask].sum(dim=-1)
            parity_bias = 1.0 - 2.0 * p_anti

            # Prefix/suffix products avoid division by zero when one incoming
            # message has exactly balanced commuting/anticommuting mass.
            product_excluding_edge = self._product_excluding_each(parity_bias, dim=1)

            syndrome_sign = 1.0 - 2.0 * syndrome[:, check]
            state_sign = 1.0 - 2.0 * anti_mask.to(dtype=dtype)
            compatible_probability = 0.5 * (
                1.0
                + syndrome_sign[:, None, None]
                * state_sign[None, None, :]
                * product_excluding_edge[:, :, None]
            )
            log_message = compatible_probability.clamp_min(tiny).log()
            outputs.append(torch.log_softmax(log_message, dim=-1))

        return torch.cat(outputs, dim=1)

    def _neural_check_update(
        self,
        exact_messages: Tensor,
        variable_messages: Tensor,
        old_check_messages: Tensor,
        syndrome: Tensor,
    ) -> Tensor:
        syndrome_per_edge = syndrome[:, self.edge_check]
        features = torch.cat(
            (
                exact_messages,
                variable_messages,
                old_check_messages,
                syndrome_per_edge.unsqueeze(-1),
            ),
            dim=-1,
        )
        residual = torch.zeros_like(exact_messages)
        for edge_type, residual_mlp in enumerate(self.residual_mlps):
            start, stop = self.edge_type_slices[edge_type]
            indices = self.edge_type_order[start:stop]
            typed_residual = self.residual_scale * torch.tanh(
                residual_mlp(features[:, indices, :])
            )
            residual = residual.index_copy(1, indices, typed_residual)

        relaxation = 1.0 + self.max_relaxation_delta * torch.tanh(self.relaxation_raw)
        coefficient = relaxation[self.edge_type].view(1, self.num_edges, 1)
        # Written around coefficient=1 so the initialized path is bitwise the
        # exact message before the common normalization below.
        relaxed = exact_messages + (coefficient - 1.0) * (
            exact_messages - old_check_messages
        )
        return torch.log_softmax(relaxed + residual, dim=-1)

    def _variable_update(
        self, prior: Tensor, check_messages: Tensor
    ) -> tuple[Tensor, Tensor]:
        aggregate = torch.zeros_like(prior).index_add(
            1, self.edge_qubit, check_messages
        )
        posterior = torch.log_softmax(prior + aggregate, dim=-1)
        variable_messages = torch.log_softmax(
            prior[:, self.edge_qubit, :]
            + aggregate[:, self.edge_qubit, :]
            - check_messages,
            dim=-1,
        )
        return posterior, variable_messages

    def forward(
        self,
        syndrome: Tensor | tuple[Tensor, Tensor],
        *,
        p: float | Tensor | None = None,
        prior_logits: Tensor | None = None,
        neural: bool = True,
        return_all: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Decode one code-capacity syndrome batch.

        Parameters
        ----------
        syndrome:
            Either a concatenated tensor ``[batch, mx + mz]`` (X checks first)
            or a tuple ``(sx[batch, mx], sz[batch, mz])``.
        p:
            Scalar or per-shot depolarizing rate.  It creates the prior
            ``[1-p, p/3, p/3, p/3]`` in ``I, X, Y, Z`` order.
        prior_logits:
            Alternative arbitrary Pauli prior with shape ``[batch, n, 4]`` or
            ``[n, 4]``.  Exactly one of ``p`` and ``prior_logits`` is required.
        neural:
            If false, bypass all learned residual and relaxation terms without
            mutating the model, yielding vanilla BP4 on exactly the same batch.
        return_all:
            If true, return ``(final, history)`` where ``final`` has shape
            ``[batch, n, 4]`` and ``history`` has shape
            ``[batch, iterations, n, 4]``.  Otherwise return only ``final``.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Normalized posterior log-probabilities in ``I, X, Y, Z`` order.
        """

        syndrome_tensor = self._parse_syndrome(syndrome)
        dtype = self.relaxation_raw.dtype
        syndrome_float = syndrome_tensor.to(dtype=dtype)
        prior = self._prior(
            batch_size=syndrome_tensor.shape[0],
            p=p,
            prior_logits=prior_logits,
            dtype=dtype,
            device=syndrome_tensor.device,
        )

        variable_messages = prior[:, self.edge_qubit, :]
        check_messages = torch.full_like(variable_messages, -float(np.log(4.0)))
        history: list[Tensor] = []
        posterior = prior

        for _ in range(self.iterations):
            exact_messages = self._exact_check_update(variable_messages, syndrome_float)
            if neural:
                check_messages = self._neural_check_update(
                    exact_messages,
                    variable_messages,
                    check_messages,
                    syndrome_float,
                )
            else:
                # Match the common normalization in the neural path.  This is
                # mathematically an identity because exact_messages are
                # already normalized, and makes initialized neural/BP paired
                # evaluation numerically identical as well.
                check_messages = torch.log_softmax(exact_messages, dim=-1)
            posterior, variable_messages = self._variable_update(prior, check_messages)
            if return_all:
                history.append(posterior)

        if return_all:
            return posterior, torch.stack(history, dim=1)
        return posterior


__all__ = ["EquivariantNeuralBP4", "PAULI_ORDER"]
