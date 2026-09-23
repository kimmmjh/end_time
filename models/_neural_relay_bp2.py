"""Neural check updates combined with Relay-BP variable memory.

The memory/relay construction follows arXiv:2506.01779, Sec. II--III.
This is a hybrid: the inherited normalized min-sum, residual and relaxation
remain in the check update. It is not a reproduction of the paper's settings.
``forward`` unrolls a fixed training budget; ``decode`` performs per-shot
early stopping and retains syndrome-valid candidates without seeing labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ._equivariant_neural_bp2 import EquivariantNeuralBP2


@dataclass
class RelayDecodeResult:
    correction: Tensor
    posterior: Tensor
    converged: Tensor
    iterations: Tensor
    legs: Tensor
    solutions: Tensor


class NeuralRelayBP2(EquivariantNeuralBP2):
    """Keep neural check messages and add disordered variable memory.

    ``iterations`` is the budget *per leg*. Memory is constant within a leg,
    and later legs draw one independent strength per shot and mechanism.
    Passing an explicit ``memory`` tensor makes comparisons reproducible.
    No new trainable parameters or persistent graph buffers are introduced.
    """

    def __init__(
        self,
        graph: Any,
        *,
        relay_legs: int = 4,
        relay_solutions: int = 1,
        relay_memory_strength: float = 0.125,
        relay_memory_min: float = -0.24,
        relay_memory_max: float = 0.66,
        **kwargs: Any,
    ) -> None:
        super().__init__(graph, **kwargs)
        if relay_legs < 1 or int(relay_legs) != relay_legs:
            raise ValueError("relay_legs must be a positive integer.")
        if (
            not 1 <= relay_solutions <= relay_legs
            or int(relay_solutions) != relay_solutions
        ):
            raise ValueError("relay_solutions must be an integer in [1, relay_legs].")
        for name, value in (
            ("relay_memory_strength", relay_memory_strength),
            ("relay_memory_min", relay_memory_min),
            ("relay_memory_max", relay_memory_max),
        ):
            if not math.isfinite(value) or not -1.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (-1, 1).")
        if relay_memory_min > relay_memory_max:
            raise ValueError("relay_memory_min must not exceed relay_memory_max.")
        self.relay_legs = int(relay_legs)
        self.relay_solutions = int(relay_solutions)
        self.relay_memory_strength = float(relay_memory_strength)
        self.relay_memory_min = float(relay_memory_min)
        self.relay_memory_max = float(relay_memory_max)

    def sample_memory(
        self, batch_size: int, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw outside checkpointed iterations so backward never resamples."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        memory = self.prior_log_odds.new_empty(
            self.relay_legs, batch_size, self.num_mechanisms
        )
        memory[0].fill_(self.relay_memory_strength)
        if self.relay_legs > 1:
            memory[1:].uniform_(
                self.relay_memory_min, self.relay_memory_max, generator=generator
            )
        return memory

    def _prepare(
        self, syndrome: Tensor, memory: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if (
            syndrome.ndim != 2
            or syndrome.shape[1] != self.num_detectors
            or syndrome.shape[0] < 1
        ):
            raise ValueError(
                f"syndrome must have shape (positive batch, {self.num_detectors})."
            )
        if syndrome.device != self.prior_log_odds.device:
            raise ValueError("syndrome and model must be on the same device.")
        syndrome = syndrome.to(dtype=self.prior_log_odds.dtype)
        if not torch.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("syndrome must contain only binary values.")
        if memory is None:
            memory = self.sample_memory(syndrome.shape[0])
        expected = (self.relay_legs, syndrome.shape[0], self.num_mechanisms)
        if memory.shape != expected or memory.device != syndrome.device:
            raise ValueError(f"memory must have shape {expected} on the model device.")
        memory = memory.to(dtype=syndrome.dtype)
        if not torch.all(torch.isfinite(memory) & (memory > -1) & (memory < 1)):
            raise ValueError("memory must be finite and in (-1, 1).")
        return syndrome, memory

    def _relay_iteration(
        self,
        variable_messages: Tensor,
        check_messages: Tensor,
        posterior: Tensor,
        syndrome: Tensor,
        memory: Tensor,
        neural: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        exact = self._check_update(variable_messages, syndrome)
        if neural:
            residual = self._residual(
                exact, variable_messages, check_messages, syndrome, posterior
            )
            coefficient = (
                1.0 + self.max_relaxation_delta * torch.tanh(self.relaxation_raw)
            )[self.sharing_index].unsqueeze(0)
            updated = exact + (coefficient - 1.0) * (exact - check_messages) + residual
        else:
            updated = exact
        updated = updated.clamp(-self.message_clip, self.message_clip)

        # One bias per variable, not per incident check, anchored to the
        # ORIGINAL physical prior even when the posterior came from another leg.
        prior = self.prior_log_odds.unsqueeze(0)
        bias = prior + memory * (posterior - prior)
        total = bias + torch.zeros_like(posterior).index_add(
            1, self.edge_mechanism, updated
        )
        # Exclude the recipient before clipping; clipping the full posterior
        # first would corrupt extrinsic messages in the saturation regime.
        variable = (total[:, self.edge_mechanism] - updated).clamp(
            -self.message_clip, self.message_clip
        )
        return variable, updated, total.clamp(-self.message_clip, self.message_clip)

    def forward(
        self,
        syndrome: Tensor,
        *,
        neural: bool = True,
        return_all: bool = False,
        memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Differentiable full-budget unrolling; history is [B, legs*T, N].

        Training keeps every step and gradients across leg boundaries. Use
        ``decode`` to obtain the inference-time selected correction/posterior.
        """
        syndrome, memory = self._prepare(syndrome, memory)
        posterior = self.prior_log_odds.unsqueeze(0).expand(syndrome.shape[0], -1)
        history: list[Tensor] = []
        for leg in range(self.relay_legs):
            # Relay passes marginals, while edge messages restart from priors.
            variable = self.prior_log_odds[self.edge_mechanism].unsqueeze(0).expand(
                syndrome.shape[0], -1
            )
            check = torch.zeros_like(variable)
            for _ in range(self.iterations):
                inputs = (variable, check, posterior, syndrome, memory[leg], neural)
                if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                    variable, check, posterior = checkpoint(
                        self._relay_iteration, *inputs, use_reentrant=False
                    )
                else:
                    variable, check, posterior = self._relay_iteration(*inputs)
                if return_all:
                    history.append(posterior)
        if return_all:
            return posterior, torch.stack(history, dim=1)
        return posterior

    @torch.no_grad()
    def decode(
        self, syndrome: Tensor, *, neural: bool = True, memory: Tensor | None = None
    ) -> RelayDecodeResult:
        """Retain the lowest physical-prior-cost valid candidate across legs.

        Each shot leaves a leg at its first valid correction and stops the
        relay after ``relay_solutions`` successful legs (possibly the same
        correction). With no valid candidate, return its final hard decision
        with ``converged=False``; never imply that memory guarantees recovery.
        """
        syndrome, memory = self._prepare(syndrome, memory)
        batch_size = syndrome.shape[0]
        latest = self.prior_log_odds.unsqueeze(0).expand(batch_size, -1).clone()
        best = latest.clone()
        best_cost = latest.new_full((batch_size,), float("inf"))
        iterations = torch.zeros(batch_size, dtype=torch.long, device=syndrome.device)
        legs = torch.zeros_like(iterations)
        solutions = torch.zeros_like(iterations)
        for leg in range(self.relay_legs):
            active = torch.where(solutions < self.relay_solutions)[0]
            if active.numel() == 0:
                break
            legs[active] += 1
            posterior = latest[active]
            variable = self.prior_log_odds[self.edge_mechanism].unsqueeze(0).expand(
                active.numel(), -1
            )
            check = torch.zeros_like(variable)
            for _ in range(self.iterations):
                variable, check, posterior = self._relay_iteration(
                    variable, check, posterior,
                    syndrome[active], memory[leg, active], neural,
                )
                iterations[active] += 1
                latest[active] = posterior
                correction = self.hard_decision(posterior)
                valid = self._syndrome_satisfied(correction, syndrome[active])
                cost = (correction * self.prior_log_odds).sum(dim=1)
                improved = valid & (cost < best_cost[active])
                best[active[improved]] = posterior[improved]
                best_cost[active[improved]] = cost[improved]
                solutions[active[valid]] += 1
                # Stop each converged shot for this leg, preserving its
                # posterior as the starting memory of its next leg.
                remaining = ~valid
                active = active[remaining]
                if active.numel() == 0:
                    break
                variable, check, posterior = (
                    variable[remaining], check[remaining], posterior[remaining]
                )
        converged = solutions > 0
        selected = torch.where(converged[:, None], best, latest)
        return RelayDecodeResult(
            correction=self.hard_decision(selected),
            posterior=selected,
            converged=converged,
            iterations=iterations,
            legs=legs,
            solutions=solutions,
        )


__all__ = ["NeuralRelayBP2", "RelayDecodeResult"]
