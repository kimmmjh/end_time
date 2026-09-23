"""Sparse periodic convolutions following the joint BB Hx/Hz Tanner graph.

The two check feature maps are X/Z checks; the two qubit maps are the left/right
BB blocks. Each edge orbit is a kernel tap (a cyclic displacement with its own
channel-mixing matrix). Stacking check/qubit convolutions expands the receptive
field. These are feed-forward learned features, not BP messages or recurrent
BP iterations. The same operation can also be described as a typed graph CNN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


class TannerShiftConv(nn.Module):
    """Learned sparse 2D convolution, with one 1x1 channel map per Tanner tap."""

    def __init__(self, code: Any, in_channels: int, out_channels: int, *, to_qubits: bool):
        super().__init__()
        self.ell, self.m = code.ell, code.m
        self.to_qubits = to_qubits
        self.taps = []
        for orbit in range(code.num_edge_orbits):
            selected = code.edge_orbit == orbit
            descriptors = np.unique(np.column_stack((
                code.edge_check_type[selected], code.edge_qubit_block[selected],
                code.edge_displacement[selected],
            )), axis=0)
            if descriptors.shape != (1, 4):
                raise ValueError("Each BB edge orbit must have a fixed type/block/displacement.")
            check_type, block, dx, dy = map(int, descriptors[0])
            # q = c + displacement. torch.roll(x, d)[q] = x[q-d].
            self.taps.append((check_type, block, dx, dy) if to_qubits else
                             (block, check_type, -dx, -dy))
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in self.taps
        ])
        self.bias = nn.Parameter(torch.zeros(2, out_channels))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != 2 or features.shape[-2:] != (self.ell, self.m):
            raise ValueError("features must have shape [batch,2,channels,ell,m].")
        sums = [None, None]
        for (source, target, dx, dy), kernel in zip(self.taps, self.kernels):
            value = kernel(torch.roll(features[:, source], shifts=(dx, dy), dims=(-2, -1)))
            sums[target] = value if sums[target] is None else sums[target] + value
        return torch.stack(sums, dim=1) / math.sqrt(6) + self.bias[None, :, :, None, None]


def _pointwise(module: nn.Module, features: Tensor) -> Tensor:
    batch, groups, channels, ell, m = features.shape
    output = module(features.reshape(batch * groups, channels, ell, m))
    return output.reshape(batch, groups, output.shape[1], ell, m)


class _SpatialBlock(nn.Module):
    def __init__(self, code: Any, width: int):
        super().__init__()
        self.qubit_to_check = TannerShiftConv(code, width, width, to_qubits=False)
        self.check_to_qubit = TannerShiftConv(code, width, width, to_qubits=True)
        self.check_self = nn.Conv2d(width, width, 1)
        self.qubit_self = nn.Conv2d(width, width, 1)
        self.activation = nn.SiLU()

    def forward(self, check: Tensor, qubit: Tensor) -> tuple[Tensor, Tensor]:
        check = self.activation(_pointwise(self.check_self, check) + self.qubit_to_check(qubit))
        qubit = self.activation(_pointwise(self.qubit_self, qubit) + self.check_to_qubit(check))
        return check, qubit


class BBTannerCNN(nn.Module):
    """Joint X/Z Tanner CNN producing [batch,n,4] Pauli logits.

    depth=1 sees the six checks touching each qubit. Each extra block performs
    Q->C->Q convolutions, adding two graph hops. No spatial pooling or absolute
    position embeddings are used. Parameters are shared over cyclic translations
    but distinguish edge roles and the two qubit blocks.
    """

    def __init__(self, code: Any, *, width: int = 64, depth: int = 2):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("CNN width and depth must be positive.")
        self.n, self.cells, self.ell, self.m = code.n, code.cells, code.ell, code.m
        self.width, self.depth = width, depth
        self.check_stem = nn.Conv2d(3, width, 1)  # syndrome plus X/Z type
        self.qubit_stem = nn.Conv2d(6, width, 1)  # four priors plus L/R block
        self.first_conv = TannerShiftConv(code, width, width, to_qubits=True)
        self.blocks = nn.ModuleList([_SpatialBlock(code, width) for _ in range(depth - 1)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU(), nn.Conv2d(width, 4, 1))
        self.activation = nn.SiLU()
        # A classifier prior offset, not a BP residual. Begin close to channel
        # probabilities without making every parity marginal exactly one half.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("node_types", torch.eye(2).reshape(1, 2, 2, 1, 1), persistent=False)

    def _inputs(self, syndrome: Tensor, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        if syndrome.ndim != 2 or syndrome.shape[1] != 2 * self.cells or syndrome.shape[0] < 1:
            raise ValueError(f"syndrome must have shape [positive batch,{2 * self.cells}].")
        syndrome = syndrome.to(device=self.node_types.device, dtype=self.node_types.dtype)
        if not torch.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("syndrome must be binary.")
        prior = torch.as_tensor(probabilities, device=syndrome.device, dtype=syndrome.dtype)
        if prior.shape == (4,):
            prior = prior[None, None, :].expand(syndrome.shape[0], self.n, 4)
        elif prior.shape == (syndrome.shape[0], 4):
            prior = prior[:, None, :].expand(-1, self.n, -1)
        elif prior.shape != (syndrome.shape[0], self.n, 4):
            raise ValueError("channel_probabilities must have shape [4], [batch,4], or [batch,n,4].")
        if not torch.isfinite(prior).all() or (prior < 0).any() or not torch.allclose(
            prior.sum(dim=-1), torch.ones_like(prior[..., 0]), atol=1e-6, rtol=1e-5
        ):
            raise ValueError("channel_probabilities must be finite, nonnegative and sum to one.")
        return syndrome, prior.clamp_min(1e-12).log()

    def encode(self, syndrome: Tensor, log_prior: Tensor) -> Tensor:
        """Return spatial features [batch,2,width,ell,m] for a separate head."""
        batch = syndrome.shape[0]
        types = self.node_types.expand(batch, -1, -1, self.ell, self.m)
        checks = (1 - 2 * syndrome).reshape(batch, 2, 1, self.ell, self.m)
        check = self.activation(_pointwise(self.check_stem, torch.cat((checks, types), dim=2)))
        prior_grid = log_prior.reshape(batch, 2, self.ell, self.m, 4).permute(0, 1, 4, 2, 3)
        qubit = _pointwise(self.qubit_stem, torch.cat((prior_grid, types), dim=2))
        qubit = self.activation(qubit + self.first_conv(check))
        for block in self.blocks:
            check, qubit = block(check, qubit)
        return qubit

    def forward(self, syndrome: Tensor, *, channel_probabilities: Tensor) -> Tensor:
        syndrome, log_prior = self._inputs(syndrome, channel_probabilities)
        features = self.encode(syndrome, log_prior)
        learned = _pointwise(self.head, features).permute(0, 1, 3, 4, 2).reshape(-1, self.n, 4)
        return learned + log_prior

    @staticmethod
    def component_probabilities(logits: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = logits.softmax(dim=-1)
        return probabilities[..., 1] + probabilities[..., 2], probabilities[..., 2] + probabilities[..., 3]

    @classmethod
    def hard_decision(cls, logits: Tensor) -> Tensor:
        qx, qz = cls.component_probabilities(logits)
        x, z = (qx > 0.5).long(), (qz > 0.5).long()
        return x + 3 * z - 2 * x * z  # I=0, X=1, Y=2, Z=3


__all__ = ["BBTannerCNN", "TannerShiftConv"]
