"""Feed-forward convolutions on the undecomposed detector/mechanism graph.

Gather/scatter replaces the periodic shifts of BBTannerCNN. An edge orbit
selects a channel gate, giving W_r = W_out diag(g_r) W_in. This factorized
kernel retains role-dependent channel mixing without allocating E x W x W
matrices. No BP update, graphlike decomposition, or temporal wrap is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class DEMConvolution(nn.Module):
    """Orbit-shared sparse convolution, normalized by destination sqrt(degree)."""

    def __init__(self, width: int, num_orbits: int, edge_chunk_size: int = 4096):
        super().__init__()
        if min(width, num_orbits, edge_chunk_size) < 1:
            raise ValueError("width, num_orbits and edge_chunk_size must be positive.")
        self.edge_chunk_size = edge_chunk_size
        self.input = nn.Linear(width, width, bias=False)
        self.gates = nn.Embedding(num_orbits, width)
        self.output = nn.Linear(width, width)
        nn.init.normal_(self.gates.weight, mean=1.0, std=0.1)

    def forward(self, features: Tensor, source: Tensor, target: Tensor,
                orbit: Tensor, degree: Tensor) -> Tensor:
        projected = self.input(features)
        result = features.new_zeros(features.shape[0], degree.numel(), features.shape[-1])
        for start in range(0, source.numel(), self.edge_chunk_size):
            end = start + self.edge_chunk_size
            messages = projected[:, source[start:end]] * self.gates(orbit[start:end])
            result.index_add_(1, target[start:end], messages)
        return self.output(result / degree.clamp_min(1).sqrt()[None, :, None])


class _DEMBlock(nn.Module):
    def __init__(self, width: int, num_orbits: int, edge_chunk_size: int):
        super().__init__()
        self.to_detector = DEMConvolution(width, num_orbits, edge_chunk_size)
        self.to_mechanism = DEMConvolution(width, num_orbits, edge_chunk_size)
        self.detector_norm = nn.LayerNorm(width)
        self.mechanism_norm = nn.LayerNorm(width)
        self.activation = nn.SiLU()

    def forward(self, detector, mechanism, edge_d, edge_m, orbit, degree_d, degree_m):
        detector = detector + self.activation(self.detector_norm(
            self.to_detector(mechanism, edge_m, edge_d, orbit, degree_d)))
        mechanism = mechanism + self.activation(self.mechanism_norm(
            self.to_mechanism(detector, edge_d, edge_m, orbit, degree_m)))
        return detector, mechanism


class BBDemCNN(nn.Module):
    """Predict binary mechanism LLRs log(P(no fault)/P(fault)).

    Depth 1 is D->M; each further block is M->D->M. Inputs are detector
    events and fixed graph features, never sampled mechanisms or observables.
    Spatial/bulk-time weight sharing uses BBDemGraph's boundary-aware orbits.
    LayerNorm mixes channels at one node only. Buffers are reconstructed from
    the fingerprint-checked graph when loading an experiment checkpoint.
    """

    def __init__(self, graph, *, width=64, depth=2, edge_chunk_size=4096,
                 gradient_checkpoint=True):
        super().__init__()
        if width < 2 or depth < 1:
            raise ValueError("DEM CNN requires width >= 2 and depth >= 1.")
        if graph.num_mechanisms < 1 or graph.num_edges < 1:
            raise ValueError("DEM CNN requires a nonempty mechanism graph.")
        self.num_detectors = int(graph.num_detectors)
        self.num_mechanisms = int(graph.num_mechanisms)
        self.width, self.depth = width, depth
        self.gradient_checkpoint = gradient_checkpoint
        edge_d = torch.as_tensor(np.array(graph.edge_detector), dtype=torch.long)
        edge_m = torch.as_tensor(np.array(graph.edge_mechanism), dtype=torch.long)
        degree_d = torch.bincount(edge_d, minlength=self.num_detectors).float()
        degree_m = torch.bincount(edge_m, minlength=self.num_mechanisms).float()
        coordinates = torch.as_tensor(np.array(graph.detector_coordinates))
        static_d = torch.stack((
            (coordinates[:, 0] == 0).float(), (coordinates[:, 0] == 1).float(),
            (coordinates[:, 3] == 0).float(),
            (coordinates[:, 3] == graph.detector_frames - 1).float(),
            degree_d.log1p(),
        ), dim=-1)
        prior = torch.as_tensor(np.array(graph.prior_log_odds), dtype=torch.float32)
        static_m = torch.stack((prior, degree_m.log1p()), dim=-1)
        for name, value in (
            ("edge_detector", edge_d), ("edge_mechanism", edge_m),
            ("edge_orbit", torch.as_tensor(np.array(graph.edge_orbit), dtype=torch.long)),
            ("detector_degree", degree_d), ("mechanism_degree", degree_m),
            ("detector_features", static_d), ("mechanism_features", static_m),
            ("prior_log_odds", prior),
        ):
            self.register_buffer(name, value, persistent=False)
        self.detector_stem = nn.Sequential(nn.Linear(6, width), nn.SiLU())
        self.mechanism_stem = nn.Sequential(nn.Linear(2, width), nn.SiLU())
        self.first = DEMConvolution(width, graph.num_orbits, edge_chunk_size)
        self.first_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            _DEMBlock(width, graph.num_orbits, edge_chunk_size) for _ in range(depth - 1)
        ])
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, detectors: Tensor) -> Tensor:
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors or detectors.shape[0] < 1:
            raise ValueError(f"Expected detectors [positive batch,{self.num_detectors}].")
        detectors = detectors.to(self.prior_log_odds)
        if not torch.all((detectors == 0) | (detectors == 1)):
            raise ValueError("detectors must be binary.")
        batch = detectors.shape[0]
        detector = self.detector_stem(torch.cat((
            (1 - 2 * detectors).unsqueeze(-1),
            self.detector_features.unsqueeze(0).expand(batch, -1, -1),
        ), dim=-1))
        mechanism = self.mechanism_stem(self.mechanism_features).unsqueeze(0).expand(batch, -1, -1)
        if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
            first = checkpoint(self.first, detector, self.edge_detector, self.edge_mechanism,
                               self.edge_orbit, self.mechanism_degree, use_reentrant=False)
        else:
            first = self.first(detector, self.edge_detector, self.edge_mechanism,
                               self.edge_orbit, self.mechanism_degree)
        mechanism = mechanism + torch.nn.functional.silu(self.first_norm(first))
        for block in self.blocks:
            inputs = (detector, mechanism, self.edge_detector, self.edge_mechanism,
                      self.edge_orbit, self.detector_degree, self.mechanism_degree)
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                detector, mechanism = checkpoint(block, *inputs, use_reentrant=False)
            else:
                detector, mechanism = block(*inputs)
        return self.prior_log_odds[None] + self.head(mechanism).squeeze(-1)

    @staticmethod
    def probabilities(log_odds: Tensor) -> Tensor:
        return torch.sigmoid(-log_odds)

    @staticmethod
    def hard_decision(log_odds: Tensor) -> Tensor:
        return (log_odds < 0).to(torch.uint8)
