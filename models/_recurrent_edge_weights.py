"""Translation-equivariant recurrent predictor for circuit DEM edge weights."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .auxiliary_components import ConvGRU, circular_conv_2d
from .residual_blocks import WideResBlock


class RecurrentEdgeWeightNetwork(nn.Module):
    """Encode all syndrome rounds and predict one conditional logit per DEM edge.

    Spatial convolutions are circular and shared over rounds.  By default a
    forward and reverse ConvGRU are concatenated because threshold evaluation is
    offline and an early-time edge should be allowed to use later detection
    events.  The edge scorer is symmetric in its endpoints and receives only
    relative toric geometry, preserving spatial translation equivariance.
    """

    def __init__(
        self,
        channels: list[int],
        depths: list[int],
        lattice_size: int,
        *,
        in_channels: int = 2,
        kernel_size: int = 3,
        gru_channels: int | None = None,
        gru_layers: int = 1,
        gru_kernel_size: int = 3,
        bidirectional: bool = True,
        edge_geometry_features: int = 8,
        edge_hidden_channels: int | None = None,
        edge_delta_scale: float = 6.0,
        edge_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if not channels or len(channels) != len(depths):
            raise ValueError("channels and depths must be non-empty and equal length.")
        if any(channel < 1 for channel in channels):
            raise ValueError("All channel widths must be positive.")
        if any(depth < 1 for depth in depths):
            raise ValueError("All block depths must be positive.")
        if lattice_size < 1:
            raise ValueError("lattice_size must be positive.")
        if edge_geometry_features < 1:
            raise ValueError("edge_geometry_features must be positive.")
        if edge_delta_scale <= 0.0:
            raise ValueError("edge_delta_scale must be positive.")
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive.")

        self.lattice_size = int(lattice_size)
        self.in_channels = int(in_channels)
        self.gru_channels = channels[-1] if gru_channels is None else gru_channels
        self.bidirectional = bool(bidirectional)
        self.edge_delta_scale = float(edge_delta_scale)
        self.edge_chunk_size = int(edge_chunk_size)
        if self.gru_channels < 1:
            raise ValueError("gru_channels must be positive.")

        self.conv_in = circular_conv_2d(
            self.in_channels, channels[0], kernel_size=kernel_size
        )
        self.blocks = nn.Sequential(
            *[
                self._make_block(
                    in_channels=channels[stage - 1] if stage > 0 else channels[0],
                    out_channels=channels[stage],
                    kernel_size=kernel_size,
                    depth=depths[stage],
                )
                for stage in range(len(channels))
            ]
        )
        self.forward_gru = ConvGRU(
            input_channels=channels[-1],
            hidden_channels=self.gru_channels,
            num_layers=gru_layers,
            kernel_size=gru_kernel_size,
        )
        self.reverse_gru = (
            ConvGRU(
                input_channels=channels[-1],
                hidden_channels=self.gru_channels,
                num_layers=gru_layers,
                kernel_size=gru_kernel_size,
            )
            if self.bidirectional
            else None
        )
        self.feature_channels = self.gru_channels * (2 if self.bidirectional else 1)
        self.output_norm = nn.BatchNorm2d(self.feature_channels)
        self.output_activation = nn.GELU()
        self.sector_embedding = nn.Embedding(2, self.feature_channels)
        self.boundary_embedding = nn.Parameter(torch.zeros(self.feature_channels))

        edge_hidden = (
            self.feature_channels if edge_hidden_channels is None else edge_hidden_channels
        )
        if edge_hidden < 1:
            raise ValueError("edge_hidden_channels must be positive.")
        self.edge_head = nn.Sequential(
            nn.LayerNorm(2 * self.feature_channels + edge_geometry_features),
            nn.Linear(2 * self.feature_channels + edge_geometry_features, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, 1),
        )
        # At initialization delta=0, so conditional logits exactly equal the
        # static DEM priors and dynamic MWPM reproduces ordinary MWPM.
        final_linear = self.edge_head[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def encode_spacetime(self, syndrome: Tensor) -> Tensor:
        """Return one feature vector for every detector node in Stim order."""

        if syndrome.ndim != 4:
            raise ValueError(
                "Expected syndrome shape (batch, 2, rounds, L^2), "
                f"got {tuple(syndrome.shape)}."
            )
        batch, sectors, rounds, stabilizers_per_sector = syndrome.shape
        if (
            sectors != self.in_channels
            or rounds < 1
            or stabilizers_per_sector != self.lattice_size**2
        ):
            raise ValueError(
                f"Expected (batch, {self.in_channels}, rounds, "
                f"{self.lattice_size**2}), got {tuple(syndrome.shape)}."
            )

        frames = syndrome.reshape(
            batch, sectors, rounds, self.lattice_size, self.lattice_size
        )
        frames = frames.permute(0, 2, 1, 3, 4).reshape(
            batch * rounds, sectors, self.lattice_size, self.lattice_size
        )
        frames = self.blocks(self.conv_in(frames))
        frames = frames.reshape(
            batch, rounds, frames.shape[1], self.lattice_size, self.lattice_size
        )

        forward, _ = self.forward_gru(frames, return_sequence=True)
        if self.reverse_gru is not None:
            reverse, _ = self.reverse_gru(
                torch.flip(frames, dims=(1,)), return_sequence=True
            )
            reverse = torch.flip(reverse, dims=(1,))
            features = torch.cat((forward, reverse), dim=2)
        else:
            features = forward

        normalized = features.reshape(
            batch * rounds,
            self.feature_channels,
            self.lattice_size,
            self.lattice_size,
        )
        normalized = self.output_activation(self.output_norm(normalized))
        features = normalized.reshape(
            batch,
            rounds,
            self.feature_channels,
            self.lattice_size,
            self.lattice_size,
        )

        # Stim detector order is (time, sector, flattened spatial position).
        spatial = features.permute(0, 1, 3, 4, 2).reshape(
            batch, rounds, self.lattice_size**2, self.feature_channels
        )
        sector_embeddings = self.sector_embedding.weight.view(
            1, 1, 2, 1, self.feature_channels
        )
        nodes = spatial.unsqueeze(2) + sector_embeddings
        return nodes.reshape(batch, rounds * 2 * self.lattice_size**2, -1)

    def forward(
        self,
        syndrome: Tensor,
        edge_endpoints: Tensor,
        edge_geometry: Tensor,
        base_logits: Tensor,
    ) -> Tensor:
        node_features = self.encode_spacetime(syndrome)
        if edge_endpoints.ndim != 2 or edge_endpoints.shape[1] != 2:
            raise ValueError("edge_endpoints must have shape (edges, 2).")
        num_edges = edge_endpoints.shape[0]
        if edge_geometry.ndim != 2 or edge_geometry.shape[0] != num_edges:
            raise ValueError("edge_geometry must have one row per edge.")
        if base_logits.shape != (num_edges,):
            raise ValueError("base_logits must have shape (edges,).")

        outputs = []
        boundary_token = self.boundary_embedding.view(1, 1, -1)
        for start in range(0, num_edges, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, num_edges)
            endpoints = edge_endpoints[start:stop]
            first = node_features[:, endpoints[:, 0], :]
            second_indices = endpoints[:, 1].clamp_min(0)
            second = node_features[:, second_indices, :]
            boundary = endpoints[:, 1].lt(0).view(1, -1, 1)
            second = torch.where(boundary, boundary_token, second)

            # Sum and absolute difference make the score invariant to swapping
            # the endpoints of an undirected matching edge.
            geometry = edge_geometry[start:stop].unsqueeze(0).expand(
                syndrome.shape[0], -1, -1
            )
            representation = torch.cat(
                (first + second, torch.abs(first - second), geometry), dim=-1
            )
            raw_delta = self.edge_head(representation).squeeze(-1)
            delta = self.edge_delta_scale * torch.tanh(raw_delta)
            outputs.append(base_logits[start:stop].unsqueeze(0) + delta)

        return torch.cat(outputs, dim=1)

    @staticmethod
    def _make_block(
        *, in_channels: int, out_channels: int, kernel_size: int, depth: int
    ) -> nn.Sequential:
        layers = [
            WideResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                layer_type=circular_conv_2d,
                norm_type=nn.BatchNorm2d,
            )
        ]
        layers.extend(
            WideResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                layer_type=circular_conv_2d,
                norm_type=nn.BatchNorm2d,
            )
            for _ in range(depth - 1)
        )
        return nn.Sequential(*layers)

