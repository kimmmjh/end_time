"""Spatially equivariant recurrent decoder for repeated syndrome rounds."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .auxiliary_components import ConvGRU, circular_conv_2d
from .residual_blocks import WideResBlock


class RecurrentEND2D(nn.Module):
    """Encode each syndrome round with a shared 2D CNN, then apply ConvGRU.

    Input shape is ``(batch, 2, rounds, L^2)``. The two detector sectors are
    treated as channels, while measurement rounds are processed sequentially.
    Circular convolutions preserve translations on the toric lattice.
    """

    def __init__(
        self,
        channels: list[int],
        depths: list[int],
        lattice_size: int,
        num_classes: int = 16,
        in_channels: int = 2,
        kernel_size: int = 3,
        gru_channels: int | None = None,
        gru_layers: int = 1,
        gru_kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        if not channels or len(channels) != len(depths):
            raise ValueError("channels and depths must be non-empty and have equal length.")
        if any(channel < 1 for channel in channels):
            raise ValueError("All channel widths must be positive.")
        if any(depth < 1 for depth in depths):
            raise ValueError("All block depths must be positive.")
        if lattice_size < 1:
            raise ValueError("lattice_size must be positive.")
        if num_classes != 16:
            raise ValueError("The toric-code equivariant head currently requires 16 classes.")

        self.lattice_size = lattice_size
        self.in_channels = in_channels
        self.gru_channels = channels[-1] if gru_channels is None else gru_channels
        if self.gru_channels < 1:
            raise ValueError("gru_channels must be positive.")

        self.conv_in = circular_conv_2d(
            in_channels, channels[0], kernel_size=kernel_size
        )
        self.blocks = nn.Sequential(
            *[
                self.make_block(
                    in_channels=channels[stage - 1] if stage > 0 else channels[0],
                    out_channels=channels[stage],
                    kernel_size=kernel_size,
                    depth=depths[stage],
                )
                for stage in range(len(channels))
            ]
        )
        self.recurrent = ConvGRU(
            input_channels=channels[-1],
            hidden_channels=self.gru_channels,
            num_layers=gru_layers,
            kernel_size=gru_kernel_size,
        )
        self.batch_norm = nn.BatchNorm2d(self.gru_channels)
        self.non_linear = nn.GELU()
        self.conv_out = circular_conv_2d(
            self.gru_channels,
            num_classes,
            kernel_size=kernel_size,
            bias=True,
        )

    def _local_class_logits(self, x: Tensor) -> Tensor:
        """Return one 16-class logit vector at every lattice position.

        Keeping the shared encoder in one method lets the absolute-logical END
        head and the matching-residual head use exactly the same spatial and
        temporal feature extractor.
        """
        if x.ndim != 4:
            raise ValueError(
                "Expected syndrome shape (batch, 2, rounds, L^2), "
                f"got {tuple(x.shape)}."
            )
        batch, sectors, rounds, stabilizers_per_sector = x.shape
        if (
            sectors != self.in_channels
            or rounds < 1
            or stabilizers_per_sector != self.lattice_size**2
        ):
            raise ValueError(
                f"Expected syndrome shape (batch, {self.in_channels}, rounds, "
                f"{self.lattice_size**2}) with rounds >= 1, got {tuple(x.shape)}."
            )

        # Apply the same spatial encoder to every round.
        x = x.reshape(
            batch, sectors, rounds, self.lattice_size, self.lattice_size
        )
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch * rounds, sectors, self.lattice_size, self.lattice_size
        )
        x = self.conv_in(x)
        x = self.blocks(x)
        x = x.reshape(
            batch, rounds, x.shape[1], self.lattice_size, self.lattice_size
        )

        # The final hidden state summarizes all measurement rounds.
        _, hidden_states = self.recurrent(x)
        x = hidden_states[-1]
        x = self.non_linear(self.batch_norm(x))
        return self.conv_out(x)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        x = self._local_class_logits(x)
        # Match the existing END head expected by TranslationalEquivariantPooling2D.
        x = torch.roll(x, (-1, -1), (2, 3))
        x = x.permute(0, 2, 3, 1)
        x = torch.flip(x, [1, 2])
        return x.reshape(batch, self.lattice_size, self.lattice_size, 4, 2, 2)

    @staticmethod
    def make_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depth: int,
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


class RecurrentResidualEND2D(RecurrentEND2D):
    """Translation-invariant logical-residual classifier.

    A matching residual is the homology class of a closed cycle: translating
    the detector pattern and the matching correction together must not change
    that class.  The circular CNN and ConvGRU produce an equivariant spatial
    logit map, and global spatial averaging turns it into an invariant
    ``(batch, 16)`` prediction.

    This differs intentionally from :class:`RecurrentEND2D`, whose output still
    requires the syndrome-dependent END pooling used for *absolute* logical
    labels.
    """

    def forward(self, x: Tensor) -> Tensor:
        local_logits = self._local_class_logits(x)
        return local_logits.mean(dim=(2, 3))
