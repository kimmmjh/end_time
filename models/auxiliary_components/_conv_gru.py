"""Convolutional GRU layers with toric (circular) spatial boundaries."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._circular_convolution_2d import circular_conv_2d


class ConvGRUCell(nn.Module):
    """A translation-equivariant GRU cell for 2D feature maps.

    Every affine operation is a circular 2D convolution. Consequently, rolling
    both the input and hidden state on the torus rolls the output by the same
    amount.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive.")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        combined_channels = input_channels + hidden_channels

        self.gate_conv = circular_conv_2d(
            combined_channels,
            2 * hidden_channels,
            kernel_size,
            bias=True,
        )
        self.candidate_conv = circular_conv_2d(
            combined_channels,
            hidden_channels,
            kernel_size,
            bias=True,
        )

        # Start slightly biased toward carrying temporal information forward.
        with torch.no_grad():
            self.gate_conv.bias.zero_()
            self.gate_conv.bias[hidden_channels:].fill_(1.0)
            self.candidate_conv.bias.zero_()

    def forward(self, x: Tensor, hidden: Tensor | None = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ConvGRUCell expects (batch, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[1]}."
            )

        if hidden is None:
            hidden = x.new_zeros(
                x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]
            )
        elif hidden.shape != (
            x.shape[0],
            self.hidden_channels,
            x.shape[2],
            x.shape[3],
        ):
            raise ValueError(
                "Hidden state must have shape "
                f"{(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])}, "
                f"got {tuple(hidden.shape)}."
            )

        reset, update = torch.sigmoid(
            self.gate_conv(torch.cat((x, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate_conv(torch.cat((x, reset * hidden), dim=1))
        )
        return update * hidden + (1.0 - update) * candidate


class ConvGRU(nn.Module):
    """A stack of ConvGRU cells operating on ``(B, T, C, H, W)`` sequences."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            ConvGRUCell(
                input_channels=input_channels if layer == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
            )
            for layer in range(num_layers)
        )

    def forward(
        self,
        sequence: Tensor,
        hidden_states: tuple[Tensor, ...] | None = None,
        return_sequence: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if sequence.ndim != 5:
            raise ValueError(
                "ConvGRU expects (batch, time, channels, height, width), "
                f"got {tuple(sequence.shape)}."
            )
        if sequence.shape[1] < 1:
            raise ValueError("ConvGRU requires at least one time step.")
        if sequence.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {sequence.shape[2]}."
            )
        if hidden_states is not None and len(hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, got {len(hidden_states)}."
            )

        states: list[Tensor | None] = (
            [None] * self.num_layers
            if hidden_states is None
            else list(hidden_states)
        )
        outputs = [] if return_sequence else None
        for frame in sequence.unbind(dim=1):
            layer_input = frame
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(layer_input, states[layer])
                layer_input = states[layer]
            if outputs is not None:
                outputs.append(layer_input)

        final_states = tuple(state for state in states if state is not None)
        output = (
            torch.stack(outputs, dim=1)
            if outputs is not None
            else final_states[-1]
        )
        return output, final_states
