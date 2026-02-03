import torch
from torch.functional import F
from torch import nn, Tensor
from ._circular_convolution_2d import circular_conv_2d


class AConvCircular2D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            attention_channels: int,
            key_depths: int,
            number_heads: int,
            bias: bool = False,

    ) -> None:
        """Initialize this module."""
        super(AConvCircular2D, self).__init__()
        self.key_depths = key_depths
        self.attention_channels = attention_channels
        self.number_heads = number_heads

        """Some sanity checks."""
        assert self.number_heads != 0, "integer division or modulo by zero, number_heads >= 1"
        assert self.key_depths % self.number_heads == 0, "dk should be divided by number_heads."
        assert self.attention_channels % self.number_heads == 0, "dv should be divided by number_heads."

        """A traditional convolutional layer."""
        self.initial_convolution = circular_conv_2d(in_channels, out_channels - attention_channels, kernel_size, bias)
        """Extract Query, Keys, and Values."""
        self.qkv_convolution = circular_conv_2d(in_channels, 2 * key_depths + attention_channels, kernel_size, bias)
        """The attention in form of another convolution."""
        self.out_attention_conv = circular_conv_2d(attention_channels, attention_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function for the module.

        :param x: The tensor (b,c,x,y)
        """
        out = self.initial_convolution(x)
        b, _, w, h = out.shape

        """Compute (flattened-) queries, keys and values."""
        query_key_values = self.qkv_convolution(x)
        bq, _, wq, hq = query_key_values.shape
        query, keys, values = torch.split(query_key_values,
                                          [self.key_depths, self.key_depths, self.attention_channels],
                                          dim=1)

        query = self.split_heads(query) * ((self.key_depths // self.number_heads) ** -0.5)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        flat_shape = (bq, self.number_heads, self.key_depths // self.number_heads, wq * hq)
        flat_queries = torch.reshape(query, flat_shape)
        flat_keys = torch.reshape(keys, flat_shape)
        flat_values = torch.reshape(values, (bq, self.number_heads, self.attention_channels // self.number_heads, wq * hq))

        """Continue with the attended convolution."""
        logits = flat_queries.transpose(2, 3) @ flat_keys
        weights = F.softmax(logits, dim=-1)

        """Compute attention."""
        attention_out = weights @ flat_values.transpose(2, 3)
        attention_out = torch.reshape(attention_out,
                                      (b, self.number_heads, self.attention_channels // self.number_heads, w, h))
        attention_out = self.combine_heads(attention_out)
        attention_out = self.out_attention_conv(attention_out)

        out = torch.cat((out, attention_out), dim=1)
        return out

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Split x into multiple heads-

        :param x: The tensor to split.
        :returns: The split tensor.
        """
        b, c, *tail = x.shape
        x = torch.reshape(x, (b, self.number_heads, c // self.number_heads, *tail))
        return x

    @staticmethod
    def combine_heads(x: Tensor) -> Tensor:
        """
        Combine heads in x into one tensor.

        :param x: The tensor to combine.
        :returns: The combined tensor.
        """
        b, heads, values, *tail = x.shape
        x = torch.reshape(x, (b, heads * values, *tail))
        return x
