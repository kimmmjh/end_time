"""Different network architectures."""

from ._transformed_end import TransformedEND
from ._mvit import MViT
from ._decoder import Decoder
from ._the_end_2d import TransformedEND2D as TheEND
from ._the_end_2d import TransformedEND2D

__all__ = [
    "TransformedEND",
    "MViT",
    "Decoder",
    "TheEND",
    "TransformedEND2D",
]
