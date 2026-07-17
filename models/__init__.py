"""Different network architectures."""

from ._decoder import Decoder
from ._the_end_2d import TransformedEND2D as TheEND
from ._the_end_2d import TransformedEND2D
from ._recurrent_end_2d import RecurrentEND2D

__all__ = [
    "Decoder",
    "TheEND",
    "TransformedEND2D",
    "RecurrentEND2D",
]
