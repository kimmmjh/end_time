"""Different network architectures."""

from ._decoder import Decoder
from ._the_end_2d import TransformedEND2D as TheEND
from ._the_end_2d import TransformedEND2D
from ._matching_residual_decoder import MatchingResidualDecoder
from ._recurrent_end_2d import RecurrentEND2D, RecurrentResidualEND2D

__all__ = [
    "Decoder",
    "TheEND",
    "TransformedEND2D",
    "RecurrentEND2D",
    "RecurrentResidualEND2D",
    "MatchingResidualDecoder",
]
