"""Different network architectures."""

from ._decoder import Decoder
from ._matching_residual_decoder import MatchingResidualDecoder
from ._neural_weighted_matching import NeuralWeightedMatchingDecoder
from ._recurrent_edge_weights import RecurrentEdgeWeightNetwork
from ._recurrent_end_2d import RecurrentEND2D, RecurrentResidualEND2D

__all__ = [
    "Decoder",
    "RecurrentEND2D",
    "RecurrentResidualEND2D",
    "MatchingResidualDecoder",
    "NeuralWeightedMatchingDecoder",
    "RecurrentEdgeWeightNetwork",
]
