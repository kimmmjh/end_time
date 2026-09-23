"""Different network architectures."""

from ._decoder import Decoder
from ._bb_tanner_cnn import BBTannerCNN
from ._equivariant_neural_bp import EquivariantNeuralBP4
from ._equivariant_neural_bp2 import BPDecodeResult, EquivariantNeuralBP2
from ._neural_relay_bp2 import NeuralRelayBP2, RelayDecodeResult
from ._matching_residual_decoder import MatchingResidualDecoder
from ._neural_weighted_matching import NeuralWeightedMatchingDecoder
from ._recurrent_edge_weights import RecurrentEdgeWeightNetwork
from ._recurrent_end_2d import RecurrentEND2D, RecurrentResidualEND2D

__all__ = [
    "Decoder",
    "BBTannerCNN",
    "EquivariantNeuralBP4",
    "EquivariantNeuralBP2",
    "BPDecodeResult",
    "NeuralRelayBP2",
    "RelayDecodeResult",
    "RecurrentEND2D",
    "RecurrentResidualEND2D",
    "MatchingResidualDecoder",
    "NeuralWeightedMatchingDecoder",
    "RecurrentEdgeWeightNetwork",
]
