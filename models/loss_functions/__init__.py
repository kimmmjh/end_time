"""Custom Loss Functions."""
from ._dynamic_ce_loss import DynamicCELoss
from ._edge_bce_loss import EdgeBCELoss

__all__ = ["DynamicCELoss", "EdgeBCELoss"]
