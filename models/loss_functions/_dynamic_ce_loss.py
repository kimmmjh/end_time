import torch
from torch import nn, Tensor
from torch.functional import F


class DynamicCELoss(nn.Module):
    """A BCE Loss adaptation, that dynamically adapts class weights."""

    def __init__(self, tensor_size: int, device: torch.device) -> None:
        """
        Initialize the DynamicBCELoss.

        :param tensor_size: The size of tensor to accept.
        :param device: The device to use the loss on.
        """
        super(DynamicCELoss, self).__init__()
        self.logit_counter = torch.zeros(size=(tensor_size,), device=device) + 1e-6 # This counts the number of occurrences for certain logicals.
        self.global_counter = torch.zeros(size=(tensor_size,), device=device)  # This counts the total number of samples viewed.
        self.num_classes = tensor_size

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        The forward pass of the loss function.

        :param output: The output tensor of the NN in form of (b, tensor_size).
        :param target: The target tensor in the same form.
        """
        """Calculating weights and updating counters."""
        """Calculating weights and updating counters."""
        with torch.no_grad():
            if target.ndim == 1:
                # Target is indices (B,)
                counts = F.one_hot(target, num_classes=self.num_classes).sum(dim=0).float()
            else:
                # Target is one-hot (B, C) - this shouldn't happen with current config but good for safety
                counts = target.sum(dim=0).float()
                
            self.logit_counter = self.logit_counter + counts
            self.global_counter = self.global_counter + target.shape[0]
        weights = (self.global_counter / (self.logit_counter * self.num_classes))

        """Calculating loss."""
        loss = F.cross_entropy(output, target, weights)
        return loss
