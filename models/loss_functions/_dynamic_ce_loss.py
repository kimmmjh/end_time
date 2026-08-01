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
        # Initialize with 1.0 (Laplace smoothing) to avoid division by zero
        self.register_buffer(
            "logit_counter", torch.ones(size=(tensor_size,), device=device)
        )
        # Start global counter at tensor_size (as if we saw 1 of each class)
        self.register_buffer(
            "global_counter", torch.tensor(float(tensor_size), device=device)
        )
        self.num_classes = tensor_size

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        The forward pass of the loss function.

        :param output: The output tensor of the NN in form of (b, tensor_size).
        :param target: The target tensor in the same form.
        """
        """Calculating weights and updating counters."""
        # Evaluation must be read-only. In the previous implementation every
        # validation pass changed the class-frequency state and therefore the
        # loss used by subsequent training epochs.
        with torch.no_grad():
            if target.ndim == 1:
                # Target is indices (B,)
                counts = F.one_hot(target, num_classes=self.num_classes).sum(dim=0).float()
            else:
                # Target is one-hot (B, C) - this shouldn't happen with current config but good for safety
                counts = target.sum(dim=0).float()
                
            if self.training:
                self.logit_counter.add_(counts)
                self.global_counter.add_(target.shape[0])
        
        # Calculate inverse frequency weights
        weights = (self.global_counter / (self.logit_counter * self.num_classes))
        
        # Clamp weights to avoid explosion for extremely rare classes
        weights = torch.clamp(weights, max=100.0)

        """Calculating loss."""
        loss = F.cross_entropy(output, target, weights)
        return loss
