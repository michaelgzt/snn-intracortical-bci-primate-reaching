"""
Loss Functions for SNN Training

Temporally-weighted MSE loss that weights later timesteps more heavily,
as SNN membrane potentials need time to stabilize.
"""

import torch
import torch.nn as nn


def temporal_weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute MSE loss with linearly increasing temporal weights.

    Early timesteps have lower weight because the SNN membrane potentials
    haven't stabilized yet. Later timesteps have higher weight.

    Weight schedule: [0, 1/window, 2/window, ..., 1]

    Implementation: loss = einsum("abc, c->abc", mse, weights).mean()

    Args:
        predictions: Model output, shape (batch, 2, window)
        targets: Ground truth, shape (batch, 2, window)
        window: Number of timesteps
        device: Torch device

    Returns:
        Scalar loss value
    """
    # Compute element-wise squared error
    mse_fn = nn.MSELoss(reduction="none")
    mse_val = mse_fn(predictions, targets)

    # Create temporal weights from 0 to 1
    weights = torch.linspace(0, 1, steps=window, device=device)

    # Apply weights along time dimension and average
    weighted_loss = torch.einsum("abc, c -> abc", mse_val, weights)

    return weighted_loss.mean()


class TemporalWeightedMSELoss(nn.Module):
    """Module wrapper for temporal weighted MSE loss."""

    def __init__(self, window: int):
        """
        Args:
            window: Number of timesteps
        """
        super().__init__()
        self.window = window
        self.register_buffer("weights", torch.linspace(0, 1, steps=window))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal weighted MSE loss.

        Args:
            predictions: Model output, shape (batch, 2, window)
            targets: Ground truth, shape (batch, 2, window)

        Returns:
            Scalar loss value
        """
        mse_fn = nn.MSELoss(reduction="none")
        mse_val = mse_fn(predictions, targets)
        weighted_loss = torch.einsum("abc, c -> abc", mse_val, self.weights)
        return weighted_loss.mean()
