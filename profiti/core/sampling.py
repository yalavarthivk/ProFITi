# File: profiti/core/sampling.py
"""Sampling utilities for ProFITi."""

import torch
from torch import Tensor
from typing import Tuple, Optional


class ProFITiSampler:
    """Efficient sampling utilities for ProFITi model."""

    def __init__(self, model: "ProFITi"):
        self.model = model

    def sample(self, mask: Tensor, num_samples: int = 100) -> Tuple[Tensor, Tensor]:
        """
        Generate samples from the model.

        Args:
            mask: [B, K] - Binary mask for valid positions
            num_samples: Number of samples to generate

        Returns:
            samples: [B, num_samples, K] - Generated samples
            log_probs: [B, num_samples] - Log probabilities
        """
        batch_size, seq_len = mask.shape
        device = mask.device

        # Generate base samples
        base_samples = torch.randn(batch_size * num_samples, seq_len, device=device)

        # Repeat mask for all samples
        mask_expanded = mask.repeat(num_samples, 1)

        # Apply inverse flow
        samples, log_det = self.model.flow.inverse(
            base_samples,
            self.model.hidden_states.repeat(num_samples, 1, 1),
            mask_expanded,
        )

        # Reshape outputs
        samples = samples.view(batch_size, num_samples, seq_len)
        log_det = log_det.view(batch_size, num_samples)

        return samples, log_det

    def mean(self, mask: Tensor, num_samples: int = 1000) -> Tensor:
        """Compute empirical mean via sampling."""
        samples, _ = self.sample(mask, num_samples)
        return torch.mean(samples, dim=1)

    def median(self, mask: Tensor) -> Tensor:
        """Compute median of univariate by transforming zero (median of standard normal)."""
        batch_size, seq_len = mask.shape
        device = mask.device

        # Median of standard normal is 0
        z_median = torch.zeros(batch_size, seq_len, device=device)

        # Transform to get median in original space
        y_median, _ = self.model.flow.inverse(z_median, self.model.hidden_states, mask)

        return y_median
