# File: profiti/core/sampling.py
"""Sampling utilities for ProFITi."""

import pdb
from typing import Tuple

import torch
from torch import Tensor


class ProFITiSampler:
    """Efficient sampling utilities for ProFITi model."""

    def __init__(self):
        pass

    def sample(
        self, model, mask: Tensor, num_samples: int = 100
    ) -> Tuple[Tensor, Tensor]:
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
        samples, log_det = model.flow.inverse(
            base_samples,
            model.hidden_states.repeat(num_samples, 1, 1),
            mask_expanded,
        )

        # Reshape outputs
        samples = samples.view(batch_size, num_samples, seq_len)
        log_det = log_det.view(batch_size, num_samples)

        return samples, log_det
