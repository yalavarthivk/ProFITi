"""Attention mechanisms for ProFITi."""

import torch
from torch import nn, Tensor
from typing import Optional


class TriangularAttention(nn.Module):
    """Efficient triangular attention mechanism for normalizing flows."""

    def __init__(self, hidden_dim: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.scale = hidden_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize with small weights for stability

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        """
        Compute triangular attention matrix.

        Args:
            hidden_states: [B, K, D] - Hidden states
            mask: [B, K] - Binary mask for valid positions

        Returns:
            Lower triangular attention matrix [B, K, K]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute queries and keys
        query = self.q_proj(hidden_states)  # [B, K, D]
        key = self.k_proj(hidden_states)  # [B, K, D]

        # Compute attention scores
        scores = torch.bmm(query, key.transpose(-2, -1)) * self.scale  # [B, K, K]

        id_tensor = (
            torch.eye(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Apply masking
        scores_masked_nondiagonal = scores.masked_fill(id_tensor == 0, -1e8)
        scores_masked_diagonal = scores.masked_fill(id_tensor == 1, 0.0)

        # Apply softplus to ensure positive values
        positive_diagonal_scores = torch.nn.functional.softplus(
            scores_masked_nondiagonal
        )

        scores_combined = positive_diagonal_scores + scores_masked_diagonal

        # Zero out invalid positions and add small epsilon for numerical stability
        attention_mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(-2))
        scores_combined = scores_combined * attention_mask + 1e-3

        return torch.tril(scores_combined)

    def log_determinant(self, attention_matrix: Tensor, mask: Tensor) -> Tensor:
        """
        Compute log determinant of triangular attention matrix.

        Args:
            attention_matrix: [B, K, K] - Triangular attention matrix
            mask: [B, K] - Binary mask for valid positions

        Returns:
            Log determinant [B]
        """
        # Extract diagonal elements
        diagonal = torch.diagonal(attention_matrix, dim1=-2, dim2=-1)  # [B, K]

        # Only consider valid positions
        masked_diagonal = diagonal * mask + 1 - mask

        # Compute log determinant (sum of log diagonal elements)
        log_det = torch.sum(torch.log(masked_diagonal), dim=-1)  # [B]

        return log_det
