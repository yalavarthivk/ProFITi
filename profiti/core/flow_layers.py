"""Flow transformation layers."""

import torch
from torch import nn, Tensor
from typing import Tuple, List
from shiesh import Shiesh

from .attention import TriangularAttention
from ..layers import dense_layers


class FlowLayer(nn.Module):
    """Single normalizing flow layer with triangular attention and affine transformation."""

    def __init__(self, hidden_dim: int, marginal_training: bool, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        # Components
        self.attention = TriangularAttention(hidden_dim, marginal_training, device)
        self.scale_net = dense_layers(2, hidden_dim, device=device)
        self.shift_net = dense_layers(2, hidden_dim, device=device)

        self.tanh = nn.Tanh()
        self.shiesh = Shiesh(t=1.0, a=1.0)
        self.shiesh_inv = Shiesh(t=-1.0, a=1.0)

    def forward(
        self, z: Tensor, hidden_states: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward flow transformation.

        Args:
            z: [B, K] - Input tensor
            hidden_states: [B, K, D] - Conditioning information
            mask: [B, K] - Binary mask

        Returns:
            Transformed z and log determinant jacobian
        """
        batch_size = z.shape[0]
        ldj = torch.zeros(batch_size, device=self.device)

        # Apply triangular attention transformation
        attention_matrix = self.attention(hidden_states, mask)
        z = torch.bmm(attention_matrix, z.unsqueeze(-1)).squeeze(-1)
        z = z * mask

        # Add attention log determinant
        ldj += self.attention.log_determinant(attention_matrix, mask)

        # Apply affine transformation
        scale = self.tanh(self.scale_net(hidden_states)).squeeze(-1)
        shift = self.shift_net(hidden_states).squeeze(-1)

        z = z * torch.exp(scale) + shift
        z = z * mask

        # Add affine log determinant
        ldj += torch.sum(scale * mask, dim=-1)

        # Apply activation function
        z, shiesh_ldj = self.shiesh(z)
        ldj += torch.sum(shiesh_ldj * mask, dim=-1)

        return z, ldj

    def inverse(
        self, z: Tensor, hidden_states: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Inverse flow transformation.

        Args:
            z: [B, K] - Input tensor
            hidden_states: [B, K, D] - Conditioning information
            mask: [B, K] - Binary mask

        Returns:
            Inverse transformed z and log determinant jacobian
        """
        batch_size = z.shape[0]
        ldj = torch.zeros(batch_size, device=self.device)

        # Inverse activation
        z, shiesh_inv_ldj = self.shiesh_inv(z)
        ldj += torch.sum(shiesh_inv_ldj * mask, dim=-1)

        # Inverse affine transformation
        scale = self.tanh(self.scale_net(hidden_states)).squeeze(-1)
        shift = self.shift_net(hidden_states).squeeze(-1)

        z = (z - shift) / torch.exp(scale)
        z = z * mask
        ldj -= torch.sum(scale * mask, dim=-1)

        # Inverse triangular attention
        diag_ones = torch.diag_embed(1 - mask)

        attention_matrix = (
            self.attention(hidden_states, mask)
        ) + diag_ones  # add 1s on diagonal for masked positions to make it non-singular

        z = torch.linalg.solve_triangular(
            attention_matrix, z.unsqueeze(-1), upper=False
        ).squeeze(-1)
        z = z * mask

        # Subtract attention log determinant
        ldj -= self.attention.log_determinant(attention_matrix, mask)

        return z, ldj


class NormalizingFlow(nn.Module):
    """Multi-layer normalizing flow."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        marginal_training: bool,
        device: torch.device,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.device = device

        self.layers = nn.ModuleList(
            [
                FlowLayer(hidden_dim, marginal_training, device)
                for _ in range(num_layers)
            ]
        )

        # Base distribution offset
        self.el0 = nn.Linear(hidden_dim, 1)

    def forward(
        self, y: Tensor, hidden_states: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply forward flow transformation."""
        z = y * mask
        ldj = torch.zeros(y.shape[0], device=self.device)

        # Apply base offset
        offset = self.el0(hidden_states).squeeze(-1)
        z = (z - offset) * mask

        # Apply flow layers
        for layer in self.layers:
            z, layer_ldj = layer.forward(z, hidden_states, mask)
            ldj += layer_ldj

        return z, ldj

    def inverse(
        self, z: Tensor, hidden_states: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply inverse flow transformation."""
        y = z * mask
        ldj = torch.zeros(z.shape[0], device=self.device)

        # Apply inverse flow layers (in reverse order)
        for layer in reversed(self.layers):
            y, layer_ldj = layer.inverse(y, hidden_states, mask)
            ldj += layer_ldj

        # Apply inverse base offset
        offset = self.el0(hidden_states).squeeze(-1)
        y = (y + offset) * mask

        return y, ldj
