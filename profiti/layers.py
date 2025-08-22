import torch
from torch import nn, Tensor


def dense_layers(
    n_layers: int = 2, latent_dim: int = 2, device: str = "cuda"
) -> nn.Sequential:
    """Creates a sequence of dense layers.
    Args:
        n_layers (int): Number of layers.
        latent_dim (int): Dimension of the latent space.
        device (str): Device to place the model on.
    Returns:
        nn.Sequential: A sequential model with dense layers.
    """
    layers = []
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(latent_dim, latent_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(latent_dim, 1))
    return nn.Sequential(*layers).to(device)
