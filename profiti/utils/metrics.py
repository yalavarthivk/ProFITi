import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import properscoring as ps
import torch.nn as nn
import torch.nn.functional as F


class compute_jnll(nn.Module):
    def __init__(self, std=1):
        super(compute_jnll, self).__init__()
        self.std = std
        self.gaussianl_nll = nn.GaussianNLLLoss(full=True, reduction="none")

    def gnll(self, z: Tensor) -> Tensor:
        mean = torch.zeros_like(z)
        var = torch.ones_like(z) * (self.std**2)
        return self.gaussianl_nll(mean, z, var)

    def forward(self, z, mask, ldj):
        gnll = self.gnll(z) * mask  # B x (K)
        jnll = gnll.sum(-1) - ldj  # B
        return jnll


class compute_mnll(nn.Module):
    def __init__(self, std=1):
        super(compute_mnll, self).__init__()
        self.var = std**2
        self.gaussianl_nll = nn.GaussianNLLLoss(full=True, reduction="none")

    def gnll(self, z: Tensor) -> Tensor:
        mean = torch.zeros_like(z)
        var = torch.ones_like(z) * (self.var)
        return self.gaussianl_nll(mean, z, var)

    def forward(self, z, mask, ldj):
        gnll = self.gnll(z) * mask  # B x (K)
        mnll = (
            gnll.sum(-1) - ldj
        )  # B ideally in mnll we should have BxK as we are trying to predict for each variable.
        # Howevery, ldj is sum of all that of univariates hence we sum gnll as well.
        return mnll


def compute_crps(
    y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute CRPS for a batch with multiple samples and masked elements.

    Args:
        y: Tensor of shape [BxK] - Ground truth values
        yhat: Tensor of shape [BxSxK] - Predicted values (ensemble)
        mask: Tensor of shape [BxK] - Binary mask indicating valid positions

    Returns:
        Tensor of shape [batch, dim] with masked CRPS values
    """

    # Compute absolute differences
    y_expanded = y.unsqueeze(1)  # [B, 1, K]
    term1 = torch.mean(torch.abs(y_expanded - yhat), dim=1)  # [B, K]

    # Compute pairwise differences between ensemble members
    yhat1 = yhat.unsqueeze(2)  # [B, S, 1, K]
    yhat2 = yhat.unsqueeze(1)  # [B, 1, S, K]
    term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(1, 2))  # [B, K]

    crps = term1 - term2

    # Apply mask
    return crps * mask


def compute_energy_score(y, yhat, mask, beta=1):
    """
    y: [batch, dim]
    yhat: [batch, nsamples, dim]
    mask: optional, [batch, dim]
    """
    _, nsamples, _ = yhat.shape

    valid_batch = mask.sum(dim=1) > 0
    y = y[valid_batch] * mask[valid_batch]
    yhat = yhat[valid_batch] * mask[valid_batch, None, :]

    # First term: mean over samples
    diff = torch.cdist(yhat, y[:, None, :], p=2)  # [batch, nsamples, 1]
    first_term = diff.pow(beta).mean(dim=1).squeeze(-1)  # mean over nsamples

    # Second term: pairwise distances, exclude diagonal
    pairwise = torch.cdist(yhat, yhat, p=2)
    diag = torch.diagonal(pairwise, dim1=1, dim2=2)
    pairwise = pairwise - torch.diag_embed(diag)
    second_term = -pairwise.pow(beta).sum(dim=(1, 2)) / (2 * nsamples**2)

    energy = first_term + second_term
    return energy  # per batch


class ComputePointErrors:
    """Utility class for computing pointwise errors."""

    def __init__(self):
        pass

    @staticmethod
    def mse(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes elementwise MSE only on valid elements.
        Args:
            y: Ground truth tensor - BxK.
            yhat: Predicted tensor - BxK.
            mask: Binary mask indicating valid positions - BxK.
        Returns:
            MSE tensor - BxK.
        """
        diff = yhat - y
        diff = diff * mask  # zero out invalid entries
        diff = diff**2
        return diff.mean()  # average over K and B

    @staticmethod
    def mae(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes elementwise MAE only on valid elements.
        Args:
            y: Ground truth tensor.
            yhat: Predicted tensor.
            mask: Binary mask indicating valid positions.
        Returns:
            MAE tensor.
        """
        diff = yhat - y
        diff = diff * mask
        return diff.abs().mean()  # average over K and B
