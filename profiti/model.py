"""Main ProFITi model implementation."""

from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from grafiti.grafiti import GraFITi
from .change_rep import obs_rep
from .core.flow_layers import NormalizingFlow
from .core.sampling import ProFITiSampler
from .core.base_model import BaseFlowModel
from .utils import compute_crps, compute_energy_score, compute_jnll, compute_mnll
import pdb


class ProFITi(nn.Module, BaseFlowModel):
    """
    ProFITi: Probabilistic Forecasting with Invertible Time-series.

    A normalizing flow model for probabilistic time series forecasting
    with conditional transformations based on historical observations.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_layers: int,
        f_layers: int,
        attn_head: int,
        marginal_training: bool,
        device: torch.device,
    ):
        """
        Initialize ProFITi model.

        Args:
            input_dim: Number of input features
            latent_dim: Hidden dimension size
            n_layers: Number of conditional layers
            f_layers: Number of flow layers
            attn_head: Number of attention heads
            device: Computation device
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Components
        self.conditioning_module = GraFITi(
            input_dim, attn_head, latent_dim, n_layers, device
        )
        self.flow = NormalizingFlow(f_layers, latent_dim, marginal_training, device)
        self.likelihood = compute_jnll()
        self.marg_likelihood = compute_mnll()
        self._cached_samples = None  # to store generated samples
        self._cached_mask = None  # to know which mask the samples correspond
        self._cached_log_det = None  # to store log_det if needed later
        # Cache for hidden states
        self._hidden_states: Optional[Tensor] = None
        self.marginal_training = marginal_training

    @property
    def hidden_states(self) -> Optional[Tensor]:
        """Get cached hidden states."""
        return self._hidden_states

    def encode_context(
        self,
        tx: Tensor,
        cx: Tensor,
        x: Tensor,
        mx: Tensor,
        tq: Tensor,
        cq: Tensor,
        mq: Tensor,
    ) -> Tensor:
        """
        Encode context information into hidden states.

        Args:
            tx, cx, x, mx: Context time, channels, values, mask
            tq, cq, mq: Query time, channels, mask

        Returns:
            Hidden states [B, K, D]
        """
        # Transform observations to compatible format
        t_updated, x_updated, m_updated, mq_updated = obs_rep(
            tx, cx, mx, x, tq, cq, mq, self.input_dim
        )

        # Compute hidden states
        hidden_states = self.conditioning_module(
            t_updated, x_updated, m_updated, mq_updated
        )

        # Cache for later use
        self._hidden_states = hidden_states

    def forward_flow(self, y: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply forward flow transformation."""
        if self._hidden_states is None:
            raise RuntimeError("Must call encode_context first")

        return self.flow.forward(y, self._hidden_states, mask)

    def inverse_flow(self, z: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply inverse flow transformation."""
        if self._hidden_states is None:
            raise RuntimeError("Must call encode_context first")

        return self.flow.inverse(z, self._hidden_states, mask)

    def distribution(
        self,
        tx: Tensor,
        cx: Tensor,
        x: Tensor,
        mx: Tensor,
        tq: Tensor,
        cq: Tensor,
        mq: Tensor,
    ) -> None:
        """
        Compute distribution parameters (for compatibility with training script).

        This method encodes the context and prepares the model for likelihood computation.
        """
        self.encode_context(tx, cx, x, mx, tq, cq, mq)
        self._cached_samples = None  # to store generated samples
        self._cached_mask = None  # to know which mask the samples correspond
        self._cached_log_det = None  # to store log_det if needed later

    def compute_njnll(self, y: Tensor, mask: Tensor) -> Tensor:
        """
        Compute normalized joint negative log-likelihood.

        Args:
            y: [B, K] - Target values
            mask: [B, K] - Binary mask for valid positions

        Returns:
            Normalized negative log-likelihood [B]
        """
        if self._hidden_states is None:
            raise RuntimeError("Must call distribution/encode_context first")

        # Apply forward flow
        z, ldj = self.forward_flow(y, mask)

        # Compute likelihood
        joint_nll = self.likelihood(z, mask, ldj)

        # Normalize by number of valid observations
        normalized_nll = joint_nll / mask.sum(dim=-1)

        return normalized_nll

    def compute_mnll(self, y: Tensor, mask: Tensor) -> Tensor:
        if self._hidden_states is None:
            raise RuntimeError("Must call distribution/encode_context first")

        # Apply forward flow
        z, ldj = self.forward_flow(y, mask)

        # Compute likelihood
        marg_nll = self.marg_likelihood(z, mask, ldj)

        return marg_nll.sum()  # B

    # Sampling interface (delegates to sampler)
    def samples(self, mask: Tensor, nsamples: int = 1000) -> Tensor:
        """Generate or return cached samples for a given mask."""
        # If we already have samples for this mask and nsamples, reuse them
        if (
            self._cached_samples is not None
            and self._cached_mask is not None
            and torch.equal(mask, self._cached_mask)
        ):
            return self._cached_samples, self._cached_log_det  # unused here

        # Otherwise, generate new samples
        batch_size, seq_len = mask.shape
        base_samples = torch.randn(batch_size * nsamples, seq_len, device=self.device)

        # Repeat mask for all samples
        mask_expanded = mask.repeat(nsamples, 1)

        # Apply inverse flow
        samples, log_det = self.flow.inverse(
            base_samples,
            self.hidden_states.repeat(nsamples, 1, 1),
            mask_expanded,
        )

        # Reshape outputs
        samples = samples.reshape(nsamples, batch_size, seq_len)
        samples = samples.permute(1, 0, 2)
        log_det = log_det.reshape(nsamples, batch_size)  # unused here
        log_det = log_det.permute(1, 0)
        self._cached_samples = samples
        self._cached_mask = mask.clone()  # store a copy
        self._cached_log_det = log_det  # store log_det if needed later
        return self._cached_samples, self._cached_log_det

    def model_mean(self, mask: Tensor, nsamples: int = 1000) -> Tensor:
        """Compute empirical mean."""
        yhat_samples, _ = self.samples(mask, nsamples=nsamples)
        yhat_mean = torch.mean(yhat_samples, dim=1)
        return yhat_mean

    def model_robust_mean(self, mask: Tensor, nsamples: int = 1000) -> Tensor:
        """Compute robust mean (currently same as regular mean)."""
        yhat_samples, _ = self.samples(mask, nsamples=nsamples)
        yhat_mean = torch.mean(yhat_samples, dim=1)
        # Placeholder for robust mean logic
        # TODO: Implement actual robust mean (e.g., trimmed mean)
        return yhat_mean

    def mse(self, y, mask, nsamples: int = 1000) -> Tensor:
        """Compute Mean Squared Error."""
        yhat_mean = self.model_mean(mask, nsamples=nsamples)
        sq_error = mask * (y - yhat_mean) ** 2
        return sq_error.sum() / mask.sum()

    def robust_mse(self, y, mask, nsamples: int = 1000) -> Tensor:
        """Compute Robust Mean Squared Error."""
        yhat_mean = self.model_robust_mean(mask, nsamples=nsamples)
        sq_error = mask * (y - yhat_mean) ** 2
        return sq_error.sum() / mask.sum()

    def crps(self, y: Tensor, mask: Tensor, nsamples: int = 1000) -> Tensor:
        """Compute CRPS."""
        yhat_samples, _ = self.samples(mask, nsamples=nsamples)
        crps_ = compute_crps(y, yhat_samples, mask)
        return crps_.sum() / mask.sum()

    def energy_score(self, y: Tensor, mask: Tensor, nsamples: int = 100) -> Tensor:
        """Compute energy score."""
        yhat_samples, _ = self.samples(mask, nsamples=nsamples)
        energy_score_ = compute_energy_score(y, yhat_samples, mask)
        return energy_score_.mean()
