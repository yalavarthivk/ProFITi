import math
import pdb

import numpy as np
import torch
from torch import Tensor, nn

from grafiti.grafiti import GraFITi
from profiti.change_rep import obs_rep
from profiti.layers import dense_layers
from profiti.utils import compute_jnll

# from profiti.shiesh_act_original import Shiesh

from shiesh import Shiesh


#


class ProFITi(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, f_layers, attn_head, device):
        """
        Initializes the ProFITi model.
        Parameters:
        input_dim: int - Number of input features.
        latent_dim: int - Size of the hidden layers.
        n_layers: int - Number of layers in the conditional module.
        f_layers: int - Number of flow layers.
        attn_head: int - Number of attention heads for the GraFITi module.
        device: str - Device to run the model on ('cuda' or 'cpu').
        """
        super(ProFITi, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.f_layers = f_layers
        self.attn_head = attn_head
        self.device = device
        self.input_dim = input_dim
        self.cond_mod = GraFITi(input_dim, attn_head, latent_dim, n_layers, device)
        self.phi_init = nn.Linear(latent_dim, 1)
        self.theta = nn.ModuleList()
        self.phi = nn.ModuleList()
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.tanh = nn.Tanh()
        self.shiesh = Shiesh(t=1.0, a=1.0)
        self.shiesh_inv = Shiesh(t=-1.0, a=1.0)
        self.jnll = compute_jnll()
        self.tri_attn = []
        self.theta_scale = []
        self.phi_transl = []
        self.z = None
        self.ldj = None
        self.z_inv = None
        self.ldj_inv = None
        self.h = None
        self.el0 = None
        self.id_tensor = None
        self.tri_attn_mask = None

        for _ in range(self.f_layers):
            self.theta.append(dense_layers(2, self.latent_dim, device=device))
            self.phi.append(dense_layers(2, self.latent_dim, device=device))
            self.q_proj.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.k_proj.append(nn.Linear(self.latent_dim, self.latent_dim))

    def sita(self, hidden_states, id_tensor, idx):
        """
        Computes the attention matrix for the ProFITi model.
        Parameters:
        hidden_states: Tensor - Hidden states from the conditional module.
        id_tensor: Tensor - Identity tensor for the attention mechanism.
        idx: int - Index for the current layer.
        Returns:
        Tensor - Attention matrix for the current layer.
        """

        query = self.q_proj[idx](hidden_states)
        key = self.k_proj[idx](hidden_states)

        scale = query.shape[-1] ** -0.5
        scores = torch.bmm(query, key.transpose(-2, -1)) * scale

        b_masked = scores.masked_fill(id_tensor == 0, -1e8)
        softplus_b = torch.nn.functional.softplus(b_masked)

        a_masked = scores.masked_fill(id_tensor == 1, 0)
        a_masked += softplus_b
        a_masked += id_tensor * 0.001

        return torch.tril(a_masked)

    def sita_det(self, tri_attn):
        """
        Computes the determinant of the lower triangular attention matrix.
        Parameters:
        tri_attn: Tensor - Lower triangular attention matrix.
        Returns:
        Tensor - Logarithmic determinant of the attention matrix.
        """
        # Create a copy to avoid modifying the original tensor
        tri_attn_masked = tri_attn * self.tri_attn_mask  # Non-in-place operation
        tri_attn_masked = (
            tri_attn_masked + (1 - self.tri_attn_mask) * self.id_tensor
        )  # Non-in-place operation

        log_det = torch.sum(
            torch.log(torch.diagonal(tri_attn_masked, dim1=-2, dim2=-1)), -1
        )  # Compute the log determinant
        return log_det

    def distribution(
        self,
        tx: Tensor,
        cx: Tensor,
        x: Tensor,
        mx: Tensor,
        tq: Tensor,
        cq: Tensor,
        mq: Tensor,
    ):
        """
        Computes the distribution parameters for the ProFITi model.
        Parameters:
        tx: Tensor - Time features for the input.
        cx: Tensor - Channel ID for the input.
        x: Tensor - Observed values for the input.
        mx: Tensor - Mask for the input features.
        tq: Tensor - Time features for the query.
        cq: Tensor - Channel ID for the query.
        mq: Tensor - Mask for the query features.
        """

        # Compute the hidden states from the conditional module
        # Make input features compatible with the conditional module
        # GraFITi takes timepoints (BxT), values (BxTxD), obs_masks (BxTxD), target masks (BxKXD)
        t_updated, x_updated, m_updated, mq_updated = obs_rep(
            tx, cx, mx, x, tq, cq, mq, self.input_dim
        )

        self.h = self.cond_mod(t_updated, x_updated, m_updated, mq_updated)

        self.el0 = self.phi_init(self.h).squeeze(-1)
        batch_size = mq.shape[0]
        id_tensor = (
            torch.eye(mq.shape[1])[None, :, :].repeat(batch_size, 1, 1).to(x.device)
        )
        self.tri_attn_mask = torch.matmul(
            mq[:, :, None], mq[:, None, :]
        )  # mask for jacobian
        self.id_tensor = id_tensor

        # Initialize as empty lists each time
        self.tri_attn = []
        self.theta_scale = []
        self.phi_transl = []

        for i in range(self.f_layers):
            self.tri_attn.append(self.sita(self.h, id_tensor, i))
            self.theta_scale.append(self.tanh(self.theta[i](self.h)).squeeze(-1))
            self.phi_transl.append(self.phi[i](self.h).squeeze(-1))

    def compute_flow(self, y, mq):
        z = y * mq  # Initialize z with y and mask y,z,mq \in [B, K]
        ldj = torch.zeros_like(
            z[:, 0]
        )  # Initialize log determinant of Jacobian ldj \in [B]
        z = z - self.el0  # Subtract the initial value from z \in [B, K]
        z = z * mq  # Apply mask to z \in [B, K]

        for i in range(self.f_layers):
            z = torch.matmul(self.tri_attn[i], z.unsqueeze(-1)).squeeze(
                -1
            )  # Apply attention matrix; z \in [B, K]
            sita_ldj = self.sita_det(
                self.tri_attn[i]
            )  # Compute log determinant of attention matrix; sita_ldj \in [B]
            ldj = ldj + sita_ldj  # Add log determinant of the attention matrix
            z = z * mq  # Apply mask to z

            z = (
                z * torch.exp(self.theta_scale[i]) + self.phi_transl[i]
            )  # Apply scaling and translation; z \in [B, K]
            ldj = ldj + torch.sum(
                self.theta_scale[i] * mq, -1
            )  # Add to log determinant of Jacobian
            z = z * mq  # Apply mask to z
            z, shiesh_ldj = self.shiesh(z)  # Apply Shiesh activation
            # print(torch.max(shiesh_ldj).item(), torch.min(shiesh_ldj).item())
            ldj = ldj + torch.sum(shiesh_ldj * mq, -1)  # Add to log determinant

        return z, ldj

    def compute_flow_inv(self, z, mq):
        """
        Computes the inverse flow transformation for the ProFITi model.
        Parameters:
        z: Tensor - Input tensor to be transformed.
        mq: Tensor - Mask for the query features.
        Returns:
        y: Tensor - Transformed tensor after applying the inverse flow.
        ldj: Tensor - Logarithmic determinant of the Jacobian.
        """
        y = z * mq  # Initialize y with z and mask
        batch_size = y.shape[0]  # Get the batch size from y
        ldj = torch.zeros(
            batch_size, device=z.device
        )  # Initialize log determinant of Jacobian

        for i in range(self.f_layers - 1, -1, -1):  # Iterate in reverse order
            y, shiesh_inv_ldj = self.shiesh_inv(y)  # Apply inverse Shiesh activation
            ldj += torch.sum(shiesh_inv_ldj * mq, -1)  # Add to log determinant

            y = (y - self.phi_transl[i]) / torch.exp(
                self.theta_scale[i]
            )  # Apply inverse scaling and translation
            ldj += torch.sum(
                -self.theta_scale[i] * mq, -1
            )  # subtract to log determinant
            y *= mq  # Apply mask to y

            y = torch.linalg.solve_triangular(
                self.tri_attn[i], y.unsqueeze(-1), upper=False
            ).squeeze(
                -1
            )  # Solve triangular system
            ldj -= self.sita_det(
                self.tri_attn[i]
            )  # Subtract log determinant of attention matrix, same as adding the inverse of the determinant
            y *= mq  # Apply mask to y

        y = y + self.el0  # Add the initial value to y
        return y, ldj

    def compute_njNLL(self, y, mq):
        """
        Computes the Normalized joint negative log-likelihood for the ProFITi model.
        Parameters:
        y: Tensor - Input tensor to be transformed.
        mq: Tensor - Mask for the query features.
        Returns:
        njnll: Tensor - Normalized joint negative log-likelihood.
        """
        z, ldj = self.compute_flow(y, mq)  # Apply the flow transformation

        # Compute the joint negative log-likelihood
        joint_neg_log_likelihood = self.jnll(
            z, mq, ldj
        )  # Compute the joint negative log-likelihood
        njnll = joint_neg_log_likelihood / mq.sum(
            -1
        )  # Normalize by the sum of the mask
        return njnll

    def samples(self, mq, nsamples=100):
        """
        Generates samples from the ProFITi model.
        Parameters:
        mq: Tensor - Mask for the query features.
        nsamples: int - Number of samples to generate.
        Returns:
        y: Tensor - Generated samples.
        ldj: Tensor - Logarithmic determinant of the Jacobian for the generated samples.
        """
        batch_size, seq_len = (
            mq.shape
        )  # Get the batch size and sequence length from the mask
        z = torch.randn(
            [batch_size * nsamples, seq_len], device=mq.device
        )  # Generate random samples from the base distribution , batch_size is repeated for the number of samples
        mq_rep = mq.repeat(nsamples, 1)  # Repeat the mask for the number of samples
        y, ldj = self.compute_flow_inv(
            z, mq_rep
        )  # Apply the inverse flow transformation
        y = y.reshape(
            batch_size, nsamples, seq_len
        )  # Reshape the output to match the batch size and number of samples
        ldj = ldj.reshape(
            batch_size, nsamples
        )  # Reshape the log determinant to match the batch size and number of samples
        return y, ldj

    def mean(self, mq):
        """
        Computes the mean of the samples generated by the ProFITi model.
        Parameters:
        mq: Tensor - Mask for the query features.
        Returns:
        y_mean: Tensor - Empirical Mean of the distribution.
        """
        y, _ = self.samples(mq, nsamples=100)  # Generate samples from the model
        y_mean = torch.mean(y, dim=1)  # Compute the mean across the samples
        return y_mean

    def robust_mean(self, mq, nsamples=100):
        """
        Computes the robust mean of the samples generated by the ProFITi model.
        Parameters:
        mq: Tensor - Mask for the query features.
        nsamples: int - Number of samples to generate.
        Returns:
        Tensor - Robust Mean of the distribution.
        """
        y, _ = self.samples(mq, nsamples=nsamples)  #   Generate samples from the model
        # todo implement robust mean
        return torch.mean(y, dim=1)

    def median(self, mq):
        """
        Computes the median of the samples generated by the ProFITi model.
        Parameters:
        mq: Tensor - Mask for the query features.
        Returns:
        Tensor - Empirical Median of each variable in the distribution.
        """
        z = torch.zeros_like(
            mq
        )  # Initialize z with zeros, median of each variable is the transformed value of median of base distribution
        y, _ = self.compute_flow_inv(z, mq)  # Apply the inverse flow transformation
        return y
