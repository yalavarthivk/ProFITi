import torch
import numpy as np
from .layers import dense_layers, grafiti
from torch import nn, Tensor
from .utils import compute_jNLL


class shiesh:
    def __init__(self, t=1.0, a=1.0):
        self.t = t
        self.a = a
        self.slope = np.exp(a * t)

    def shiesh_(self, x):
        shiesh_x = torch.arcsinh(self.slope * torch.sinh(self.a * x)) / self.a
        return shiesh_x

    def shiesh_log_jac(self, x):
        den = (1 + self.slope * torch.sinh(self.a * x)) ** 2
        num = self.slope * torch.cosh(self.a * x)
        return torch.log(num) - torch.log(den)

    def forward(self, x):
        large_inds = torch.abs(x) > 5
        activation_out = torch.where(large_inds, x + t*torch.sign(x), self.shiesh_(x))
        activation_log_jac = torch.where(
            large_inds, torch.zeros_like(x), self.shiesh_log_jac(x)
        )
        return activation_out, activation_log_jac


class ProFITi(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, f_layers, attn_head, device):
        super(ProFITi, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.f_layers = f_layers
        self.attn_head = attn_head
        self.device = device
        self.cond_mod = grafiti(
            input_dim, latent_dim, n_layers, f_layers, attn_head, device
        )
        self.phi_init = nn.Linear(latent_dim, 1)
        self.theta = nn.ModuleList()
        self.phi = nn.ModuleList()
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.tanh = nn.Tanh()
        self.shiesh = shiesh(t=1.0, a=1.0).forward
        self.shiesh_inv = shiesh(t=-1.0, a=1.0).forward
        self.jnll = compute_jNLL()
        self.tri_attn = []
        self.theta_scale = []
        self.phi_transl = []
        self.Z = None
        self.LDJ = None
        self.Z_inv = None
        self.LDJ_inv = None
        self.H = None
        self.EL0 = None
        for i in range(self.f_layers):
            self.theta.append(dense_layers(2, self.latent_dim, device=device))
            self.phi.append(dense_layers(2, self.latent_dim, device=device))
            self.q_proj.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.k_proj.append(nn.Linear(self.latent_dim, self.latent_dim))

    def sita(self, hidden_states, mq, id_tensor, idx):
        query = self.q_proj[idx](hidden_states)
        key = self.k_proj[idx](hidden_states)

        scale = query.shape[-1] ** -0.5
        scores = torch.bmm(query, key.transpose(-2, -1)) * scale

        # Masking logic
        b_masked = scores.masked_fill(id_tensor == 0, -1e8)
        softplus_b = torch.nn.functional.softplus(b_masked)

        a_masked = scores.masked_fill(id_tensor == 1, 0)
        a_masked += softplus_b
        a_masked += id_tensor * 0.001

        return torch.tril(a_masked)

    def distribution(
        self,
        TX: Tensor,
        CX: Tensor,
        X: Tensor,
        MX: Tensor,
        TQ: Tensor,
        CQ: Tensor,
        MQ: Tensor,
    ):
        self.H = self.cond_mod(TX, CX, X, MX, TQ, CQ, MQ)
        self.EL0 = self.phi_init(self.H).sequeeze(-1)
        batch_size = MQ.shape[0]
        id_tensor = (
            torch.eye(MQ.shape[1])[None, :, :].repeat(batch_size, 1, 1).to(X.device)
        )
        for i in range(self.f_layers):
            self.tri_attn.append(self.sita(self.H, MQ, id_tensor, i))
            self.theta_scale.append(torch.exp(nn.Tanh()(self.theta[i](self.H))))
            self.phi_transl.append(self.phi[i](self.H))

    def compute_flow(self, Y, MQ):
        Z = Y * MQ
        LDJ = torch.oneslike(Z)
        Z = Z - self.EL0
        Z *= MQ
        for i in range(self.f_layers):
            Z = torch.matmul(self.tri_attn[i], Z)
            LDJ += self.sita_det(self.tri_attn[i], MQ)
            Z *= MQ

            Z = Z * self.theta_scale[i] + self.phi_scale[i]
            LDJ += torch.sum(self.theta_scale[i] * MQ, -1)

            Z, shiesh_ldj = self.shiesh(Z)
            LDJ += shiesh_ldj * MQ
        return Z, LDJ

    def compute_flow_inv(self, Z, MQ):
        Y = Z * MQ
        batch_size = Y.shape[0]
        LDJ = torch.zeros(batch_size, device=Z.device)

        for i in range(self.f_layers - 1, -1, -1):
            Y, shiesh_inv_ldj = self.shiesh_inv(Y)
            LDJ += torch.sum(shiesh_inv_ldj * MQ, -1)

            Y = Y / self.phi_scale[i] - self.phi_transl[i]
            LDJ += torch.sum(self.phi_scale[i] * MQ, -1)
            Y *= MQ

            Y = torch.torch.linalg.solve_triangular(self.tri_attn[i], Y, upper=False)
            LDJ -= self.sita_det(self.tri_attn, MQ)
            Y *= MQ
        Y = Y + self.EL0
        return Y, LDJ

    def compute_njNLL(self, Y, MQ):
        Z, LDJ = self.compute_flow(Y, MQ)
        neg_loglikelihood = self.jnll(Z, MQ, LDJ)
        return neg_loglikelihood

    def samples(self, MQ, nsamples=100):
        batch_size = MQ.shape[0]
        seq_len = MQ.shape[1]
        Z = torch.random.randn([batch_size * nsamples, seq_len], device=MQ.device)
        MQ_ = MQ.repeat(nsamples, 1)
        Y, LDJ = self.compute_flow_inv(Z, MQ_)
        Y = Y.reshape(batch_size, -1, seq_len)
        LDJ = LDJ.reshape(batch_size, -1)
        return Y, LDJ

    def mean(self, MQ):
        Y, _ = self.samples(MQ, nsamples=100)
        return torch.mean(Y, dim=1)

    def robust_mean(self, MQ, nsamples=100):
        Y, _ = self.samples(MQ, nsamples=nsamples)
        return torch.mean(Y, dim=1)
