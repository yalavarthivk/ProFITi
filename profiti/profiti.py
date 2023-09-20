import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from .layers import dense_layers, grafiti
from .utils import preprocess, reshape_

class ProFITi(nn.Module):
    def __init__(
        self,
        input_dim=41,
        attn_head=2,
        latent_dim = 32,
        n_layers=4,
        f_layers=2,
        device='cuda'):
        super(ProFITi, self).__init__()
        self.f_layers=f_layers
        self.device=device
        self.CM = grafiti(input_dim, latent_dim, n_layers,f_layers, attn_head, device=device)
        self.mu = nn.Linear(latent_dim, 1)
        self.theta = nn.ModuleList()
        self.phi = nn.ModuleList()
        self.q_proj = nn.ModuleList()
        self.k_proj = nn.ModuleList()
        self.tanh = nn.Tanh()
        self.mu = nn.ModuleList()
        for i in range(f_layers):
            self.theta.append(dense_layers(2, latent_dim))
            self.phi.append(dense_layers(2, latent_dim))
            self.q_proj.append(nn.Linear(latent_dim, latent_dim))
            self.k_proj.append(nn.Linear(latent_dim, latent_dim))


    def alpha(self, x, t=1):
        return torch.arcsinh(np.exp(t)*torch.sinh(x))
    
    def jfa(self, x, t=1):
        den = 1+(np.exp(t)*torch.sinh(x))**2
        jac = np.exp(t)*torch.cosh(x)/(den**0.5)
        return jac
    
    def fa(self, x, mask, t = 1):
        asd = torch.abs(x) > 30
        int_val = torch.zeros_like(x)
        jac = torch.ones_like(x)
        int_val[asd] = x[asd] + torch.sign(x[asd])
        int_val[~asd] = self.alpha(x[~asd], t)
        jac[~asd] = self.jfa(x[~asd])
        return int_val*mask[:,:,None], jac.squeeze(-1)*mask
    
    def lrelu_fn(self, x, mask):
        jac = torch.where(x<0, 0.01, 1)
        x *= jac
        return x*mask[:,:,None], jac.squeeze(-1)*mask

    def fc_det(self, J, Jmask, idtensor):
        asd = (1-Jmask)*idtensor
        J_ = J + asd
        det = torch.log(torch.diagonal(J_, dim1 = -2, dim2 = -1))
        return torch.sum(det, -1)
    
    def fc(self, U, QM, id_tensor, i):
        query = self.q_proj[i](U)
        key = self.k_proj[i](U)
        scores = torch.bmm(query, key.transpose(-2, -1))
        A = scores.masked_fill(id_tensor == 1, 0) / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        B = scores.masked_fill(id_tensor == 0, -1e8) / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        A += torch.nn.Softplus()(B)
        return torch.tril(A)

    def z(self, TX: Tensor, X: Tensor, MX: Tensor, TY: Tensor, Y: Tensor, MY: Tensor, marginal=0):
        U_ = self.CM(TX, X, MX, MY)
        S, SM, Q, QM, Z, Tr_My, U = preprocess(TX, X, MX, MY, U_, Y)
        bs = S.shape[0]
        LJD = torch.zeros_like(Z[:,0,0]) #initialize log jacobian determinent
        idtensor =  torch.eye(Z.shape[1])[None,:,:].repeat(bs,1,1).to(S.device) # an identity matrix
        J_mask = torch.matmul(QM[:,:,None], QM[:,None,:]) # mask for jacobian
        LJ_diag_mask = 1 - QM
        Z *= QM[:,:,None].repeat(1,1,Z.shape[-1]) # initial observation
        Z = Z - self.mu(U)

        for i in range(self.f_layers):  # repeat for l many flow layers
            # Invertible attention layer
            A = self.fc(U, QM, idtensor, i)*J_mask + 0.1*idtensor # invertible attention
            Z = torch.matmul(A, Z)
            J = A*J_mask
            LJD += self.fc_det(J, J_mask, idtensor) # Computing log det of jac

            # Tranformation layer
            theta = torch.exp(nn.Tanh()(self.theta[i](U))) # Scaling
            phi = self.phi[i](U)    # translation
            Z = Z*theta + phi # Transformation
            Z *= QM[:,:,None].repeat(1,1,Z.shape[-1])
            J_cc = theta.squeeze(-1)*QM + LJ_diag_mask # jacobian of transformation
            LJ_cc = torch.log(torch.abs(J_cc)) # Log Jacobian of transofrmation
            LJD += LJ_cc.sum((1))

            # Non-linear invertible activation function
            Z, J = self.fa(Z, QM.bool()) # activation function and the jacobian
            J *= QM # multiplying with jacobian mask
            J += LJ_diag_mask
            LJ = torch.log(torch.abs(J))
            LJD  += LJ.sum((1))

        Z_ = reshape_(Z, MY, QM, Tr_My)
        return Z_.squeeze(-1), LJD
    
    def forward(self, TX: Tensor, X: Tensor, MX: Tensor, TY: Tensor, Y: Tensor, MY: Tensor):
        z, J, yhat, Jdet = self.z(TX,X,MX,TY,Y,MY)
        return z, J, yhat, Jdet