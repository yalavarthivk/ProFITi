import torch
from typing import NamedTuple
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import properscoring as ps
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
import pdb

class compute_losses():
    def __init__(self, std=1, marginal=False):
        self.std = std
        self.marginal = marginal
        print('marginals: ', self.marginal)
        self.nlloss = nn.GaussianNLLLoss(full=True, reduction='none')
        super(compute_losses).__init__()

    def nll(self, y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
        mean = torch.zeros_like(yhat)
        var = torch.ones_like(yhat)*(self.std**2)  
        LL2 = self.nlloss(mean, yhat, var)
        return LL2
    
    def loss(self, Y, YHAT, MASK, Z, J, Jdet):
        gnll = self.nll(Y, Z, MASK) * MASK  # B x (T x C)
        gnll = gnll.sum((1,2)) # B x 1
        mask_sum = MASK.sum((1,2))  # B x 1
        mask_sum[mask_sum==0] = 1
        ynll = (gnll - Jdet)/mask_sum
        gnll = gnll/mask_sum
        mask_sum2 = MASK.sum((1,2)).bool().sum()
        return ynll.sum()/mask_sum2

def reshape_inference(Z, MY, QM, Tr_My):
    X = torch.zeros_like(MY)[:,:,:,None].repeat(1,1,1,Z.shape[-1]).to(Z.device)
    values = Z[QM.to(torch.bool)]
    X[Tr_My] = values
    return X
def reshape_(Z, MY, QM, Tr_My):
    X = torch.zeros_like(MY).unsqueeze(-1).repeat(1,1,1,Z.shape[-1]).to(Z.device)
    values = Z[QM.to(torch.bool)]
    X[Tr_My] = values
    return X


def preprocess(TX, X, MX, MY, U_, Y=None):
    ndims = X.shape[-1]
    T = TX[:,:,None].repeat(1,1,ndims)
    MX_bool = MX.bool()
    MY_bool = MY.bool()
    C = torch.ones_like(X).cumsum(-1).to(torch.int64).to(TX.device) - 1

    full_len = torch.max(MX.sum((1,2)).to(torch.int64))
    pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

    X_flat = torch.stack([pad(r[m]) for r, m in zip(X, MX_bool)]).contiguous() # Flattening X
    TX_flat = torch.stack([pad(r[m]) for r, m in zip(T, MX_bool)]).contiguous() # Flattening TX
    CX_flat = torch.stack([pad(r[m]) for r, m in zip(C, MX_bool)]).contiguous() # Flattening Channels
    SM = torch.stack([pad(r[m]) for r, m in zip(MX, MX_bool)]).contiguous() # Flattenting MX to get the series mask SM
    
    CX_onehot = F.one_hot(CX_flat, ndims) # Onehot encoding channels

    S = torch.cat([X_flat[:,:,None], TX_flat[:,:,None], CX_onehot], -1) # Concatenating time, channel and value for S


    full_len = torch.max(MY.sum((1,2)).to(torch.int64))
    pady = lambda v: F.pad(v, [0, full_len - len(v)], value=0)
    padu = lambda u: F.pad(u, [0,0, 0, full_len - len(u)], value=0)

    
    TY_flat = torch.stack([pady(r[m]) for r, m in zip(T, MY_bool)]).contiguous()
    CY_flat = torch.stack([pady(r[m]) for r, m in zip(C, MY_bool)]).contiguous()
    QM = torch.stack([pady(r[m]) for r, m in zip(MY, MY_bool)]).contiguous() # Flattenting MX to get the series mask SM
    # pdb.set_trace()
    if U_ is None:
        U = None
    else:
        U = torch.stack([padu(r[m]) for r, m in zip(U_, MY_bool)]).contiguous()
    CY_onehot = F.one_hot(CY_flat, ndims)

    Q = torch.cat([TY_flat[:,:,None], CY_onehot], -1)
    Tr_My = torch.where(MY)
    if Y == None:
        return S, SM, Q, QM, None, Tr_My
    else:
        Y_flat = torch.stack([pady(r[m]) for r, m in zip(Y, MY_bool)]).contiguous()
        return S, SM, Q, QM, Y_flat.unsqueeze(-1), Tr_My, U


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor



def profiti_collate_fn(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []


    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets
        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)


        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)
        
        context_x.append(torch.cat([t, t_target], dim = 0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )