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
        gnll = gnll.sum(-1)  # B
        jnll = gnll - ldj
        return jnll


class Batch(NamedTuple):
    r"""A single sample of the data."""

    tobs: Tensor  # N
    cobs: Tensor  # N
    xobs: Tensor  # N
    mobs: Tensor  # N
    tq: Tensor  # K
    cq: Tensor  # K
    y: Tensor  # K
    mq: Tensor  # K


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
    batch_tobs: list[Tensor] = []
    batch_xobs: list[Tensor] = []
    batch_cobs: list[Tensor] = []
    batch_mobs: list[Tensor] = []
    batch_tq: list[Tensor] = []
    batch_cq: list[Tensor] = []
    batch_mq: list[Tensor] = []
    batch_y: list[Tensor] = []
    for sample in batch:

        t, x, t_target = sample.inputs
        if t_target.shape[0] == 0:
            continue
        if t.shape[0] == 0:
            continue
        y = sample.targets
        obs_seq_len = len(t)
        qu_seq_len = len(t_target)
        nchannels = x.shape[-1]

        obs_channels = torch.arange(nchannels).expand(obs_seq_len, nchannels)
        qu_channels = torch.arange(nchannels).expand(qu_seq_len, nchannels)
        tobs = t[:, None].repeat(1, nchannels)
        tq = t_target[:, None].repeat(1, nchannels)

        mask_y = y.isfinite().to(x.dtype)
        mask_x = x.isfinite().to(x.dtype)

        mask_x_bool = mask_x.bool()
        mask_y_bool = mask_y.bool()
        if mask_y_bool.sum() == 0:
            continue
        if mask_x_bool.sum() == 0:
            continue

        xobs = x[mask_x_bool]
        y = y[mask_y_bool]

        tobs = tobs[mask_x_bool]
        tq = tq[mask_y_bool]

        cobs = obs_channels[mask_x_bool]
        cq = qu_channels[mask_y_bool]

        mobs = torch.ones_like(cobs)
        mq = torch.ones_like(cq)

        batch_xobs.append(xobs)
        batch_mobs.append(mobs)
        batch_cobs.append(cobs)
        batch_tobs.append(tobs)
        batch_tq.append(tq)
        batch_mq.append(mq)
        batch_cq.append(cq)
        batch_y.append(y)

    return Batch(
        tobs=pad_sequence(batch_tobs, batch_first=True, padding_value=0),
        cobs=pad_sequence(batch_cobs, batch_first=True, padding_value=0),
        xobs=pad_sequence(batch_xobs, batch_first=True, padding_value=0),
        mobs=pad_sequence(batch_mobs, batch_first=True, padding_value=0),
        tq=pad_sequence(batch_tq, batch_first=True, padding_value=0),
        cq=pad_sequence(batch_cq, batch_first=True, padding_value=0),
        y=pad_sequence(batch_y, batch_first=True, padding_value=0),
        mq=pad_sequence(batch_mq, batch_first=True, padding_value=0),
    )
