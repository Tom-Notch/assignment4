#!/usr/bin/env python3
import torch
from torch.amp import custom_bwd
from torch.amp import custom_fwd
from torch.autograd import Function


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15))


trunc_exp = _trunc_exp.apply


def biased_softplus(x, bias=0):
    return torch.nn.functional.softplus(x - bias)
