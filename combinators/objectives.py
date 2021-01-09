#!/usr/bin/env python
#
from torch import Tensor

def nvo_avo(lv: Tensor, sample_dims=0) -> Tensor:
    # values = -lv
    # log_weights = torch.zeros_like(lv)

    # nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    # loss = (nw * values).sum(dim=(sample_dims,), keepdim=False)
    loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
    return loss
