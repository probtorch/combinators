import pickle
from torch import Tensor
from typing import Optional
from typeguard import typechecked
import base64
import hashlib

@typechecked
def thash(aten:Tensor, length=6)->str:
    hasher = hashlib.sha1(pickle.dumps(aten))
    return base64.urlsafe_b64encode(hasher.digest()[:length]).decode('ascii')

@typechecked
def show(aten:Tensor, fix_width=True)->str:
    t = str(aten.dtype).split(".")[-1]
    g = "∇" if aten.requires_grad else (" " if fix_width else "")
    h = thash(aten)
    s = "×".join(map(str, aten.shape))
    return f"{t}{g}[{s}]#{h}"
