import pickle
from torch import Tensor
from typing import Optional
from typeguard import typechecked
import base64
import hashlib

@typechecked
def thash(aten:Tensor, length=6, no_grad_char=" ")->str:
    g = "∇" if aten.requires_grad else no_grad_char
    hasher = hashlib.sha1(pickle.dumps(aten.detach()))
    h = base64.urlsafe_b64encode(hasher.digest()[:length]).decode('ascii')
    return f'#{g}{h}'

@typechecked
def show(aten:Tensor, fix_width=True)->str:
    t = str(aten.dtype).split(".")[-1]
    s = "×".join(map(str, aten.shape))
    return f"{t}[{s}]{thash(aten)}"
