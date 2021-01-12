import pickle
import torch
from torch import Tensor
from typing import Optional, Union, Dict
from typeguard import typechecked
import base64
import hashlib

@typechecked
def autodevice(preference:Union[int, str, None]=None)->str:
    if isinstance(preference, str):
        return preference
    else:
        if not torch.cuda.is_available():
            return "cpu"
        elif isinstance(preference, int):
            return f"cuda:{preference}"
        else:
            return "cuda"

@typechecked
def kw_autodevice(preference:Union[int, str, None]=None)->Dict[str, Optional[str]]:
    return dict(device=autodevice(preference))

@typechecked
def thash(aten:Tensor, length:int=6, no_grad_char:str=" ")->str:
    g = "∇" if aten.requires_grad else no_grad_char
    hasher = hashlib.sha1(pickle.dumps(aten.detach()))
    h = base64.urlsafe_b64encode(hasher.digest()[:length]).decode('ascii')
    return f'#{g}{h}'

@typechecked
def show(aten:Tensor, fix_width:bool=True)->str:
    t = str(aten.dtype).split(".")[-1]
    s = "×".join(map(str, aten.shape))
    return f"{t}[{s}]{thash(aten)}"


def dim_contract(fn, expected_dims):
    def wrapper(*args, **kwargs):
        tr, out = fn(*args, **kwargs)
        assert len(out.shape) == expected_dims, f"expected shape should be of length {expected_dims}"
        return tr, out
    return wrapper
