import pickle
import torch
from torch import Tensor
from typing import Optional, Union, Dict
from typeguard import typechecked
import base64
import hashlib
import os

@typechecked
def autodevice(preference:Union[int, str, None, torch.device]=None)->torch.device:
    if isinstance(preference, torch.device):
        return preference
    else:
        if isinstance(preference, str):
            devstr = preference
        else:
            if not torch.cuda.is_available() or os.getenv('CUDA_VISIBLE_DEVICES') == '':
                devstr = "cpu"
            elif isinstance(preference, int):
                devstr = f"cuda:{preference}"
            else:
                devstr = "cuda"
        return torch.device(devstr)

@typechecked
def kw_autodevice(preference:Union[int, str, None]=None)->Dict[str, torch.device]:
    return dict(device=autodevice(preference))

def _hash(t:Tensor, length:int):
    hasher = hashlib.sha1(pickle.dumps(t))
    return base64.urlsafe_b64encode(hasher.digest()[:length]).decode('ascii')

@typechecked
def thash(aten:Tensor, length:int=8, with_ref=False, no_grad_char:str=" ")->str:
    g = "∇" if aten.requires_grad else no_grad_char
    save_ref = aten.detach()
    if with_ref:
        r = _hash(save_ref, (length // 4))
        v = _hash(save_ref.cpu().numpy(), 3*(length // 4))
        return f'#{g}{r}{v}'
    else:
        v = _hash(save_ref.cpu().numpy(), length)
        return f'#{g}{v}'

def prettyshape(size):
    return "[]" if len(size) == 0 else f"[{'×'.join(map(str, size))}]"

@typechecked
def show(aten:Tensor, fix_width:bool=True)->str:
    t = str(aten.dtype).split(".")[-1]
    return f"{t}{prettyshape(aten.size())}{thash(aten)}"

@typechecked
def copy(aten:Tensor, requires_grad=False, deepcopy=False)->Tensor:
    """
    A copy that will deep- or shallow- copy depending on if you want the output tensor on the computation
    graph.

    Ref: https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    """
    if (requires_grad and not aten.requires_grad):
        ret = aten.clone().detach()
        ret.requires_grad_(True)
        return ret
    elif (requires_grad and aten.requires_grad) or (not requires_grad and not aten.requires_grad):
        return aten.clone() if deepcopy else aten
    else: # (not requires_grad and rv.value.requires_grad):
        return aten.detach().clone() if deepcopy else aten.detach()

def dim_contract(fn, expected_dims):
    def wrapper(*args, **kwargs):
        tr, out = fn(*args, **kwargs)
        assert len(out.shape) == expected_dims, f"expected shape should be of length {expected_dims}"
        return tr, out
    return wrapper
