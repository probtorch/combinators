from torch import Tensor
from typing import Any, Optional
from probtorch.stochastic import Trace
import combinators.tensor.utils as tensor_utils


class PropertyDict(dict):
    """quick hack to prototype program outputs"""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        return "<PropertyDict>:\n" + show_nest(self, pd_header="<PropertyDict>")


def show_nest(
    p: PropertyDict,
    nest_level=0,
    indent_len: Optional[int] = None,
    pd_header="<PropertyDict>",
):
    """show the nesting level of a PropertyDict with indentation"""
    _max_len = max(map(len, p.keys()))
    max_len = _max_len + nest_level * (_max_len if indent_len is None else indent_len)
    delimiter = "\n  "

    def showitem(v):
        if isinstance(v, Tensor):
            return tensor_utils.show(v)
        elif isinstance(v, dict):
            return "dict({})".format(
                ", ".join(["{}={}".format(k, showitem(v)) for k, v in v.items()])
            )
        else:
            return repr(v)

    unnested = dict(filter(lambda kv: not isinstance(kv[1], PropertyDict), p.items()))
    unnested_str = delimiter.join(
        [
            *[
                ("{:>" + str(max_len) + "}: {}").format(k, showitem(v))
                for k, v in unnested.items()
            ]
        ]
    )

    nested = dict(filter(lambda kv: isinstance(kv[1], PropertyDict), p.items()))
    nested_str = delimiter.join(
        [
            *[
                ("{:>" + str(max_len) + "}: {}").format(
                    k + pd_header, "\n" + show_nest(v, nest_level=nest_level + 1)
                )
                for k, v in nested.items()
            ]
        ]
    )

    return unnested_str + delimiter + nested_str


class GlobalStore(dict):
    # keys, just a smidge more user-friendly than enums
    PreGet, PostGet = "pre_get", "post_get"
    PreSet, PostSet = "pre_set", "post_set"
    ResampleUpdate = "resample_update"

    # with no model combinators, a global store makes things less painful
    def __init__(self):
        super().__init__()

        def noop(*args):
            return None

        self._hooks = dict(
            pre_get=noop,
            pre_set=noop,
            post_get=noop,
            post_set=noop,
            resample_update=noop,
        )

    def __getitem__(self, k):
        self._hooks["pre_get"](self, k)
        v = super().__getitem__(k)
        o = self._hooks["post_get"](self, k, v)
        return v if o is None else o

    def set_hook(self, ty, f):
        ty_ = ty.split("_")
        assert ty in self._hooks.keys()
        self._hooks[ty] = f

    def __setitem__(self, k, v):
        self._hooks["pre_set"](self, k, v)
        super().__setitem__(k, v)
        self._hooks["post_set"](self, k, v)
        return None

    def resample_update(self, trace, resampled):
        self._hooks["resample_update"](self, trace, resampled)
        return None

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, super().__repr__())


global_store = GlobalStore()


class Out(PropertyDict):
    """
    Prototype of a Program's return type. This will be replaced with the
    inference state. Programs always return three things: a trace, log_weight,
    and output, but in some cases (like propose) we need to pass around more
    information. We place this in 'extras' and merge this directly into the
    dict.
    """

    def __init__(
        self,
        trace: Trace,
        log_weight: Optional[Tensor],
        output: Any,
        extras: dict = dict(),
    ):
        self.trace = trace  # τ; ρ
        self.log_weight = log_weight  # w
        self.output = output  # c
        for k, v in extras.items():
            self[k] = v

    def __iter__(self):
        optionals = ["ix", "program", "kernel", "proposal"]
        extras = dict()
        for o in optionals:
            if hasattr(self, o):
                extras[o] = getattr(self, o)

        for x in [self.trace, self.log_weight, self.output, extras]:
            yield x
