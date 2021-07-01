import weakref

from combinators.out import GlobalStore
from collections import namedtuple
from combinators import global_store


# Adjust the global store
# ==============================================


def _resample_hook(gs, tr, retr):
    for k in tr.keys():
        if (
            k in gs
            and k in retr
            and isinstance(gs[k], weakref.ref)
            and gs[k]() is tr[k].value
        ):
            gs[k] = weakref.ref(retr[k].value)


global_store.set_hook(GlobalStore.ResampleUpdate, _resample_hook)


def _pre_set_hook(gs, k, v):
    assert isinstance(v, weakref.ref), "only use weakrefs to avoid a space leak"


global_store.set_hook(GlobalStore.PreSet, _pre_set_hook)


# Prevent typos
# ==============================================


class key:
    @staticmethod
    def z_where(t=None, sweep=None):
        assert t is not None and sweep is not None
        return f"z_where_{t}_{sweep}"

    @staticmethod
    def z_what(sweep=None):
        assert sweep is not None
        return f"z_what_{sweep}"


# Define an index
# ==============================================


apg_ix = namedtuple("apg_ix", ["t", "sweep", "dir"])


def check_dir(ix):
    if ix.dir not in ["forward", "reverse"]:
        raise ValueError("Kernel must be run either forward or reverse")


def is_forward(ix):
    check_dir(ix)
    return ix.dir == "forward"
