#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
from typing import Iterable

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs
from combinators.trace.utils import RequiresGrad, copytrace
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Observable
from inspect import signature
import inspect
from combinators.objectives import nvo_avo

@typechecked
class State(Iterable):
    def __init__(self, trace:Optional[Trace], output:Optional[Output]):
        self.trace = trace
        self.output = output
    def __repr__(self):
        if self.trace is None:
            return "None"
        else:
            out_str = tensor_utils.show(self.output) if isinstance(self.output, Tensor) else self.output
            return "tr: {}; out: {}".format(trace_utils.show(self.trace), out_str)
    def __iter__(self):
        for x in [self.trace, self.output]:
            yield x

@typechecked
class KCache:
    def __init__(self, program:Optional[State], kernel:Optional[State]):
        self.program = program
        self.kernel = kernel
    def __repr__(self):
        return "Kernel Cache:" + \
            "\n  program: {}".format(self.program) + \
            "\n  kernel:  {}".format(self.kernel)

@typechecked
class PCache:
    def __init__(self, target:Optional[State], proposal:Optional[State]):
        self.target = target
        self.proposal = proposal
    def __repr__(self):
        return "Propose Cache:" + \
            "\n  proposal: {}".format(self.proposal) + \
            "\n  target:   {}".format(self.target)

class Inf:
    pass

class KernelInf(nn.Module): # , Observable):
    def __init__(self, _step:Optional[int]=None, _permissive_arguments:bool=True):
        nn.Module.__init__(self)
        # Observable.__init__(self)
        self._cache = KCache(None, None)
        self._step = _step
        self._permissive_arguments = _permissive_arguments

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))

    def _program_args(self, fn, *args):
        if self._permissive_arguments:
            assert args is None or len(args) == 0, "need to filter this list, but currently don't have an example"
            # return [v for k,v in args.items() if check_passable_arg(k, fn)]
            return args
        else:
            return args

    def _program_kwargs(self, fn, **kwargs):
        if self._permissive_arguments and isinstance(fn, Program):
            return {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.model)}
        else:
            return kwargs


class Reverse(KernelInf, Inf):
    def __init__(self, program: Union[Program, KernelInf], kernel: Kernel, _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel

    def forward(self, *program_args:Any, cond_trace:Optional[Trace]=None, sample_dims=None, **program_kwargs:Any) -> Tuple[Trace, Output]:
        if cond_trace is not None:
            if isinstance(self.program, Program):
                self.program.with_observations(trace_utils.copytrace(cond_trace))# cond_trace.keys()))
            else:
                raise NotImplementedError("propagation of observes is not defined, but this is handled in the greenfield-lazy branch")

        program_state = State(*self.program(
            *self._program_args(self.program, *program_args),
            sample_dims=sample_dims,
            **self._program_kwargs(self.program, **program_kwargs)))

        if cond_trace is not None and isinstance(self.program, Program):
            self.program.clear_observations()

        # # validate conditions are kept
        # _observed_keys, _program_keys = set(self.observations.keys()), set(program_state.trace.keys())
        #
        # for k in _observed_keys.intersection(_program_keys):
        #     oval = self.observations[k]
        #     pval = program_state.trace[k].value
        #     assert torch.equal(oval, pval), \
        #         f'{k}: {tensor_utils.show(oval)} vs. {tensor_utils.show(pval)}'
        #
        # # add new conditions
        # for k in (_program_key - _observed_keys):
        #     self.observations[k] = program_state.trace[k].value

        #self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, sample_dims=sample_dims))
        # self.kernel.clear_conditions()

        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, None


class Forward(KernelInf, Inf):
    def __init__(self, kernel: Kernel, program: Union[Program, KernelInf], _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs) -> Tuple[Trace, Output]:
        program_state = State(*self.program(
            *self._program_args(self.program, *program_args),
            sample_dims=sample_dims,
            **self._program_kwargs(self.program, **program_kwargs)))

        # stop gradients for nesting. FIXME: is this the correct location? if so, then traces conditioned on this fail to have a gradient
        # for k, v in program_state.trace.items():
        #     program_state.trace[k]._value = program_state.trace[k].value.detach()

        # self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, sample_dims=sample_dims))
        # self.kernel.clear_conditions()
        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, kernel_state.output


class Propose(nn.Module, Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Program, Inf], validate:bool=True, _step=None):
        super().__init__()
        self.target = target
        self.proposal = proposal
        self._cache = PCache(None, None)
        self.validate = validate
        self._step = _step # used for debugging

    def forward(self, *shared_args, sample_dims=None, **shared_kwargs):
        # FIXME: target and proposal args can / should be separated
        proposal_state = State(*self.proposal(*shared_args, sample_dims=sample_dims, **shared_kwargs))

        # self.target.condition_on(proposal_state.trace)
        # target_state = State(*self.target(*shared_args, **shared_kwargs))
        # self.target.clear_conditions()


        conditions = dict(cond_trace=copytrace(proposal_state.trace, requires_grad=RequiresGrad.YES)) if isinstance(self.target, (Reverse, Kernel)) else dict()

        target_state = State(*self.target(*shared_args, sample_dims=sample_dims, **shared_kwargs, **conditions))

        joint_proposal_trace = proposal_state.trace
        joint_target_trace = target_state.trace
        self._cache = PCache(target_state, proposal_state)
        state = self._cache

        lv = Propose.log_weights(joint_target_trace, joint_proposal_trace, validate=self.validate, sample_dims=sample_dims)
        # print(self._cache)

        if self.proposal._cache is not None:
            # FIXME: this is a hack for the moment and should be removed somehow.
            # NOTE: this is unnecessary for the e2e/test_1dgaussians.py, but I am a bit nervous about double-gradients
            if isinstance(self.proposal._cache, PCache):
                raise NotImplemented("If this is the case (which it can be) i will actually need a way to propogate these detachments in a smarter way")

            # can we always assume we are in NVI territory?
            for k, rv in self.proposal._cache.program.trace.items():
                joint_proposal_trace[k]._value = rv.value.detach()

            proposal_keys = set(self.proposal._cache.program.trace.keys())
            target_keys = set(self.target._cache.kernel.trace.keys()) - proposal_keys
            state = PCache(
                proposal=State(trace=trace_utils.copysubtrace(proposal_state.trace, proposal_keys), output=proposal_state.output),
                target=State(trace=trace_utils.copysubtrace(target_state.trace, target_keys), output=target_state.output),
            )

        return state, lv

    @classmethod
    def log_weights(cls, target_trace, proposal_trace, sample_dims=None, validate=True):
        if validate:
            assert trace_utils.valeq(proposal_trace, target_trace, nodes=target_trace._nodes, check_exist=True)

        assert sample_dims != -1, "seems to be a bug in probtorch which blocks this behavior"

        batch_dim=None # TODO
        q_tar = target_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)
        p_tar = proposal_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)
        if validate:
            arv = list(target_trace.values())[0]
            dim = 0 if sample_dims is None else sample_dims
            lp_shape = q_tar.shape[0] if len(q_tar.shape) > 0 else 1
            rv_shape = arv.value.shape[dim] if len(arv.value.shape) > 0 else 1
            if rv_shape != lp_shape:
                raise RuntimeError("shape mismatch between log weight and elements in trace, you are probably missing sample_dims")

        return q_tar - p_tar
