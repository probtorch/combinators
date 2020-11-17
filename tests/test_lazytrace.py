#!/usr/bin/env python3

import pytest
import hypothesis.strategies as st
from hypothesis import given

from combinators.lazytrace import lazytrace

ascii_characters = st.characters(whitelist_categories=("L"), min_codepoint=0, max_codepoint=128)
st_variable_name = st.text(alphabet=ascii_characters, min_size=1)

@pytest.fixture(autouse=True)
def run_around_tests(mocker):
    mocker.patch('combinators.lazytrace.LazyTrace.backend', dict)

@given(st.dictionaries(st_variable_name, st.floats()))
def test_lazytrace_creation(mocker, vals):
    trace = lazytrace(**vals)
    assert len(trace.keys()) == len(vals)

def test_lazytrace_append1():
    trace1 = lazytrace(a="A")
    trace2 = lazytrace(b="B")
    trace3 = trace1 + trace2
    assert all(map(lambda k: k in trace3.keys() and trace3[k] == k.upper(), ["a", "b"]))

def test_lazytrace_append2():
    """
    this is a wrinkle that offers us performance speed ups, maybe? run the same thing twice with only overlapping
    samples changed?
    """
    trace1 = lazytrace(a="A1")
    trace2 = lazytrace(b="B", a="A2")
    trace3 = trace1 + trace2
    assert all(map(lambda k: k in trace3.keys(), ["a", "b"]))
    assert trace3.a == "A2"

def test_lazytrace_condition0():
    trace1 = lazytrace()
    trace2 = lazytrace()
    trace3 = trace1 | trace2
    assert all(map(lambda t: len(t.keys()) ==0, [trace1, trace2, trace3]))

def test_lazytrace_condition0():
    trace1 = lazytrace(a="A")
    trace2 = lazytrace(b="B")
    empty  = lazytrace()
    assert len(( empty | trace1 ).keys()) == 0
    assert len(( trace1 | empty ).keys()) == 1 and "a" in (trace1 | empty)
    assert len(( trace2 | empty ).keys()) == 1 and "b" in (trace2 | empty)
