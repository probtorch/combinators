from pytest import fixture
import combinators.utils as debug

@fixture
def is_smoketest():
    return debug.is_smoketest()

@fixture(autouse=True)
def seed():
    debug.seed(1)
