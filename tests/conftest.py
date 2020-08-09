import sys
import os
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


@pytest.fixture(scope="function")
def tfb():
    tc.set_backend("tensorflow")
    yield
    tc.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def jaxb():
    tc.set_backend("jax")
    yield
    tc.set_backend("numpy")


@pytest.fixture(scope="function")
def torchb():
    tc.set_backend("pytorch")
    tc.set_dtype("float64")
    yield
    tc.set_backend("numpy")
    tc.set_dtype("complex64")


@pytest.fixture(scope="function")
def highp():
    tc.set_dtype("complex128")
    yield
    tc.set_dtype("complex64")
