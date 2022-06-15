import sys
import os
import pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc


@pytest.fixture(scope="function")
def npb():
    tc.set_backend("numpy")
    yield
    tc.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def tfb():
    tc.set_backend("tensorflow")
    yield
    tc.set_backend("numpy")  # default backend


@pytest.fixture(scope="function")
def jaxb():
    try:
        tc.set_backend("jax")
        yield
        tc.set_backend("numpy")

    except ImportError as e:
        print(e)
        tc.set_backend("numpy")
        pytest.skip("****** No jax backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def torchb():
    try:
        tc.set_backend("pytorch")
        yield
        tc.set_backend("numpy")
    except ImportError as e:
        print(e)
        tc.set_backend("numpy")
        pytest.skip("****** No torch backend found, skipping test suit *******")


@pytest.fixture(scope="function")
def highp():
    tc.set_dtype("complex128")
    yield
    tc.set_dtype("complex64")
