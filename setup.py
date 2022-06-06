import setuptools

from tensorcircuit import __version__, __author__
from tensorcircuit.utils import is_m1mac

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numpy", "scipy", "tensornetwork", "networkx"]

if not is_m1mac():
    install_requires.append("tensorflow")
    # avoid the embarassing macos M1 chip case, where the package is called tensorflow-macos

setuptools.setup(
    name="tensorcircuit",
    version=__version__,
    author=__author__,
    author_email="shixinzhang@tencent.com",
    description="Quantum circuits on top of tensor network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tencent-quantum-lab/tensorcircuit",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=[
        "pytest",
        "pytest-lazy-fixture",
        "pytest-cov",
        "pytest-benchmark",
        "pytest-xdist",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
