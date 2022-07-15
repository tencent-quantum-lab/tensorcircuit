import setuptools

from tensorcircuit import __version__, __author__

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tensorcircuit-nightly",
    version=__version__,
    author=__author__,
    author_email="znfesnpbh.tc@gmail.com",
    description="nightly release for tensorcircuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/refraction-ray/tensorcircuit-dev",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "tensornetwork", "networkx"],
    extras_require={
        "tensorflow": ["tensorflow"],
        "jax": ["jax", "jaxlib"],
        "torch": ["torch"],
        "qiskit": ["qiskit"],
    },
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
