from typing import List, Union
from dataclasses import dataclass

ParamType = Union[float, str]

__all__ = [
    "Gaussian",
    "GaussianSquare",
    "Drag",
    "Constant",
    "Sine",
    "Cosine",
    "CosineDrag",
    "Flattop",
]

@dataclass
class Gaussian:
    amp: ParamType
    duration: int
    sigma: ParamType
    def qasm_name(self) -> str:
        return "gaussian"

@dataclass
class GaussianSquare:
    amp: ParamType
    duration: int
    sigma: ParamType
    width: ParamType
    def qasm_name(self) -> str:
        return "gaussian_square"

@dataclass
class Drag:
    amp: ParamType
    duration: int
    sigma: ParamType
    beta: ParamType
    def qasm_name(self) -> str:
        return "drag"

@dataclass
class Constant:
    amp: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "constant"


@dataclass
class Sine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "sine"


@dataclass
class Cosine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "cosine"


@dataclass
class CosineDrag:
    amp: ParamType
    duration: int
    phase: ParamType
    alpha: ParamType
    def qasm_name(self) -> str:
        return "cosine_drag"


@dataclass
class Flattop:
    amp: ParamType
    width: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "flattop"
