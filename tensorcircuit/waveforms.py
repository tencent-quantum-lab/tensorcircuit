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
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma]

@dataclass
class GaussianSquare:
    amp: ParamType
    duration: int
    sigma: ParamType
    width: ParamType
    def qasm_name(self) -> str:
        return "gaussian_square"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma, self.width]

@dataclass
class Drag:
    amp: ParamType
    duration: int
    sigma: ParamType
    beta: ParamType
    def qasm_name(self) -> str:
        return "drag"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma, self.beta]

@dataclass
class Constant:
    amp: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "constant"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration]

@dataclass
class Sine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "sine"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.frequency, self.duration]

@dataclass
class Cosine:
    amp: ParamType
    frequency: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "cosine"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.frequency, self.duration]

@dataclass
class CosineDrag:
    amp: ParamType
    duration: int
    phase: ParamType
    alpha: ParamType
    def qasm_name(self) -> str:
        return "cosine_drag"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.phase, self.alpha]

@dataclass
class Flattop:
    amp: ParamType
    width: ParamType
    duration: int
    def qasm_name(self) -> str:
        return "flattop"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.width, self.duration]
