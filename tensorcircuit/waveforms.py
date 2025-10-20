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
    duration: int
    amp: ParamType
    sigma: ParamType
    angle: ParamType
    def qasm_name(self) -> str:
        return "gaussian"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.sigma, self.angle]

@dataclass
class GaussianSquare:
    duration: int
    amp: ParamType
    sigma: ParamType
    width: ParamType
    def qasm_name(self) -> str:
        return "gaussian_square"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.sigma, self.width]

@dataclass
class Drag:
    duration: int
    amp: ParamType
    sigma: ParamType
    beta: ParamType
    def qasm_name(self) -> str:
        return "drag"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.sigma, self.beta]

@dataclass
class Constant:
    duration: int
    amp: ParamType
    def qasm_name(self) -> str:
        return "constant"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp]

@dataclass
class Sine:
    duration: int
    amp: ParamType
    phase: ParamType
    frequency: ParamType
    angle: ParamType
    def qasm_name(self) -> str:
        return "sin"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.phase, self.frequency, self.angle]
    
@dataclass
class Cosine:
    duration: int
    amp: ParamType
    frequency: ParamType
    def qasm_name(self) -> str:
        return "cosine"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.frequency]

@dataclass
class CosineDrag:
    duration: int
    amp: ParamType
    phase: ParamType
    alpha: ParamType
    def qasm_name(self) -> str:
        return "cosine_drag"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.phase, self.alpha]

@dataclass
class Flattop:
    duration: int
    amp: ParamType
    width: ParamType
    def qasm_name(self) -> str:
        return "flattop"
    def to_args(self) -> List[ParamType]:
        return [self.duration, self.amp, self.width]
