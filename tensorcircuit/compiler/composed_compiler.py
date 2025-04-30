"""
object oriented compiler pipeline
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils import is_sequence
from ..abstractcircuit import AbstractCircuit
from .qiskit_compiler import qiskit_compile
from .simple_compiler import simple_compile


class Compiler:
    def __init__(
        self,
        compile_funcs: Union[Callable[..., Any], List[Callable[..., Any]]],
        compiled_options: Optional[List[Dict[str, Any]]] = None,
    ):
        if not is_sequence(compile_funcs):
            self.compile_funcs = [compile_funcs]
        else:
            self.compile_funcs = list(compile_funcs)  # type: ignore
        self.add_options(compiled_options)

    def add_options(
        self, compiled_options: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        if compiled_options is None:
            self.compiled_options = [{} for _ in range(len(self.compile_funcs))]  # type: ignore
        elif not is_sequence(compiled_options):
            self.compiled_options = [compiled_options for _ in self.compile_funcs]  # type: ignore
        else:
            assert len(compiled_options) == len(  # type: ignore
                self.compile_funcs
            ), "`compiled_options` must have the same list length as `compile_funcs`"
            self.compiled_options = list(compiled_options)  # type: ignore
            for i, c in enumerate(self.compiled_options):
                if c is None:
                    self.compiled_options[i] = {}

    def __call__(
        self, circuit: AbstractCircuit, info: Optional[Dict[str, Any]] = None
    ) -> Any:
        for f, d in zip(self.compile_funcs, self.compiled_options):
            result = f(circuit, info, compiled_options=d)  # type: ignore
            if not isinstance(result, tuple):
                result = (result, info)
            circuit, info = result
        return circuit, info


class DefaultCompiler(Compiler):
    def __init__(self, qiskit_compiled_options: Optional[Dict[str, Any]] = None):
        """
        A fallback choice to compile circuit running on tencent quantum cloud with rz as native gate

        :param qiskit_compiled_options: qiskit compiled options to be added
            options documented in `qiskit.transpile` method,
            to use tencent quantum cloud, `{"coupling_map": d.topology()}` is in general enough,
            where d is a device object,
            defaults to None, i.e. no qubit mapping is applied
        :type qiskit_compiled_options: Optional[Dict[str, Any]], optional
        """
        compiled_options = {
            "optimization_level": 3,
            "basis_gates": ["u3", "h", "cx", "cz"],
        }
        # rz target is bad for qiskit
        if qiskit_compiled_options:
            compiled_options.update(qiskit_compiled_options)
        super().__init__(
            [qiskit_compile, simple_compile],
            [compiled_options, None],  # type: ignore
        )


def default_compile(
    circuit: AbstractCircuit,
    info: Optional[Dict[str, Any]] = None,
    compiled_options: Optional[Dict[str, Any]] = None,
) -> Tuple[AbstractCircuit, Dict[str, Any]]:
    dc = DefaultCompiler(compiled_options)
    c, info = dc(circuit, info)
    return c, info  # type: ignore
