from typing import List, Dict, Any


class Param:
    def __init__(self, name: str):
        self.name = name

class Frame:
    def __init__(self, name: str):
        self.name = name

class DefcalBuilder:
    def __init__(self, circuit, name: str, parameters: List["Param"]):
        self.circuit = circuit
        self.name = name
        self.parameters = parameters
        self.instructions = []

    def new_frame(self, frame_name: str, param: "Param"):
        frame = Frame(frame_name)
        self.instructions.append({
            "type": "frame",
            "frame": frame,
            "qubit": param.name,
        })
        return frame

    def play(self, frame: Frame, waveform: Any, start_time: int = None):
        if not hasattr(waveform, "__dataclass_fields__"):
            raise TypeError("Unsupported waveform type")

        waveform_type = waveform.qasm_name()
        args = waveform.to_args()
        if start_time is not None:
            args = [start_time] + args

        self.instructions.append({
            "type": "play",
            "frame": frame.name,
            "waveform_type": waveform_type,
            "args": args,
        })
        return self

    def build(self):
        self.circuit.def_calibration(
            name=self.name,
            parameters=[p.name for p in self.parameters],
            instructions=self.instructions,
        )
