// Extended gate set supported in TensorCircuit with OpenQASM3.0 stdgate

// OpenQASM 3 standard gate library

// phase gate
gate p(λ) a { ctrl @ gphase(λ) a; }

// Pauli gate: bit-flip or NOT gate
gate x a { U(π, 0, π) a; }
// Pauli gate: bit and phase flip
gate y a { U(π, π/2, π/2) a; }
// Pauli gate: phase flip
gate z a { p(π) a; }

// Clifford gate: Hadamard
gate h a { U(π/2, 0, π) a; }
// Clifford gate: sqrt(Z) or S gate
gate s a { pow(1/2) @ z a; }
// Clifford gate: inverse of sqrt(Z)
gate sdg a { inv @ pow(1/2) @ z a; }

// sqrt(S) or T gate
gate t a { pow(1/2) @ s a; }
// inverse of sqrt(S)
gate tdg a { inv @ pow(1/2) @ s a; }

// sqrt(NOT) gate
gate sx a { pow(1/2) @ x a; }

// Rotation around X-axis
gate rx(θ) a { U(θ, -π/2, π/2) a; }
// rotation around Y-axis
gate ry(θ) a { U(θ, 0, 0) a; }
// rotation around Z axis
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }

// controlled-NOT
gate cx a, b { ctrl @ x a, b; }
// controlled-Y
gate cy a, b { ctrl @ y a, b; }
// controlled-Z
gate cz a, b { ctrl @ z a, b; }
// controlled-phase
gate cp(λ) a, b { ctrl @ p(λ) a, b; }
// controlled-rx
gate crx(θ) a, b { ctrl @ rx(θ) a, b; }
// controlled-ry
gate cry(θ) a, b { ctrl @ ry(θ) a, b; }
// controlled-rz
gate crz(θ) a, b { ctrl @ rz(θ) a, b; }
// controlled-H
gate ch a, b { ctrl @ h a, b; }

// swap
gate swap a, b { cx a, b; cx b, a; cx a, b; }

// Toffoli
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
// controlled-swap
gate cswap a, b, c { ctrl @ swap a, b, c; }

// four parameter controlled-U gate with relative phase γ
gate cu(θ, φ, λ, γ) a, b { p(γ) a; ctrl @ U(θ, φ, λ) a, b; }

// Gates for OpenQASM 2 backwards compatibility
// CNOT
gate CX a, b { ctrl @ U(π, 0, π) a, b; }
// phase gate
gate phase(λ) q { U(0, 0, λ) q; }
// controlled-phase
gate cphase(λ) a, b { ctrl @ phase(λ) a, b; }
// identity or idle gate
gate id a { U(0, 0, 0) a; }
// IBM Quantum experience gates
gate u1(λ) q { U(0, 0, λ) q; }
gate u2(φ, λ) q { gphase(-(φ+λ)/2); U(π/2, φ, λ) q; }
gate u3(θ, φ, λ) q { gphase(-(φ+λ)/2); U(θ, φ, λ) q; }



// Extended gate set for TensorCircuit
// Implemented by @refraction-ray

// Aliases
gate toffoli a, b, c {ccx a, b, c; }
gate ccnot a, b, c {ccx a, b, c; }
gate fredkin a, b, c {cswap a, b, c; }
gate cnot a, b {cx a, b; }
gate sd a, b {sdg a, b; }
gate td a, b {tdg a, b; }
gate phase(λ) a { ctrl @ gphase(λ) a; }


// Neg control gates
gate ox a, b { negctrl @ x a, b; }
gate oy a, b { negctrl @ y a, b; }
gate oz a, b { negctrl @ z a, b; }

// Neg ctrol rotation
gate orx(θ) a, b { negctrl @ rx(θ) a, b; }
gate ory(θ) a, b { negctrl @ ry(θ) a, b; }
gate orz(θ) a, b { negctrl @ rz(θ) a, b; }

// exponential gate of pauli product
gate rxx(θ) a, b { h a; h b; cx a, b; rz(θ) b; cx a, b; h a; h b; }
gate ryy(θ) a, b { rx(-π/2) a; rx(-π/2) b; cx a, b; rz(θ) b; cx a, b; rx(π/2) a; rx(π/2) b; }
gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }

// iswap: implementation according to https://github.com/quantumlib/Cirq/blob/v1.0.0/cirq-core/cirq/ops/swap_gates.py#L168-L291
gate iswap(θ) a, b { cx a, b; h a; cx b, a; phase(θ*π/2) a; cx b, a; phase(-θ*π/2) a; h a; cx a,b; }

// wroot
gate wroot a { U(π/2, -π/4, π/4) a; }
