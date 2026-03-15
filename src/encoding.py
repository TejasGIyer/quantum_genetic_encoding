"""Step 2: Quantum Encoding — Amplitude encoding and Angle encoding of codon weights."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def build_amplitude_vector(weight_vector):
    amps = weight_vector.astype(complex)
    norm = np.linalg.norm(amps)
    if norm > 0:
        amps /= norm
    return amps


def amplitude_encode(step1_result):
    weight_vec = step1_result['weight_vector']
    n_q = step1_result['num_qubits']
    amps = build_amplitude_vector(weight_vec)

    qc = QuantumCircuit(n_q)
    qc.initialize(amps, range(n_q))
    qc_m = QuantumCircuit(n_q, n_q)
    qc_m.initialize(amps, range(n_q))
    qc_m.measure(range(n_q), range(n_q))

    return {
        'encoding_type': 'amplitude', 'circuit': qc, 'circuit_meas': qc_m,
        'amplitudes': amps, 'initial_sv': Statevector(amps),
        'initial_dm': DensityMatrix(Statevector(amps)), 'num_qubits': n_q,
        'logical_cnot': 2 ** (n_q - 1), 'logical_ry': 2 ** (n_q - 1),
        'logical_total': 2 ** n_q,
    }


def rescale_weights_to_angles(weights):
    max_w = np.max(weights[weights > 0]) if np.any(weights > 0) else 1.0
    angles = np.zeros_like(weights, dtype=float)
    for i in range(len(weights)):
        if weights[i] > 0:
            angles[i] = (weights[i] / max_w) * 2 * np.pi
    return angles


def angle_encode(step1_result):
    unique_reg = step1_result['unique_register']
    n_q = step1_result['num_unique']
    weights = np.array([e['weight'] for e in unique_reg], dtype=float)
    angles = rescale_weights_to_angles(weights)

    qc = QuantumCircuit(n_q)
    for i in range(n_q):
        qc.ry(angles[i], i)
    qc_m = QuantumCircuit(n_q, n_q)
    for i in range(n_q):
        qc_m.ry(angles[i], i)
    qc_m.measure(range(n_q), range(n_q))

    initial_sv = Statevector.from_instruction(qc)
    return {
        'encoding_type': 'angle', 'circuit': qc, 'circuit_meas': qc_m,
        'angles': angles, 'weights': weights,
        'initial_sv': initial_sv, 'initial_dm': DensityMatrix(initial_sv),
        'num_qubits': n_q, 'logical_cnot': 0,
        'logical_ry': n_q, 'logical_total': n_q,
    }


def print_step2(step1_result, step2_result):
    enc = step2_result['encoding_type']
    if enc == 'amplitude':
        _print_amplitude(step1_result, step2_result)
    elif enc == 'angle':
        _print_angle(step1_result, step2_result)


def _print_amplitude(s1, s2):
    n_q = s2['num_qubits']
    amps = s2['amplitudes']

    print("\n" + "=" * 65)
    print("STEP 2: AMPLITUDE ENCODING")
    print("=" * 65)
    print(f"\n  Qubits: {n_q}   Hilbert space: 2^{n_q} = {2**n_q}")
    print(f"  Logical gates: {s2['logical_cnot']} CNOT + {s2['logical_ry']} Ry = {s2['logical_total']} total")
    print(f"\n  {'Basis':>8}  {'Weight':>6}  {'Amplitude':>10}  {'Prob':>8}  Codon")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*6}")
    for e in s1['unique_register']:
        idx = e['unique_index']
        print(f"  |{e['binary']}>  {e['weight']:6d}  {amps[idx].real:10.6f}  {abs(amps[idx])**2:8.6f}  {e['codon']}")
    for i in range(s1['num_unique'], 2 ** n_q):
        print(f"  |{format(i, f'0{n_q}b')}>  {0:6d}  {0:10.6f}  {0:8.6f}  (unused)")


def _print_angle(s1, s2):
    n_q = s2['num_qubits']
    angles, weights = s2['angles'], s2['weights']

    print("\n" + "=" * 65)
    print("STEP 2: ANGLE ENCODING")
    print("=" * 65)
    print(f"\n  Qubits: {n_q}   Product state (no entanglement)")
    print(f"  Logical gates: {s2['logical_ry']} Ry, 0 CNOT, depth 1")
    print(f"\n  {'Qubit':>6}  {'Codon':>6}  {'Weight':>6}  {'Angle(rad)':>10}  {'Angle(°)':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")
    for i, e in enumerate(s1['unique_register']):
        print(f"  q[{i:2d}]  {e['codon']:>6}  {weights[i]:6.0f}  {angles[i]:10.4f}  {np.degrees(angles[i]):10.2f}")
