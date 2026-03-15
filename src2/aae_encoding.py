"""Step 2 (AAE): Approximate Amplitude Encoding — Brickwall ansatz + L-BFGS training."""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def build_brickwall_ansatz(n_qubits, n_layers, params):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)
    return qc


def statevector_from_params(params, n_qubits, n_layers):
    return np.array(Statevector.from_instruction(build_brickwall_ansatz(n_qubits, n_layers, params)).data)


def cost_function(params, n_qubits, n_layers, target_state):
    sv = statevector_from_params(params, n_qubits, n_layers)
    return 1.0 - np.real(np.vdot(target_state, sv))


def train_pqc(n_qubits, n_layers, target_state, n_trials=6, maxiter=5000):
    n_params = n_qubits * n_layers
    best_params, best_cost = None, float('inf')

    for trial in range(n_trials):
        params_init = np.random.uniform(0, 1, n_params)
        result = minimize(cost_function, params_init, args=(n_qubits, n_layers, target_state),
                          method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
        print(f"  Trial {trial+1}/{n_trials}: cost={result.fun:.8f}, iters={result.nit}, overlap={1-result.fun:.6f}")
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x.copy()

    return best_params, best_cost


def aae_encode(step1_result, n_layers=4, n_trials=6, maxiter=5000):
    n_q = step1_result['num_qubits']
    d = step1_result['d_normalized']

    print(f"\n  Config: {n_q} qubits, {n_layers} layers, {n_q * n_layers} params, {n_trials} trials")
    best_params, best_cost = train_pqc(n_q, n_layers, d, n_trials, maxiter)

    trained_circuit = build_brickwall_ansatz(n_q, n_layers, best_params)
    trained_circuit_meas = QuantumCircuit(n_q, n_q)
    trained_circuit_meas.compose(trained_circuit, inplace=True)
    trained_circuit_meas.measure(range(n_q), range(n_q))

    trained_sv = Statevector.from_instruction(trained_circuit)
    gc = dict(trained_circuit.count_ops())

    return {
        'encoding_type': 'aae', 'circuit': trained_circuit, 'circuit_meas': trained_circuit_meas,
        'initial_sv': trained_sv, 'initial_dm': DensityMatrix(trained_sv),
        'target_sv': Statevector(d), 'target_dm': DensityMatrix(Statevector(d)),
        'num_qubits': n_q, 'best_params': best_params, 'best_cost': best_cost,
        'overlap': abs(np.vdot(d, trained_sv.data)), 'n_layers': n_layers,
        'logical_cnot': gc.get('cx', 0), 'logical_ry': gc.get('ry', 0),
        'logical_total': gc.get('cx', 0) + gc.get('ry', 0),
    }


def print_step2(step1_result, step2_result):
    n_q = step2_result['num_qubits']
    d = step1_result['d_normalized']
    sv = step2_result['initial_sv']

    print("\n" + "=" * 65)
    print("STEP 2: APPROXIMATE AMPLITUDE ENCODING (AAE)")
    print("=" * 65)
    print(f"\n  Qubits: {n_q}   Layers: {step2_result['n_layers']}   Params: {n_q * step2_result['n_layers']}")
    print(f"  Gates: {step2_result['logical_ry']} Ry + {step2_result['logical_cnot']} CNOT = {step2_result['logical_total']}")
    print(f"  Depth: {step2_result['circuit'].depth()}")
    print(f"  Cost: {step2_result['best_cost']:.8f}   Overlap: {step2_result['overlap']:.6f}")

    probs_t = d ** 2
    probs_a = np.abs(sv.data) ** 2
    max_delta = 0

    print(f"\n  {'Basis':>9}  {'p_target':>9}  {'p_actual':>9}  {'|Δp|':>8}  Codon")
    print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*6}")
    for e in step1_result['unique_register']:
        idx = e['unique_index']
        delta = abs(probs_t[idx] - probs_a[idx])
        max_delta = max(max_delta, delta)
        print(f"  |{e['binary']}>  {probs_t[idx]:9.6f}  {probs_a[idx]:9.6f}  {delta:8.6f}  {e['codon']}")
    print(f"\n  Max |Δp|: {max_delta:.6f}")
