"""
Approximate Amplitude Encoding (AAE) Pipeline
PhysisTechne Symposium 2026
"""

import os, sys, json, time
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src2.compression2 import DNA_SEQUENCE, build_classical_register, print_step1
from src2.aae_encoding import aae_encode, print_step2
from src2.simulation2 import run_dual_simulation
from src2.reconstruction2 import reconstruct_dna, compute_accuracy
from src2.fidelity2 import compute_all_fidelities

N_LAYERS = 6
N_TRIALS = 8
MAXITER = 5000
SHOTS_SIM = 8192
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def run_step3(s1, s2):
    print("\n" + "=" * 65)
    print("STEP 3: SIMULATION + RECONSTRUCTION")
    print("=" * 65)
    sim = run_dual_simulation(s2, shots=SHOTS_SIM)

    for label, key in [("Aer", 'aer'), ("FakeSherbrooke", 'sherbrooke')]:
        top5 = Counter(sim[key]['counts']).most_common(5)
        print(f"\n  {label} — Top 5: {dict(top5)}")

    n_q = s2['num_qubits']
    aer_recon = reconstruct_dna(sim['aer']['counts'], s1, n_q, SHOTS_SIM)
    sher_recon = reconstruct_dna(sim['sherbrooke']['counts'], s1, n_q, SHOTS_SIM)
    aer_acc = compute_accuracy(DNA_SEQUENCE, aer_recon['reconstructed_dna'])
    sher_acc = compute_accuracy(DNA_SEQUENCE, sher_recon['reconstructed_dna'])

    print(f"\n  Reconstruction:")
    print(f"    Aer:        {aer_acc['exact_match']} ({aer_acc['char_accuracy']:.2%})")
    print(f"    Sherbrooke: {sher_acc['exact_match']} ({sher_acc['char_accuracy']:.2%})")

    return {
        'aer': sim['aer'], 'sherbrooke': sim['sherbrooke'],
        'aer_recon': {**aer_recon, **aer_acc}, 'sherbrooke_recon': {**sher_recon, **sher_acc},
    }


def main():
    print("=" * 65)
    print("  APPROXIMATE AMPLITUDE ENCODING (AAE)")
    print("  PhysisTechne Symposium 2026")
    print("=" * 65)
    print(f"\n  Target: {DNA_SEQUENCE[:60]}... ({len(DNA_SEQUENCE)} bases)")
    print(f"  Ansatz: Brickwall ({N_LAYERS} layers) | Optimizer: L-BFGS | Trials: {N_TRIALS}\n")

    t0 = time.time()

    s1 = build_classical_register(DNA_SEQUENCE)
    print_step1(s1)

    print("\n" + "=" * 65)
    print("STEP 2: AAE TRAINING")
    print("=" * 65)
    s2 = aae_encode(s1, n_layers=N_LAYERS, n_trials=N_TRIALS, maxiter=MAXITER)
    print_step2(s1, s2)

    s3 = run_step3(s1, s2)
    fid = compute_all_fidelities(s2, s3)

    elapsed = time.time() - t0
    m = s3['sherbrooke']['metrics']

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"""
  DNA: {len(DNA_SEQUENCE)} bases | {s1['num_codons']} codons | {s1['num_unique']} unique

  Encoding:  {s2['num_qubits']} qubits, {s2['n_layers']} layers, {s2['num_qubits']*s2['n_layers']} params
  Gates:     {s2['logical_ry']} Ry + {s2['logical_cnot']} CNOT = {s2['logical_total']} logical
  Transpiled: depth {m['depth']}, {m['total_gates']} gates, {m['two_qubit_gates']} two-qubit

  Overlap:  {s2['overlap']:.6f}
  F(target, trained):     {fid['f_target_trained']:.6f}
  F(trained, Sherbrooke): {fid['f_trained_sherbrooke']:.6f}
  F(target, Sherbrooke):  {fid['f_target_sherbrooke']:.6f}
  Noise drop:             {fid['noise_drop']:.6f}

  Reconstruction: Aer={'PASS' if s3['aer_recon']['exact_match'] else 'FAIL'} | Sherbrooke={'PASS' if s3['sherbrooke_recon']['exact_match'] else 'FAIL'}
  Runtime: {elapsed:.1f}s
""")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        'sequence_length': len(DNA_SEQUENCE), 'num_codons': s1['num_codons'],
        'unique_codons': s1['num_unique'], 'qubits': s2['num_qubits'],
        'ansatz_layers': s2['n_layers'], 'params': s2['num_qubits'] * s2['n_layers'],
        'logical_gates': s2['logical_total'], 'transpiled_depth': m['depth'],
        'overlap': s2['overlap'], 'f_target_trained': fid['f_target_trained'],
        'f_trained_sherbrooke': fid['f_trained_sherbrooke'],
        'f_target_sherbrooke': fid['f_target_sherbrooke'],
        'noise_drop': fid['noise_drop'], 'runtime': elapsed,
    }
    with open(os.path.join(RESULTS_DIR, 'summary_aae.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/summary_aae.json")


if __name__ == "__main__":
    main()
