"""Step 3b (AAE): Reconstruction — DNA from classical position register."""

import numpy as np


def reconstruct_dna(counts, step1_result, num_qubits, shots):
    n_states = 2 ** num_qubits
    probs = np.zeros(n_states)
    total = sum(counts.values())
    for bs, c in counts.items():
        idx = int(bs, 2)
        if idx < n_states:
            probs[idx] = c / total
    return {'reconstructed_dna': ''.join(e['codon'] for e in step1_result['position_register']), 'probabilities': probs}


def compute_accuracy(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    matches = sum(1 for a, b in zip(original[:min_len], reconstructed[:min_len]) if a == b)
    accuracy = matches / len(original) if len(original) > 0 else 0.0
    return {'exact_match': reconstructed == original, 'char_accuracy': accuracy}
