"""Step 1 (AAE): Classical Bit Register + target distributions for AAE training."""

import os
import numpy as np
from collections import Counter, OrderedDict

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
_SEQ_FILE = os.path.join(_DATA_DIR, 'dna_12000.txt')

with open(_SEQ_FILE, 'r') as f:
    DNA_SEQUENCE = f.read().strip().replace('\n', '').replace(' ', '').replace('\r', '')


def divide_into_codons(sequence):
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]


def build_classical_register(sequence):
    codon_sequence = divide_into_codons(sequence)
    freq = Counter(codon_sequence)

    seen = OrderedDict()
    for codon in codon_sequence:
        if codon not in seen:
            seen[codon] = len(seen)

    n_unique = len(seen)
    n_qubits = int(np.ceil(np.log2(max(n_unique, 2))))

    unique_register = []
    for codon, idx in seen.items():
        unique_register.append({
            'unique_index': idx, 'codon': codon,
            'weight': freq[codon], 'binary': format(idx, f'0{n_qubits}b'),
        })

    position_register = []
    for pos, codon in enumerate(codon_sequence):
        position_register.append({
            'position': pos, 'codon': codon,
            'unique_index': seen[codon], 'binary': format(seen[codon], f'0{n_qubits}b'),
        })

    n_states = 2 ** n_qubits
    weight_vector = np.zeros(n_states)
    for entry in unique_register:
        weight_vector[entry['unique_index']] = entry['weight']

    d = weight_vector.copy()
    norm = np.linalg.norm(d)
    if norm > 0:
        d /= norm

    p_comp = d ** 2

    N = n_states
    d_H = np.zeros(N)
    for j in range(N):
        val = 0.0
        for k in range(N):
            val += d[k] * ((-1) ** bin(j & k).count('1'))
        d_H[j] = val / np.sqrt(N)
    p_hadamard = d_H ** 2

    return {
        'sequence': sequence, 'codon_sequence': codon_sequence,
        'num_codons': len(codon_sequence), 'unique_codons': seen,
        'num_unique': n_unique, 'weights': dict(freq),
        'unique_register': unique_register, 'position_register': position_register,
        'num_qubits': n_qubits, 'weight_vector': weight_vector,
        'd_normalized': d, 'p_comp': p_comp, 'p_hadamard': p_hadamard, 'd_hadamard': d_H,
    }


def print_step1(result):
    seq = result['sequence']
    n_q = result['num_qubits']
    d = result['d_normalized']

    print("=" * 65)
    print("STEP 1: CLASSICAL BIT REGISTER")
    print("=" * 65)
    print(f"\n  Sequence:        {seq[:50]}...")
    print(f"  Length:          {len(seq)} bases")
    print(f"  Total codons:   {result['num_codons']}")
    print(f"  Unique codons:  {result['num_unique']}")
    print(f"  Qubits:         {n_q}   Hilbert space: 2^{n_q} = {2**n_q}")

    print(f"\n  Top 10 codons:")
    sorted_reg = sorted(result['unique_register'], key=lambda e: e['weight'], reverse=True)
    for e in sorted_reg[:10]:
        idx = e['unique_index']
        print(f"    {e['codon']:>6}  weight={e['weight']:4d}  p(j)={d[idx]**2:.6f}")
