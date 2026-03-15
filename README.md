# Quantum Encoding of Genetic Information

### PhysisTechne Symposium 2026 — Quantum Computing Track

<p align="center">
  <b>Encoding DNA sequences into quantum circuits using three encoding strategies,<br>benchmarked on IBM's FakeSherbrooke (127-qubit Eagle processor)</b>
</p>

---

## Overview

This project implements a complete pipeline for encoding DNA sequences into optimized quantum circuits. The DNA is divided into codons (triplets), their frequencies are computed, and the resulting weight distribution is encoded into quantum states using three different strategies. Each encoding is simulated on both an ideal backend and a realistic noisy backend to measure fidelity degradation.

Two pipelines are provided:

| Pipeline | File | Encodings | Target |
|---|---|---|---|
| **Pipeline 1** | `main.py` | Amplitude + Angle | 50-base sequence |
| **Pipeline 2** | `main_aae.py` | Approximate Amplitude (AAE) | 12,001-base sequence |

---

## Results

### Pipeline 1 — Amplitude vs Angle Encoding (50 bases)

| Metric | Amplitude | Angle |
|---|---|---|
| Qubits | 4 | 12 |
| Logical CNOT gates | 8 | 0 |
| Logical Ry gates | 8 | 12 |
| Transpiled depth | 67 | 5 |
| Two-qubit gates (transpiled) | 15 | 0 |
| F(initial, Aer) | 1.000 | 1.000 |
| F(initial, Sherbrooke) | 0.959 | 0.985 |
| Noise drop | 0.041 | 0.015 |
| Reconstruction | 100% | 100% |

### Pipeline 2 — Approximate Amplitude Encoding (12,001 bases)

| Metric | Value |
|---|---|
| Qubits | 7 |
| Ansatz | Brickwall (6 layers) |
| Parameters | 42 |
| Logical gates | 60 (42 Ry + 18 CNOT) |
| Transpiled depth | 39 |
| Two-qubit gates (transpiled) | 18 |
| Overlap O | 0.973 |
| F(target, trained) | 0.947 |
| F(trained, Sherbrooke) | 0.941 |
| F(target, Sherbrooke) | 0.890 |
| Reconstruction | 100% |

---

## Pipeline Architecture

### Step 1 — Classical Bit Register (Codon Division)

The DNA sequence is divided into codons (groups of 3 bases). The frequency of each unique codon is counted — this is its weight. Two registers are built:

- **Unique register**: maps each unique codon → index and weight (used for quantum encoding)
- **Position register**: records which codon appears at every position (used for reconstruction)

### Step 2 — Quantum Encoding

Three encoding strategies are implemented:

**Amplitude Encoding** (`src/encoding.py`)
- Encodes weights as amplitudes of basis states: `|ψ⟩ = (1/N) Σᵢ weightᵢ |i⟩`
- `ceil(log₂(N_unique))` qubits, `2^(n-1)` CNOT + `2^(n-1)` Ry gates
- Exact encoding via `initialize()`

**Angle Encoding** (`src/encoding.py`)
- Each unique codon → its own qubit with Ry rotation proportional to weight
- `N_unique` qubits, `N_unique` Ry gates, 0 CNOTs, depth 1
- Product state, no entanglement

**Approximate Amplitude Encoding** (`src2/aae_encoding.py`)
- Trains a shallow parameterized circuit (brickwall ansatz) to approximate the target state
- Direct fidelity cost function: `C(θ) = 1 - Re⟨target|U(θ)|0⟩`
- L-BFGS optimizer with statevector simulation
- Scales to large sequences with O(poly(n)) circuit depth

### Step 3 — Simulation and Reconstruction

The encoded circuit is transpiled for IBM FakeSherbrooke and executed on:

- **Aer Simulator**: ideal, no noise (baseline)
- **FakeSherbrooke**: IBM Eagle r3 noise model (realistic hardware)

Density matrices are extracted for fidelity computation. The DNA is reconstructed using the classical position register.

### Fidelity Metrics

- **F(initial/target, Aer)**: does the ideal backend reproduce the target state?
- **F(initial/target, Sherbrooke)**: end-to-end fidelity including noise
- **Noise drop**: fidelity lost due to hardware noise

---

## Project Structure

```
├── main.py                    # Pipeline 1: Amplitude + Angle encoding (50 bases)
├── main_aae.py                # Pipeline 2: AAE encoding (12,001 bases)
├── requirements.txt
│
├── src/                       # Pipeline 1 modules
│   ├── compression.py         # Codon division + classical register
│   ├── encoding.py            # Amplitude & Angle encoding
│   ├── simulation.py          # Aer + FakeSherbrooke simulation
│   ├── reconstruction.py      # DNA reconstruction from classical register
│   └── fidelity.py            # Fidelity calculations
│
├── src2/                      # Pipeline 2 modules (AAE)
│   ├── compression2.py        # Codon division + target distributions (p, p^H)
│   ├── aae_encoding.py        # Brickwall ansatz + L-BFGS training
│   ├── simulation2.py         # Aer + FakeSherbrooke simulation
│   ├── reconstruction2.py     # DNA reconstruction
│   └── fidelity2.py           # Fidelity (target vs trained vs noisy)
│
├── data/
│   └── dna_12000.txt          # 12,001-base Rhesus macaque chr16 fragment
│
└── results/
    ├── summary.json            # Pipeline 1 output
    └── summary_aae.json        # Pipeline 2 output
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Usage

```bash
# Pipeline 1: Amplitude + Angle encoding (50-base sequence)
python main.py

# Pipeline 2: AAE encoding (12,001-base sequence)
python main_aae.py
```

---

## DNA Sequences

**Pipeline 1 — 50 bases:**
```
ATGCGTACGTTAGCGTACGATCGTAGCTAGCTTGACGATCGTACGTTAGC
```

**Pipeline 2 — 12,001 bases:**
Rhesus macaque (*Macaca mulatta*) chromosome 16 fragment
`NC_133421.1:91056922-91068922` — gene LOC144335571

---

## Requirements

- Python 3.10+
- Qiskit ≥ 1.0
- Qiskit Aer ≥ 0.14
- Qiskit IBM Runtime ≥ 0.20
- NumPy ≥ 1.24
- SciPy ≥ 1.10

---

## References

1. IBM Quantum Learning — [Data Encoding](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/data-encoding)
2. Nakaji et al. — [Approximate Amplitude Encoding in Shallow Parameterized Quantum Circuits](https://doi.org/10.1103/PhysRevResearch.4.023136), Phys. Rev. Research 4, 023136 (2022)
3. IBM Qiskit — [FakeSherbrooke Backend](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider)

## License

[MIT](LICENSE)
