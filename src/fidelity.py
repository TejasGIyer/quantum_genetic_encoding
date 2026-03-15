"""Fidelity Calculations — Raw fidelity and simulator comparison."""

from qiskit.quantum_info import state_fidelity


def compute_all_fidelities(step2_result, step3_result):
    initial_dm = step2_result['initial_dm']
    aer_dm = step3_result['aer']['dm']
    sherbrooke_dm = step3_result['sherbrooke']['dm']

    f_aer = state_fidelity(initial_dm, aer_dm) if aer_dm is not None else 0.0
    f_sher = state_fidelity(initial_dm, sherbrooke_dm) if sherbrooke_dm is not None else 0.0
    f_inter = state_fidelity(aer_dm, sherbrooke_dm) if (aer_dm is not None and sherbrooke_dm is not None) else 0.0

    print("\n" + "=" * 65)
    print("FIDELITY")
    print("=" * 65)
    print(f"  F(initial, Aer)        = {f_aer:.6f}")
    print(f"  F(initial, Sherbrooke)  = {f_sher:.6f}")
    print(f"  F(Aer, Sherbrooke)     = {f_inter:.6f}")
    print(f"  Noise drop             = {f_aer - f_sher:.6f}")

    return {
        'raw_fidelity_aer': f_aer, 'raw_fidelity_sherbrooke': f_sher,
        'fidelity_drop': f_aer - f_sher, 'inter_simulator_fidelity': f_inter,
    }
