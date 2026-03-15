"""Fidelity (AAE) — Target vs trained, trained vs noisy, end-to-end."""

from qiskit.quantum_info import state_fidelity


def compute_all_fidelities(step2_result, step3_result):
    trained_dm = step2_result['initial_dm']
    target_dm = step2_result['target_dm']
    aer_dm = step3_result['aer']['dm']
    sherbrooke_dm = step3_result['sherbrooke']['dm']

    f_tt = state_fidelity(target_dm, trained_dm)
    f_ta = state_fidelity(trained_dm, aer_dm) if aer_dm is not None else 0.0
    f_ts = state_fidelity(trained_dm, sherbrooke_dm) if sherbrooke_dm is not None else 0.0
    f_es = state_fidelity(target_dm, sherbrooke_dm) if sherbrooke_dm is not None else 0.0

    print("\n" + "=" * 65)
    print("FIDELITY")
    print("=" * 65)
    print(f"  F(target, trained)      = {f_tt:.6f}   (training quality)")
    print(f"  F(trained, Aer)         = {f_ta:.6f}   (sanity check)")
    print(f"  F(trained, Sherbrooke)  = {f_ts:.6f}   (noise impact)")
    print(f"  F(target, Sherbrooke)   = {f_es:.6f}   (end-to-end)")
    print(f"  Noise drop             = {f_ta - f_ts:.6f}")

    return {
        'f_target_trained': f_tt, 'f_trained_aer': f_ta,
        'f_trained_sherbrooke': f_ts, 'f_target_sherbrooke': f_es,
        'noise_drop': f_ta - f_ts, 'overlap': step2_result['overlap'],
    }
