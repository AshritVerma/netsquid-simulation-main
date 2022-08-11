from netsquid_simulationtools.linear_algebra import _fidelity_between_single_qubit_states, _perform_pauli_correction, \
    _convert_to_density_matrix
import numpy as np
import netsquid as ns
from netsquid.qubits.ketstates import b00, b01, b10, b11, s0, h0, y0, BellIndex
import netsquid.qubits.qubitapi as qapi


def test_fidelity_between_single_qubit_states(depolar_prob=0.3):
    """Test if fidelity determination function gives expected results."""

    # first test if both are kets
    ket_with_itself = _fidelity_between_single_qubit_states(state_1=s0, state_2=s0)
    assert isinstance(ket_with_itself, float)
    assert np.isclose(ket_with_itself, 1)
    zero_with_plus = _fidelity_between_single_qubit_states(state_1=s0, state_2=h0)
    assert isinstance(zero_with_plus, float)
    assert np.isclose(zero_with_plus, 1 / 2)
    plus_with_zero = _fidelity_between_single_qubit_states(state_1=h0, state_2=s0)
    assert isinstance(plus_with_zero, float)
    assert np.isclose(plus_with_zero, 1 / 2)

    # use depolarizing channel to create a density matrix
    formalism = ns.get_qstate_formalism()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)
    [qubit] = qapi.create_qubits(1, no_state=True)
    qapi.assign_qstate([qubit], y0)  # using Y eigenstate to verify complex amplitudes are handled correctly
    qapi.depolarize(qubit, prob=depolar_prob)
    depolarized_state = qapi.reduced_dm(qubit)
    ns.set_qstate_formalism(formalism)

    # ket with dm
    dm_with_ket = _fidelity_between_single_qubit_states(depolarized_state, y0)
    assert isinstance(dm_with_ket, float)
    assert np.isclose(dm_with_ket, 1 - depolar_prob / 2)
    ket_with_dm = _fidelity_between_single_qubit_states(y0, depolarized_state)
    assert isinstance(ket_with_dm, float)
    assert np.isclose(ket_with_dm, 1 - depolar_prob / 2)
    assert np.isclose(ket_with_dm, 1 - depolar_prob / 2)

    # dm with dm
    dm_with_itself = _fidelity_between_single_qubit_states(depolarized_state, depolarized_state)
    assert isinstance(dm_with_itself, float)
    assert np.isclose(dm_with_itself, 1)
    dm_with_dm = _fidelity_between_single_qubit_states(s0 @ s0.conj().T, h0 @ h0.conj().T)
    assert isinstance(dm_with_dm, float)
    assert np.isclose(dm_with_dm, 1 / 2)


def test_perform_pauli_correction():
    """Assert that Pauli corrections bring all Bell states to the |Phi+> = (|00> + |11>) / sqrt(2) Bell state."""
    for bell_state, bell_index in zip([b00, b01, b10, b11],
                                      [BellIndex.B00, BellIndex.B01, BellIndex.B10, BellIndex.B11]):
        assert np.isclose(b00, _perform_pauli_correction(state=bell_state, bell_index=bell_index)).all()
        assert np.isclose(b00 @ b00.conj().T,
                          _perform_pauli_correction(state=bell_state @ bell_state.conj().T,
                                                    bell_index=bell_index)
                          ).all()
    assert np.isclose(b00, _perform_pauli_correction([0, 1, 1, 0] / np.sqrt(2), bell_index=BellIndex.B01)).all()


def test_convert_to_density_matrix():
    dm_2_qubits = np.array([[0, 0], [0, 1]])
    for ket in [np.array([0, 1]), np.array([[0], [1]])]:
        assert np.isclose(_convert_to_density_matrix(ket), dm_2_qubits).all()
    assert np.isclose(_convert_to_density_matrix(dm_2_qubits), dm_2_qubits).all()
    dm_4_qubits = np.zeros([4, 4])
    dm_4_qubits[3][3] = 1
    for ket in [np.array([0, 0, 0, 1]), np.array([[0], [0], [0], [1]])]:
        assert np.isclose(_convert_to_density_matrix(ket), dm_4_qubits).all()
    assert np.isclose(_convert_to_density_matrix(dm_4_qubits), dm_4_qubits).all()
