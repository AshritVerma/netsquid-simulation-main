import unittest
from netsquid_simulationtools.linear_algebra import XYZEigenstateIndex, pauli_x, pauli_y, pauli_z
from netsquid_simulationtools.process_teleportation import determine_teleportation_output_state, \
    determine_teleportation_fidelities_of_xyz_eigenstates_from_data, estimate_average_teleportation_fidelity_from_data, \
    estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data, estimate_teleportation_fidelity_from_data, \
    teleportation_fidelity_optimized_over_local_unitaries_from_data, \
    teleportation_fidelity_optimized_over_local_unitaries, determine_teleportation_fidelity
import netsquid as ns
from netsquid.qubits.ketstates import b00, b01, b10, b11, s0, h0, y0, s1, h1, y1, BellIndex
import netsquid.qubits.qubitapi as qapi
import pandas
import numpy as np


def test_determine_teleportation_fidelities_of_xyz_eigenstates_from_data():
    """Check format of fidelities dictionary, and assert that eigenstates are teleported as expected."""
    data = {"state": [b00, b01, b10, b11],
            "midpoint_outcome_0": [BellIndex.B00] * 4,
            }
    df = pandas.DataFrame(data)

    fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)

    # Z eigenstates are unaffected by Z errors
    for information_state in [XYZEigenstateIndex.Z0, XYZEigenstateIndex.Z1]:
        assert np.isclose(fidelities[information_state], [1, 0, 1, 0]).all()
    # X eigenstates are unaffected by X errors
    for information_state in [XYZEigenstateIndex.X0, XYZEigenstateIndex.X1]:
        assert np.isclose(fidelities[information_state], [1, 1, 0, 0]).all()
    # Y eigenstates are unaffected by Y errors
    for information_state in [XYZEigenstateIndex.Y0, XYZEigenstateIndex.Y1]:
        assert np.isclose(fidelities[information_state], [1, 0, 0, 1]).all()


class TestEstimateAverageTeleportationFidelityFromData(unittest.TestCase):
    def test_perfect_state(self):
        """Assert fidelity is unit when having perfect Bell state."""
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome": [BellIndex.B00, BellIndex.B01, BellIndex.B10, BellIndex.B11]
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        average_fidelity, error = estimate_average_teleportation_fidelity_from_data(fidelities)
        assert np.isclose(average_fidelity, 1.)
        assert np.isclose(error, 0.)

    def test_maximally_mixed(self):
        """Assert fidelity is 0.5 with maximally mixed state.

        Performing quantum teleportation with a maximally mixed resource state results in a maximally mixed output
        state. A single-qubit maximally mixed state has fidelity 0.5 to any pure information state.
        Therefore, the teleportation fidelity (which is an average over all pure information states) should be 0.5.

        """
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome_0": [BellIndex.B00] * 4,
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        average_fidelity, error = estimate_average_teleportation_fidelity_from_data(fidelities)
        assert np.isclose(average_fidelity, .5)
        assert error > 0

    def test_small_deviation_from_perfection(self):
        """Assert deviation and error are small when state is almost perfect (1% chance of bit flip)."""
        data = {"state": [b00] * 100,
                "midpoint_outcome_0": [BellIndex.B00] * 99 + [BellIndex.B01]}
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        average_fidelity, error = estimate_average_teleportation_fidelity_from_data(fidelities)
        assert average_fidelity < 1.
        assert average_fidelity > 0.99
        assert error > 0.
        assert error < 0.01


class TestEstimateMinimumTeleportationFidelityOverXYZEigenstatesFromData(unittest.TestCase):
    def test_perfect_state(self):
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome": [BellIndex.B00, BellIndex.B01, BellIndex.B10, BellIndex.B11]
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        label, minimum, error = estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities)
        assert isinstance(label, XYZEigenstateIndex)
        assert np.isclose(minimum, 1.)
        assert np.isclose(error, 0.)

    def test_maximally_mixed(self):
        """Assert fidelity is 0.5 with maximally mixed state for all eigenstates.

        Note
        ----
        There is no test for which there is a unique minimum among all eigenstates.
        This is because at the time of writing these tests I cannot come up with a resource state that has
        different teleportation fidelity for the different eigenstates of the same Pauli operator.
        I suspect such a resource state in fact does not exist, as applying a Pauli operation at the
        information state can be also considered to act on the POVM instead of on the information state,
        effectively mapping the POVM to another POVM.
        For a Bell-state measurement, the POVM is mapped to itself.
        Furthermore, because the Pauli correction that is performed at the end is exactly the "shift" between the
        POVMs (i.e. because having outcome P|Phi+> leads to correction P, for some Pauli P),
        the state will be teleported with the exact same fidelity.

        """
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome_0": [BellIndex.B00] * 4,
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        label, minimum, error = estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities)
        assert isinstance(label, XYZEigenstateIndex)
        assert np.isclose(minimum, .5)
        assert error > 0

    def test_x_and_y_errors(self):
        """Assert that when there always are X or Y error, Z states have smallest fidelity."""
        data = {"state": [b11, b01, b11, b01],
                "midpoint_outcome_0": [BellIndex.B00] * 4,
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        label, minimum, error = estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities)
        assert label in [XYZEigenstateIndex.Z0, XYZEigenstateIndex.Z1]
        assert np.isclose(minimum, 0)
        assert np.isclose(error, 0)  # Z states should have zero fidelity in each case

    def test_x_and_z_errors(self):
        """Assert that when there always are X or Z error, Y states have smallest fidelity."""
        data = {"state": [b10, b01, b10, b01],
                "midpoint_outcome_0": [BellIndex.B00] * 4,
                }
        df = pandas.DataFrame(data)

        fidelities = determine_teleportation_fidelities_of_xyz_eigenstates_from_data(df)
        label, minimum, error = estimate_minimum_teleportation_fidelity_over_xyz_eigenstates_from_data(fidelities)
        assert label in [XYZEigenstateIndex.Y0, XYZEigenstateIndex.Y1]
        assert np.isclose(minimum, 0)
        assert np.isclose(error, 0)  # Y states should have zero fidelity in each case


def test_determine_teleportation_fidelity_from_data():
    """Assert that determining teleportation fidelity from data gives correct results."""

    # data with 50% prob X error; this should give Z and Y eigenstates 50% fidelity, but X eigenstates fidelity 1
    data = {"state": [b00, b01, b00, b01],
            "midpoint_outcome_0": [BellIndex.B00] * 4,
            }
    df = pandas.DataFrame(data)

    assert np.isclose(estimate_teleportation_fidelity_from_data(df, information_state=s0), .5)
    assert np.isclose(estimate_teleportation_fidelity_from_data(df, information_state=h0), 1.)


def test_teleportation_fidelity_optimized_over_local_unitaries_from_data():
    data = {"state": [b11] * 4,
            "midpoint_outcome_0": [BellIndex.B00] * 4,
            }
    df = pandas.DataFrame(data)

    assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries_from_data(df), 1)


def test_teleportation_fidelity_optimized_over_local_unitaries(depolar_prob=0.3):

    # Should work perfectly for any maximally entangled state
    for bell_state in [b00, b01, b10, b11]:
        assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries(bell_state), 1)
    graph_state = (np.kron(s0, h0) + np.kron(s1, h1)) / np.sqrt(2)
    assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries(graph_state), 1)

    formalism = ns.get_qstate_formalism()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    # For any depolarized Bell state, the maximal fidelity should be the same as when using a Werner state
    for bell_state in [b00, b01, b10, b11]:
        qubits = qapi.create_qubits(2, no_state=True)
        qapi.assign_qstate(qubits, bell_state)
        qapi.depolarize(qubits[0], depolar_prob)
        depolarized_bell_state = qapi.reduced_dm(qubits)
        assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries(depolarized_bell_state),
                          1 - .5 * depolar_prob)

    # Also check for maximally mixed state
    max_mixed_state = np.eye(4)
    assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries(max_mixed_state), .5)

    # Density matrix engineered to have det(T) > 0, so that case is also tested
    dm = (np.kron(np.eye(2), np.eye(2)) +
          np.kron(pauli_x, pauli_x) / 6 - np.kron(pauli_y, pauli_y) / 6 - np.kron(pauli_z, pauli_z) / 6) / 4
    # assert it is a valid density matrix (tr = 1, rho > 0)
    eigenvalues, eigenvectors = np.linalg.eig(dm)
    for eigenvalue in eigenvalues:
        assert eigenvalue >= 0
    assert np.isclose(np.trace(dm), 1)

    # Compare to value calculated from formula by hand
    assert np.isclose(teleportation_fidelity_optimized_over_local_unitaries(dm), .5 + 1 / 36)

    ns.set_qstate_formalism(formalism)


class TestDetermineTeleportationOutputState(unittest.TestCase):

    information_states = [s0, s1, h0, h1, y0, y1]
    information_state_dms = [information_state @ information_state.conj().T
                             for information_state in information_states]

    def test_perfect_states(self):
        """Assert that pure states are teleported correctly for a perfect resource state."""

        for information_state, information_state_dm in zip(self.information_states, self.information_state_dms):
            assert np.isclose(information_state_dm,
                              determine_teleportation_output_state(information_state=information_state,
                                                                   resource_state=b00)
                              ).all()

    def test_unentangled_resource_state(self):
        # If the resource state is |00>, the output state is always |0> up to correction.
        # If the information state is |0>, only |Phi+> and |Phi-> have nonzero measurement probability,
        # with corresponding correction I and Z, leaving |0> invariant.
        # If the information state is |1>, outcome will be |Psi+-> with correction X or Y, bringing |0> to |1>.
        s00 = np.array([1, 0, 0, 0])  # |00>
        assert np.isclose(s0 @ s0.conj().T,
                          determine_teleportation_output_state(information_state=s0, resource_state=s00)
                          ).all()
        assert np.isclose(s1 @ s1.conj().T,
                          determine_teleportation_output_state(information_state=s1, resource_state=s00)
                          ).all()

    def test_depolarized_states(self, depolar_prob=0.3):
        """Test teleportation of depolarized information state and/or Werner resource state.

        Tests the following:
        1. Teleporting a depolarized information state using perfect |Phi+> should be the same as first
        teleporting the qubit and then depolarizing it.
        2. Depolarization on the information state should be effectively equivalent to depolarization on
        the |Phi+> resource state (since the noise can be "moved" to the output qubit).
        3. Depolarizing both the information state and |Phi+> resource state should be equivalent
        to depolarizing the output qubit twice.

        """
        # prepare Werner resource state density matrix
        ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)
        qubit_resource_state_1, qubit_resource_state_2 = qapi.create_qubits(2, no_state=True)
        qapi.assign_qstate([qubit_resource_state_1, qubit_resource_state_2], b00)
        qapi.depolarize(qubit_resource_state_1, depolar_prob)
        depolarized_resource_state = qapi.reduced_dm([qubit_resource_state_1, qubit_resource_state_2])

        for information_state, information_state_dm in zip(self.information_states, self.information_state_dms):

            # prepare depolarized information state density matrix
            [qubit_information_state] = qapi.create_qubits(1, no_state=True)
            qapi.assign_qstate([qubit_information_state], information_state)
            qapi.depolarize(qubit_information_state, depolar_prob)
            depolarized_information_state = qapi.reduced_dm([qubit_information_state])
            qapi.depolarize(qubit_information_state, depolar_prob)
            twice_depolarized_information_state = qapi.reduced_dm([qubit_information_state])

            # Assert depolarizing information state has same effect as depolarizing output state
            output_state_from_depolarized_information_state = \
                determine_teleportation_output_state(information_state=depolarized_information_state,
                                                     resource_state=b00)
            assert np.isclose(output_state_from_depolarized_information_state, depolarized_information_state).all()

            # Assert using depolarized information state and depolarized resource state give same result
            output_state_from_depolarized_resource_state = \
                determine_teleportation_output_state(information_state=information_state,
                                                     resource_state=depolarized_resource_state)
            assert np.isclose(output_state_from_depolarized_information_state,
                              output_state_from_depolarized_resource_state
                              ).all()

            # Assert depolarizing information state and resource state is equivalent to depolarizing final state twice
            output_state_from_depolarized_information_state_and_resource_state = \
                determine_teleportation_output_state(information_state=depolarized_information_state,
                                                     resource_state=depolarized_resource_state)
            assert np.isclose(output_state_from_depolarized_information_state_and_resource_state,
                              twice_depolarized_information_state,
                              ).all()

    def test_conversion(self):
        """Check the function doesn't crash when using [a, b] instead of [[a], [b]] vector format."""
        determine_teleportation_output_state(information_state=np.array([1, 0]),
                                             resource_state=b00)


def test_determine_teleportation_fidelity():
    for information_state in [s0, h0, y0]:
        for bell_state, bell_index in zip([b00, b01, b10, b11],
                                          [BellIndex.B00, BellIndex.B01, BellIndex.B10, BellIndex.B11]):
            assert determine_teleportation_fidelity(information_state=information_state,
                                                    resource_state=bell_state,
                                                    bell_index=bell_index)
