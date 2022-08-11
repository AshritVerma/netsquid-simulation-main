import unittest

import pandas
import numpy as np
from netsquid.qubits.ketstates import BellIndex

from netsquid.qubits.ketstates import b00, b01, b10, b11
from netsquid.qubits.operators import I, X, Y, Z

from netsquid_simulationtools.repchain_data_functions import _expected_target_state, end_to_end_fidelity, \
    estimate_density_matrix_from_data


class ProcessingFunctionsTestBasis(unittest.TestCase):
    """basis for unittests to verify the functionality of the processing tools for the repchain_dataframe_holder."""
    @staticmethod
    def bell_state(index):
        """static method to get KetState corresponding to bell index."""
        bell_index_to_state = {BellIndex.B00: b00,
                               BellIndex.B01: b01,
                               BellIndex.B11: b11,
                               BellIndex.B10: b10,
                               }
        return bell_index_to_state[index]

    @staticmethod
    def pauli_correction(index):
        # switched version for applying Pauli to A in stead of B
        bell_index_to_correction = {BellIndex.PHI_PLUS: I ^ I,
                                    BellIndex.PSI_PLUS: I ^ X,
                                    BellIndex.PSI_MINUS: I ^ Y,
                                    BellIndex.PHI_MINUS: I ^ Z,
                                    }
        return bell_index_to_correction[index]

    @staticmethod
    def operator(index):
        bell_index_to_operator = {BellIndex.PHI_PLUS: I,
                                  BellIndex.PSI_PLUS: X,
                                  BellIndex.PSI_MINUS: Y,
                                  BellIndex.PHI_MINUS: Z,
                                  }
        return bell_index_to_operator[index]

    @staticmethod
    def SimData():
        data = pandas.DataFrame({"basis_A": [],
                                 "basis_B": [],
                                 "outcome_A": [],
                                 "outcome_B": [],
                                 "generation_duration": [],
                                 "midpoint_outcome_0": [],
                                 "midpoint_outcome_1": [],
                                 "midpoint_outcome_2": [],
                                 "swap_outcome_0": [],
                                 "swap_outcome_1": [],
                                 "state": [],
                                 })
        #params = {}
        return data


class TestExpectedTargetState(ProcessingFunctionsTestBasis):
    """Test if expected target states are determined correctly."""

    def test_expected_target_state_one_midpoint_outcome(self):
        for bell_index in BellIndex:
            row = pandas.Series({"midpoint_outcome_0": bell_index})
            assert _expected_target_state(row) == bell_index

    def test_expected_target_state_one_swap_outcome(self):
        for bell_index in BellIndex:
            row = pandas.Series({"swap_outcome_0": bell_index})
            assert _expected_target_state(row) == bell_index

    def test_expected_target_state(self):
        """Tests Correctness of expected target state first, as other functions are based on this function."""
        data = self.SimData()

        # see if we labelled the Bell states correctly :)
        self.assertTrue((self.bell_state(0) == (self.pauli_correction(0) * b00).arr).all())
        self.assertTrue((self.bell_state(1) == (self.pauli_correction(1) * b00).arr).all())
        self.assertTrue((self.bell_state(2) == - 1j * (self.pauli_correction(2) * b00).arr).all())
        self.assertTrue((self.bell_state(3) == (self.pauli_correction(3) * b00).arr).all())

        # fill outcomes in Dataframe with all possible combinations
        outcome_list = []
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        for e in range(4):
                            outcome_list.append([a, b, c, d, e])
        for i in range(len(outcome_list)):
            data.loc[i] = ["X", "X", 1, 1, int(1)] + outcome_list[i] + [1]

        # compare state that is predicted by _expected_target_state with state calculated with pauli corrections
        for i in range(len(outcome_list)):
            expected_index = _expected_target_state(data.iloc[i])
            ii = outcome_list[i].count(0)
            x = outcome_list[i].count(1)
            y = outcome_list[i].count(2)
            z = outcome_list[i].count(3)
            pauli_corrected_state = np.linalg.matrix_power(self.pauli_correction(0).arr, ii) @ \
                np.linalg.matrix_power(self.pauli_correction(1).arr, x) @ \
                np.linalg.matrix_power(self.pauli_correction(2).arr, y) @ \
                np.linalg.matrix_power(self.pauli_correction(3).arr, z) @ b00
            try:
                self.assertTrue((pauli_corrected_state == self.bell_state(expected_index)).all())
            except AssertionError:
                # global complex phase but still the same state
                self.assertTrue((pauli_corrected_state == 1j * self.bell_state(expected_index)).all())


class TestEndToEndFidelity(ProcessingFunctionsTestBasis):
    def test_end_to_end_fidelity(self):
        """Tests correct implementation of the end-to-end fidelity calculation."""

        # test functionality with perfect bell states
        for i in range(4):
            data = {"state": [self.bell_state(i)]*10,
                    "midpoint_outcome_0": [1]*10,
                    "generation_duration": [1]*10,
                    }
            df = pandas.DataFrame(data)

            f, f_err = end_to_end_fidelity(df)

            if i == 1:
                self.assertAlmostEqual(f, 1.)
            else:
                self.assertAlmostEqual(f, 0.)
            self.assertAlmostEqual(f_err, 0.)


class TestEstimateDensityMatrixFromData(unittest.TestCase):
    def test_multiple_copies_pure_state(self):
        data = {"state": [b00] * 100,
                "midpoint_outcome_0": [BellIndex.B00] * 100,
                }
        df = pandas.DataFrame(data)

        state = estimate_density_matrix_from_data(df)
        assert np.isclose(state, b00 @ b00.conj().T).all()

    def test_perfect_bell_states(self):
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome": [BellIndex.B00, BellIndex.B01, BellIndex.B10, BellIndex.B11]
                }
        df = pandas.DataFrame(data)

        state = estimate_density_matrix_from_data(df)
        assert np.isclose(state, b00 @ b00.conj().T).all()

    def test_maximally_mixed_state(self):
        data = {"state": [b00, b01, b10, b11],
                "midpoint_outcome_0": [BellIndex.B00] * 4,
                }
        df = pandas.DataFrame(data)

        state = estimate_density_matrix_from_data(df)
        assert np.isclose(state, np.eye(4) / 4).all()


if __name__ == "__main__":
    unittest.main()
