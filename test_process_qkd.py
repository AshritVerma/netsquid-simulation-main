from netsquid_simulationtools.process_qkd import _do_we_expect_correlation, agreement_with_expected_outcome, qber, \
    _estimate_bb84_secret_key_rate, _binary_entropy, estimate_bb84_secret_key_rate_from_data
from netsquid_simulationtools.repchain_data_functions import _expected_target_state
from tests.test_repchain_data_functions import ProcessingFunctionsTestBasis
import pandas
import numpy as np
from statistics import mean, stdev


class TestDoWeExpectCorrelation(ProcessingFunctionsTestBasis):

    def test_expected_correlations_unequal_bases(self):
        row = pandas.Series({"basis_A": "Z", "basis_B": "X"})
        with self.assertRaises(ValueError):
            _do_we_expect_correlation(row)

    def test_expected_correlations(self):
        """Tests Correctness of expected correlation for different states and basis choices."""
        # create list that contains each pauli correction once
        outcome_list = []
        for i in range(4):
            outcome_list.append([i, 0, 0, 0, 0])
        # basis dictionary
        basis = {1: "X",
                 2: "Y",
                 3: "Z"}
        # loop over all three basis
        for basis_index in [1, 2, 3]:
            data = self.SimData()

            # fill dataframe to get each bellstate once
            for l in range(len(outcome_list)):
                data.loc[l] = [basis[basis_index], basis[basis_index], 1, 1, 1] + outcome_list[l] + [1]

            # check if predicted correlations agree with expectation value of state
            for i in range(len(outcome_list)):
                do_we_expect_corr = _do_we_expect_correlation(data.iloc[i])
                bell_state = self.bell_state(_expected_target_state(data.iloc[i]))

                bell_state_c = np.conjugate(bell_state)
                bell_state_T = np.transpose(bell_state_c)
                meas, = bell_state_T @ (self.operator(basis_index) ^ self.operator(basis_index)).arr @ bell_state
                if int(np.round(meas[0])) == -1:
                    self.assertFalse(do_we_expect_corr)
                else:
                    self.assertTrue(do_we_expect_corr)

            # while we have the dataframe, test agreement and qber
            agreement_list = agreement_with_expected_outcome(data, basis[basis_index])
            qb, qber_error = qber(data, basis[basis_index])
            self.assertTrue(mean(agreement_list) == 0.5)
            self.assertTrue(qb == 0.5)
            # TODO: think about whether qber should return stdev or error
            self.assertTrue(qber_error == stdev(agreement_list) / np.sqrt(len(agreement_list)))


class TestEstimateBB84SecretKeyRateFromData(ProcessingFunctionsTestBasis):
    def test_secret_key_rate(self):
        """Tests Correctness of calculated secret key rate."""

        for qber_x in np.linspace(0, 1, 100):
            for qber_z in np.linspace(0, 1, 100):
                skr, skr_min, skr_max, skr_error = _estimate_bb84_secret_key_rate(qber_x, 0., qber_z, 0., 1, 0)
                self.assertEqual((max(0., 1 - _binary_entropy(qber_x) - _binary_entropy(qber_z))), skr)
                for i in [skr, skr_min, skr_max]:
                    self.assertTrue(max(i, 1.) == 1.)
                    self.assertTrue(min(i, 0.) == 0.)

        data = self.SimData()
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
        secret_key_rate, skr_min, skr_max, skr_error = estimate_bb84_secret_key_rate_from_data(data)
        self.assertEqual(secret_key_rate, 0.)
