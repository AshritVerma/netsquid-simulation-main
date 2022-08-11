from statistics import StatisticsError

import os
import pickle
import random
import shutil
import unittest
from copy import deepcopy

import netsquid.qubits.ketstates as ketstates
import numpy as np
import pandas

from netsquid_simulationtools.linear_algebra import XYZEigenstateIndex
from netsquid_simulationtools.repchain_data_process import process_data, process_repchain_dataframe_holder, \
    secret_key_capacity, process_data_fidelity, convert_rdfh_with_number_of_rounds, \
    process_data_bb84, process_data_duration, process_data_teleportation_fidelity
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder


class TestRepchainDataProcess(unittest.TestCase):
    """Unittests to verify the functionality of `repchain_dataframe_process`."""

    test_data_dir = "tests/test_data"

    def setUp(self):
        """Creates mock dataframes for testing purposes."""
        self.number_of_results = 6

        baseline_parameters_dict = {"probability": 0.42, "fidelity": 0.42, "generation_duration_unit": "seconds"}

        bell_states = [np.outer(ketstates.b00, ketstates.b00), np.outer(ketstates.b01, ketstates.b01),
                       np.outer(ketstates.b10, ketstates.b10), np.outer(ketstates.b11, ketstates.b11)]

        different_bases = {
            "state": random.choices(population=bell_states, k=self.number_of_results),
            "basis_A": ["X", "Y", "Z", "X", "Y", "Z"],
            "basis_B": ["Y", "X", "Y", "Z", "X", "Y"],
            "outcome_A": [0, 1, 0, 1, 1, 1],
            "outcome_B": [1, 0, 1, 0, 1, 0],
            "generation_duration": [1, 1, 1, 1, 1, 1],
            "midpoint_outcome_0": [0, 1, 2, 3, 0, 1],
            "midpoint_outcome_1": [3, 2, 1, 0, 3, 2],
            "swap_outcome_0": [0, 1, 2, 3, 0, 1],
            "length": [1, 2, 3, 4, 5, 6]
        }

        same_bases = {
            "state": random.choices(population=bell_states, k=self.number_of_results),
            "basis_A": ["X", "X", "Z", "Z", "Z", "Z"],
            "basis_B": ["X", "X", "Z", "Z", "Z", "Z"],
            "outcome_A": [0, 1, 0, 1, 0, 1],
            "outcome_B": [1, 0, 1, 0, 1, 0],
            "generation_duration": [1, 1, 1, 1, 1, 1],
            "midpoint_outcome_0": [0, 1, 2, 3, 0, 2],
            "midpoint_outcome_1": [3, 2, 1, 0, 3, 1],
            "swap_outcome_0": [0, 0, 0, 0, 0, 0],
            "length": [1, 1, 2, 2, 3, 3]
        }

        same_bases_without_state = deepcopy(same_bases)
        del same_bases_without_state["state"]

        same_bases_with_without_length = deepcopy(same_bases)
        del same_bases_with_without_length["length"]
        same_bases_with_without_length["error_rate"] = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]

        self.fidelity_data = {
            "state": [bell_states[1]] * 6,
            "generation_duration": [1] * 6,
            "midpoint_outcome_0": [1] * 6,
            "midpoint_outcome_1": [1] * 6,
            "swap_outcome_0": [1] * 6,
            "length": [1, 1, 2, 2, 3, 3]
        }

        self.fidelity_data_with_kets = {
            "state": [ketstates.b00] * 3,
            "generation_duration": [1] * 3,
            "midpoint_outcome_0": [0] * 3,
        }

        self.mock_different_bases = RepchainDataFrameHolder(number_of_nodes=3,
                                                            baseline_parameters=baseline_parameters_dict,
                                                            data=different_bases)

        self.mock_same_bases = RepchainDataFrameHolder(number_of_nodes=3,
                                                       baseline_parameters=baseline_parameters_dict,
                                                       data=same_bases)

        self.mock_same_bases_without_state = RepchainDataFrameHolder(number_of_nodes=3,
                                                                     baseline_parameters=baseline_parameters_dict,
                                                                     data=same_bases_without_state)

        self.mock_same_bases_without_length = RepchainDataFrameHolder(number_of_nodes=3,
                                                                      baseline_parameters=baseline_parameters_dict,
                                                                      data=same_bases_with_without_length)

        self.mock_fid = RepchainDataFrameHolder(number_of_nodes=3,
                                                baseline_parameters=baseline_parameters_dict,
                                                data=self.fidelity_data)

        self.mock_fid_with_kets = RepchainDataFrameHolder(number_of_nodes=2,
                                                          baseline_parameters=baseline_parameters_dict,
                                                          data=self.fidelity_data_with_kets)

        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

        pickle.dump(self.mock_same_bases, open(self.test_data_dir + "/mock_dataframe_qkd.pickle", "wb"))
        pickle.dump(self.mock_fid, open(self.test_data_dir + "/mock_dataframe_fidelity.pickle", "wb"))

    def tearDown(self):
        """Deletes test data directory with all files in it."""
        shutil.rmtree(self.test_data_dir)

    def test_secret_key_capacity_positive_length(self):
        """Check results for secret key capacity with positive length."""

        attenuation = 5 / (11 * np.log(10))  # corresponds to attenuation length of 22 km
        sk_capacity, tgw_bound = secret_key_capacity(length=4.2, attenuation=attenuation)
        self.assertAlmostEqual(sk_capacity, 2.5245638855922863)
        self.assertAlmostEqual(tgw_bound, 3.3934147378943593)

    def test_secret_key_capacity_zero_length(self):
        """Check results for secret key capacity with zero length and values very close to zero."""
        cap1, bound1 = secret_key_capacity(0)
        cap2, bound2 = secret_key_capacity(10e-16)

        # check results are equal
        self.assertEqual(cap1, cap2)
        self.assertEqual(bound1, bound2)

        # check values of results
        self.assertEqual(cap1, 13.287712379549609)
        self.assertEqual(bound1, 14.287640242994135)

    def test_secret_key_capacity_negative_length(self):
        """Check results for secret key capacity with negative length."""
        with self.assertRaises(ValueError):
            _, _ = secret_key_capacity(-1.0)

    def test_number_of_successes(self):
        """Check that number of successes does not depend on whether same basis was used."""
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases,
                                                           processing_functions=[],
                                                           some_arguement=10)
        assert np.isclose(processed_data.number_of_successes, 2.).all()
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_different_bases,
                                                           processing_functions=[])
        assert np.isclose(processed_data.number_of_successes, 1.).all()

    def test_process_duration_same_bases(self):
        """Check that processing of generation duration is performed correctly when bases are the same."""
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases,
                                                           processing_functions=[process_data_duration],
                                                           some_argument="string")
        assert np.isclose(processed_data.duration_per_success, 1.).all()
        assert np.isclose(processed_data.duration_per_success_error, 0.).all()

    def test_process_duration_without_qkd_measurement_data(self):
        """Check that processing of generation duration is performed correctly when no QKD measurement data is
        available, i.e. there are no bases and outcomes, only states."""
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_fid,
                                                           processing_functions=[process_data_duration])
        assert (processed_data.duration_per_success == 1.).all()
        assert (processed_data.duration_per_success_error == 0.).all()

    def test_process_bb84_different_bases(self):
        """Check that an error is raised when the measurement bases are different (i.e. not enough data to
        estimate QKD performance)."""
        with self.assertRaises(StatisticsError):
            process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_different_bases,
                                              processing_functions=[process_data_bb84])

    def test_process_bb84_same_bases(self):
        """Check that sk rate is calculated properly with perfect correlation."""
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases,
                                                           processing_functions=[process_data_bb84])

        assert "plob_bound" in processed_data.columns
        assert "tgw_bound" in processed_data.columns

        self.assertEqual(len(processed_data), 3)  # there are three unique value of varied parameters

        self.assertTrue((processed_data.sk_rate == 1).all())
        self.assertTrue((processed_data.sk_rate_upper_bound == 1).all())
        self.assertTrue((processed_data.sk_rate_lower_bound == 1).all())

    def test_process_bb84_no_states(self):
        """Check that bb84 processing works when there is no state information in data."""
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases_without_state,
                                                           processing_functions=[process_data_bb84])

        assert "plob_bound" in processed_data.columns
        assert "tgw_bound" in processed_data.columns

        self.assertEqual(len(processed_data), 3)  # there are three unique value of varied parameters

        self.assertTrue((processed_data.sk_rate == 1).all())
        self.assertTrue((processed_data.sk_rate_upper_bound == 1).all())
        self.assertTrue((processed_data.sk_rate_lower_bound == 1).all())

    def test_process_bb84_without_length(self):
        """Check that length is not a mandatory parameter, and that capacity bounds are not calculated when there is
        no length specified."""
        processed_data = \
            process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases_without_length,
                                              processing_functions=[process_data_bb84])
        assert "plob_bound" not in processed_data.columns
        assert "tgw_bound" not in processed_data.columns

    def test_bb84_sifting_factor(self):
        """Check that the constant sifting factor is applied correctly."""
        sifting_factor = random.random()
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_same_bases,
                                                           processing_functions=[process_data_bb84],
                                                           sifting_factor=sifting_factor)

        self.assertEqual(len(processed_data), 3)  # there are three unique value of varied parameters

        self.assertTrue((processed_data.sk_rate == 1 * sifting_factor).all())
        self.assertTrue((processed_data.sk_rate_upper_bound == 1 * sifting_factor).all())
        self.assertTrue((processed_data.sk_rate_lower_bound == 1 * sifting_factor).all())

    def test_process_data_bb84(self):
        """Check that correct output files for processed data are created."""
        process_data(raw_data_dir=self.test_data_dir, suffix="qkd.pickle",
                     output=self.test_data_dir + "/repchain_data_process_qkd_results.pickle",
                     csv_output_filename=self.test_data_dir + "/csv_output.csv",
                     process_bb84=True, plot_processed_data=False)

        df = pandas.read_csv(self.test_data_dir + "/csv_output.csv")

        self.assertEqual(df.shape, (self.number_of_results / 2, 15))
        # divide by 2 because there are two results per unique parameter set

    def test_process_teleportation_fidelity(self):
        """Check that teleportation fidelities are calculated correctly for perfect Bell states."""

        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_fid,
                                                           processing_functions=[process_data_teleportation_fidelity])

        self.assertEqual(len(processed_data), 3)

        self.assertTrue((np.isclose(processed_data.teleportation_fidelity_average, 1)).all())
        self.assertTrue((np.isclose(processed_data.teleportation_fidelity_average_error, 0)).all())
        for index in processed_data.teleportation_fidelity_minimum_xyz_eigenstate_index:
            XYZEigenstateIndex(index)  # fails if not interpretable as XYZ index
        self.assertTrue((np.isclose(processed_data.teleportation_fidelity_minimum_xyz_eigenstates, 1)).all())
        self.assertTrue((np.isclose(processed_data.teleportation_fidelity_minimum_xyz_eigenstates_error, 0)).all())
        self.assertTrue((np.isclose(processed_data.teleportation_fidelity_average_optimized_local_unitaries, 1)).all())

    def test_process_teleportation_fidelity_with_kets(self):
        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_fid_with_kets,
                                                           processing_functions=[process_data_teleportation_fidelity])
        assert np.isclose(processed_data.teleportation_fidelity_average, [1.]).all()
        assert np.isclose(processed_data.teleportation_fidelity_average_error, [0.]).all()
        assert np.isclose(processed_data.teleportation_fidelity_minimum_xyz_eigenstates, [1.]).all()
        assert np.isclose(processed_data.teleportation_fidelity_minimum_xyz_eigenstates_error, [0.]).all()
        assert np.isclose(processed_data.teleportation_fidelity_average_optimized_local_unitaries, [1.]).all()

    def test_process_data_teleportation_fidelity(self):

        process_data(raw_data_dir=self.test_data_dir, suffix="fidelity.pickle",
                     output=self.test_data_dir + "/repchain_data_process_fid_results.pickle",
                     csv_output_filename=self.test_data_dir + "/output.csv",
                     process_teleportation=True, plot_processed_data=False)

        df = pandas.read_csv(self.test_data_dir + "/output.csv")

        self.assertTrue((np.isclose(df["teleportation_fidelity_average"], [1, 1, 1])).all())
        self.assertTrue((np.isclose(df["teleportation_fidelity_average_error"], [0, 0, 0])).all())
        for index in df["teleportation_fidelity_minimum_xyz_eigenstate_index"]:
            XYZEigenstateIndex(index)  # fails if not interpretable as XYZ index
        self.assertTrue((np.isclose(df["teleportation_fidelity_minimum_xyz_eigenstates"], [1, 1, 1])).all())
        self.assertTrue((np.isclose(df["teleportation_fidelity_minimum_xyz_eigenstates_error"], [0, 0, 0])).all())
        self.assertTrue((np.isclose(df["teleportation_fidelity_average_optimized_local_unitaries"],
                                    [1, 1, 1])).all())

    def test_process_fidelity(self):
        """Check that fidelity and fidelity error is calculated properly."""

        processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=self.mock_fid,
                                                           processing_functions=[process_data_fidelity])

        self.assertEqual(len(processed_data), 3)

        self.assertTrue((np.isclose(processed_data.fidelity, 1.0)).all())
        self.assertTrue((np.isclose(processed_data.fidelity_error, 0.0)).all())

    def test_process_data_fidelity(self):
        """Check that the output file holds correct number of datapoints."""
        process_data(raw_data_dir=self.test_data_dir, suffix="fidelity.pickle",
                     output=self.test_data_dir + "/repchain_data_process_fid_results.pickle",
                     csv_output_filename=self.test_data_dir + "/output.csv",
                     process_fidelity=True, plot_processed_data=False)

        df = pandas.read_csv(self.test_data_dir + "/output.csv")
        assert np.isclose(df["fidelity"], [1., 1., 1.]).all()
        assert np.isclose(df["fidelity_error"], [0., 0., 0.]).all()

    def test_convert_rdfh_with_number_of_rounds_suitable_rdfh(self):
        """Check that RDFH with `number_of_rounds` is correctly converted."""
        # set up fake 'outdated' RDFH
        baseline_parameters = {"probability": 0.42, "fidelity": 0.42}
        outdated = RepchainDataFrameHolder(number_of_nodes=3,
                                           baseline_parameters=baseline_parameters,
                                           data=self.fidelity_data)
        del outdated.baseline_parameters['generation_duration_unit']
        outdated.dataframe = outdated.dataframe.rename(columns={"generation_duration": "number_of_rounds"})

        converted = convert_rdfh_with_number_of_rounds(outdated)

        self.assertTrue('generation_duration' in converted.dataframe.columns)
        self.assertFalse('number_of_rounds' in converted.dataframe.columns)
        self.assertEqual(converted.baseline_parameters,
                         {"probability": 0.42, "fidelity": 0.42, "generation_duration_unit": "rounds"})

    def test_convert_rdfh_with_number_of_rounds_not_rdfh(self):
        """Check that inputs that are not RepchainDataFrameHolder are rejected."""
        with self.assertRaises(ValueError):
            _ = convert_rdfh_with_number_of_rounds(pandas.DataFrame(data=self.fidelity_data))

    def test_convert_rdfh_with_number_of_rounds_wrong_rdfh(self):
        """Check that input RDFH must contain `number_of_rounds` column."""
        with self.assertRaises(ValueError):
            _ = convert_rdfh_with_number_of_rounds(self.mock_different_bases)


if __name__ == "__main__":
    unittest.main()
