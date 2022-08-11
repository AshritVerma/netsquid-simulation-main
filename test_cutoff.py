import shutil
import unittest
import os
import pickle
from copy import deepcopy
from netsquid_simulationtools.repchain_data_cutoff import implement_cutoff_duration, scan_cutoff
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from .test_repchain_dataframe_holder import create_mock_rc_dataframe_holder


class TestCutoffImplementationDefault(unittest.TestCase):

    @staticmethod
    def implement_cutoff_duration(repchain_dataframe_holder, max_duration):
        return implement_cutoff_duration(repchain_dataframe_holder=repchain_dataframe_holder,
                                         max_duration=max_duration,
                                         use_default_method=True)

    @classmethod
    def setUpClass(cls):

        cls.InitialData = create_mock_rc_dataframe_holder(number_of_nodes=2,
                                                          varied_parameters=["duration_between_alice_and_bob"],
                                                          number_of_results=4)

        generation_duration = [3, 6, 10, 8]
        cls.InitialData.baseline_parameters.update({"generation_duration_unit": "rounds"})
        rounds_between_alice_and_bob = [2, 4, 7, 1]

        basis_A = ["X", "Y", "Z", "X"]

        cls.InitialData.dataframe["duration_between_alice_and_bob"] = rounds_between_alice_and_bob
        cls.InitialData.dataframe["generation_duration"] = generation_duration
        cls.InitialData.dataframe["basis_A"] = basis_A
        cls.InitialDataframe = deepcopy(cls.InitialData.dataframe)

    def tearDown(self) -> None:
        self.assertTrue(self.InitialDataframe.equals(self.InitialData.dataframe))

    def test_cutoff_all_results(self):
        # test whether a cutoff time smaller than any of the memory rounds empties the dataframe.

        results = self.implement_cutoff_duration(self.InitialData, 0)
        self.assertEqual(results.number_of_results, 0)

    def test_cutoff_no_results(self):
        # test whether a cutoff time larger than any of the memory rounds leaves it unaltered.

        results = self.implement_cutoff_duration(self.InitialData, 8)
        self.assertTrue(results.dataframe.equals(self.InitialData.dataframe))

    def test_cutoff_single_result(self):
        # test whether results are processed correctly if only one result (in the center) hits the cutoff time

        results = self.implement_cutoff_duration(self.InitialData, 5)
        expected_basis_A = ["X", "Y", "X"]

        # In the third attempt, there were 3 rounds before Alice succeeded and another 7 before Bob succeeded.
        # If a cutoff was implemented of 5, it would have been 3 + 5 = 8, and then a new one would have started.
        # This effectively adds 8 rounds to the number of rounds of the result that is obtained after.
        expected_duration = [3, 6, 16]

        result_basis_A = results.dataframe["basis_A"].tolist()
        result_duration = results.dataframe["generation_duration"].tolist()
        self.assertListEqual(result_basis_A, expected_basis_A)
        self.assertListEqual(result_duration, expected_duration)

    def test_cutoff_adjacent_results(self):
        # test whether results are processed correctly if two adjacent results hit the cutoff time

        results = self.implement_cutoff_duration(self.InitialData, 3)
        expected_basis_A = ["X", "X"]
        expected_duration = [3, 19]

        result_basis_A = results.dataframe["basis_A"].tolist()
        result_duration = results.dataframe["generation_duration"].tolist()
        self.assertListEqual(result_basis_A, expected_basis_A)
        self.assertListEqual(result_duration, expected_duration)

    def test_cutoff_last_result(self):
        # test whether results are processed correctly if the last result if cut off (this is a special case)

        initial_data = deepcopy(self.InitialData)
        initial_data.dataframe["duration_between_alice_and_bob"] = [1, 2, 3, 10]
        results = self.implement_cutoff_duration(initial_data, 5).dataframe
        expected_results = initial_data.dataframe.drop(3)
        self.assertTrue(results.equals(expected_results))

    def test_cutting_off_duration_equal_to_max(self):
        # test whether results are processed correctly when one of the memory rounds is equal to the cutoff

        results = self.implement_cutoff_duration(self.InitialData, 4)
        expected_basis_A = ["X", "Y", "X"]
        expected_duration = [3, 6, 15]

        result_basis_A = results.dataframe["basis_A"].tolist()
        result_duration = results.dataframe["generation_duration"].tolist()
        self.assertListEqual(result_basis_A, expected_basis_A)
        self.assertListEqual(result_duration, expected_duration)


class TestCutoffImplementationNondefault(TestCutoffImplementationDefault):

    @staticmethod
    def implement_cutoff_duration(repchain_dataframe_holder, max_duration):
        return implement_cutoff_duration(repchain_dataframe_holder=repchain_dataframe_holder,
                                         max_duration=max_duration,
                                         use_default_method=False)


def test_scan_cutoff():

    test_data_dir = "tests/test_data"

    dataframe_holder = create_mock_rc_dataframe_holder(number_of_nodes=2,
                                                       varied_parameters=["duration_between_alice_and_bob"],
                                                       number_of_results=10)
    dataframe_holder.dataframe["duration_between_alice_and_bob"] = list(range(1, 11))
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    pickle.dump(dataframe_holder, open(test_data_dir + "/mock_dataframe_cutoff.pickle", "wb"))
    scan_cutoff(cutoff_duration_min=3, cutoff_duration_max=9, stepsize=3,
                target_dir=test_data_dir, filename=test_data_dir + "/mock_dataframe_cutoff.pickle")
    num_cutoffs = 0
    for file in os.listdir(test_data_dir):
        if file[:7] == "cutoff=":
            assert file[-7:] == ".pickle"
            num_cutoffs += 1
            cutoff = int(file[-8])
            assert cutoff in [3, 6, 9]
            repchain_with_cutoff = pickle.load(open(test_data_dir + "/" + file, "rb"))
            assert isinstance(repchain_with_cutoff, RepchainDataFrameHolder)
            assert repchain_with_cutoff.baseline_parameters["cutoff_round"] == cutoff
            assert (repchain_with_cutoff.dataframe["duration_between_alice_and_bob"] <= cutoff).all()
    assert num_cutoffs == 3
    shutil.rmtree(test_data_dir)


if __name__ == "__main__":
    unittest.main()
