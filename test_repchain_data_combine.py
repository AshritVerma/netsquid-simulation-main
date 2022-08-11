import io
import os
import pickle
import shutil
import unittest
from contextlib import redirect_stdout

from netsquid_simulationtools.repchain_data_combine import combine_data
from .test_repchain_dataframe_holder import create_mock_rc_dataframe_holder


class TestRepchainDataCombine(unittest.TestCase):
    """Unittests to verify the functionality of repchain_dataframe_combine."""

    test_data_dir = "tests/test_data"

    def setUp(self):
        """Creates mock dataframes for testing purposes."""
        self.results_a = 10
        self.results_b = 3

        mock_a = create_mock_rc_dataframe_holder(number_of_results=self.results_a,
                                                 baseline_parameters=["fidelity", "test_a"])
        mock_b = create_mock_rc_dataframe_holder(number_of_results=self.results_b,
                                                 baseline_parameters=["fidelity", "test_b"])

        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

        pickle.dump(mock_a, open(self.test_data_dir + "/mock_dataframe_a_combine.pickle", "wb"))
        pickle.dump(mock_a, open(self.test_data_dir + "/mock_dataframe_a_combine_wrong_suffix.pickle", "wb"))
        pickle.dump(mock_b, open(self.test_data_dir + "/mock_dataframe_b_combine.pickle", "wb"))

    def tearDown(self):
        """Deletes test data directory with all files in it."""
        shutil.rmtree(self.test_data_dir)

    def test_invalid_directory(self):
        """Checks that NotADirectoryError is raised when given directory that is not there."""
        with self.assertRaises(NotADirectoryError):
            combine_data(raw_data_dir="this_test_directory_does_not_exist")

    def test_two_files(self):
        """Checks that the correct files are combined from a given directory."""
        # check what gets printed out in the function
        f = io.StringIO()
        with redirect_stdout(f):
            combine_data(raw_data_dir=self.test_data_dir, suffix="_combine.pickle",
                         output=self.test_data_dir + "/repchain_data_combine_results.pickle")

        printed_output = f.getvalue().strip().split(" ")
        self.assertEqual(printed_output[1], "2")  # check number of files
        self.assertEqual(printed_output[4], str(self.results_a + self.results_b))  # check number of successes
