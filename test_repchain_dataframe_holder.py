import io
import unittest
import random
import numpy as np
from contextlib import redirect_stdout

import netsquid.qubits.ketstates as ketstates
from netsquid.qubits.ketstates import BellIndex

from netsquid_simulationtools.repchain_dataframe_holder import VersionMismatchError
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder


def create_mock_rc_dataframe_holder(number_of_nodes=5, baseline_parameters=None, additional_packages=None,
                                    varied_parameters=None, number_of_results=5, kets=False):
    """Create mock instance of RepchainDataFrameHolder for testing.

    Parameters
    ----------
    number_of_nodes : int
        Number of nodes used in repeater chain simulation
        (including both repeater nodes and end nodes, but not heralding stations).
    baseline_parameters : list of str
        Names of baseline parameters.
    additional_packages : list of str
        Names of additional packages that are used to generate the data.
    varied_parameters : list of str
        Names of varied parameters.
    number_of_results : int
        Number of fake results to be included.
    kets : bool
        Set true to have ket-vector states instead of density-matrix states.

    """

    if varied_parameters is None:
        varied_parameters = ["length", "cycle_time"]
    if baseline_parameters is None:
        baseline_parameters = ["probability", "fidelity"]
    if number_of_nodes < 2:
        raise ValueError("Can only create mock results for more than 1 node.")

    bell_states_ket = [ketstates.b00, ketstates.b01, ketstates.b10, ketstates.b11]
    bell_states_dm = [np.outer(ketstates.b00, ketstates.b00), np.outer(ketstates.b01, ketstates.b01),
                      np.outer(ketstates.b10, ketstates.b10), np.outer(ketstates.b11, ketstates.b11)]
    bell_states = bell_states_ket if kets else bell_states_dm

    data = {"state": random.choices(population=bell_states, k=number_of_results),
            "basis_A": random.choices(population=["X", "Y", "Z"], k=number_of_results),
            "basis_B": random.choices(population=["X", "Y", "Z"], k=number_of_results),
            "outcome_A": random.choices(population=[0, 1], k=number_of_results),
            "outcome_B": random.choices(population=[0, 1], k=number_of_results),
            "generation_duration": random.choices(population=range(1, 100), k=number_of_results),
            }
    data["generation_duration"][0] = 4.2  # check that generation_duration can be float

    for i in range(number_of_nodes - 2):
        data.update({"midpoint_outcome_{}".format(i): random.choices(population=[BellIndex.PHI_PLUS,
                                                                                 BellIndex.PSI_PLUS,
                                                                                 BellIndex.PSI_MINUS,
                                                                                 BellIndex.PHI_MINUS],
                                                                     k=number_of_results),
                     "swap_outcome_{}".format(i): random.choices(population=[BellIndex.PHI_PLUS,
                                                                             BellIndex.PSI_PLUS,
                                                                             BellIndex.PSI_MINUS,
                                                                             BellIndex.PHI_MINUS],
                                                                 k=number_of_results)})
    data.update({"midpoint_outcome_{}".format(number_of_nodes - 2): random.choices(population=[BellIndex.PHI_PLUS,
                                                                                               BellIndex.PSI_PLUS,
                                                                                               BellIndex.PSI_MINUS,
                                                                                               BellIndex.PHI_MINUS],
                                                                                   k=number_of_results)})

    for param in varied_parameters:
        param_values = []
        for i in range(number_of_results):
            param_values.append(random.random())
        data.update({param: param_values})

    baseline_parameters_dict = {}
    for baseline_param in baseline_parameters:
        baseline_parameters_dict.update({baseline_param: random.random()})
    baseline_parameters_dict.update({'generation_duration_unit': 'seconds'})

    return RepchainDataFrameHolder(number_of_nodes=number_of_nodes,
                                   baseline_parameters=baseline_parameters_dict,
                                   additional_packages=additional_packages,
                                   data=data)


class TestRepchainDataFrameHolder(unittest.TestCase):

    def setUp(self) -> None:
        self.rc_dataframe_holder = create_mock_rc_dataframe_holder(kets=False)

    def check_rc_dataframe_holder(self, expected_number_of_results=10,
                                  expected_baseline_parameters=None,
                                  expected_varied_parameters=None):
        if expected_baseline_parameters is None:
            expected_baseline_parameters = ["probability", "fidelity", "generation_duration_unit"]
        if expected_varied_parameters is None:
            expected_varied_parameters = ["length", "cycle_time"]
        self.rc_dataframe_holder.check_dataframe_correctness()
        # check if expected number of results
        self.assertEqual(self.rc_dataframe_holder.number_of_results, expected_number_of_results)
        # check if all baseline parameters are present
        self.assertEqual(set(self.rc_dataframe_holder.baseline_parameters.keys()), set(expected_baseline_parameters))
        # check if no baseline parameters in the columns
        self.assertEqual(set(expected_baseline_parameters) & set(self.rc_dataframe_holder.dataframe.columns), set())
        # check if varied parameters are in the columns
        self.assertTrue(set(expected_varied_parameters) <= set(self.rc_dataframe_holder.dataframe.columns))
        self.assertEqual(set(expected_varied_parameters),
                         set(self.rc_dataframe_holder.varied_parameters))
        # check if netsquid and netsquid-simulationtools are by default included in the baseline packages
        self.assertIn("netsquid", self.rc_dataframe_holder.packages)
        self.assertIn("netsquid-simulationtools", self.rc_dataframe_holder.packages)


class TestMockRepchainDataFrameHolder(TestRepchainDataFrameHolder):

    def test_mock_rc_dataframe_holder(self):

        self.check_rc_dataframe_holder(expected_number_of_results=5)

    def test_invalid_number_of_nodes(self):

        with self.assertRaises(TypeError):
            create_mock_rc_dataframe_holder(number_of_nodes=2.5)


class TestMockRepchainDataFrameHolderWithKets(TestMockRepchainDataFrameHolder):

    def setUp(self) -> None:
        self.rc_dataframe_holder = create_mock_rc_dataframe_holder(kets=True)


class TestRepchainDataFrameHolderChecks(TestRepchainDataFrameHolder):

    def tearDown(self) -> None:
        # test succeeds only if an error is raised
        failed = False
        try:
            self.rc_dataframe_holder.check_dataframe_correctness()
        except ValueError:
            failed = True
        self.assertTrue(failed)

#    def test_invalid_state_ket(self):
#        self.rc_dataframe_holder.dataframe["state"].update(pandas.Series([ketstates.b00], index=[0]))

    def test_invalid_basis_A(self):
        self.rc_dataframe_holder.dataframe.at[0, "basis_A"] = 1

    def test_invalid_basis_B(self):
        self.rc_dataframe_holder.dataframe.at[0, "basis_B"] = "W"

    def test_invalid_outcome_A(self):
        self.rc_dataframe_holder.dataframe.at[0, "outcome_A"] = -1

    def test_invalid_outcome_B(self):
        self.rc_dataframe_holder.dataframe.at[0, "outcome_B"] = 2

    def test_invalid_generation_duration(self):
        self.rc_dataframe_holder.dataframe.at[0, "generation_duration"] = -10

    def test_invalid_midpoint_outcome(self):
        self.rc_dataframe_holder.dataframe.at[0, "midpoint_outcome_0"] = 4

    def test_invalid_swap_outcome(self):
        self.rc_dataframe_holder.dataframe.at[2, "swap_outcome_0"] = -1

    def test_too_many_midpoint_outcomes(self):
        self.rc_dataframe_holder.dataframe["midpoint_outcome_4"] = random.choices(population=[0, 1, 2, 3], k=5)

    def test_too_few_midpoint_outcomes(self):
        self.rc_dataframe_holder.dataframe = self.rc_dataframe_holder.dataframe.drop(columns=["midpoint_outcome_3"])

    def test_too_many_swap_outcomes(self):
        self.rc_dataframe_holder.dataframe["swap_outcome_3"] = random.choices(population=[0, 1, 2, 3], k=5)

    def test_too_few_swap_outcomes(self):
        self.rc_dataframe_holder.dataframe = self.rc_dataframe_holder.dataframe.drop(columns=["swap_outcome_2"])


class TestRepchainDataFrameHolderDirectlyAddingData(TestRepchainDataFrameHolder):

    def test_add_data_homogeneous(self):
        rcdfholder = create_mock_rc_dataframe_holder(number_of_results=9)
        new_data = [dict(row) for __, row in rcdfholder.dataframe.iterrows()]
        self.rc_dataframe_holder.update_dataframe_by_appending(other=new_data, ignore_index=True)
        self.check_rc_dataframe_holder(expected_number_of_results=14)


class TestRepchainDataFrameHolderCombine(TestRepchainDataFrameHolder):

    def try_combine(self, number_of_nodes_2=5, baseline_parameters_2=None, varied_parameters_2=None):
        with self.assertRaises(Exception):
            self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder(number_of_nodes=number_of_nodes_2,
                                                                         baseline_parameters=baseline_parameters_2,
                                                                         varied_parameters=varied_parameters_2)
            self.rc_dataframe_holder.combine(self.rc_dataframe_holder_2)

    def test_different_number_of_nodes(self):
        self.try_combine(number_of_nodes_2=2)
        self.try_combine(number_of_nodes_2=4)

    def test_missing_baseline(self):
        self.try_combine(baseline_parameters_2=["probability"])

    def test_too_many_baseline(self):
        self.try_combine(baseline_parameters_2=["probability", "fidelity", "gate_time"])

    def test_missing_varied(self):
        self.try_combine(varied_parameters_2=["length"])

    def test_too_many_varied(self):
        self.try_combine(varied_parameters_2=["length", "cycle_time", "gate_time"])

    def test_equal_baseline_parameters(self):
        # absorbing a rc_dataframe_holder with same baseline/varied parameters
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder(number_of_results=8)
        self.rc_dataframe_holder_2.baseline_parameters.update(self.rc_dataframe_holder.baseline_parameters)
        self.rc_dataframe_holder.combine(self.rc_dataframe_holder_2, assert_equal_baseline_parameters=True)
        self.check_rc_dataframe_holder(expected_number_of_results=13)

    def test_unequal_baseline_parameters_raise_error(self):
        # absorbing a rc_dataframe_holder with one varied parameter which is baseline for the host
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder(baseline_parameters=["probability"],
                                                                     varied_parameters=["length", "cycle_time",
                                                                                        "fidelity"],
                                                                     number_of_results=8)
        self.rc_dataframe_holder_2.baseline_parameters.update(
            {"probability": self.rc_dataframe_holder.baseline_parameters["probability"]})
        with self.assertRaises(Exception):
            self.rc_dataframe_holder.combine(self.rc_dataframe_holder_2)

    def check_combine_with_unequal_baseline_packages(self):
        """Check that merge between self.rc_dataframe_holder and self.rc_dataframe_holder_2 has the right
        behaviour when baseline packages are different."""
        # when we set `assert_equal_baseline_packages` to True, we expect an Exception to be raised
        with self.assertRaises(VersionMismatchError):
            self.rc_dataframe_holder.combine(self.rc_dataframe_holder_2, assert_equal_packages=True)
        # otherwise, we expect to only have some warnings to be printed, but no Exceptions raised
        f = io.StringIO()
        with redirect_stdout(f):
            self.rc_dataframe_holder.combine(self.rc_dataframe_holder_2, assert_equal_packages=False)
        printed_output = f.getvalue().strip().split(" ")[0]
        self.assertEqual(printed_output, 'Warning:')

    def test_unequal_baseline_packages(self):
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder()
        self.rc_dataframe_holder_2.baseline_parameters.update(self.rc_dataframe_holder.baseline_parameters)
        # change the minor to some nonsense value
        nonsense_version = "10.-1.0"
        self.rc_dataframe_holder_2.packages["netsquid"] = [nonsense_version]
        self.check_combine_with_unequal_baseline_packages()
        # check whether the baseline packages have been updated correctly
        self.assertIn(nonsense_version, self.rc_dataframe_holder.packages["netsquid"])

    def test_unequal_baseline_packages_missing_package(self):
        # add numpy package to second dataframe holder that is not used in the first dataframe holder
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder(additional_packages=["numpy"])
        self.rc_dataframe_holder_2.baseline_parameters.update(self.rc_dataframe_holder.baseline_parameters)
        self.check_combine_with_unequal_baseline_packages()
        # check whether the baseline packages have been updated correctly
        self.assertIsNotNone(self.rc_dataframe_holder.packages.get("numpy", None))
        self.assertIn("missing", self.rc_dataframe_holder.packages["numpy"])
        # check whether "missing" is not included multiple times if it's missing from a third dataframe holder
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder()
        self.rc_dataframe_holder_2.baseline_parameters.update(self.rc_dataframe_holder.baseline_parameters)
        # holder 1 numpy: ["missing", "X.X.X"], holder 2 has no numpy in baseline_packages
        self.check_combine_with_unequal_baseline_packages()
        self.assertEqual(self.rc_dataframe_holder.packages["numpy"].count("missing"), 1)

    def test_unequal_baseline_packages_partly_same_packages(self):
        # first dataframe holder contains data of two different netsquid versions
        self.rc_dataframe_holder.packages["netsquid"] = [str(1), str(2)]
        self.rc_dataframe_holder_2 = create_mock_rc_dataframe_holder()
        self.rc_dataframe_holder_2.baseline_parameters.update(self.rc_dataframe_holder.baseline_parameters)
        # second dataframe holder contains data of two different netsquid versions, of which one is shared with the
        # baseline packages of first dataframe holder
        self.rc_dataframe_holder_2.packages["netsquid"] = [str(2), str(3)]
        self.check_combine_with_unequal_baseline_packages()
        self.assertEqual(self.rc_dataframe_holder.packages["netsquid"], [str(3), str(2), str(1)])


if __name__ == "__main__":
    unittest.main()
