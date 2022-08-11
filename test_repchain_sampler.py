import unittest
import random
import numpy as np
from scipy import sparse

import netsquid.qubits.ketstates as ketstates

from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from netsquid_simulationtools.repchain_sampler import RepchainSampler


def create_mock_repchain_dataframe(number_of_nodes=2, baseline_parameters=None, varied_parameters=None,
                                   number_of_results=5, is_sparse=False):

    if varied_parameters is None:
        varied_parameters = ["length", "cycle_time"]
    if baseline_parameters is None:
        baseline_parameters = ["probability", "num_repeaters"]
    if number_of_nodes < 2:
        raise ValueError("Can only create mock results for more than 1 node.")

    bell_states = [np.outer(ketstates.b00, ketstates.b00), np.outer(ketstates.b01, ketstates.b01),
                   np.outer(ketstates.b10, ketstates.b10), np.outer(ketstates.b11, ketstates.b11)]
    if is_sparse:
        sparse_states = []
        for b in bell_states:
            sparse_states.append(sparse.csr_matrix(b))
        bell_states = sparse_states

    data = {"state": random.choices(population=bell_states, k=number_of_results),
            "basis_A": random.choices(population=["X", "Y", "Z"], k=number_of_results),
            "basis_B": random.choices(population=["X", "Y", "Z"], k=number_of_results),
            "outcome_A": random.choices(population=[0, 1], k=number_of_results),
            "outcome_B": random.choices(population=[0, 1], k=number_of_results),
            "generation_duration": random.choices(population=range(1, 100), k=number_of_results),
            }

    for i in range(number_of_nodes - 2):
        data.update({"midpoint_outcome_{}".format(i): random.choices(population=[0, 1, 2, 3], k=number_of_results),
                     "swap_outcome_{}".format(i): random.choices(population=[0, 1, 2, 3], k=number_of_results)})
    data.update({"midpoint_outcome_{}".format(number_of_nodes - 2): random.choices(population=[0, 1, 2, 3],
                                                                                   k=number_of_results)})

    for param in varied_parameters:
        param_values = []
        for i in range(number_of_results):
            param_values.append(1)
        data.update({param: param_values})

    baseline_parameters_dict = {}
    for baseline_param in baseline_parameters:
        if baseline_param == 'num_repeaters':
            baseline_parameters_dict.update({baseline_param: 0})
        else:
            baseline_parameters_dict.update({baseline_param: random.random()})
    baseline_parameters_dict.update({'generation_duration_unit': 'seconds'})

    return RepchainDataFrameHolder(number_of_nodes=number_of_nodes,
                                   baseline_parameters=baseline_parameters_dict,
                                   data=data)


quick = False
skip_reason = "Skipping probabilistic tests"


class TestRepchainSampler(unittest.TestCase):

    def setUp(self) -> None:

        self.repchain_dataframe = create_mock_repchain_dataframe()
        self.length = self.repchain_dataframe.dataframe.at[0, 'length']


class TestMockRepchainSampler(TestRepchainSampler):

    def test_asserts_pass_with_suitable_dataframe(self):
        """Test that no asserts are thrown when suitable dataframe is supplied."""
        failed = False
        try:
            _ = RepchainSampler(self.repchain_dataframe, self.length)
        except AssertionError:
            failed = True
        self.assertFalse(failed)

    def test_more_nodes_cause_asserts_to_fail(self):
        """Test that simulation data not from elementary link throws AssertionError."""
        self.repchain_dataframe = create_mock_repchain_dataframe(
            number_of_nodes=5,
            baseline_parameters=["probability", "num_repeaters"],
            number_of_results=5)
        failed = False
        try:
            _ = RepchainSampler(self.repchain_dataframe, self.length)
        except AssertionError:
            failed = True
        self.assertTrue(failed)


class TestRepchainSamplerChecks(TestRepchainSampler):

    def test_sampled_properties_present_in_tree(self):
        """Test that sampled properties are present in the original data."""
        repchain_sampler = RepchainSampler(self.repchain_dataframe, self.length)
        state, midpoint_outcome, generation_duration = repchain_sampler.sample()

        # needs to be converted to tuple again to check if the values are in the tree
        state = repchain_sampler.array_to_tuple(state.dm)
        self.assertFalse(repchain_sampler._tree.get(state) is None)
        self.assertFalse(repchain_sampler._tree.get(state)[1].get(midpoint_outcome) is None)
        self.assertFalse(repchain_sampler._tree.get(state)[1].get(midpoint_outcome)[1].get(generation_duration) is None)

    @unittest.skipIf(quick, reason=skip_reason)
    def test_sampled_properties_follow_distribution_small(self):
        """Test that from a dataframe with two states, each is equally likely to be sampled."""
        for is_sparse in [True, False]:
            self.repchain_dataframe = create_mock_repchain_dataframe(number_of_results=2, is_sparse=is_sparse)
            dataframe = self.repchain_dataframe.dataframe
            repchain_sampler = RepchainSampler(self.repchain_dataframe, self.length)

            val0 = [dataframe.at[0, 'state'], dataframe.at[0, 'midpoint_outcome_0'],
                    dataframe.at[0, 'generation_duration']]
            val1 = [dataframe.at[1, 'state'], dataframe.at[1, 'midpoint_outcome_0'],
                    dataframe.at[1, 'generation_duration']]
            counter = [0, 0]

            for val in [val0, val1]:
                if type(val[0]) == sparse.csr_matrix:
                    # Convert to dense matrix and then to array
                    val[0] = np.squeeze(np.asarray(sparse.csr_matrix.todense(val[0])))

            for i in range(1000):
                state, midpoint, round = repchain_sampler.sample()
                if state.dm.all() == val0[0].all() and midpoint == val0[1] and round == val0[2]:
                    counter[0] += 1
                if state.dm.all() == val1[0].all() and midpoint == val1[1] and round == val1[2]:
                    counter[1] += 1

            if val0[0].all() == val1[0].all() and val0[1] == val1[1] and val0[2] == val1[2]:
                self.assertTrue(counter[0] == 1000)
                self.assertTrue(counter[1] == 1000)
            else:
                self.assertEqual(1000, counter[0] + counter[1])
                # this allows for 5 % inaccuracy (50 can be "misclassified")
                self.assertAlmostEqual(counter[0], counter[1], delta=100)


if __name__ == "__main__":
    unittest.main()
