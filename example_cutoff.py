from netsquid_simulationtools.repchain_data_cutoff import implement_cutoff_duration
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder


def main(no_output=False):

    data = {"outcome_A": [0, 1, 0], "basis_A": ["Z", "Z", "X"],
            "outcome_B": [0, 1, 0], "basis_B": ["Z", "Z", "X"],
            "generation_duration": [100, 100, 100],
            "midpoint_outcome_0": [0, 0, 0], "midpoint_outcome_1": [0, 0, 0], "swap_outcome_0": [0, 0, 0],
            "duration_between_alice_and_bob": [10, 30, 15]}

    repchain_dataframe_holder = RepchainDataFrameHolder(data, generation_duration_unit="rounds")
    repchain_dataframe_holder_with_cutoff = implement_cutoff_duration(repchain_dataframe_holder, 20)

    if not no_output:
        print(repchain_dataframe_holder_with_cutoff)
        print(repchain_dataframe_holder_with_cutoff.baseline_parameters)

    # output:
    #
    #    outcome_A basis_A  outcome_B basis_B  generation_duration  midpoint_outcome_0  midpoint_outcome_1
    # 0          0       Z          0       Z                100.0                   0                   0
    # 2          0       X          0       X                190.0                   0                   0
    #    swap_outcome_0   duration_between_alice_and_bob
    # 0          0           10
    # 2          0           15
    #
    # {'cutoff_round': 20, 'generation_duration_unit': 'rounds'}

    # If there were a cutoff of 20 rounds, during what is the second entanglement distribution (row)
    # in the original data, entanglement would have been discarded 10 rounds before entanglement was successfully
    # swapped.
    # At this point, the setup would have to start all over again.
    # To represent this in the data when implementing a cutoff time, it is assumed that at this point we are at the
    # beginning of what in in the original data the third entanglement distribution.
    # In the data with cutoff time, there are now only two entanglement distributions,
    # the second of which is basically the third of the original data,
    # but it also includes rounds that were "wasted" by discarding entanglement.


if __name__ == "__main__":
    main()
