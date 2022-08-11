import os
import pickle
import pandas as pd

from netsquid_simulationtools.repchain_data_process import process_data
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder


def main(no_output=False):
    data_1 = {"basis_A": ["X", "X", "Z", "Z", "Z", "Z"],
              "basis_B": ["X", "X", "Z", "Z", "Z", "Z"],
              "outcome_A": [0, 1, 0, 1, 0, 1],
              "outcome_B": [1, 0, 1, 0, 1, 0],
              "generation_duration": [1, 2, 1, 4, 1, 2],
              "midpoint_outcome_0": [0, 1, 2, 3, 0, 2],
              "midpoint_outcome_1": [3, 2, 1, 0, 3, 1],
              "swap_outcome_0": [0, 0, 0, 0, 0, 0],
              "length": [1, 1, 1, 1, 1, 1]
              }
    data_2 = {"basis_A": ["X", "X", "X", "Z", "Z", "Z"],
              "basis_B": ["X", "X", "X", "Z", "Z", "Z"],
              "outcome_A": [0, 1, 1, 0, 1, 0],
              "outcome_B": [0, 1, 1, 0, 1, 0],
              "generation_duration": [3, 2, 1, 1, 2, 1],
              "midpoint_outcome_0": [0, 0, 0, 0, 0, 0],
              "midpoint_outcome_1": [0, 0, 0, 0, 0, 0],
              "swap_outcome_0": [0, 0, 0, 0, 0, 0],
              "length": [2, 2, 2, 2, 2, 2]
              }
    data_dir = "examples/example_bb84_data"
    for data, name in zip([data_1, data_2], ["data1", "data2"]):
        df = pd.DataFrame(data)
        repchain_dataframe_holder = RepchainDataFrameHolder(data=df, number_of_nodes=3)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        pickle.dump(repchain_dataframe_holder, open(data_dir + f"/{name}.pickle", "wb"))

    process_data(raw_data_dir=data_dir, output=data_dir + "/combined_data.pickle",
                 csv_output_filename=data_dir + "/output.csv", process_bb84=True,
                 plot_processed_data=not no_output)


if __name__ == "__main__":
    main(no_output=False)
