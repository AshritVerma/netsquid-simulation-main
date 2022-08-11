import os
import pickle
import pandas as pd

from netsquid_simulationtools.repchain_data_process import process_data
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder

from netsquid.qubits.ketstates import b00, b10, b01, b11


def main(no_output=False):
    data_1 = {
        "state": [b00, b00, b00, b11, b11, b11],
        "generation_duration": [1, 2, 1, 4, 1, 2],
        "midpoint_outcome_0": [0, 0, 0, 0, 0, 0],
        "midpoint_outcome_1": [0, 0, 0, 0, 0, 0],
        "swap_outcome_0": [0, 0, 1, 2, 2, 3],
        "length": [1, 1, 1, 1, 1, 1]
    }
    data_2 = {
        "state": [b00, b00, b00, b00, b00, b00],
        "generation_duration": [3, 2, 1, 1, 2, 1],
        "midpoint_outcome_0": [0, 0, 0, 0, 0, 0],
        "midpoint_outcome_1": [0, 0, 0, 0, 0, 0],
        "swap_outcome_0": [0, 0, 0, 0, 0, 0],
        "length": [2, 2, 2, 2, 2, 2]
    }
    data_dir = "examples/example_teleportation_data"
    for data, name in zip([data_1, data_2], ["data1", "data2"]):
        df = pd.DataFrame(data)
        repchain_dataframe_holder = RepchainDataFrameHolder(data=df, number_of_nodes=3)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        pickle.dump(repchain_dataframe_holder, open(data_dir + f"/{name}.pickle", "wb"))

    process_data(raw_data_dir=data_dir, output=data_dir + "/combined_data.pickle",
                 csv_output_filename=data_dir + "/output.csv", process_teleportation=True,
                 plot_processed_data=not no_output)


if __name__ == "__main__":
    main(no_output=False)