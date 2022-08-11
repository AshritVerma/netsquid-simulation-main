CHANGELOG
=========

2021-05-20 (2.0.0)
------------------
- Refactoring of `repchain_data_process.py`. To process data, you should now call `process_data()`, which has multiple options for what kind of processing should be performed. The new modular setup makes it easy to implement new processing functions.
- Implemented quantum-teleportation-based data processing. If data contains state information, `process_data()` with `process_teleportation=True` can be used to obtain estimates of how well quantum teleportation can be performed using the generated entanglement. Added files `process_teleportation.py` and `linear_algebra.py` with functions required to implement the teleportation-based processing.
- Migrated BB84-proccessing code from `repchain_data_functions.py` to new file `process_qkd.py`.
- Removed creation of `csv_output.csv` file, which was used by Stopos, when processing data.
- When calling `repchain_data_plot.py`, you can now use the `--mode`/`-m` argument to choose a plotting type (currently teleportation or BB84, with teleportation as default).
- States in `RepchainDataFrameHolder` can now also be ket vectors instead of density matrices.

2021-03-19 (1.0.0)
------------------
- Updated to public release of `nestquid==1.0.0`.
- Replaced `number_of_rounds` in `RepchainDataFrameHolder` by `generation_duration` and `generation_duration_unit`, to allow users to define their own time scales. Old dataframe holders can be converted to the new format using `repchain_data_process.convert_rdfh_with_number_of_rounds`.
- Renamed `secret_key_rate_from_outcomes` in `repchain_data_functions.py` to `estimate_bb84_secret_key_rate_from_data`, added better documentation with references to literature, and included `sifting_factor` as optional parameter.
- Removed `normalized_skr` (etc.) from `repchain_data_process.py` to remove hardware dependency due to normalization.
- Renamed `sk_capacity` to `plob_bound` to avoid confusion.
- Changed degree of rate fit in `repchain_data_plot.py` to correctly reproduce behavior.
- Extended `repchain_data_plot.py` to allow multiple sets of QKD to be plotted with QBER and attempts.
- Added argparsers to `repchain_data_process.py`, `repchain_data_combine.py` and `repchain_data_plot.py`.
- Updated readme with small explanation of usage and list of contributors.
- Fixed units for secret key rate and its error in `repchain_data_functions.py`.
- Fixed automatically added parameters picked up as varied parameters in `repchain_data_process.py`.
- Updated docker image to use python 3.8.


2020-09-18 (0.4.1)
------------------

- `number_of_rounds` in `RepChainDataframeHolder` can now be either float or int (instead of just int).


2020-08-31 (0.4.0)
------------------

- Using NetSquid's BellIndex for swap/midpoint outcomes.
- Fixed bug in data type/value checking of RepChainDataFrameHolder (it never raised an exception).
- `repchain_data_plot.py` contains standardized methods to plot QKD data and fidelity against varied parameter.
- Method for processing fidelity data that can then be used for plotting is added.

2020-07-28 (0.3.0)
------------------

- version info of used packages is now available in RepChainDataFrameHolder

2020-07-22 (0.2.0)
------------------

- cutoff time can now be retroactively implemented on results of single-repeater simulations

2020-06-08 (0.1.0)
------------------

- attenuation coefficient can now be specified when calculating secret-key rate capacity
- error bars on QBER can now be scaled depending on desired confidence interval
- secret key rate now has min/max and symmetric error bars
- added several utility function to plotting (e.g. fitting)


2020-01-28 (0.0.5)
------------------

- added _NOT_SPECIFIED to ParameterSet


2020-01-27 (0.0.4)
------------------

- hot-fix to repchain_sampler, fixing comparison of large states failing


2020-01-22 (0.0.3)
------------------

- added repchain_sampler


2019-12-25 (0.0.2)
------------------

- added function to calculate end-to-end fidelity 
- small fix to parameter set class to add exception for "infinite integers"


2019-11-29 (0.0.1)
------------------

- added repchain_dataframe_holder and associated functions


2019-11-29 (0.0.0)
------------------

- Created this snippet
