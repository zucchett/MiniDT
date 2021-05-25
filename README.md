# FortyMHz

### Requirements

The scripts run with python3; previous python versions also work but require minimal changes to the printouts, but form instance cannot use numba for performance improvements.

The following packages are also required: `pandas` `numpy` `scipy` `matplotlib` `seaborn` `mplhep` `bokeh`. The packege `numba` is used for testing.

They can be installed with `pip` using the command:

`python3 -m pip install package1 package2 ...`


# Analysis

The script ```analysis.py``` reads a datafile (binary or txt) written to disk by the FortyMHz DAQ and performs a basic analysis on the data taken. The output consists of plots showing the basic DT chamber quantities (timebox, position, ...). If the input is a binary file, the data is unpacked on-the-fly before being loaded into the pandas dataframe. The script is able to use the trigger BX assignment, if present (faster solution, does not require loops); otherwise, the meantimer equations are applyed layer-wise to determine the crossing time.

The full list of options is available by typing `python3 analysis.py --help`.

# Generation

The script ```generation.py``` performs a random generation of muons in each superlayer, and fills a dataframe with the same structure as the data in output from the DAQ. The dataframe can be saved in many formats, including ```.csv```. If requested, plots can be also produced to show the basic kinematic distributions and the quantities filled in the dataframe.