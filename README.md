# FortyMHz

# Analysis

The script ```analysis.py``` reads a datafile (binary or txt) written to disk by the FortyMHz DAQ and performs a basic analysis on the data taken. The output consists of plots showing the basic DT chamber quantities (timebox, position, ...). If the input is a binary file, the data is unpacked on-the-fly before veing loaded into the pandas dataframe. The analysis script, if a BX assignment is also provided, performs all operations column-wise without any loop on the events. In the future, data visualization and fitting will be also implemented.

# Generation

The script ```generation.py``` performs a random generation of muons in each superlayer, and fills a dataframe with the same structure as the data in output from the DAQ. The dataframe can be saved in many formats, including ```.csv```. If requested, plots can be also produced to show the basic kinematic distributions and the quantities filled in the dataframe.