"""
Functions for handling data at the generative system level.

Author: Alexander TJ Barron
Date Created: 2014-09-24

"""

import numpy as np
import pdb

def load_data(path, transpose=False):
    """
    Load data to a numpy array and reshape it for the generative system.  The
    data should not have headers and should be whitespace delimited.

    Args:
      path (str): path to the data file

    Returns:
      np.array: n by t array, where t_i indexes the support and n_i indexes
        the variable

    """

    a_data = np.loadtxt(path, dtype=int, ndmin=2)

    if transpose:
        return a_data.transpose()
    else:
        return a_data
