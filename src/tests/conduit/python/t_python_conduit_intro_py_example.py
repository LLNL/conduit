# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_conduit_intro_py_example.py
 description: Example usage in python

"""

import sys
import unittest

from conduit import Node
from conduit import DataType

from numpy import *


class Test_Conduit_Intro_Talk_Ex(unittest.TestCase):
    def test_intro_py_example(self):
        import conduit
        import numpy as np
        
        den = np.ones(4,dtype="float64")
        
        n = conduit.Node()
        n["fields/density/values"] = den
        n["fields/density/units"] = "g/cc"
        
        n_density = n.fetch("fields/density")
        
        den_vals  = n_density["values"]
        den_units = n_density["units"]
        
        print(n_density)
        
        print("\nDensity ({0}):\n{1})".format(den_units, den_vals))


if __name__ == '__main__':
    unittest.main()


