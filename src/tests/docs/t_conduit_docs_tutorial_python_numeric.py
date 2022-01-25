# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_conduit_docs_tutorial_python_examples.py
"""

import sys
import unittest
import inspect
import numpy
import conduit

def BEGIN_EXAMPLE(tag):
    print('BEGIN_EXAMPLE("' + tag + '")')

def END_EXAMPLE(tag):
    print('END_EXAMPLE("' + tag + '")')

class Conduit_Tutorial_Python_Numeric(unittest.TestCase):
    
    def test_001_numeric_via_value(self):
        BEGIN_EXAMPLE("py_numeric_via_value")
        n = conduit.Node()
        n["test"] = 10 
        print(n.fetch("test").value())
        END_EXAMPLE("py_numeric_via_value")
