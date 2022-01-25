# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_conduit_docs_tutorial_python_examples.py
"""

##############################################################
# make sure windows can pick up dlls for python 3.8 and newer
import os
if "PATH" in os.environ:
    for dll_path in os.environ["PATH"].split(";"):
        os.add_dll_directory(dll_path)
##############################################################

import sys
import unittest
import inspect
import numpy
import conduit

def BEGIN_EXAMPLE(tag):
    print('BEGIN_EXAMPLE("' + tag + '")')

def END_EXAMPLE(tag):
    print('END_EXAMPLE("' + tag + '")')

class Conduit_Tutorial_Python_Ownership(unittest.TestCase):
    def test_001_mem_ownership_external(self):
        BEGIN_EXAMPLE("py_mem_ownership_external")
        vals = numpy.zeros((5,),dtype=numpy.float64)
        n = conduit.Node()
        n["v_owned"].set(vals)
        n["v_external"].set_external(vals)

        print(n.info())
        print(n)
        vals[0] = 3.1415
        print(n)
        print(vals)
        END_EXAMPLE("py_mem_ownership_external")

