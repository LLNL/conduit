# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_conduit_smoke.py
 description: Simple unit test for the basic conduit python module interface.

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

import conduit

class Test_Condut_Basic(unittest.TestCase):
    def test_about(self):
        print(conduit.about())

if __name__ == '__main__':
    unittest.main()


