# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_conduit_docs_tutorial_python_utils.py
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

class Conduit_Tutorial_Python_Move_and_Swap(unittest.TestCase):
    def test_001_move(self):
        BEGIN_EXAMPLE("py_move")
        import conduit

        n_a = conduit.Node()
        n_b = conduit.Node()
        info = conduit.Node()
        
        n_b["path/to/data"] = 42
        
        print("- Before Move -")
        print("n_a contents:")
        print(n_a)
        print("n_a memory details:")
        n_a.info(info)
        print(info)

        print("n_b contents:")
        print(n_b)
        print("n_b memory details:")
        n_b.info(info)
        print(info)

        n_a.move(n_b)

        print("- After Move -")
        print("n_a contents:")
        print(n_a)
        print("n_a memory details:")
        n_a.info(info)
        print(info)

        print("n_b contents:")
        print(n_b)
        print("n_b memory details:")
        n_b.info(info)
        print(info)

        END_EXAMPLE("py_move")

    def test_002_swap(self):
        BEGIN_EXAMPLE("py_swap")
        import conduit

        n_a = conduit.Node()
        n_b = conduit.Node()
        info = conduit.Node()
        
        n_a["data"] = 10
        n_b["path/to/data"] = 20
        
        print("- Before Swap -")
        print("n_a contents:")
        print(n_a)
        print("n_a memory details:")
        n_a.info(info)
        print(info)

        print("n_b contents:")
        print(n_b)
        print("n_b memory details:")
        n_b.info(info)
        print(info)

        n_a.move(n_b)

        print("- After Swap -")
        print("n_a contents:")
        print(n_a)
        print("n_a memory details:")
        n_a.info(info)
        print(info)

        print("n_b contents:")
        print(n_b)
        print("n_b memory details:")
        n_b.info(info)
        print(info)

        END_EXAMPLE("py_swap")
