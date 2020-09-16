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

class Conduit_Tutorial_Python_Basics(unittest.TestCase):

    def test_001_basics_very_basic(self):
        BEGIN_EXAMPLE("py_basics_very_basic")
        import conduit
        n = conduit.Node()
        n["my"] = "data"
        print(n)
        END_EXAMPLE("py_basics_very_basic")

    def test_002_basics_hierarchial(self):
        BEGIN_EXAMPLE("py_basics_hierarchial")
        n = conduit.Node()
        n["my"] = "data";
        n["a/b/c"] = "d";
        n["a"]["b"]["e"] = 64.0;
        print(n)
        print("total bytes: {}\n".format(n.total_strided_bytes()))
        END_EXAMPLE("py_basics_hierarchial")

    def test_003_basics_object_and_list(self):
        BEGIN_EXAMPLE("py_basics_object_and_list")
        n = conduit.Node()
        n["object_example/val1"] = "data"
        n["object_example/val2"] = 10
        n["object_example/val3"] = 3.1415

        for i in range(5):
            l_entry = n["list_example"].append()
            l_entry.set(i)
        print(n)
        END_EXAMPLE("py_basics_object_and_list")

    def test_004_basics_object_and_list_itr(self):
        BEGIN_EXAMPLE("py_basics_object_and_list_itr")
        n = conduit.Node()
        n["object_example/val1"] = "data"
        n["object_example/val2"] = 10
        n["object_example/val3"] = 3.1415

        for i in range(5):
            l_entry = n["list_example"].append()
            l_entry.set(i)
        print(n)
        
        for v in n["object_example"].children():
            print("{}: {}".format(v.name(),str(v.node())))
        print()
        for v in n["list_example"].children():
            print(v.node())
        END_EXAMPLE("py_basics_object_and_list_itr")

    def test_005_basics_mem_spaces(self):
        BEGIN_EXAMPLE("py_basics_mem_spaces")
        n = conduit.Node()
        n["my"] = "data"
        n["a/b/c"] = "d"
        n["a"]["b"]["e"] = 64.0
        print(n.info())
        END_EXAMPLE("py_basics_mem_spaces")

    def test_006_basics_bw_style(self):
        BEGIN_EXAMPLE("py_basics_bw_style")
        n = conduit.Node()
        n["test"] = numpy.uint32(100)
        print(n)
        END_EXAMPLE("py_basics_bw_style")

    def test_007_basics_bw_style_from_native(self):
        BEGIN_EXAMPLE("py_basics_bw_style_from_native")
        n = conduit.Node()
        n["test"] = 10
        print(n.schema())
        END_EXAMPLE("py_basics_bw_style_from_native")


