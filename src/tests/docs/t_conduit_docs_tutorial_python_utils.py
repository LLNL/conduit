# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_conduit_docs_tutorial_python_utils.py
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

class Conduit_Tutorial_Python_Utils(unittest.TestCase):
    def test_001_using_utils_fmt_args_obj(self):
        BEGIN_EXAMPLE("py_using_utils_fmt_args_obj")
        import conduit
        import conduit.utils

        args = conduit.Node()
        args["answer"] = 42

        print(conduit.utils.format("The answer is {answer:04}.", args = args))

        args.reset()
        args["adjective"] =  "other";
        args["answer"] = 3.1415;

        print(conduit.utils.format("The {adjective} answer is {answer:0.4f}.",
                                   args = args))
        END_EXAMPLE("py_using_utils_fmt_args_obj")

    def test_002_using_utils_fmt_args_list(self):
        BEGIN_EXAMPLE("py_using_utils_fmt_args_list")
        import conduit
        import conduit.utils

        args = conduit.Node()
        args.append().set(42)

        print(conduit.utils.format("The answer is {:04}.",args = args))

        args.reset()
        args.append().set("other")
        args.append().set(3.1415)

        print(conduit.utils.format("The {} answer is {:0.4f}.", args =args))
        END_EXAMPLE("py_using_utils_fmt_args_list")

    def test_003_using_utils_fmt_maps_obj(self):
        BEGIN_EXAMPLE("py_using_utils_fmt_maps_obj")
        import conduit
        import conduit.utils
        import numpy as np

        maps = conduit.Node()
        maps["answer"].set(np.array([42.0, 3.1415]))

        print(conduit.utils.format("The answer is {answer:04}.",
                                    maps = maps, map_index = 0))

        print(conduit.utils.format("The answer is {answer:04}.",
                                    maps = maps, map_index = 1))
        print()

        maps.reset()
        maps["answer"].set(np.array([42.0, 3.1415]));
        slist = maps["position"];
        slist.append().set("first")
        slist.append().set("second")

        print(conduit.utils.format("The {position} answer is {answer:0.4f}.",
                                    maps = maps, map_index = 0))

        print(conduit.utils.format("The {position} answer is {answer:0.4f}.",
                                    maps = maps, map_index = 1))
        END_EXAMPLE("py_using_utils_fmt_maps_obj")

    def test_004_using_utils_fmt_maps_list(self):
        BEGIN_EXAMPLE("py_using_utils_fmt_maps_list")
        import conduit
        import conduit.utils
        import numpy as np

        maps = conduit.Node()
        vals = np.array([42.0, 3.1415])
        maps.append().set(vals)

        print(conduit.utils.format("The answer is {}.",
                                   maps = maps, map_index = 0))

        print(conduit.utils.format("The answer is {}.",
                                   maps = maps, map_index = 1))
        print()
        
        maps.reset()
        # first arg
        slist = maps.append();
        slist.append().set("first")
        slist.append().set("second")

        # second arg
        maps.append().set(vals)

        print(conduit.utils.format("The {} answer is {:0.4f}.",
                                   maps = maps, map_index = 0))

        print(conduit.utils.format("The {} answer is {:0.4f}.",
                                   maps = maps, map_index = 1))
        END_EXAMPLE("py_using_utils_fmt_maps_list")
