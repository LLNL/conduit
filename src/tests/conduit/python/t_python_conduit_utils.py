# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_conduit_node.py
 description: Unit tests for conduit::Node python module interface.

"""

import sys
import unittest

import conduit
import conduit.utils


from conduit import Node

import numpy as np

class Test_Conduit_Utils(unittest.TestCase):
    def test_args(self):
        args = Node()
        # named args test
        args["a"] = "something about";
        args["b"] = "and";
        args["c"] = 3.1415;
        args["d"] = 42;

        t = conduit.utils.format("{a} {c:0.3} {b} {d:04}",
                                 args = args)
        print(t)
        self.assertTrue(t == "something about 3.14 and 0042")
        
        # ordered args test
        args.reset()
        args.append().set("something about")
        args.append().set(3.1415)
        args.append().set("and")
        args.append().set(42)

        t = conduit.utils.format("{} {:0.3} {} {:04}", args);
        print(t)
        self.assertTrue(t == "something about 3.14 and 0042")

    def test_maps(self):
        maps = Node()
        
        # named args test
        maps["a"] = np.array([1.1415, 2.1415, 3.1415])
        maps["b"] = np.array([0,0,42])

        t = conduit.utils.format("something about {a:0.3} and {b:04}",
                                 maps=maps, map_index =2)
        print(t)
        self.assertTrue(t == "something about 3.14 and 0042")

        # ordered args test
        maps.reset()
        maps.append().set(np.array([1.1415, 2.1415, 3.1415]))
        maps.append().set(np.array([0,0,42]));

        t = conduit.utils.format("something about {:0.3} and {:04}",
                                 maps=maps,map_index =2)
        print(t)
        self.assertTrue(t == "something about 3.14 and 0042")

        #string list case
        maps.reset()
        slist = maps.append()
        slist.append().set("hi 0")
        slist.append().set("hi 1")
        slist.append().set("hi 2")
        
        t = conduit.utils.format("{} is it",maps=maps,map_index =0)
        print(t)
        self.assertTrue(t == "hi 0 is it")

        t = conduit.utils.format("{} is it",maps=maps,map_index =1)
        print(t)
        self.assertTrue(t == "hi 1 is it")

        t = conduit.utils.format("{} is it",maps=maps,map_index =2)
        print(t)
        self.assertTrue(t == "hi 2 is it")


    def test_improper_calls(self):
        args = Node()
        maps = Node()

        # check for error with trying to call with no args or map
        with self.assertRaises(Exception):
            conduit.utils.format(pattern= "")

        # check for error with trying to call both styles
        with self.assertRaises(Exception):
            conduit.utils.format(pattern= "",
                                 args = args,
                                 maps = maps,
                                 map_index = 0)
    
    def test_bad_args(self):
        args = Node()
        # args empty, throw exception
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 args = args)
        # args is not a node
        args = 32
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 args = args)

        # args not an obj or list, throw exception
        args = Node()
        args.set(32)
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 args = args)

        # too little args for fmt string
        args.reset()
        args["a"] = 10;
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                  args = args)

    def test_bad_maps(self):
        maps = Node()
        # args empty, throw exception
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 maps = maps)
        # maps is not a node
        maps = 32
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 maps = maps)

        # maps not an obj or list, throw exception
        maps = Node()
        maps.set(32)
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                 maps = maps)
        # too little args for fmt string
        maps.reset()
        vec = np.array(range(100), np.uint32)
        maps["a"].set(vec);
        with self.assertRaises(Exception):
            conduit.utils.format("{a} {b} {c} {d} {e} {f}",
                                  args = args)

        # bad index, (negative)
        maps.reset()
        vec = np.array(range(10), np.uint32)
        maps["a"].set(vec);
        with self.assertRaises(Exception):
            conduit.utils.format("{a}",
                                  maps = maps,
                                  map_index = -100)

        # bad index, (negative)
        with self.assertRaises(Exception):
            conduit.utils.format("{a}",
                                  maps = maps,
                                  map_index = -1)

        # bad index, (too big)
        with self.assertRaises(Exception):
            conduit.utils.format("{a}",
                                  maps = maps,
                                  map_index = 10)
        # bad index, (too big)
        with self.assertRaises(Exception):
            conduit.utils.format("{a}",
                                  maps = maps,
                                  map_index = 100)

        # bad arg (entry is object)
        maps.reset()
        maps["a"].append()["im/an/object"].set(32);
        with self.assertRaises(Exception):
            conduit.utils.format("{a}",
                                 maps = maps,
                                 map_index = 0)





if __name__ == '__main__':
    unittest.main()


