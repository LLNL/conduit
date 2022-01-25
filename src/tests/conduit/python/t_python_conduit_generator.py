# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: python_conduit_generator.py
 description: Unit tests for conduit::Generator python module interface.

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

from conduit import Node
from conduit import Schema
from conduit import Generator

from numpy import *

def default_node():
    a_val = int64(10)
    b_val = int64(20)
    c_val = float64(30.0)

    n = Node()
    n['a'] = a_val
    n['b'] = b_val
    n['c'] = c_val
    return n;

class Test_Conduit_Generator(unittest.TestCase):
    def test_simple(self):
        n = default_node()
        n_schema = n.to_json("conduit_json");
        print("result detailed json", n_schema)
        g = Generator(schema=n_schema);
        ng = Node();
        sg = Schema()
        g.walk(node=ng);
        g.walk(schema=sg);
        print(ng)
        print(sg)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)

    def test_json(self):
        n = default_node()
        n_schema = n.to_json("json");
        print("result json", n_schema)
        g = Generator(schema=n_schema,protocol="yaml");
        ng = Node();
        g.walk(node=ng);
        print(ng)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)

    def test_yaml(self):
        n = default_node()
        n_schema = n.to_yaml();
        print("result yaml", n_schema)
        g = Generator(schema=n_schema,protocol="yaml");
        ng = Node();
        g.walk(node=ng);
        print(ng)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)

    def test_base64(self):
        n = default_node()
        print(n)
        n_schema = n.to_json("conduit_base64_json");
        print("result base64 json", n_schema)
        g = Generator(n_schema,"conduit_base64_json");
        ng = Node();
        g.walk(node=ng);
        print("Generator result")
        print(ng)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)


if __name__ == '__main__':
    unittest.main()


