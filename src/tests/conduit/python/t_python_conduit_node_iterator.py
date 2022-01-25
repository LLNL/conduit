# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: python_conduit_node_iterator.py
 description: Unit tests for conduit::NodeIterator python module interface.

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
from conduit import NodeIterator


from numpy import *


class Test_Conduit_Node_Iterator(unittest.TestCase):
    def test_simple(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)

        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
  
        itr = NodeIterator()
        self.assertFalse(itr.has_next())
        itr = n.children()
        self.assertTrue(itr.has_next())
        print(itr.has_next());
        for v in itr:
            print(v.name(), v.node())
            idx = v.index()
            if idx == 0:
                self.assertEqual(v.node().value(),a_val)
            elif idx == 1:
                self.assertEqual(v.node().value(),b_val)
            elif idx == 2:
                self.assertEqual(v.node().value(),c_val)

#
# TODO: sensible itr use cases:
# for v in itr:
# ?
# for k,v in itr.items():
# for i,v in itr.children():
#

if __name__ == '__main__':
    unittest.main()


