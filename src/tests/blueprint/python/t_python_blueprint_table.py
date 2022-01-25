# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_blueprint_table.py
 description: Simple unit test for the conduit blueprint table python module.

"""

import sys
import unittest

import conduit.blueprint as blueprint

from conduit import Node

class Test_Blueprint_Table(unittest.TestCase):
    def test_basic_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.table.verify(n, info))
        blueprint.table.examples.basic(5, 4, 3, n)
        self.assertTrue(blueprint.verify("table", n, info))
        self.assertTrue(blueprint.table.verify(n, info))

        # No sub protocols
        self.assertFalse(blueprint.verify("table/subproto", n, info))
        self.assertFalse(blueprint.table.verify(n, info, "subproto"))

    def test_functions_exist(self):
        table_funcs = dir(blueprint.table)
        self.assertGreater(table_funcs.count("verify"), 0)

        examples_funcs = dir(blueprint.table.examples)
        self.assertGreater(examples_funcs.count("basic"), 0)

if __name__ == '__main__':
    unittest.main()
