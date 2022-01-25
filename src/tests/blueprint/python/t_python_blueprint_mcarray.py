# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_blueprint_mcarray.py
 description: Simple unit test for the conduit blueprint mcarray python module.

"""

import sys
import unittest

import conduit.blueprint as blueprint

import conduit.blueprint.mcarray
import conduit.blueprint.mcarray.examples

from conduit import Node

class Test_Blueprint_MCArray(unittest.TestCase):
    def test_xyz_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mcarray",n,info))
        blueprint.mcarray.examples.xyz("separate",5,n);
        self.assertTrue(blueprint.verify("mcarray",n,info))
        self.assertTrue(blueprint.mcarray.verify(n,info))
        n_int = Node()
        blueprint.mcarray.to_interleaved(n,n_int)
        self.assertTrue(blueprint.mcarray.is_interleaved(n_int))

if __name__ == '__main__':
    unittest.main()


