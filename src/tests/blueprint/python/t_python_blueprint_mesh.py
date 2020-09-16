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

import conduit.blueprint.mesh
import conduit.blueprint.mesh.examples

from conduit import Node

class Test_Blueprint_Mesh(unittest.TestCase):

    def test_basic_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mesh",n,info))
        self.assertFalse(blueprint.mesh.verify(n,info))
        blueprint.mesh.examples.basic("hexs",2,2,2,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_braid_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mesh",n,info))
        self.assertFalse(blueprint.mesh.verify(n,info))
        blueprint.mesh.examples.braid("hexs",2,2,2,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_julia_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mesh",n,info))
        self.assertFalse(blueprint.mesh.verify(n,info))
        blueprint.mesh.examples.julia(200,200,
                                      -2.0, 2.0,
                                      -2.0, 2.0,
                                      0.285, 0.01,
                                      n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_spiral_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mesh",n,info))
        self.assertFalse(blueprint.mesh.verify(n,info))
        blueprint.mesh.examples.spiral(4,n);
        self.assertTrue(blueprint.mesh.verify(n["domain_000000"],info))
        n_idx = Node()
        blueprint.mesh.generate_index(n["domain_000000"],"",4,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

if __name__ == '__main__':
    unittest.main()


