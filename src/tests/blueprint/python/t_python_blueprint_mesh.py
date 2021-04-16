# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_blueprint_mesh.py
 description: Simple unit test for the conduit blueprint mesh python module.

"""

import sys
import unittest

import conduit.blueprint as blueprint

import conduit.blueprint.mesh
import conduit.blueprint.mesh.examples

from conduit import Node

class Test_Blueprint_Mesh(unittest.TestCase):

    def has_empty_warning(self, info):
        res = False

        if info.dtype().is_object() and info.has_child("info"):
            iinfo = info.fetch('info')
            if iinfo.dtype().is_object() or iinfo.dtype().is_list():
                iitr = iinfo.children()
                for ival in iitr:
                    inode = ival.node()
                    res = res or (inode.dtype().is_string() and "is an empty mesh" in inode.to_string())

        return res

    def test_basic_and_verify(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        blueprint.mesh.examples.basic("hexs",2,2,2,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_braid_and_verify(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        blueprint.mesh.examples.braid("hexs",2,2,2,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_julia_and_verify(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        blueprint.mesh.examples.julia(200,200,
                                      -2.0, 2.0,
                                      -2.0, 2.0,
                                      0.285, 0.01,
                                      n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_spiral_and_verify(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        blueprint.mesh.examples.spiral(4,n);
        self.assertTrue(blueprint.mesh.verify(n["domain_000000"],info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n["domain_000000"],"",4,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_julia_nestsets(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        # simple case
        blueprint.mesh.examples.julia_nestsets_simple(-2.0, 2.0,
                                                      -2.0, 2.0,
                                                      0.285, 0.01,
                                                      n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n["domain_000000"],"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))
        # complex case
        blueprint.mesh.examples.julia_nestsets_complex(50,50,
                                                       -2.0, 2.0,
                                                       -2.0, 2.0,
                                                       0.285, 0.01,
                                                       3,
                                                       n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        n_idx = Node()
        blueprint.mesh.generate_index(n["domain_000000"],"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))

    def test_venn(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        for matset_type in ['full', 
                            'sparse_by_material',
                            'sparse_by_element' ]:
            blueprint.mesh.examples.venn(matset_type,
                                         10,10,
                                         .5,
                                         n);
            self.assertTrue(blueprint.mesh.verify(n,info))
            self.assertFalse(self.has_empty_warning(info))
            n_idx = Node()
            blueprint.mesh.generate_index(n,"",1,n_idx)
            self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
            print(info)
            self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))
            print(info)

    def test_polytess(self):
        n = Node()
        info = Node()
        self.assertTrue(blueprint.verify("mesh",n,info))
        self.assertTrue(self.has_empty_warning(info))
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertTrue(self.has_empty_warning(info))
        # simple case
        blueprint.mesh.examples.polytess(3,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        self.assertFalse(self.has_empty_warning(info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))



if __name__ == '__main__':
    unittest.main()


