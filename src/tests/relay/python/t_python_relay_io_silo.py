# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

"""
 file: t_python_relay_io_silo.py
 description: Unit tests for the conduit relay io python module interface.

"""

import sys
import os
import unittest

from numpy import *
from conduit import Node


import conduit
import conduit.relay as relay
import conduit.blueprint as blueprint
import conduit.relay.io
import conduit.relay.io.silo


class Test_Relay_IO_Silo(unittest.TestCase):
    def test_relay_io_silo_write_read_mesh(self):
        # only run if we have silo
        if not relay.io.about()["protocols/conduit_silo_mesh"] == "enabled":
            return

        data = Node()
        blueprint.mesh.examples.julia(200,200,
                                      -2.0, 2.0,
                                      -2.0, 2.0,
                                      0.285, 0.01,
                                      data);

        data["state/cycle"] =0
        tbase = "tout_python_relay_io_silo_mesh_t1"
        tout = tbase + ".cycle_000000.root"
        if os.path.isfile(tout):
            os.remove(tout)
        print("saving to {0}".format(tout))
        relay.io.silo.write_mesh(data, tbase)
        self.assertTrue(os.path.isfile(tout))

        n_load = Node()
        info = Node()
        relay.io.silo.read_mesh(n_load, tout)

        print(n_load[0])

        # make changes for clean diff
        data["coordsets"].rename_child("coords", "mesh_topo")
        data["topologies"].rename_child("topo", "mesh_topo")
        data["topologies"]["mesh_topo"]["coordset"] = "mesh_topo"
        data["fields"].rename_child("iters", "mesh_iters")
        data["fields"]["mesh_iters"]["topology"] = "mesh_topo"
        data["state"]["domain_id"] = 0

        data.diff(n_load[0],info)
        print(info)

    def test_relay_io_silo_save_load_mesh(self):
        # only run if we have silo
        if not relay.io.about()["protocols/conduit_silo_mesh"] == "enabled":
            return

        data = Node()
        blueprint.mesh.examples.julia(200,200,
                                      -2.0, 2.0,
                                      -2.0, 2.0,
                                      0.285, 0.01,
                                      data);
        data["state/cycle"] =0
        tbase = "tout_python_relay_io_silo_mesh_t1"
        tout = tbase + ".cycle_000000.root"
        if os.path.isfile(tout):
            os.remove(tout)
        print("saving to {0}".format(tout))
        relay.io.silo.save_mesh(data, tbase)
        self.assertTrue(os.path.isfile(tout))

        n_load = Node()
        info = Node()
        n_load["HERE_IS_SOMETHING_BEFORE_WE_LOAD"] = 1
        relay.io.silo.load_mesh(n_load, tout)
        # with load sematics, the input node is cleared, so this 
        # child should no longer exist
        self.assertFalse(n_load.has_child("HERE_IS_SOMETHING_BEFORE_WE_LOAD"))
        print(n_load[0])

        # make changes for clean diff
        data["coordsets"].rename_child("coords", "mesh_topo")
        data["topologies"].rename_child("topo", "mesh_topo")
        data["topologies"]["mesh_topo"]["coordset"] = "mesh_topo"
        data["fields"].rename_child("iters", "mesh_iters")
        data["fields"]["mesh_iters"]["topology"] = "mesh_topo"
        data["state"]["domain_id"] = 0

        data.diff(n_load[0],info)
        print(info)

if __name__ == '__main__':
    unittest.main()


