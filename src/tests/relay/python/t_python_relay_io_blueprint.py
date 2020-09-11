###############################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################
"""
 file: t_python_relay_io_blueprint.py
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
import conduit.relay.io.blueprint


class Test_Relay_IO_Blueprint(unittest.TestCase):
    def test_relay_io_blueprint_save_load_mesh(self):
        #self.assertTrue(False)
        print(os.getcwd())
        # only run if we have hdf5
        if not relay.io.about()["protocols/hdf5"] == "enabled":
            return
        data = Node()
        blueprint.mesh.examples.julia(200,200,
                                      -2.0, 2.0,
                                      -2.0, 2.0,
                                      0.285, 0.01,
                                      data);
        data["state/cycle"] =0
        tbase = "tout_python_relay_io_blueprint_mesh_t1"
        tout = tbase + ".cycle_000000.root"
        if os.path.isfile(tout):
            os.remove(tout)
        print("saving to {0}".format(tout))
        relay.io.blueprint.save_mesh(data, tbase, "hdf5")
        self.assertTrue(os.path.isfile(tout))

        n_load = Node()
        info = Node()
        relay.io.blueprint.load_mesh(n_load, tout)
        print(n_load)
        data.diff(n_load,info)
        print(info)

if __name__ == '__main__':
    unittest.main()


