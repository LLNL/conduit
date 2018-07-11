###############################################################################
# Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
 file: python_relay_io.py
 description: Unit tests for the conduit relay io python module interface.

"""


import sys
import os
import unittest

from numpy import *
from conduit import Node
from conduit import DataType

import conduit
import conduit.relay as relay
import conduit.relay.io

class Test_Relay_IO(unittest.TestCase):
    def test_load_save(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)
        #
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['b'] == b_val)
        self.assertTrue(n['c'] == c_val)
        relay.io.save(n,"tout_python_relay_io_save_load.conduit_bin")
        #  now load the value
        n_load = Node()
        relay.io.load(n_load,"tout_python_relay_io_save_load.conduit_bin")
        self.assertTrue(n_load['a'] == a_val)
        self.assertTrue(n_load['b'] == b_val)
        self.assertTrue(n_load['c'] == c_val)

    def test_load_save_protocols(self):
        a_val = int64(10)
        b_val = int64(20)
        c_val = float64(30.0)
        #
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['b'] == b_val)
        self.assertTrue(n['c'] == c_val)

        protos = ["conduit_bin",
                  "json",
                  "conduit_json",
                  "conduit_base64_json"]
        
        # only test hdf5 if relay was built with hdf5 support
        if relay.about()["io/protocols/hdf5"] == "enabled":
            protos.append("hdf5")
        
        for proto in protos:
            print("testing protocol: ", proto)
            ftest = "tout_python_relay_io_save_load_proto." + proto
            relay.io.save(n,ftest)
            #  now load the value
            n_load = Node()
            relay.io.load(n_load,ftest)
            self.assertTrue(n_load['a'] == a_val)
            self.assertTrue(n_load['b'] == b_val)
            self.assertTrue(n_load['c'] == c_val)
        # only test silo if relay was built with hdf5 support
        if relay.about()["io/protocols/conduit_silo"] == "enabled":
            # silo needs a subpath
            print("testing protocol: silo")
            ftest = "tout_python_relay_io_save_load_proto.silo:obj"
            relay.io.save(n,ftest)
            #  now load the value
            n_load = Node()
            relay.io.load(n_load,ftest)
            self.assertTrue(n_load['a'] == a_val)
            self.assertTrue(n_load['b'] == b_val)
            self.assertTrue(n_load['c'] == c_val)

    def test_load_merged(self):
       a_val = int64(10)
       b_val = int64(20)
       c_val = float64(30.0)
       d_val = a_val * 4
       #
       n = Node()
       n['a'] = a_val
       n['b'] = b_val
       n['c'] = c_val
       tout = "tout_python_relay_io_save_merged.conduit_bin";
       
       relay.io.save_merged(n,tout);
       self.assertTrue(os.path.isfile(tout))
       n_load = Node();
       n_load['d'] = d_val
       relay.io.load_merged(n_load,tout);
       print(n_load)
       self.assertTrue(n_load.has_child("d"))
       self.assertTrue(n_load.has_child("a"))
       self.assertTrue(n_load['a'] == a_val)
       self.assertTrue(n_load['d'] == d_val)
   
    def test_save_merged(self):
       a_val = int64(10)
       b_val = int64(20)
       c_val = float64(30.0)
       d_val = a_val * 4
       #
       n = Node()
       n['a'] = a_val
       n['b'] = b_val
       n['c'] = c_val
       tout = "tout_python_relay_io_save_merged.conduit_bin";
       
       relay.io.save_merged(n,tout);
       self.assertTrue(os.path.isfile(tout))
       n2 = Node();
       n2['d'] = d_val
       relay.io.save_merged(n2,tout);
       
       self.assertTrue(os.path.isfile(tout))
       n_load = Node()
       relay.io.load(n_load,tout)
       print(n_load)
       self.assertTrue(n_load.has_child("d"))
       self.assertTrue(n_load.has_child("a"))
       self.assertTrue(n_load['a'] == a_val)
       self.assertTrue(n_load['d'] == d_val)
   
    def test_hdf5_generic_save_opts(self):
        # only run if we have hdf5
        if not relay.about()["io/protocols/hdf5"] == "enabled":
            return
        # 5k float64 zeros, will compress well, but below default 
        # thresh
        n = Node()
        opts = Node()
        opts["hdf5/chunking/threshold"]  = 2000
        opts["hdf5/chunking/chunk_size"] = 2000
        # hdf5 requires object at top level ... 
        n['value'].set(DataType.float64(5000))
        tout_std = "tout_python_relay_io_hdf5_generic_std.hdf5"
        tout_cmp = "tout_python_relay_io_hdf5_generic_cmp.hdf5"
        relay.io.save(n,tout_std);
        relay.io.save(n,tout_cmp,options=opts);
        tout_std_fs  = os.path.getsize(tout_std);
        tout_std_cmp = os.path.getsize(tout_cmp);
        self.assertTrue(os.path.isfile(tout_std))
        self.assertTrue(os.path.isfile(tout_cmp))
        print("fs compare: std = ", tout_std_fs, " cmp = ", tout_std_cmp)
        self.assertTrue(tout_std_cmp < tout_std_fs)


if __name__ == '__main__':
    unittest.main()


