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
 file: t_python_relay_io.py
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

class Test_Relay_IO_Handle(unittest.TestCase):
    def test_io_handle(self):
        tbase = "tout_python_relay_io_handle."
        protos = ["conduit_bin",
                  "json",
                  "conduit_json",
                  "conduit_base64_json"]
        
        # only test hdf5 if relay was built with hdf5 support
        if relay.io.about()["protocols/hdf5"] == "enabled":
            protos.append("hdf5")
        for proto in protos:
            test_file = tbase + proto
            
            n = conduit.Node();
            n["a"] = int64(20);
            n["b"] = int64(8);
            n["c"] = int64(12);
            n["d/here"] = int64(10);

            h = conduit.relay.io.IOHandle();
            h.open(test_file)
            h.write(n)
            self.assertTrue(h.has_path("d/here"))
            cnames = h.list_child_names()
            self.assertTrue(cnames[0] == "a")
            self.assertTrue(cnames[1] == "b")
            self.assertTrue(cnames[2] == "c")
            self.assertTrue(cnames[3] == "d")
            cnames = h.list_child_names("d")
            self.assertTrue(cnames[0] == "here")
            h.remove("d");
            self.assertFalse(h.has_path("d"))
            self.assertFalse(h.has_path("d/here"))
            h.close();
            n2 = conduit.Node()
            h2 = conduit.relay.io.IOHandle();
            h2.open(test_file);
            cnames = h2.list_child_names();
            self.assertTrue(cnames[0] == "a")
            self.assertTrue(cnames[1] == "b")
            self.assertTrue(cnames[2] == "c")
            n_val = conduit.Node()
            n_val.set(int64(10))
            h2.write(n_val,"d/here")
            h2.read(n2);
            info = conduit.Node()
            self.assertFalse(n.diff(n2,info,0.0))
            n_val.reset()
            h2.read(n_val,"c");
            self.assertTrue(n_val.value() == 12)
            h2.close()

    def test_io_handle_exceptions(self):
            h = conduit.relay.io.IOHandle()
            n = conduit.Node()
            # call to un-opend
            with self.assertRaises(IOError):
                h.read(n);
            with self.assertRaises(IOError):
                h.write(n);
            with self.assertRaises(IOError):
                h.has_path("here");
            with self.assertRaises(IOError):
                h.remove("here");
            with self.assertRaises(IOError):
                h.list_child_names();

            # bad path
            with self.assertRaises(IOError):
                h.open("here/is/a/garbage/file/path.json");

if __name__ == '__main__':
    unittest.main()


