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
 file: t_conduit_docs_tutorial_python_examples.py
"""

import sys
import unittest
import inspect
import numpy
import conduit
import conduit.relay
import os
from  os.path import join as pjoin

def relay_test_data_path(fname):
    return pjoin(os.path.split(os.path.abspath(__file__))[0],
          "..",
          "relay",
          "data",
          fname);

def BEGIN_EXAMPLE(tag):
    print('\nBEGIN_EXAMPLE("' + tag + '")')

def END_EXAMPLE(tag):
    print('\nEND_EXAMPLE("' + tag + '")')

class Conduit_Tutorial_Python_Relay_IO_Handle(unittest.TestCase):

    def test_001_io_handle(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle")
        import conduit
        import conduit.relay.io

        n = conduit.Node()
        n["a/data"]   = 1.0
        n["a/more_data"] = 2.0
        n["a/b/my_string"] = "value"
        print("\nNode to write:")
        print(n)

        # save to hdf5 file using the path-based api
        conduit.relay.io.save(n,"my_output.hdf5")

        # inspect and modify with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open("my_output.hdf5")

        # check for and read a path we are interested in
        if h.has_path("a/data"):
             nread = conduit.Node()
             h.read(nread,"a/data")
             print('\nValue at "a/data" = {0}'.format(nread.value()))

        # check for and remove a path we don't want
        if h.has_path("a/more_data"):
            h.remove("a/more_data")
            print('\nRemoved "a/more_data"')

        # verify the data was removed
        if not h.has_path("a/more_data"):
            print('\nPath "a/more_data" is no more')

        # write some new data
        print('\nWriting to "a/c"')
        n.set(42.0)
        h.write(n,"a/c")
        
        # find the names of the children of "a"
        cnames = h.list_child_names("a")
        print('\nChildren of "a": {0}'.format(cnames))

        nread = conduit.Node()
        # read the entire contents
        h.read(nread)

        print("\nRead Result:")
        print(nread)
        END_EXAMPLE("py_relay_io_handle")


    def test_002_io_handle_sidre(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle_sidre")
        import conduit
        import conduit.relay.io

        # this example reads a sample hdf5 sidre style file
        input_fname = relay_test_data_path("texample_sidre_basic_ds_demo.sidre_hdf5")

        # open our sidre file for read with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open(input_fname,"sidre_hdf5")

        # find the names of the children at the root
        cnames = h.list_child_names()
        print('\nChildren at root {0}'.format(cnames))

        nread = conduit.Node()
        # read the entire contents
        h.read(nread);

        print("Read Result:")
        print(nread)

        END_EXAMPLE("py_relay_io_handle_sidre")



    def test_003_io_handle_sidre_root(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle_sidre_root")
        import conduit
        import conduit.relay.io

        # this example reads a sample hdf5 sidre datastore,
        # grouped by a root file
        input_fname = relay_test_data_path("out_spio_blueprint_example.root")

        # open our sidre datastore for read via root file with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open(input_fname,"sidre_hdf5")

        # find the names of the children at the root
        # the "root" (/) of the Sidre-based IOHandle to the datastore provides
        # access to the root file itself, and all of the data groups
        cnames = h.list_child_names()
        print('\nChildren at root {0}'.format(cnames))
        
        nroot = conduit.Node();
        # read the entire root file contents
        h.read(path="root",node=nroot);

        print("Read 'root' Result:")
        print(nroot)

        nread = conduit.Node();
        # read all of data group 0
        h.read(path="0",node=nread);

        print("Read '0' Result:")
        print(nread)

        #reset, or trees will blend in this case
        nread.reset();

        # read a subpath of data group 1
        h.read(path="1/mesh",node=nread);

        print("Read '1/mesh' Result:")
        print(nread)

        END_EXAMPLE("py_relay_io_handle_sidre_root")


