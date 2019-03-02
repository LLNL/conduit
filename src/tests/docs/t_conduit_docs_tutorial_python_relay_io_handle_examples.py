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

def echo_src(s,fname,lineno):
    print("\n{}: {},{}".format(s,fname,lineno))

class Conduit_Tutorial_Python_Relay_IO_Handle(unittest.TestCase):

    def test_001_io_handle(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        echo_src("begin",inspect.stack()[0][3],inspect.currentframe().f_lineno)
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
        echo_src("end",inspect.stack()[0][3],inspect.currentframe().f_lineno)

