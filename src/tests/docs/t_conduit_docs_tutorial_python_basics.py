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

def BEGIN_EXAMPLE(tag):
    print('BEGIN_EXAMPLE("' + tag + '")')

def END_EXAMPLE(tag):
    print('END_EXAMPLE("' + tag + '")')

class Conduit_Tutorial_Python_Basics(unittest.TestCase):

    def test_001_basics_very_basic(self):
        BEGIN_EXAMPLE("py_basics_very_basic")
        import conduit
        n = conduit.Node()
        n["my"] = "data"
        print(n)
        END_EXAMPLE("py_basics_very_basic")

    def test_002_basics_hierarchial(self):
        BEGIN_EXAMPLE("py_basics_hierarchial")
        n = conduit.Node()
        n["my"] = "data";
        n["a/b/c"] = "d";
        n["a"]["b"]["e"] = 64.0;
        print(n)
        print("total bytes: {}\n".format(n.total_strided_bytes()))
        END_EXAMPLE("py_basics_hierarchial")

    def test_003_basics_object_and_list(self):
        BEGIN_EXAMPLE("py_basics_object_and_list")
        n = conduit.Node()
        n["object_example/val1"] = "data"
        n["object_example/val2"] = 10
        n["object_example/val3"] = 3.1415

        for i in range(5):
            l_entry = n["list_example"].append()
            l_entry.set(i)
        print(n)
        END_EXAMPLE("py_basics_object_and_list")

    def test_004_basics_object_and_list_itr(self):
        BEGIN_EXAMPLE("py_basics_object_and_list_itr")
        n = conduit.Node()
        n["object_example/val1"] = "data"
        n["object_example/val2"] = 10
        n["object_example/val3"] = 3.1415

        for i in range(5):
            l_entry = n["list_example"].append()
            l_entry.set(i)
        print(n)
        
        for v in n["object_example"].children():
            print("{}: {}".format(v.name(),str(v.node())))
        print()
        for v in n["list_example"].children():
            print(v.node())
        END_EXAMPLE("py_basics_object_and_list_itr")

    def test_005_basics_mem_spaces(self):
        BEGIN_EXAMPLE("py_basics_mem_spaces")
        n = conduit.Node()
        n["my"] = "data"
        n["a/b/c"] = "d"
        n["a"]["b"]["e"] = 64.0
        print(n.info())
        END_EXAMPLE("py_basics_mem_spaces")

    def test_006_basics_bw_style(self):
        BEGIN_EXAMPLE("py_basics_bw_style")
        n = conduit.Node()
        n["test"] = numpy.uint32(100)
        print(n)
        END_EXAMPLE("py_basics_bw_style")

    def test_007_basics_bw_style_from_native(self):
        BEGIN_EXAMPLE("py_basics_bw_style_from_native")
        n = conduit.Node()
        n["test"] = 10
        print(n.schema())
        END_EXAMPLE("py_basics_bw_style_from_native")


