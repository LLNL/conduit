###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://llnl.github.io/conduit/.
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
 file: python_conduit_generator.py
 description: Unit tests for conduit::Geneartor python module interface.

"""

import sys
import unittest

from conduit import Node
from conduit import Schema
from conduit import Generator

from numpy import *

def default_node():
    a_val = uint32(10)
    b_val = uint32(20)
    c_val = float64(30.0)

    n = Node()
    n['a'] = a_val
    n['b'] = b_val
    n['c'] = c_val
    return n;
    

class Test_Conduit_Geneartor(unittest.TestCase):
    def test_simple(self):
        n = default_node()
        n_schema = n.to_json("conduit");
        print("result detailed json", n_schema)
        g = Generator(json_schema=n_schema);
        ng = Node();
        sg = Schema()
        g.walk(node=ng);
        g.walk(schema=sg);
        print(ng)
        print(sg)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)
        

    def test_base64(self):
        n = default_node()
        print(n)
        n_schema = n.to_json("base64_json");
        print("result base64 json", n_schema)
        g = Generator(n_schema,"base64_json");
        ng = Node();
        g.walk(node=ng);
        print("Generator result")
        print(ng)
        for p in ["a","b","c"]:
            orig = n.fetch(p).value()
            curr = ng.fetch(p).value()
            print(ng)
            print(p, orig, curr)
            orig = n[p]
            curr = ng[p]
            print(ng)
            print(p, orig, curr)
            self.assertEqual(orig,curr)



if __name__ == '__main__':
    unittest.main()


