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
 file: python_conduit_schema.py
 description: Unit tests for conduit::Schema python module interface.

"""

import sys
import unittest

from conduit import Node
from conduit import Schema, DataType

from numpy import *


class Test_Conduit_Schema(unittest.TestCase):
    def test_simple(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        s = n.schema();
        self.assertEqual(s.total_strided_bytes(),16)

    def test_simple(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        s = n.schema();
        self.assertTrue(s.is_root())
        self.assertFalse(n.fetch('a').schema().is_root())

    def test_create_node_using_schema_object(self):
        s = Schema()
        s["a"] = DataType.float64(10)
        s["b"] = DataType.float32(10)
        n = Node()
        n.set(s)
        sr = n.schema()
        self.assertEqual(sr.total_strided_bytes(), 8 * 10 + 4 * 10)
        self.assertEqual(sr["a"].total_strided_bytes(),8 * 10)
        self.assertEqual(sr["b"].total_strided_bytes(),4 * 10)



if __name__ == '__main__':
    unittest.main()


