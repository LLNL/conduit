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

    def test_schema_remove(self):
        s = Schema()
        s['a'] = DataType.float64(1)
        s['b'] = DataType.float64(1)
        s['c'] = DataType.float64(1)
        self.assertEqual(s.number_of_children(),3)
        s.remove(path='c')
        self.assertEqual(s.number_of_children(),2)
        paths = s.child_names()
        for v in ['a','b']:
            self.assertTrue(v in paths)
        s.remove(index=0)
        paths = s.child_names()
        for v in ['b']:
            self.assertTrue(v in paths)

    def test_schema_child_rename(self):
        s = Schema()
        with self.assertRaises(Exception):
            s.rename_child('a','b')
        s["a"] = DataType.float64(10)
        s["b"] = DataType.float32(10)

        with self.assertRaises(Exception):
            s.rename_child('bad','good')

        with self.assertRaises(Exception):
            s.rename_child('b','a')

        s.rename_child("b","c")
        n = Node()
        n.set(s)
        sr = n.schema()
        self.assertEqual(sr.total_strided_bytes(), 8 * 10 + 4 * 10)
        self.assertEqual(sr["a"].total_strided_bytes(),8 * 10)
        self.assertEqual(sr["c"].total_strided_bytes(),4 * 10)

    def test_key_with_slash(self):
        s = Schema()
        s["normal/path"] = DataType.float64(1)
        s.add_child("child_with_/_inside").set(DataType.float64(1))
        print(s)
        self.assertTrue(s.has_path("normal/path"))
        self.assertFalse(s.has_child("normal/path"))
        self.assertFalse(s.has_path("child_with_/_inside"))
        self.assertTrue(s.has_child("child_with_/_inside"))
        self.assertEqual(2,s.number_of_children())
        self.assertEqual(s.child(1).dtype().id(),DataType.float64().id())
        self.assertEqual(s.child(name="child_with_/_inside").dtype().id(),DataType.float64().id())
        s["normal"].remove_child("path")
        self.assertFalse(s.has_path("normal/path"))

    def test_compact_to(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        s = n.schema()

        n_info = n.info()
        print(n_info)
        self.assertEqual(n_info["mem_spaces"].number_of_children(),3)

        s2 = Schema()
        s.compact_to(s2);

        n2 = Node()
        n2.set(s2)
        n2_info = n2.info()
        print(n2_info)
        self.assertEqual(n2_info["mem_spaces"].number_of_children(),1)


    def test_to_string_and_friends(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)
        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
        s = n.schema()

        print("yaml rep")
        print(s.to_string("yaml"))

        print("json rep")
        print(s.to_string("json"))

        self.assertEqual(s.to_string("yaml"),s.to_yaml())
        self.assertEqual(s.to_string("json"),s.to_json())

if __name__ == '__main__':
    unittest.main()


