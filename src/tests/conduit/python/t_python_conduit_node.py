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
 file: t_python_conduit_node.py
 description: Unit tests for conduit::Node python module interface.

"""

import sys
import unittest

from conduit import Node

from numpy import *


class Test_Conduit_Node(unittest.TestCase):
    def test_simple(self):
        a_val = uint32(10)
        b_val = uint32(20)
        c_val = float64(30.0)

        n = Node()
        n['a'] = a_val
        n['b'] = b_val
        n['c'] = c_val
  
        self.assertTrue(n['a'] == a_val)
        self.assertTrue(n['b'] == b_val)
        self.assertTrue(n['c'] == c_val)

    def test_nested(self):
        val = uint32(10)
        n = Node()
        n['a']['b'] = val
        print(n['a']['b'])
        self.assertEqual(n['a']['b'],val)

    def test_vector(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        self.assertEqual(n['a'][99], 99)

    def test_fetch(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        na_val = na.value()
        self.assertEqual(na_val[99], 99)
        
    def test_child(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.child(0)
        na_val = na.value()
        self.assertEqual(na_val[99], 99)
        n['b'] = vec
        self.assertEqual(n.number_of_children(),2)
      
    def test_save_load(self):
        # on windows, this breaks at 27 !?
        alen = 26
        vec = array(range(alen), uint32)
        n = Node()
        n['a'] = vec
        print(n)
        n.save("test_pyconduit_node_save_load.conduit_bin")
        nl = Node()
        nl.load("test_pyconduit_node_save_load.conduit_bin")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)
        
        n.save("test_pyconduit_node_json_save_load.json",protocol="json")
        nl = Node()
        nl.load("test_pyconduit_node_json_save_load.json", protocol="json")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)
        
        n.save("test_pyconduit_node_base64_json_save_load.conduit_base64_json", protocol="conduit_base64_json")
        nl = Node()
        nl.load("test_pyconduit_node_base64_json_save_load.conduit_base64_json", protocol="conduit_base64_json")
        print(nl)
        self.assertEqual(nl['a'][alen-1], alen-1)

    def test_parent(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        self.assertFalse(na.is_root())
        # todo: test parent()

    def test_total_bytes(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        self.assertEqual(n.total_strided_bytes(),4 * 100)
        self.assertEqual(n.total_bytes_compact(),4 * 100)
        # TODO: check if n.is_compact() should pass as well?
        # it doesn't currently
        self.assertTrue(n.fetch('a').is_compact())

    def test_paths(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        for v in ['a','b','c']:
            self.assertTrue(n.has_path(v))
        paths = n.child_names()
        for v in ['a','b','c']:
            self.assertTrue(v in paths)

    def test_list(self):
        n = Node()
        n.append().set(1)
        self.assertTrue(n.child(0).value(),1)
        self.assertTrue(n[0],1)
        n2 = Node()
        n2_c = n2.append()
        n2_c.set(2)
        self.assertEqual(n2.child(0).value(),2)

        n3 = Node()
        n3.fetch("here").append().set("a")
        n3.fetch("here").append().set("b")
        self.assertTrue(n3.fetch("here").child(0).value(),"a")
        self.assertTrue(n3.fetch("here").child(1).value(),"b")

        n4 = Node()
        n4["here"].append().set("a")
        n5 = n4["here"]
        n5.append().set("b")
        self.assertTrue(n4["here"].child(0).value(),"a")
        self.assertTrue(n4["here"].child(1).value(),"b")
        self.assertTrue(n4["here"][0],"a")
        self.assertTrue(n4["here"][1],"b")


    
    def test_remove(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        self.assertEqual(n.number_of_children(),3)
        n.remove(path='c')
        self.assertEqual(n.number_of_children(),2)
        paths = n.child_names()
        for v in ['a','b']:
            self.assertTrue(v in paths)
        n.remove(index=0)
        paths = n.child_names()
        for v in ['b']:
            self.assertTrue(v in paths)

    def test_info(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        ni = n.info();
        #print ni
        self.assertEqual(ni["total_strided_bytes"],n.total_strided_bytes())

    def test_set_external(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            ext_data = array(range(10), dtype=type)
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[8] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

    def test_set_external_basic_slice(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            base_data = array(range(20), dtype=type)
            ext_data  = base_data[1:16]
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[6] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

    def test_set_external_basic_strides(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        for type in types:
            base_data = array(range(20), dtype=type)
            ext_data  = base_data[1:16:2]
            n = Node()
            n.set_external(ext_data)
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])
            ext_data[5] = 11
            n.value()[6] = 77
            n.value()[2] = 8
            for i in range(len(ext_data)):
                self.assertEqual(n.value()[i], ext_data[i])

if __name__ == '__main__':
    unittest.main()


