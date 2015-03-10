###############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see https://lc.llnl.gov/conduit/.
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
 file: test_conduit_python_node.py
 description: Unit tests for conduit::Node python module interface.

"""

import sys
import unittest

import conduit
Node = conduit.Node.Node

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
        print n['a']['b']
        self.assertTrue(n['a']['b'] == val)

    def test_vector(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        self.assertTrue(n['a'][99], 99)

    def test_fetch(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        na_val = na.data()
        self.assertTrue(na_val[99], 99)
        
    def test_child(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.child(0)
        na_val = na.data()
        self.assertTrue(na_val[99], 99)
        n['b'] = vec
        self.assertTrue(n.number_of_children(),2)
      
    def test_save_load(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        n.save("test_pyconduit_node_save_load")
        nl = Node()
        nl.load("test_pyconduit_node_save_load")
        self.assertTrue(nl['a'][99], 99)

    def test_parent(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        na = n.fetch('a')
        self.assertTrue(na.has_parent())
        # todo: test parent()

    def test_total_bytes(self):
        vec = array(range(100), uint32)
        n = Node()
        n['a'] = vec
        self.assertTrue(n.total_bytes(),4 * 100)
        self.assertTrue(n.total_bytes_compact(),4 * 100)
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
        paths = n.paths()
        for v in ['a','b','c']:
            self.assertTrue(v in paths)

    def test_list(self):
        n = Node()
        n.append().set(1)
        self.assertTrue(n.child(0).data(),1)
        # TODO: this needs to work but doesn't
        #self.assertTrue(n[0],1)
        n2 = Node()
        n2_c = n2.append()
        n2_c.set(2)
        self.assertTrue(n2.child(0).data(),2)
        

    def test_remove(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        self.assertTrue(n.number_of_children(),3)
        n.remove(path='c')
        self.assertTrue(n.number_of_children(),2)
        paths = n.paths()
        for v in ['a','b']:
            self.assertTrue(v in paths)
        n.remove(index=0)
        paths = n.paths()
        for v in ['b']:
            self.assertTrue(v in paths)

    def test_info(self):
        n = Node()
        n['a'] = 1
        n['b'] = 2
        n['c'] = 3
        ni = n.info();
        #print ni
        self.assertTrue(ni["total_bytes"],n.total_bytes())


if __name__ == '__main__':
    unittest.main()


