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
 file: python_conduit_datatype.py
 description: Unit tests for conduit::DataType python module interface.

"""

import sys
import unittest

from conduit import Node
from conduit import DataType

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
        print(n)
        d = n.fetch('a').dtype()
        self.assertEqual(d.id(),DataType.name_to_id("uint32"))
        print(d)

    def test_id_to_name(self):
        names = [DataType.id_to_name(i) for i in range(14)]
        ids   = [DataType.name_to_id(n) for n in names]
        self.assertEqual(ids,list(range(14)))

    def test_explicit_set(self):
        d = DataType()
        d.set(dtype_id = DataType.name_to_id("uint32"),
              num_elements = 1,
              offset = 0,
              stride = 4,
              element_bytes = 4)
        print(d)
        self.assertEqual(d.id(),DataType.name_to_id("uint32"))
        self.assertEqual(d.number_of_elements(),1)
        self.assertEqual(d.offset(),0)
        self.assertEqual(d.stride(),4)
        self.assertEqual(d.element_bytes(),4)
        self.assertEqual(d.endianness(),0)

    def test_construction(self):
        dt = DataType();
        dt.set_id(DataType.name_to_id("uint32"))
        dt.set_number_of_elements(10);
        dt.set_offset(0);
        dt.set_stride(4);
        dt.set_element_bytes(4);
        
        dt2 = DataType(dt)
        self.assertEqual(dt.id(),dt2.id())
        self.assertEqual(dt.number_of_elements(),dt2.number_of_elements())
        self.assertEqual(dt.offset(),dt2.offset())
        self.assertEqual(dt.stride(),dt2.stride())
        self.assertEqual(dt.element_bytes(),dt2.element_bytes())        
        self.assertEqual(dt.endianness(),dt2.endianness())

        dt3 = DataType()
        dt3.set(dtype_name="uint32",
                num_elements=10,
                offset=0,
                stride=4,
                element_bytes=4)
        self.assertEqual(dt2.id(),dt3.id())
        self.assertEqual(dt2.number_of_elements(),dt3.number_of_elements())
        self.assertEqual(dt2.offset(),dt3.offset())
        self.assertEqual(dt2.stride(),dt3.stride())
        self.assertEqual(dt2.element_bytes(),dt3.element_bytes())
        self.assertEqual(dt2.endianness(),dt3.endianness())
        
        print(dt)
        print(dt2)
        print(dt3)

    def test_constructor_helpers(self):
        # objs
        print(DataType.empty());
        print(DataType.object());
        print(DataType.list());
        # signed integers
        print(DataType.int8());
        print(DataType.int16());
        print(DataType.int32());
        print(DataType.int64());
        # unsigned integers
        print(DataType.uint8());
        print(DataType.uint16());
        print(DataType.uint32());
        print(DataType.uint64());
        # floating point
        print(DataType.float32());
        print(DataType.float64());
        # signed integers
        print(DataType.c_char());
        print(DataType.c_short());
        print(DataType.c_int());
        print(DataType.c_long());
        # unsigned integers
        print(DataType.c_unsigned_char());
        print(DataType.c_unsigned_short());
        print(DataType.c_unsigned_int());
        print(DataType.c_unsigned_long());
        # floating point
        print(DataType.c_float());
        print(DataType.c_double());




if __name__ == '__main__':
    unittest.main()


