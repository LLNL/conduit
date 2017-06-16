###############################################################################
# Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
 file: t_python_conduit_endianness.py
 description: Unit tests for conduit::Endianness python module interface.

"""

import sys
import unittest

from conduit import Node, Endianness

from numpy import *


class Test_Conduit_Endianness(unittest.TestCase):
    def test_endianness(self):
        n = Node()
        n.set(100)
        if Endianness.machine_is_little_endian():
            print "machine is little endian"
            self.assertTrue(Endianness.little_id() == Endianness.machine_default())
            # big and back
            n.endian_swap_to_big()
            v = n.value()
            self.assertTrue(v != 100)
            n.endian_swap_to_little()
            v = n.value()
            self.assertTrue(v == 100)

            # big and back w/ default swap
            n.endian_swap_to_big()
            v = n.value()
            self.assertTrue(v != 100)
            n.endian_swap_to_machine_default()
            v = n.value()
            self.assertTrue(v == 100)

        elif Endianness.machine_is_big_endian() :
            print "machine is big endian"
            self.assertTrue(Endianness.big_id() == Endianness.machine_default())
            # little and back
            n.endian_swap_to_little()
            v = n.value()
            self.assertTrue(v != 100)
            n.endian_swap_to_big()
            v = n.value()
            self.assertTrue(v == 100)

            # little and back w/ default swap
            n.endian_swap_to_little()
            v = n.value()
            self.assertTrue(v != 100)
            n.endian_swap_to_machine_default()
            v = n.value()
            self.assertTrue(v == 100)
        n.set(20)
        n.endian_swap(Endianness.big_id())
        n.endian_swap(Endianness.little_id())
        v = n.value()
        self.assertTrue(v == 20)


if __name__ == '__main__':
    unittest.main()


