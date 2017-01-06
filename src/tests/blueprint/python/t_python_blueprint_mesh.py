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
 file: t_python_blueprint_mcarray.py
 description: Simple unit test for the conduit blueprint mcarray python module.

"""

import sys
import unittest

import conduit.blueprint as blueprint

import conduit.blueprint.mesh
import conduit.blueprint.mesh.examples

from conduit import Node

class Test_Blueprint_Mesh(unittest.TestCase):
        
    def test_braid_and_verify(self):
        n = Node()
        info = Node()
        self.assertFalse(blueprint.verify("mesh",n,info))
        self.assertFalse(blueprint.mesh.verify(n,info))
        blueprint.mesh.examples.braid("hexs",2,2,2,n);
        self.assertTrue(blueprint.mesh.verify(n,info))
        n_idx = Node()
        blueprint.mesh.generate_index(n,"",1,n_idx)
        self.assertTrue(blueprint.verify("mesh/index",n_idx,info))
        self.assertTrue(blueprint.mesh.verify(protocol="index",node=n_idx,info=info))


if __name__ == '__main__':
    unittest.main()


