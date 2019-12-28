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

class Conduit_Tutorial_Python_Parse(unittest.TestCase):

    def test_docs_tutorial_yaml(self):
        BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml")
        yaml_txt = "mykey: 42.0"

        n = conduit.Node()
        n.parse(yaml_txt,"yaml")

        print(n["mykey"])
        print(n.schema())
        END_EXAMPLE("t_py_conduit_docs_tutorial_yaml")

    def test_docs_tutorial_json(self):
        BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_json")
        json_txt = '{"mykey": 42.0}'

        n = conduit.Node()
        n.parse(json_txt,"json")

        print(n["mykey"])
        print(n.schema())
        END_EXAMPLE("t_py_conduit_docs_tutorial_json")

    def test_docs_tutorial_yaml_inline_array(self):
        BEGIN_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")
        yaml_txt = "myarray: [0.0, 10.0, 20.0, 30.0]"

        n = conduit.Node()
        n.parse(yaml_txt,"yaml")

        print(n["myarray"])

        print(n.fetch("myarray").schema())
        END_EXAMPLE("t_py_conduit_docs_tutorial_yaml_inline_array")

    def test_json_generator_std(self):
        BEGIN_EXAMPLE("t_py_json_generator_std")
        g = conduit.Generator("{test: {dtype: float64, value: 100.0}}",
                              "conduit_json")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("t_py_json_generator_std")

    def test_json_generator_pure_json(self):
        BEGIN_EXAMPLE("t_py_json_generator_pure_json")
        g = conduit.Generator("{test: 100.0}",
                              "json")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("t_py_json_generator_pure_json")

    def test_json_generator_pure_yaml(self):
        BEGIN_EXAMPLE("t_py_json_generator_pure_yaml")
        g = conduit.Generator("test: 100.0",
                              "yaml")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("t_py_json_generator_pure_yaml")



