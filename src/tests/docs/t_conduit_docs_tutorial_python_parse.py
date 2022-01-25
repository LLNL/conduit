# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
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
        BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml")
        yaml_txt = "mykey: 42.0"

        n = conduit.Node()
        n.parse(yaml_txt,"yaml")

        print(n["mykey"])
        print(n.schema())
        END_EXAMPLE("py_conduit_docs_tutorial_yaml")

    def test_docs_tutorial_json(self):
        BEGIN_EXAMPLE("py_conduit_docs_tutorial_json")
        json_txt = '{"mykey": 42.0}'

        n = conduit.Node()
        n.parse(json_txt,"json")

        print(n["mykey"])
        print(n.schema())
        END_EXAMPLE("py_conduit_docs_tutorial_json")

    def test_docs_tutorial_yaml_inline_array(self):
        BEGIN_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")
        yaml_txt = "myarray: [0.0, 10.0, 20.0, 30.0]"

        n = conduit.Node()
        n.parse(yaml_txt,"yaml")

        print(n["myarray"])

        print(n.fetch("myarray").schema())
        END_EXAMPLE("py_conduit_docs_tutorial_yaml_inline_array")

    def test_json_generator_std(self):
        BEGIN_EXAMPLE("py_json_generator_std")
        g = conduit.Generator("{test: {dtype: float64, value: 100.0}}",
                              "conduit_json")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("py_json_generator_std")

    def test_json_generator_pure_json(self):
        BEGIN_EXAMPLE("py_json_generator_pure_json")
        g = conduit.Generator("{test: 100.0}",
                              "json")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("py_json_generator_pure_json")

    def test_json_generator_pure_yaml(self):
        BEGIN_EXAMPLE("py_json_generator_pure_yaml")
        g = conduit.Generator("test: 100.0",
                              "yaml")
        n = conduit.Node()
        g.walk(n)
        print(n["test"])
        print(n)
        END_EXAMPLE("py_json_generator_pure_yaml")



