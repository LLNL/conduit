# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_relay_web.py
 description: Unit tests for the conduit relay web python module interface.

"""

##############################################################
# make sure windows can pick up dlls for python 3.8 and newer
import os
if "PATH" in os.environ:
    for dll_path in os.environ["PATH"].split(";"):
        os.add_dll_directory(dll_path)
##############################################################

import sys
import unittest

from numpy import *
from conduit import Node

import conduit
import conduit.relay as relay
import conduit.relay.web

class Test_Relay_Web(unittest.TestCase):
    def test_webserver(self):
        ws = relay.web.WebServer()
        self.assertFalse(ws.is_running())


if __name__ == '__main__':
    unittest.main()


