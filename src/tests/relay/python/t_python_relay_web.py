# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_relay_web.py
 description: Unit tests for the conduit relay web python module interface.

"""

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


