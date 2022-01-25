# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_relay_smoke.py
 description: Simple unit test for the conduit relay python module interface.

"""

import sys
import unittest

import conduit.relay as relay

class Test_Relay_Basic(unittest.TestCase):
    def test_about(self):
        print(relay.about())

if __name__ == '__main__':
    unittest.main()


