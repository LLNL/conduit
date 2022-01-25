# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
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
            print("machine is little endian")
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
            print("machine is big endian")
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


