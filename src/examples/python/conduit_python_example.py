# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

"""
 file: conduit_python_example.py

 description:
   Basic conduit python usage

"""

import conduit
import conduit.relay
import conduit.blueprint

# print details about conduit
print(conduit.about())
print(conduit.relay.about())
print(conduit.blueprint.about())

# create a conduit node
n = conduit.Node()
n["a/b/c/d/e/f/g"] = 42
print(n)





