# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
# file: __init__.py
# Purpose: Main init for the conduit relay io module.
###############################################################################

# silo support is optional, so drive on if we can't import
try:
    from .conduit_relay_io_silo_python import *
except:
    pass
