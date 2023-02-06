#! /usr/bin/env python
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
# note: run in build dir, expects src to be ../src from build dir
import subprocess
import os

from os.path import join as pjoin

tests = ["t_conduit_docs_tutorial_basics",
         "t_conduit_docs_tutorial_numeric",
         "t_conduit_docs_tutorial_parse",
         "t_conduit_docs_tutorial_ownership",
         "t_conduit_docs_tutorial_utils",
         "t_conduit_docs_tutorial_move_and_swap",
         "t_conduit_docs_tutorial_errors",
         # bp
         "t_conduit_docs_blueprint_demos",
         "t_conduit_docs_blueprint_examples",
         "t_conduit_docs_relay_io_generic_examples",
         "t_conduit_docs_relay_io_handle_examples",
         "t_conduit_docs_relay_io_hdf5_examples",
         # python
         "t_conduit_docs_tutorial_python_basics",
         "t_conduit_docs_tutorial_python_numeric",
         "t_conduit_docs_tutorial_python_ownership",
         "t_conduit_docs_tutorial_python_parse",
         "t_conduit_docs_tutorial_python_move_and_swap",
         "t_conduit_docs_tutorial_python_utils",
         "t_conduit_docs_tutorial_python_relay_io_handle_examples",
         "t_conduit_docs_tutorial_python_relay_mpi_examples"]


dest_dir = "../src/docs/sphinx/"
out_suffix = "_out.txt"

for test in tests:
    if "python" in test:
        cmd = "ctest -VV -R {} > {}_out.txt".format(test,pjoin(dest_dir,test))
    else:
        cmd = "tests/docs/{} > {}_out.txt".format(test,pjoin(dest_dir,test))
    print(cmd)
    subprocess.call(cmd,shell=True)
