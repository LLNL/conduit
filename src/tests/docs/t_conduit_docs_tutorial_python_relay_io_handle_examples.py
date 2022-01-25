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
import conduit.relay
import os
from  os.path import join as pjoin

def relay_test_data_path(fname):
    return pjoin(os.path.split(os.path.abspath(__file__))[0],
          "..",
          "relay",
          "data",
          fname);

def BEGIN_EXAMPLE(tag):
    print('\nBEGIN_EXAMPLE("' + tag + '")')

def END_EXAMPLE(tag):
    print('\nEND_EXAMPLE("' + tag + '")')

class Conduit_Tutorial_Python_Relay_IO_Handle(unittest.TestCase):

    def test_001_io_handle(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle")
        import conduit
        import conduit.relay.io

        n = conduit.Node()
        n["a/data"]   = 1.0
        n["a/more_data"] = 2.0
        n["a/b/my_string"] = "value"
        print("\nNode to write:")
        print(n)

        # save to hdf5 file using the path-based api
        conduit.relay.io.save(n,"my_output.hdf5")

        # inspect and modify with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open("my_output.hdf5")

        # check for and read a path we are interested in
        if h.has_path("a/data"):
             nread = conduit.Node()
             h.read(nread,"a/data")
             print('\nValue at "a/data" = {0}'.format(nread.value()))

        # check for and remove a path we don't want
        if h.has_path("a/more_data"):
            h.remove("a/more_data")
            print('\nRemoved "a/more_data"')

        # verify the data was removed
        if not h.has_path("a/more_data"):
            print('\nPath "a/more_data" is no more')

        # write some new data
        print('\nWriting to "a/c"')
        n.set(42.0)
        h.write(n,"a/c")
        
        # find the names of the children of "a"
        cnames = h.list_child_names("a")
        print('\nChildren of "a": {0}'.format(cnames))

        nread = conduit.Node()
        # read the entire contents
        h.read(nread)

        print("\nRead Result:")
        print(nread)
        END_EXAMPLE("py_relay_io_handle")


    def test_002_io_handle_sidre(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle_sidre")
        import conduit
        import conduit.relay.io

        # this example reads a sample hdf5 sidre style file
        input_fname = relay_test_data_path("texample_sidre_basic_ds_demo.sidre_hdf5")

        # open our sidre file for read with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open(input_fname,"sidre_hdf5")

        # find the names of the children at the root
        cnames = h.list_child_names()
        print('\nChildren at root {0}'.format(cnames))

        nread = conduit.Node()
        # read the entire contents
        h.read(nread);

        print("Read Result:")
        print(nread)

        END_EXAMPLE("py_relay_io_handle_sidre")



    def test_003_io_handle_sidre_root(self):
        import conduit.relay
        if conduit.relay.io.about()["protocols/hdf5"] != "enabled":
            return
        BEGIN_EXAMPLE("py_relay_io_handle_sidre_root")
        import conduit
        import conduit.relay.io

        # this example reads a sample hdf5 sidre datastore,
        # grouped by a root file
        input_fname = relay_test_data_path("out_spio_blueprint_example.root")

        # open our sidre datastore for read via root file with an IOHandle
        h = conduit.relay.io.IOHandle()
        h.open(input_fname,"sidre_hdf5")

        # find the names of the children at the root
        # the "root" (/) of the Sidre-based IOHandle to the datastore provides
        # access to the root file itself, and all of the data groups
        cnames = h.list_child_names()
        print('\nChildren at root {0}'.format(cnames))
        
        nroot = conduit.Node();
        # read the entire root file contents
        h.read(path="root",node=nroot);

        print("Read 'root' Result:")
        print(nroot)

        nread = conduit.Node();
        # read all of data group 0
        h.read(path="0",node=nread);

        print("Read '0' Result:")
        print(nread)

        #reset, or trees will blend in this case
        nread.reset();

        # read a subpath of data group 1
        h.read(path="1/mesh",node=nread);

        print("Read '1/mesh' Result:")
        print(nread)

        END_EXAMPLE("py_relay_io_handle_sidre_root")


