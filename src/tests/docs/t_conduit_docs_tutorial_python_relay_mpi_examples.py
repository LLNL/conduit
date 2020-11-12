# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_conduit_docs_tutorial_python_relay_mpi_examples.py
"""

import sys
import unittest
import inspect
import numpy


def BEGIN_EXAMPLE(tag):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.rank == 0:
        print('\nBEGIN_EXAMPLE("' + tag + '")')
    MPI.COMM_WORLD.Barrier()

def END_EXAMPLE(tag):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.rank == 0:
        print('\nEND_EXAMPLE("' + tag + '")')
    MPI.COMM_WORLD.Barrier()

class Conduit_Tutorial_Python_Relay_IO_Handle(unittest.TestCase):

    def test_001_mpi_send_and_recv_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_send_and_recv_using_schema")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # Note: example expects 2 mpi tasks

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        # send a node and its schema from rank 0 to rank 1
        n = conduit.Node()
        if comm_rank == 0:
            # setup node to send on rank 0
            n["a/data"]   = 1.0
            n["a/more_data"] = 2.0
            n["a/b/my_string"] = "value"

        # show node data on rank 0
        if comm_rank == 0:
            print("[rank: {}] sending: {}".format(comm_rank,n.to_yaml()))

        if comm_rank == 0:
            relay.mpi.send_using_schema(n,dest=1,tag=0,comm=comm_id)
        else:
            relay.mpi.recv_using_schema(n,source=0,tag=0,comm=comm_id)

        # show received node data on rank 1
        if comm_rank == 1:
            print("[rank: {}] received: {}".format(comm_rank,n.to_yaml()))

        END_EXAMPLE("py_mpi_send_and_recv_using_schema")


    def test_002_mpi_send_and_recv(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_send_and_recv")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # Note: example expects 2 mpi tasks

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        # send data from a node on rank 0 to rank 1
        # (both ranks have nodes with compatible schemas)
        n = conduit.Node(conduit.DataType.int64(4))
        if comm_rank == 0:
            # setup node to send on rank 0
            vals = n.value()
            for i in range(4):
                vals[i] = i * i

        # show node data on rank 0
        if comm_rank == 0:
            print("[rank: {}] sending: {}".format(comm_rank,n.to_yaml()))

        if comm_rank == 0:
            relay.mpi.send(n,dest=1,tag=0,comm=comm_id)
        else:
            relay.mpi.recv(n,source=0,tag=0,comm=comm_id)

        # show received node data on rank 1
        if comm_rank == 1:
            print("[rank: {}] received: {}".format(comm_rank,n.to_yaml()))

        END_EXAMPLE("py_mpi_send_and_recv")

    def test_003_mpi_bcast_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_bcast_using_schema")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # Note: example expects 2 mpi tasks

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        # send a node and its schema from rank 0 to rank 1
        n = conduit.Node()
        if comm_rank == 0:
            # setup node to broadcast on rank 0
            n["a/data"]   = 1.0
            n["a/more_data"] = 2.0
            n["a/b/my_string"] = "value"

        # show node data on rank 0
        if comm_rank == 0:
            print("[rank: {}] broadcasting: {}".format(comm_rank,n.to_yaml()))

        relay.mpi.broadcast_using_schema(n,root=0,comm=comm_id)

        # show received node data on rank 1
        if comm_rank == 1:
            print("[rank: {}] received: {}".format(comm_rank,n.to_yaml()))

        END_EXAMPLE("py_mpi_bcast_using_schema")

    def test_004_mpi_bcast_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_bcast")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # Note: example expects 2 mpi tasks

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        # send data from a node on rank 0 to rank 1
        # (both ranks have nodes with compatible schemas)
        n = conduit.Node(conduit.DataType.int64(4))
        if comm_rank == 0:
            # setup node to send on rank 0
            vals = n.value()
            for i in range(4):
                vals[i] = i * i

        # show node data on rank 0
        if comm_rank == 0:
            print("[rank: {}] broadcasting: {}".format(comm_rank,n.to_yaml()))

        relay.mpi.broadcast_using_schema(n,root=0,comm=comm_id)

        # show received node data on rank 1
        if comm_rank == 1:
            print("[rank: {}] received: {}".format(comm_rank,n.to_yaml()))

        END_EXAMPLE("py_mpi_bcast")

    def test_005_mpi_sum_all_reduce(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_sum_all_reduce")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        # gather data all ranks
        # (ranks have nodes with compatible schemas)
        n = conduit.Node(conduit.DataType.int64(4))
        n_res = conduit.Node(conduit.DataType.int64(4))
        # data to reduce
        vals = n.value()
        for i in range(4):
            vals[i] = 1

        relay.mpi.sum_all_reduce(n,n_res,comm=comm_id)
        # answer should be an array with each value == comm_size
        # show result on rank 0
        if comm_rank == 0:
            print("[rank: {}] sum reduce result: {}".format(comm_rank,n_res.to_yaml()))

        END_EXAMPLE("py_mpi_sum_all_reduce")

    def test_006_mpi_all_gather(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        BEGIN_EXAMPLE("py_mpi_all_gather_using_schema")
        import conduit
        import conduit.relay as relay
        import conduit.relay.mpi
        from mpi4py import MPI

        # get a comm id from mpi4py world comm
        comm_id   = MPI.COMM_WORLD.py2f()
        # get our rank and the comm's size
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)

        n = conduit.Node(conduit.DataType.int64(4))
        n_res = conduit.Node()
        # data to gather
        vals = n.value()
        for i in range(4):
            vals[i] = comm_rank

        relay.mpi.all_gather_using_schema(n,n_res,comm=comm_id)
        # show result on rank 0
        if comm_rank == 0:
            print("[rank: {}] all gather using schema result: {}".format(comm_rank,n_res.to_yaml()))

        END_EXAMPLE("py_mpi_all_gather_using_schema")
