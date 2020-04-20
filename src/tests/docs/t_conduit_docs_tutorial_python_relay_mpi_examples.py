###############################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################
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
