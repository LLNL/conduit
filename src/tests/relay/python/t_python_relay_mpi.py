# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
 file: t_python_relay_mpi.py
 description: Unit tests for relay mpi

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
import os
import math

import numpy as np

import conduit
import conduit.blueprint
import conduit.relay as relay
import conduit.relay.mpi

# TODO:
# from mpi4py import MPI

class Test_Relay_MPI_Module(unittest.TestCase):

    def test_about(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        print(relay.mpi.about())
        self.assertTrue(True)

    def test_rank_and_size(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id = MPI.COMM_WORLD.py2f()
        rank    = relay.mpi.rank(comm_id)
        size    = relay.mpi.size(comm_id)
        self.assertEqual(rank,MPI.COMM_WORLD.rank)
        self.assertEqual(size,MPI.COMM_WORLD.size)

    def test_send_recv_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        n = conduit.Node()
        comm_id = MPI.COMM_WORLD.py2f()
        rank    = relay.mpi.rank(comm_id)
        in_vals = np.zeros(3,dtype=np.float64)

        if rank == 0:
            in_vals[0] = rank + 1;
            in_vals[1] = 3.4124 * rank
            in_vals[2] = 10.7 - rank
            n.set(in_vals);
            relay.mpi.send_using_schema(n,1,0,comm_id);
        else:
            relay.mpi.recv_using_schema(n,0,0,comm_id);

        res_vals = n.value();
        self.assertEqual(res_vals[0], 1)
        self.assertEqual(res_vals[1], 0)
        self.assertEqual(res_vals[2], 10.7)
        n.reset()
        if rank == 0:
            n["value/a"] = 1;
            n["value/b"] = 2;
            relay.mpi.send_using_schema(n,1,0,comm_id);
        else:
            relay.mpi.recv_using_schema(n,0,0,comm_id);

        val_a = n["value/a"]
        val_b = n["value/b"]

        self.assertEqual(val_a, 1)
        self.assertEqual(val_b, 2)

    def test_send_recv_without_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        n = conduit.Node()
        comm_id = MPI.COMM_WORLD.py2f()
        rank    = relay.mpi.rank(comm_id)
        vals = np.zeros(3,dtype=np.float64)
        n.set_external(vals);
        print(rank)
        if rank == 0:
            vals[0] = rank + 1
            vals[1] = 3.4124 * rank
            vals[2] = 10.7 - rank
            relay.mpi.send(n,1,0,comm_id);
        else:
            relay.mpi.recv(n,0,0,comm_id);

        res_vals = n.value();
        self.assertEqual(res_vals[0], 1)
        self.assertEqual(res_vals[1], 0)
        self.assertEqual(res_vals[2], 10.7)

    def test_reduce_helpers(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        snd = conduit.Node(conduit.DataType.int64(5))
        rcv = conduit.Node(conduit.DataType.int64(5))

        snd_vals = snd.value()
        rcv_vals = rcv.value()

        # sum
        print("sum")
        for i in range(5):
            snd_vals[i] = 10
        relay.mpi.sum_reduce(snd, rcv, 0, comm_id);
        if comm_rank == 0:
            print(rcv_vals)
            for i in range(5):
                self.assertEqual(rcv_vals[i], 10 * comm_size)

        # prod
        print("prod")
        for i in range(5):
            snd_vals[i] = 2
        relay.mpi.prod_reduce(snd, rcv, 0, comm_id)
        if comm_rank == 0:
            print(rcv_vals)
            for i in range(5):
                self.assertEqual(rcv_vals[i], math.pow(comm_size,2) )

        # max
        print("max")
        for i in range(5):
            snd_vals[i] = comm_rank * 10 +1
        relay.mpi.max_reduce(snd, rcv, 0, comm_id)
        if comm_rank == 0:
            print(rcv_vals)
            for i in range(5):
                self.assertEqual(rcv_vals[i], 10 * (comm_size-1) + 1 )

        # min 
        print("min")
        for i in range(5):
            snd_vals[i] = comm_rank * 10 +1
        relay.mpi.min_reduce(snd, rcv, 0, comm_id)
        if comm_rank == 0:
            print(rcv_vals)
            for i in range(5):
                self.assertEqual(rcv_vals[i], 1)

    def test_all_reduce_helpers(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        snd = conduit.Node(conduit.DataType.int64(5))
        rcv = conduit.Node(conduit.DataType.int64(5))

        snd_vals = snd.value()
        rcv_vals = rcv.value()

        # sum
        print("sum")
        for i in range(5):
            snd_vals[i] = 10
        relay.mpi.sum_all_reduce(snd, rcv, comm_id);
        print(rcv_vals)
        for i in range(5):
            self.assertEqual(rcv_vals[i], 10 * comm_size)

        # prod
        print("prod")
        for i in range(5):
            snd_vals[i] = 2
        relay.mpi.prod_all_reduce(snd, rcv, comm_id)
        print(rcv_vals)
        for i in range(5):
            self.assertEqual(rcv_vals[i], math.pow(comm_size,2) )

        # max
        print("max")
        for i in range(5):
            snd_vals[i] = comm_rank * 10 +1
        relay.mpi.max_all_reduce(snd, rcv, comm_id)
        print(rcv_vals)
        for i in range(5):
            self.assertEqual(rcv_vals[i], 10 * (comm_size-1) + 1 )

        # min 
        print("min")
        for i in range(5):
            snd_vals[i] = comm_rank * 10 +1
        relay.mpi.min_all_reduce(snd, rcv, comm_id)
        rcv_vals = rcv.value()
        print(rcv_vals)
        for i in range(5):
            self.assertEqual(rcv_vals[i], 1)

    def test_gather_simple(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        n = conduit.Node()
        n["values/a"] = comm_rank+1;
        n["values/b"] = comm_rank+2;
        n["values/c"] = comm_rank+3;

        rcv = conduit.Node()
        relay.mpi.gather(n,rcv,0,comm_id)
        if comm_rank == 0:
            print(rcv)
            self.assertEqual(rcv[0]["values/a"],1)
            self.assertEqual(rcv[0]["values/b"],2)
            self.assertEqual(rcv[0]["values/c"],3)
            self.assertEqual(rcv[1]["values/a"],2)
            self.assertEqual(rcv[1]["values/b"],3)
            self.assertEqual(rcv[1]["values/c"],4)

    def test_all_gather_simple(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        n = conduit.Node()
        n["values/a"] = comm_rank+1;
        n["values/b"] = comm_rank+2;
        n["values/c"] = comm_rank+3;

        rcv = conduit.Node()
        relay.mpi.all_gather(n,rcv,comm_id)
        print(rcv)
        self.assertEqual(rcv[0]["values/a"],1)
        self.assertEqual(rcv[0]["values/b"],2)
        self.assertEqual(rcv[0]["values/c"],3)
        self.assertEqual(rcv[1]["values/a"],2)
        self.assertEqual(rcv[1]["values/b"],3)
        self.assertEqual(rcv[1]["values/c"],4)

    def test_gather_using_schema_simple(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        n = conduit.Node()
        n["values/a"] = comm_rank+1;
        n["values/b"] = comm_rank+2;
        n["values/c"] = comm_rank+3;

        rcv = conduit.Node()
        relay.mpi.gather_using_schema(n,rcv,0,comm_id)
        if comm_rank == 0:
            print(rcv)
            self.assertEqual(rcv[0]["values/a"],1)
            self.assertEqual(rcv[0]["values/b"],2)
            self.assertEqual(rcv[0]["values/c"],3)
            self.assertEqual(rcv[1]["values/a"],2)
            self.assertEqual(rcv[1]["values/b"],3)
            self.assertEqual(rcv[1]["values/c"],4)

    def test_all_gather_using_schema_simple(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        n = conduit.Node()
        n["values/a"] = comm_rank+1;
        n["values/b"] = comm_rank+2;
        n["values/c"] = comm_rank+3;

        rcv = conduit.Node()
        relay.mpi.all_gather_using_schema(n,rcv,comm_id)
        print(rcv)
        self.assertEqual(rcv[0]["values/a"],1)
        self.assertEqual(rcv[0]["values/b"],2)
        self.assertEqual(rcv[0]["values/c"],3)
        self.assertEqual(rcv[1]["values/a"],2)
        self.assertEqual(rcv[1]["values/b"],3)
        self.assertEqual(rcv[1]["values/c"],4)

    def test_bcast(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        
        for root in range(comm_size):
            n = conduit.Node()
            n.set(conduit.DataType.int64(3))
            vals = n.value()
            if comm_rank == root:
                vals[0] = 11
                vals[1] = 22
                vals[2] = 33
            relay.mpi.broadcast(n,root,comm_id)
            print(n)
            self.assertEqual(vals[0],11)
            self.assertEqual(vals[1],22)
            self.assertEqual(vals[2],33)

        for root in range(comm_size):
            n = conduit.Node()
            if comm_rank == root:
                n["a/b/c/d/e/f"].set(np.int64(10))
            else:
                n["a/b/c/d/e/f"].set(np.int64(0))
            relay.mpi.broadcast(n,root,comm_id)
            val = n["a/b/c/d/e/f"]
            self.assertEqual(val,10)

    def test_bcast_using_schema(self):
        # skip tests on windows until we work out proper
        # mpi4py install for our windows ci
        if sys.platform == "win32":
            return
        from mpi4py import MPI
        comm_id   = MPI.COMM_WORLD.py2f()
        comm_rank = relay.mpi.rank(comm_id)
        comm_size = relay.mpi.size(comm_id)
        
        for root in range(comm_size):
            n = conduit.Node()
            if comm_rank == root:
                n.set(conduit.DataType.int64(3))
                vals = n.value()
                vals[0] = 11
                vals[1] = 22
                vals[2] = 33
            relay.mpi.broadcast_using_schema(n,root,comm_id)
            print(n)
            vals = n.value()
            self.assertEqual(vals[0],11)
            self.assertEqual(vals[1],22)
            self.assertEqual(vals[2],33)

        for root in range(comm_size):
            n = conduit.Node()
            if comm_rank == root:
                n["a/b/c/d/e/f"].set(np.int64(10))
            relay.mpi.broadcast_using_schema(n,root,comm_id)
            val = n["a/b/c/d/e/f"]
            self.assertEqual(val,10)


if __name__ == '__main__':
    unittest.main()



