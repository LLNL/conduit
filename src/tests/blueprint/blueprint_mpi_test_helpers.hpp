// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: blueprint_mpi_test_helpers.hpp
///
//-----------------------------------------------------------------------------

#ifndef BLUEPRINT_MPI_TEST_HELPERS_HPP
#define BLUEPRINT_MPI_TEST_HELPERS_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include <conduit.hpp>
#include <conduit_node.hpp>
#include <conduit_blueprint_mesh_examples.hpp>
#include <conduit_blueprint_table.hpp>
#include <conduit_log.hpp>

#include <gtest/gtest.h>

#include <mpi.h>
//-----------------------------------------------------------------------------
template <typename Func>
void in_rank_order(MPI_Comm comm, Func &&func)
{
    int rank = 0, size = 1, buf = 0, tag = 11223344;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank > 0)
    {
        MPI_Status status;
        MPI_Recv(&buf, 1, MPI_INT, rank - 1, tag, comm, &status);
    }

    func(rank);

    if(rank < size - 1)
    {
        buf = rank;
        MPI_Send(&buf, 1, MPI_INT, rank + 1, tag, comm);
    }

    MPI_Barrier(comm);
}

#endif
