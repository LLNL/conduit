// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_parmetis_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <mpi.h>
#include <iostream>
#include <fstream>

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Initialize MPI and get rank and comm size
    MPI_Init(&argc, &argv);

    int par_rank, par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
