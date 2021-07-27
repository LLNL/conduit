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

#include <parmetis.h>

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Initialize MPI and get rank and comm size
    MPI_Init(&argc, &argv);

    int par_rank, par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    //
    //   Simple example mesh
    //
    //   0 ---- 1 ---- 2
    //   |  e0  |  e1  |
    //   3 ---- 4 ---  5
    //   |  e2  |  e3  |
    //   6 ---- 7 ---- 8


    // elems across processors,
    // first rank has [0,1,2], second rank has [3]
    idx_t eldist[] = {0, 3, 4};

    /*
    The eptr and eind arrays are similar in nature to the xadj and adjncy
    arrays used to specify the adjacency list of a graph but now for each
    element they specify the set of nodes that make up each element.
    Specifically, the set of nodes that belong to element i is stored in
    array eind starting at index eptr[i] and ending at (but not including)
    index eptr[i + 1] (in other words, eind[eptr[i]] up through and including
    eind[eptr[i + 1]-1]). Hence, the node lists for each element are stored
    consecutively in the array eind.
    */

    // e0 vertices 0,1,3,4
    // e1 vertices 1,2,4,5
    // e2 vertices 3,4,6,7

    idx_t eind_rank_0[] = {0,1,3,4,
                           1,2,4,5,
                           3,4,6,7};

    idx_t eptr_rank_0[] = {0,4,8,12};

    // e3 vertices 4,5,7,8
    idx_t eind_rank_1[] = {4,5,7,8};
    idx_t eptr_rank_1[] = {0,4};

    idx_t wgtflag = 0; // weights are NULL
    idx_t numflag = 0; // C-style numbering
    idx_t ncon = 1; // the number of weights per vertex
    idx_t ncommonnodes = 4; // we have quads
    idx_t nparts = 2; //
    // equal weights for each proc
    real_t tpwgts[] = {0.5,0.5};
    real_t ubvec=1.050000;

    // options == extra output
    idx_t options[] = {1,
                       PARMETIS_DBGLVL_TIME |
                       PARMETIS_DBGLVL_INFO |
                       PARMETIS_DBGLVL_PROGRESS |
                       PARMETIS_DBGLVL_REFINEINFO |
                       PARMETIS_DBGLVL_MATCHINFO |
                       PARMETIS_DBGLVL_RMOVEINFO |
                       PARMETIS_DBGLVL_REMAP,
                       0};
    // outputs
    idx_t edgecut = 0; // will hold # of cut edges

    // each proc will have its local answer 
    // rank 0 has 3 eles to label
    idx_t part_rank_0[] = {10,10,10};

    // rank 1 has 1 ele to label
    idx_t part_rank_1[] = {20};

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int res = -1;

    // make sure everything is ok 
    if(par_rank == 0)
    {
        std::cout << "before:" << std::endl;
        std::cout << "part_rank_0: ";
        for(int i=0;i<3;i++)
        {
            std::cout << part_rank_0[i] << " ";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(par_rank == 1)
    {
        std::cout << std::endl;
        std::cout << "part_rank_1: ";
        for(int i=0;i<1;i++)
        {
            std::cout << part_rank_1[i] << " ";
        }
        std::cout << std::endl;
    }

    if(par_rank == 0)
    {
        res = ParMETIS_V3_PartMeshKway(eldist,
                                       eptr_rank_0,
                                       eind_rank_0,
                                       NULL,
                                       &wgtflag,
                                       &numflag,
                                       &ncon,
                                       &ncommonnodes,
                                       &nparts,
                                       tpwgts,
                                       &ubvec,
                                       options,
                                       &edgecut,
                                       part_rank_0,
                                       &mpi_comm);
    }
    else // rank == 1
    {
        res = ParMETIS_V3_PartMeshKway(eldist,
                                       eptr_rank_1,
                                       eind_rank_1,
                                       NULL,
                                       &wgtflag,
                                       &numflag,
                                       &ncon,
                                       &ncommonnodes,
                                       &nparts,
                                       tpwgts,
                                       &ubvec,
                                       options,
                                       &edgecut,
                                       part_rank_1,
                                       &mpi_comm);
    }

    // make sure everything is ok 
    if(res == METIS_ERROR)
    {
        std::cout <<  "METIS_ERROR!" << std::endl;
    }

    // print results
    if(par_rank == 0)
    {
        std::cout << "after:" << std::endl;
        std::cout << "part_rank_0: ";
        for(int i=0;i<3;i++)
        {
            std::cout << part_rank_0[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(par_rank == 1)
    {
        std::cout << "part_rank_1: ";
        for(int i=0;i<1;i++)
        {
            std::cout << part_rank_1[i] << " ";
        }
        std::cout << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    if(res == METIS_OK)
        return 0;
    else // (res == METIS_ERROR)
        return -1;
}

