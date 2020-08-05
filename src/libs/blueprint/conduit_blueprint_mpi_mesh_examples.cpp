//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-666778
//
// All rights reserved.
//
// This file is part of Conduit.
//
// For details, see: http://software.llnl.gov/conduit/.
//
// Please also read conduit/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_examples.cpp
///
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mpi_mesh_examples.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_blueprint_mesh_examples.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::examples --
//-----------------------------------------------------------------------------
namespace examples
{

//---------------------------------------------------------------------------//
void
braid_uniform_multi_domain(Node &res, MPI_Comm comm)
{
    int par_rank = relay::mpi::rank(comm);

    index_t npts_x = 10;
    index_t npts_y = 10;
    index_t npts_z = 10;

    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      res);

    // the example data set has the bounds -10 to 10 in all dims
    // Offset this along x to create mpi 'pencil'

    res["coordsets/coords/origin/x"] = -10.0 + 20.0 * par_rank;
    res["state/domain_id"] = par_rank;
    // set cycle to 0, so we can construct the correct root file
    res["state/cycle"] = 0;
    
    // add field with domain_id
    // radial is element centered as well, borrow details
    res["fields/rank"].set(res["fields/radial"]);

    float64_array rank_vals = res["fields/rank/values"].value();

    for(index_t i=0; i < rank_vals.number_of_elements(); i++)
    {
        rank_vals[i] = (float64) par_rank;
    }
}


//---------------------------------------------------------------------------//
void
spiral_round_robin(conduit::index_t ndomains,
                   conduit::Node &res,
                   MPI_Comm comm)
{
    res.reset();

    int par_rank = relay::mpi::rank(comm);
    int par_size = relay::mpi::size(comm);

    /// TODO: We gen full on all ranks, not ideal but prob ok for example
    Node dset;
    blueprint::mesh::examples::spiral(ndomains,dset);

    index_t dom_rank = 0;
    // pick out local doms or this rank
    for(index_t i=0; i < ndomains; i++)
    {
        if(dom_rank == par_rank)
        {
            res.append().set(dset.child(i));
        }

        dom_rank++;
        if(dom_rank >= par_size)
        {
            dom_rank = 0;
        }
    }

    NodeIterator itr = res.children();
    while(itr.has_next())
    {
        Node &dom = itr.next();
        // also add field with domain_id
        // dist is vertex centered as well, borrow details
        dom["fields/rank"].set(dom["fields/dist"]);
        float64_array rank_vals = dom["fields/rank/values"].value();
        for(index_t i=0; i < rank_vals.number_of_elements(); i++)
        {
            rank_vals[i] = (float64) par_rank;
        }
    }
    
    std::cout << " HERE!" << std::endl;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::examples --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit --
//-----------------------------------------------------------------------------
