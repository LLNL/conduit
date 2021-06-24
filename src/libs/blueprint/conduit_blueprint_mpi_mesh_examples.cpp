// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
    res.set(DataType::list());

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
