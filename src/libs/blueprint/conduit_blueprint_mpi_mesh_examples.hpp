// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_examples.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_EXAMPLES_HPP
#define CONDUIT_BLUEPRINT_MPI_MESH_EXAMPLES_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_exports.h"

#include <mpi.h>

//-----------------------------------------------------------------------------
// -- begin conduit::--
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
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
/// Methods that generate example meshes.
//-----------------------------------------------------------------------------
namespace examples
{
    /// Generates a uniform grid per MPI task using
    /// blueprint::mesh::examples::braid
    /// Adds an element-associated scalar field painted with the domain id
    void CONDUIT_BLUEPRINT_API braid_uniform_multi_domain(conduit::Node &res,
                                                          MPI_Comm comm);

    /// Generates a multi-domain fibonacci estimation of a golden spiral.
    /// Domains are assigned round-robin to MPI tasks
    void CONDUIT_BLUEPRINT_API spiral_round_robin(conduit::index_t ndomains,
                                                  conduit::Node &res,
                                                  MPI_Comm comm);
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



#endif



