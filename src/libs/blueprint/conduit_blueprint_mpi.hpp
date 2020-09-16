// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_HPP
#define CONDUIT_BLUEPRINT_MPI_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "conduit_blueprint_exports.h"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_blueprint_mpi_mesh_examples.hpp"


#include <mpi.h>


//-----------------------------------------------------------------------------
// -- begin conduit:: --
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
/// The about methods construct human readable info about how blueprint was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_BLUEPRINT_API about();
void        CONDUIT_BLUEPRINT_API about(conduit::Node &n);

//-----------------------------------------------------------------------------
/// blueprint verify interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint protocol.
/// Messages related to the verification are be placed in the "info" node.
//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const std::string &protocol,
                                  const conduit::Node &n,
                                  conduit::Node &info,
                                  MPI_Comm comm);

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif



