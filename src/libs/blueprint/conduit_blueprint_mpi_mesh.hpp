// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_HPP
#define CONDUIT_BLUEPRINT_MPI_MESH_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"

#include <mpi.h>

//-----------------------------------------------------------------------------
// -- begin conduit --
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
/// blueprint protocol verify interface
//-----------------------------------------------------------------------------

// mesh verify
//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info,
                                  MPI_Comm comm);

//-----------------------------------------------------------------------------
/// blueprint mesh property and transform methods
/// 
/// These methods can be called on any verified blueprint mesh.
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_index(const conduit::Node &mesh,
                                          const std::string &ref_path,
                                          Node &index_out,
                                          MPI_Comm comm);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_domain_to_rank_map(
                                              const conduit::Node &mesh,
                                              Node &domain_to_rank_map,
                                              MPI_Comm comm);

//-------------------------------------------------------------------------
index_t CONDUIT_BLUEPRINT_API number_of_domains(const conduit::Node &mesh,
                                                MPI_Comm comm);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API partition(const conduit::Node &mesh,
                                     const conduit::Node &options,
                                     conduit::Node &output,
                                     MPI_Comm comm);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API flatten(const conduit::Node &mesh,
                                   const conduit::Node &options,
                                   conduit::Node &output,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------

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
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif 



