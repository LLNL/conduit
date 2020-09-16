// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_blueprint.hpp
///
//-----------------------------------------------------------------------------

// NOTE: This functionality is a placeholder for more general functionality in
// future versions of Blueprint. That said, the functions in this header are
// subject to change and could be moved with any future iteration of Conduit,
// so use this header with caution!

#ifndef CONDUIT_RELAY_IO_MPI_BLUEPRINT_HPP
#define CONDUIT_RELAY_IO_MPI_BLUEPRINT_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

#include <mpi.h>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi::io::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// Save a blueprint mesh to root + file set
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 int number_of_files,
                                 MPI_Comm comm);


//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const std::string &protocol,
                                 MPI_Comm comm);


void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const std::string &protocol,
                                 int number_of_files,
                                 MPI_Comm comm);


//-----------------------------------------------------------------------------
// Load a blueprint mesh from root + file set
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
                                 conduit::Node &mesh,
                                 MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Variant with explicit mesh name, for cases where bp data includes
/// more than one mesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
                                 const std::string &mesh_name,
                                 conduit::Node &mesh,
                                 MPI_Comm comm);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi::io::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi::io --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi::io_blueprint -- (DEPRECATED!)
//-----------------------------------------------------------------------------
namespace io_blueprint
{
    
//////////////////////////////////////////////////////////////////////////////
// DEPRECATED FUNCTIONS
//////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const conduit::Node &mesh,
                            const std::string &path,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const conduit::Node &mesh,
                            const std::string &path,
                            const std::string &protocol,
                            MPI_Comm comm);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi::io_blueprint --
//-----------------------------------------------------------------------------

}

//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
