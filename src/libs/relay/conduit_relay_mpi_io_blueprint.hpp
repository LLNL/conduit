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
                                 const std::string &protocol,
                                 MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const std::string &protocol,
                                 const conduit::Node &opts,
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
                                 const conduit::Node &opts,
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
