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
/// file: conduit_relay_mpi_io_adios.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_MPI_IO_ADIOS_HPP
#define CONDUIT_RELAY_MPI_IO_ADIOS_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios_mpi.h>

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

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

// Define argument macros that add a communicator argument.
#define CONDUIT_RELAY_COMMUNICATOR_ARG0(ARG) ARG
#define CONDUIT_RELAY_COMMUNICATOR_ARG(ARG) ,ARG

// Functions are provided by this include file.
#include "conduit_relay_io_adios_api.hpp"

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi::io --
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

