//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_adios_api.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_ADIOS_API_HPP
#define CONDUIT_RELAY_ADIOS_API_HPP

//-----------------------------------------------------------------------------
/// Write node data to a given path
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save(const Node &node,
                                  const std::string &path
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm));

//-----------------------------------------------------------------------------
/// Write node data to a given path in an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_append(const Node &node,
                                    const std::string &path
                                    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm));

//-----------------------------------------------------------------------------
/// Read adios data from given path into the output node 
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  Node &node
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm));

//-----------------------------------------------------------------------------
/// Read a given time step and domain of adios data from given path into the
//  output node.
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  int time_step,
                                  int domain,
                                  Node &node
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm));

//-----------------------------------------------------------------------------
/// Pass a Node to set adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_set_options(const Node &opts);

//-----------------------------------------------------------------------------
/// Get a Node that contains adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_options(Node &opts);

#endif

