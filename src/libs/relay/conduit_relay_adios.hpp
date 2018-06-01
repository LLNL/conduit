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
/// file: conduit_relay_adios.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_ADIOS_HPP
#define CONDUIT_RELAY_ADIOS_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_relay_io.hpp"


//-----------------------------------------------------------------------------
/// The CONDUIT_CHECK_ADIOS_ERROR macro is used to check error codes from ADIOS.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_ADIOS_ERROR( adios_err, msg    )                \
{                                                                   \
    if( adios_err < 0 )                                              \
    {                                                               \
        std::ostringstream adios_err_oss;                            \
        adios_err_oss << "ADIOS Error code"                           \
            <<  adios_err                                            \
            << " " << msg;                                          \
        CONDUIT_ERROR( adios_err_oss.str());                         \
    }                                                               \
}

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
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

/* UPDATE THIS!!!! */
//-----------------------------------------------------------------------------
//
/// When writing a node to a ADIOS hierarchy there are two steps:
/// (1) Given a destination path, find (or create) the proper ADIOS parent group
/// (2) Write the node's data into the parent group
/// 
///
/// 1) To find (or create) the parent group there are two cases to finalize
///    the desired parent's ADIOS path:
///
///  A) If the input node is a leaf type, the last part of the path 
///     (the portion after the last slash '/', or the entire path if 
///     the path doesn't contain any slashes '/') is reserved for the
///     name of the ADIOS dataset that will be used to store the node's data. 
///     The rest of the path will be used to find or create the proper parent 
///     group.
/// 
///  B) If the input node is an Object, the full input path will be used to 
///     find or create the proper parent group.
///
///  Given the desired parent path: 
/// 
///  If the desired parent path exists in the ADIOS tree and resolves to 
///  a group, that group will be used as the parent.
///
///  If the desired parent path does not exist, or partially exists 
///  relay will attempt to create a hierarchy of ADIOS groups to represent it.
///
///  During this process, if any part of the path exists and is not a ADIOS 
///  group, an error is thrown and nothing will be modified.
///  (This error check happens before anything is modified in ADIOS)
///
///
/// 2) For writing the data, there are two cases:
///   A) If the input node is an Object, the children of the node will be
///      written to the parent group. If children correspond to existing 
///      ADIOS entires and they are incompatible (e.g. a group vs a dataset)
//       an error is thrown and nothing will be written.
///      (This error check happens before anything is modified in ADIOS)
///
///   B) If the input node is a leaf type, the last part of the path will be 
///      used as the name for a ADIOS dataset that will hold the node's data. 
///      If a child with this name already exists in in the parent group, and
///      it is not compatible (e.g. a group vs a dataset) an error is thrown 
///     and nothing will be written.
///
//-----------------------------------------------------------------------------
///  Note: ADIOS I/O is not implemented for Conduit Nodes in the List role.
///        We believe this will require a non-standard ADIOS convention, and
///        we want to focus on the Object and Leave cases since they are 
///        compatible with ADIOS's data model.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Write node data to a given path
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save(const Node &node,
                                  const std::string &path);

//-----------------------------------------------------------------------------
/// Write node data to a given path in an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_append(const Node &node,
                                    const std::string &path);

//-----------------------------------------------------------------------------
/// Read adios data from given path into the output node 
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  Node &node);

//-----------------------------------------------------------------------------
/// Pass a Node to set adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_set_options(const Node &opts);

//-----------------------------------------------------------------------------
/// Get a Node that contains adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_options(Node &opts);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
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

