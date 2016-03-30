//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://llnl.github.io/conduit/.
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
/// file: conduit_io.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_IO_HPP
#define CONDUIT_IO_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "Conduit_IO_Exports.hpp"
#include "Conduit_IO_Config.hpp"


#include "conduit_web.hpp"
#include "conduit_web_visualizer.hpp"

// include optional libs

#ifdef CONDUIT_IO_HDF5_ENABLED
#include "conduit_hdf5.hpp"
#endif

// include optional libs
#ifdef CONDUIT_IO_SILO_ENABLED
#include "conduit_silo.hpp"
#endif

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::io --
//-----------------------------------------------------------------------------
namespace io
{

///
/// ``save`` works like a 'set' to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save(Node &node,
                         const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save(const std::string &protocol,
                         Node &node,
                         const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save(const std::string &protocol,
                         Node &node,
                         const std::string &file_path,
                         const std::string &protocol_path);

///
/// ``save_merged`` works like an update to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save_merged(Node &node,
                                const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save_merged(const std::string &protocol,
                                Node &node,
                                const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save_merged(const std::string &protocol,
                                Node &node,
                                const std::string &file_path,
                                const std::string &protocol_path);

///
/// ``load`` works like a 'set', the node is reset and then populated
///

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load(const std::string &path,
                         Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load(const std::string &protocol,
                         const std::string &path,
                         Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load(const std::string &protocol,
                         const std::string &file_path,
                         const std::string &protocol_path,
                         Node &node);


///
/// ``load_merged`` works like an update, for the object case, entries are read
///  into the node. If the node is already in the OBJECT_T role, children are 
///  added
///

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load_merged(const std::string &path,
                                Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load_merged(const std::string &protocol,
                                const std::string &path,
                                Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API load_merged(const std::string &protocol,
                                const std::string &file_path,
                                const std::string &protocol_path,
                                Node &node);

//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how conduit_io was
/// configured.
//-----------------------------------------------------------------------------
 std::string CONDUIT_IO_API about();
 void        CONDUIT_IO_API about(Node &);


}
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------



}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif

