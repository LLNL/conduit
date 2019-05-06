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
/// file: conduit_log.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_log.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
// -- begin conduit::utils::log --
//-----------------------------------------------------------------------------
namespace log
{

//-----------------------------------------------------------------------------
void
info(Node &info,
     const std::string &proto_name,
     const std::string &msg)
{
    info["info"].append().set(proto_name + ": " + msg);
}

//-----------------------------------------------------------------------------
void
optional(Node &info,
         const std::string &proto_name,
         const std::string &msg)
{
    info["optional"].append().set(proto_name + ": " + msg);
}

//-----------------------------------------------------------------------------
void
error(Node &info,
      const std::string &proto_name,
      const std::string &msg)
{
    info["errors"].append().set(proto_name + ": " + msg);
}


//-----------------------------------------------------------------------------
void
validation(Node &info,
           bool res)
{
    // NOTE: if the given node already has a "valid" child, it's updated to
    // be the logical and of its current value and the new value so that
    // invalid nodes aren't changed to valid by later updates
    bool info_res = info.has_child("valid") ? info["valid"].as_string() == "true" : true;
    info["valid"].set(res && info_res ? "true" : "false");
}


//-----------------------------------------------------------------------------
std::string
quote(const std::string &str,
      bool pad_before)
{
    std::ostringstream oss;
    oss << (pad_before ? " " : "") << "'" << str << "'" << (pad_before ? "" : " ");
    return (str != "") ? oss.str() : "";
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::utils::log --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::utils --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

