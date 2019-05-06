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
/// file: conduit_core.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_CORE_HPP
#define CONDUIT_CORE_HPP

//-----------------------------------------------------------------------------
// -- standard cpp lib includes -- 
//-----------------------------------------------------------------------------
#include <string>
#include <iostream>

//-----------------------------------------------------------------------------
// -- configure time defines -- 
//-----------------------------------------------------------------------------
#include "conduit_config.h"

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms -- 
//-----------------------------------------------------------------------------
#include "conduit_exports.h"

//-----------------------------------------------------------------------------
// -- include bit width style types mapping header  -- 
//-----------------------------------------------------------------------------
#include "conduit_bitwidth_style_types.h"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

class Node;

//-----------------------------------------------------------------------------
/// typedefs that map bit width style types into conduit::
//-----------------------------------------------------------------------------

/// unsigned integer typedefs
typedef conduit_uint8   uint8;
typedef conduit_uint16  uint16;
typedef conduit_uint32  uint32;
typedef conduit_uint64  uint64;

/// signed integer typedefs
typedef conduit_int8    int8;
typedef conduit_int16   int16;
typedef conduit_int32   int32;
typedef conduit_int64   int64;

/// floating point typedefs
typedef conduit_float32 float32;
typedef conduit_float64 float64;

/// index typedefs
typedef conduit_index32_t index32_t;
typedef conduit_index64_t index64_t;
// conduit_index_t is defined in Bitwidth_Style_Types.h
// it will be index64_t, unless CONDUIT_INDEX_32 is defined
typedef conduit_index_t   index_t;

//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how conduit was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_API about();
void        CONDUIT_API about(Node &);

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif

