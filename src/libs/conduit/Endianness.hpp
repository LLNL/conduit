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
// For details, see: http://scalability-llnl.github.io/conduit/.
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
/// file: Endianness.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_ENDIANNESS_HPP
#define CONDUIT_ENDIANNESS_HPP

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <vector>
#include <sstream>

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "Endianness_Types.h"

#include "Core.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::Endianness --
//-----------------------------------------------------------------------------
///
/// class: conduit::Endianness
///
/// description:
///  Class for endian info and conversation. 
///
//-----------------------------------------------------------------------------
class CONDUIT_API Endianness
{
public:
//-----------------------------------------------------------------------------
//
// -- conduit::Endianness public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// EndianEnum is an Enumeration used to hold endian states:
///  *DEFAULT_T - represents the current machine's endianness
///  *BIG_T     - represents is big endian 
///  *LITTLE_T  - represents little endian
//-----------------------------------------------------------------------------
    typedef enum
    {
        DEFAULT_T = CONDUIT_ENDIANNESS_DEFAULT_T, // default
        BIG_T     = CONDUIT_ENDIANNESS_BIG_T,
        LITTLE_T  = CONDUIT_ENDIANNESS_LITTLE_T,
    } EndianEnum;

//-----------------------------------------------------------------------------
/// Returns the current machine's endianness: BIG_T or LITTLE_T
//-----------------------------------------------------------------------------
    static index_t          machine_default();
//-----------------------------------------------------------------------------
/// Convert human readable string {big|little|default} to an EndianEnum id.
//-----------------------------------------------------------------------------
    static index_t          name_to_id(const std::string &name);
//-----------------------------------------------------------------------------
/// Converts an EndianEnum id to a human readable string.
//-----------------------------------------------------------------------------
    static std::string      id_to_name(index_t endianness);
    
//-----------------------------------------------------------------------------
/// Helpers for endianness transforms
//-----------------------------------------------------------------------------
    /// swaps for 16 bit types
    static void             swap16(void *data);
    static void             swap16(void *src,void *dest);

    /// swaps for 32 bit types
    static void             swap32(void *data);
    static void             swap32(void *src,void *dest);

    /// swaps for 64 bit types    
    static void             swap64(void *data);
    static void             swap64(void *src,void *dest);

};
//-----------------------------------------------------------------------------
// -- end conduit::Endianness --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
