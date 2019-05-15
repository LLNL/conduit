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
/// file: conduit_endianness.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_endianness.hpp"

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

//---------------------------------------------------------------------------//
index_t
Endianness::machine_default()
{
    union{uint8  vbyte; uint32 vuint;} test;
    test.vuint = 1;
    if(test.vbyte ^ 1)
        return BIG_ID;
    else
        return LITTLE_ID;
}

//---------------------------------------------------------------------------//
bool
Endianness::machine_is_little_endian()
{
    return machine_default() == LITTLE_ID;
}

//---------------------------------------------------------------------------//
bool
Endianness::machine_is_big_endian()
{
    return machine_default() == BIG_ID;
}


//-----------------------------------------------------------------------------
/// Enum id to string and string to enum id helpers.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t
Endianness::name_to_id(const std::string &name)
{
    if(name == "big")
        return BIG_ID;
    else if(name =="little")
        return LITTLE_ID;
    return DEFAULT_ID;

}

//---------------------------------------------------------------------------//
std::string      
Endianness::id_to_name(index_t endianness)
{
    std::string res = "default";
    if(endianness == BIG_ID) 
        res = "big";
    else if(endianness == LITTLE_ID) 
        res = "little";
    return res;
};

//-----------------------------------------------------------------------------
/// Helpers for endianness transforms
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Endianness::swap16(void *data)
{
    uint8 tmp = *((uint8*)data);
    ((uint8*)data)[0] = ((uint8*)data)[1];
    ((uint8*)data)[1] = tmp;
}

//---------------------------------------------------------------------------//
void             
Endianness::swap16(void *src,void *dest)
{
    ((uint8*)dest)[0] = ((uint8*)src)[1];
    ((uint8*)dest)[1] = ((uint8*)src)[0];
}

//---------------------------------------------------------------------------//
void
Endianness::swap32(void *data)
{
    union{uint8 vbytes[4]; uint32 vdata;} swp;
    
    swp.vbytes[3] = ((uint8*)data)[0];
    swp.vbytes[2] = ((uint8*)data)[1];
    swp.vbytes[1] = ((uint8*)data)[2];
    swp.vbytes[0] = ((uint8*)data)[3];

    *((uint32*)data) = swp.vdata;
}

//---------------------------------------------------------------------------//
void             
Endianness::swap32(void *src,void *dest)
{
    ((uint8*)dest)[0] = ((uint8*)src)[3];
    ((uint8*)dest)[1] = ((uint8*)src)[2];
    ((uint8*)dest)[2] = ((uint8*)src)[1];
    ((uint8*)dest)[3] = ((uint8*)src)[0];
}

//---------------------------------------------------------------------------//
void
Endianness::swap64(void *data)
{
    union{uint8 vbytes[8]; uint64 vdata;} swp;
    
    swp.vbytes[7] = ((uint8*)data)[0];
    swp.vbytes[6] = ((uint8*)data)[1];
    swp.vbytes[5] = ((uint8*)data)[2];
    swp.vbytes[4] = ((uint8*)data)[3];
    swp.vbytes[3] = ((uint8*)data)[4];
    swp.vbytes[2] = ((uint8*)data)[5];
    swp.vbytes[1] = ((uint8*)data)[6];
    swp.vbytes[0] = ((uint8*)data)[7];

    *((uint64*)data) = swp.vdata;
}

//---------------------------------------------------------------------------//
void
Endianness::swap64(void *src,void *dest)
{
    ((uint8*)dest)[0] = ((uint8*)src)[7];
    ((uint8*)dest)[1] = ((uint8*)src)[6];
    ((uint8*)dest)[2] = ((uint8*)src)[5];
    ((uint8*)dest)[3] = ((uint8*)src)[4];
    ((uint8*)dest)[4] = ((uint8*)src)[3];
    ((uint8*)dest)[5] = ((uint8*)src)[2];
    ((uint8*)dest)[6] = ((uint8*)src)[1];
    ((uint8*)dest)[7] = ((uint8*)src)[0];

}


}
//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------

