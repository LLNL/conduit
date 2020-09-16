// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

