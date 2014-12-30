//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: Endianness.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_ENDIANNESS_H
#define __CONDUIT_ENDIANNESS_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Core.h"

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <vector>
#include <sstream>

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
class Endianness
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
        DEFAULT_T = 0, // default
        BIG_T,
        LITTLE_T,
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
