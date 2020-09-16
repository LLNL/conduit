// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_endianness.hpp
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
#include "conduit_core.hpp"
#include "conduit_endianness_types.h"


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
///  *DEFAULT_ID - represents the current machine's endianness
///  *BIG_ID     - represents is big endian 
///  *LITTLE_ID  - represents little endian
//-----------------------------------------------------------------------------
    typedef enum
    {
        DEFAULT_ID = CONDUIT_ENDIANNESS_DEFAULT_ID, // default
        BIG_ID     = CONDUIT_ENDIANNESS_BIG_ID,
        LITTLE_ID  = CONDUIT_ENDIANNESS_LITTLE_ID
    } EndianEnum;

//-----------------------------------------------------------------------------
/// Returns the current machine's endianness: BIG_ID or LITTLE_ID
//-----------------------------------------------------------------------------
    static index_t          machine_default();
    
//-----------------------------------------------------------------------------
/// Convenience checks for machine endianness. 
//-----------------------------------------------------------------------------
    static bool             machine_is_little_endian();
    static bool             machine_is_big_endian();

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
    /// executes direct copy from src to dest. 
    /// src and dest must not be the same location.
    static void             swap16(void *src, void *dest);

    /// swaps for 32 bit types
    static void             swap32(void *data);
    /// executes direct copy from src to dest. 
    /// src and dest must not be the same location.
    static void             swap32(void *src, void *dest);

    /// swaps for 64 bit types    
    static void             swap64(void *data);
    /// executes direct copy from src to dest. 
    /// src and dest must not be the same location.
    static void             swap64(void *src, void *dest);

};
//-----------------------------------------------------------------------------
// -- end conduit::Endianness --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
