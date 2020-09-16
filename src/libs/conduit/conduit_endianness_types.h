// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_endianness_types.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_ENDIANNESS_TYPES_H
#define CONDUIT_ENDIANNESS_TYPES_H

//-----------------------------------------------------------------------------
/// conduit_endianness_type_id is an Enumeration used to describe the  
/// endianness cases supported by conduit
//-----------------------------------------------------------------------------
typedef enum
{
    CONDUIT_ENDIANNESS_DEFAULT_ID  = 0, // (machine default)
    CONDUIT_ENDIANNESS_BIG_ID,          // big endian
    CONDUIT_ENDIANNESS_LITTLE_ID        // little endian
} conduit_endianness_type_id;


#endif

