// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_datatype.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_DATATYPE_H
#define CONDUIT_DATATYPE_H

#include <stdlib.h>
#include <stddef.h>

#include "conduit_bitwidth_style_types.h"
#include "conduit_endianness_types.h"
#include "conduit_exports.h"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- typedef for conduit_datatype --
//-----------------------------------------------------------------------------

typedef void  conduit_datatype;

CONDUIT_API int conduit_datatype_is_empty(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_object(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_list(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_number(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_floating_point(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_integer(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_signed_integer(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_unsigned_integer(const conduit_datatype *cdatatype);
    
CONDUIT_API int conduit_datatype_is_int8(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_int16(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_int32(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_int64(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_uint8(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_uint16(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_uint32(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_uint64(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_float32(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_float64(const conduit_datatype *cdatatype);

// skipping b/c we don't provide this yet, unsure if we will
// CONDUIT_API int conduit_datatype_is_index_t(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_char(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_short(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_int(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_long(const conduit_datatype *cdatatype);
    
CONDUIT_API int conduit_datatype_is_unsigned_char(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_unsigned_short(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_unsigned_int(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_unsigned_long(const conduit_datatype *cdatatype);
 
CONDUIT_API int conduit_datatype_is_float(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_double(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_string(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_char8_str(const conduit_datatype *cdatatype);

CONDUIT_API int conduit_datatype_is_little_endian(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_is_big_endian(const conduit_datatype *cdatatype);
CONDUIT_API int conduit_datatype_endianness_matches_machine(const conduit_datatype *cdatatype);

#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


#endif
