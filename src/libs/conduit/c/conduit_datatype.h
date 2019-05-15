//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
