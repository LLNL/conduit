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
/// file: conduit_datatype_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_datatype.h"

#include "conduit.hpp"
#include "conduit_cpp_to_c.hpp"

#include <stdlib.h>

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

int conduit_datatype_is_empty(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_empty() ? 1 : 0;
}

int conduit_datatype_is_object(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_object() ? 1 : 0;
}

int conduit_datatype_is_list(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_list() ? 1 : 0;
}

int conduit_datatype_is_number(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_number() ? 1 : 0;
}

int conduit_datatype_is_floating_point(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_floating_point() ? 1 : 0;
}

int conduit_datatype_is_integer(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_integer() ? 1 : 0;
}

int conduit_datatype_is_signed_integer(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_signed_integer() ? 1 : 0;
}

int conduit_datatype_is_unsigned_integer(const conduit_datatype *cdatatype)
   {
    return cpp_datatype_ref(cdatatype).is_unsigned_integer() ? 1 : 0;
}
 
int conduit_datatype_is_int8(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_int8() ? 1 : 0;
}

int conduit_datatype_is_int16(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_int16() ? 1 : 0;
}

int conduit_datatype_is_int32(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_int32() ? 1 : 0;
}

int conduit_datatype_is_int64(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_int64() ? 1 : 0;
}

int conduit_datatype_is_uint8(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_uint8() ? 1 : 0;
}

int conduit_datatype_is_uint16(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_uint16() ? 1 : 0;
}

int conduit_datatype_is_uint32(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_uint32() ? 1 : 0;
}

int conduit_datatype_is_uint64(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_uint64() ? 1 : 0;
}

int conduit_datatype_is_float32(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_float32() ? 1 : 0;
}

int conduit_datatype_is_float64(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_float64() ? 1 : 0;
}

// int conduit_datatype_is_index_t(const conduit_datatype /*cdatatype*/)
// {
// // NOTE: conduit::DataType::is_index_t() const does not seem to exist
// //    return cpp_datatype_ref(cdatatype).is_index_t() ? 1 : 0;
//     return 0;
// }

int conduit_datatype_is_char(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_char() ? 1 : 0;
}

int conduit_datatype_is_short(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_short() ? 1 : 0;
}

int conduit_datatype_is_int(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_int() ? 1 : 0;
}

int conduit_datatype_is_long(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_long() ? 1 : 0;
}
   
int conduit_datatype_is_unsigned_char(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_unsigned_char() ? 1 : 0;
}

int conduit_datatype_is_unsigned_short(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_unsigned_short() ? 1 : 0;
}

int conduit_datatype_is_unsigned_int(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_unsigned_int() ? 1 : 0;
}

int conduit_datatype_is_unsigned_long(const conduit_datatype *cdatatype)
 {
    return cpp_datatype_ref(cdatatype).is_unsigned_long() ? 1 : 0;
}

int conduit_datatype_is_float(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_float() ? 1 : 0;
}

int conduit_datatype_is_double(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_double() ? 1 : 0;
}

int conduit_datatype_is_string(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_string() ? 1 : 0;
}

int conduit_datatype_is_char8_str(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_char8_str() ? 1 : 0;
}

int conduit_datatype_is_little_endian(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_little_endian() ? 1 : 0;
}

int conduit_datatype_is_big_endian(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).is_big_endian() ? 1 : 0;
}

int conduit_datatype_endianness_matches_machine(const conduit_datatype *cdatatype)
{
    return cpp_datatype_ref(cdatatype).endianness_matches_machine() ? 1 : 0;
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

