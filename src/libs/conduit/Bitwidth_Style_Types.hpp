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
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: Bitwidth_Style_Types.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BITWIDTH_STYLE_TYPES_HPP
#define CONDUIT_BITWIDTH_STYLE_TYPES_HPP

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <limits.h>

//-----------------------------------------------------------------------------
// -- include sizes from cmake configure tests -- 
//-----------------------------------------------------------------------------
#include "Conduit_Config.hpp"

//-----------------------------------------------------------------------------
/// Bit width annotated Style Standard Data Types
/// Derived from numpy (which provides very comprehensive support 
/// for these types)
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// OSX Specific Sizes:
//-----------------------------------------------------------------------------
// On Mac OS X, because there is only one configuration stage for all the 
// architectures in universal builds, any macro which depends on the 
// architecture needs to be hardcoded.
//-----------------------------------------------------------------------------
#ifdef __APPLE__
    #undef SIZEOF_LONG
    #ifdef __LP64__
        #define SIZEOF_LONG      8
    #else
        #define SIZEOF_LONG      4
    #endif
#endif

//-----------------------------------------------------------------------------
// native types
//-----------------------------------------------------------------------------

typedef signed char         conduit_byte;
typedef unsigned char       conduit_ubyte;
typedef unsigned short      conduit_ushort;
typedef unsigned int        conduit_uint;
typedef unsigned long       conduit_ulong;

#ifdef CONDUIT_HAS_LONG_LONG
typedef unsigned long long  conduit_ulong_long;
#endif

typedef char                conduit_char;
typedef short               conduit_short;
typedef int                 conduit_int;
typedef long                conduit_long;

#ifdef CONDUIT_HAS_LONG_LONG
typedef long long           conduit_long_long;
#endif

typedef float               conduit_float;
typedef double              conduit_double;

#if CONDUIT_SIZEOF_LONG_DOUBLE == CONDUIT_SIZEOF_DOUBLE
        typedef double conduit_long_double;
#else
#ifdef CONDUIT_HAS_LONG_DOUBLE
        typedef long double conduit_long_double;
#endif
#endif

//-----------------------------------------------------------------------------
/// conduit_datatype_type_id is an Enumeration used to describe the type 
/// roles supported by conduit:
//-----------------------------------------------------------------------------
typedef enum
{
    CONDUIT_EMPTY_T = 0, // empty (default type)
    CONDUIT_OBJECT_T,    // object
    CONDUIT_LIST_T,      // list
    CONDUIT_INT8_T,      // int8 and int8_array
    CONDUIT_INT16_T,     // int16 and int16_array
    CONDUIT_INT32_T,     // int32 and int32_array
    CONDUIT_INT64_T,     // int64 and int64_array
    CONDUIT_UINT8_T,     // int8 and int8_array
    CONDUIT_UINT16_T,    // uint16 and uint16_array
    CONDUIT_UINT32_T,    // uint32 and uint32_array
    CONDUIT_UINT64_T,    // uint64 and uint64_array
    CONDUIT_FLOAT32_T,   // float32 and float32_array
    CONDUIT_FLOAT64_T,   // float64 and float64_array
    CONDUIT_CHAR8_STR_T, // char8 string (incore c-string)
} conduit_datatype_type_id;
    
//-----------------------------------------------------------------------------
// bytes to bits size definitions
//-----------------------------------------------------------------------------
#define CONDUIT_BITSOF_CHAR        CHAR_BIT

#define CONDUIT_BITSOF_BYTE        (CONDUIT_SIZEOF_BYTE * CHAR_BIT)
#define CONDUIT_BITSOF_SHORT       (CONDUIT_SIZEOF_SHORT * CHAR_BIT)
#define CONDUIT_BITSOF_INT         (CONDUIT_SIZEOF_INT * CHAR_BIT)
#define CONDUIT_BITSOF_LONG        (CONDUIT_SIZEOF_LONG * CHAR_BIT)

#ifdef CONDUIT_HAS_LONG_LONG
#define CONDUIT_BITSOF_LONG_LONG   (CONDUIT_SIZEOF_LONG_LONG * CHAR_BIT)
#endif

#define CONDUIT_BITSOF_FLOAT       (CONDUIT_SIZEOF_FLOAT * CHAR_BIT)
#define CONDUIT_BITSOF_DOUBLE      (CONDUIT_SIZEOF_DOUBLE * CHAR_BIT)

#ifdef CONDUIT_HAS_LONG_DOUBLE
#define CONDUIT_BITSOF_LONG_DOUBLE (CONDUIT_SIZEOF_LONG_DOUBLE * CHAR_BIT)
#endif

#define CONDUIT_BITSOF_VOID_P      (CONDUIT_SIZEOF_VOID_P * CHAR_BIT)

//-----------------------------------------------------------------------------
// -- long size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_LONG == 8
    #define CONDUIT_INT8 CONDUIT_LONG
    #define CONDUIT_UINT8 CONDUIT_ULONG
    #define CONDUIT_INT8_NATIVE_TYPENAME "long"
    #define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned long"
    typedef conduit_long  conduit_int8;
    typedef conduit_ulong conduit_uint8;
    // native mapping defs for long
    #define CONDUIT_NATIVE_LONG conduit_int8
    #define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint8
    #define CONDUIT_NATIVE_LONG_DATATYPE_ID CONDUIT_INT8_T
    #define CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID CONDUIT_UINT8_T
#elif CONDUIT_BITSOF_LONG == 16
    #define CONDUIT_INT16 CONDUIT_LONG
    #define CONDUIT_UINT16 CONDUIT_ULONG
    #define CONDUIT_INT16_NATIVE_TYPENAME "long"
    #define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned long"
    typedef conduit_long  conduit_int16;
    typedef conduit_ulong conduit_uint16;
    // native mapping defs for long
    #define CONDUIT_NATIVE_LONG conduit_int16
    #define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint16
    #define CONDUIT_NATIVE_LONG_DATATYPE_ID CONDUIT_INT16_T
    #define CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID CONDUIT_UINT16_T
#elif CONDUIT_BITSOF_LONG == 32
    #define CONDUIT_INT32 CONDUIT_LONG
    #define CONDUIT_UINT32 CONDUIT_ULONG
    #define CONDUIT_INT32_NATIVE_TYPENAME "long"
    #define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned long"
    typedef conduit_long  conduit_int32;
    typedef conduit_ulong conduit_uint32;
    // native mapping defs for long
    #define CONDUIT_NATIVE_LONG conduit_int32
    #define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint32
    #define CONDUIT_NATIVE_LONG_DATATYPE_ID CONDUIT_INT32_T
    #define CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID CONDUIT_UINT32_T
#elif CONDUIT_BITSOF_LONG == 64
    #define CONDUIT_INT64 CONDUIT_LONG
    #define CONDUIT_UINT64 CONDUIT_ULONG
    #define CONDUIT_INT64_NATIVE_TYPENAME "long"
    #define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned long"
    typedef conduit_long  conduit_int64;
    typedef conduit_ulong conduit_uint64;
    // native mapping defs for long
    #define CONDUIT_NATIVE_LONG conduit_int64
    #define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint64
    #define CONDUIT_NATIVE_LONG_DATATYPE_ID CONDUIT_INT64_T
    #define CONDUIT_NATIVE_UNSIGNED_LONG_DATATYPE_ID CONDUIT_UINT64_T
#endif

//-----------------------------------------------------------------------------
// -- long long size checks --
//-----------------------------------------------------------------------------
#ifdef CONDUIT_HAS_LONG_LONG
#if CONDUIT_BITSOF_LONG_LONG == 8
    #ifndef CONDUIT_INT8
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_INT8 CONDUIT_LONG_LONG
        #define CONDUIT_UINT8 CONDUIT_ULONG_LONG
        #define CONDUIT_INT8_NATIVE_TYPENAME "long long"
        #define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned long long"
        typedef conduit_long_long conduit_int8;
        typedef conduit_ulong_long conduit_uint8;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_LONG
        #define CONDUIT_NATIVE_LONG_LONG conduit_int8
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint8
        #define CONDUIT_NATIVE_LONG_LONG_DATATYPE_ID CONDUIT_INT8_T
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG_DATATYPE_ID CONDUIT_UINT8_T
    #endif
#elif CONDUIT_BITSOF_LONG_LONG == 16
    #ifndef CONDUIT_INT16
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_INT16 CONDUIT_LONG_LONG
        #define CONDUIT_UINT16 CONDUIT_ULONG_LONG
        #define CONDUIT_INT16_NATIVE_TYPENAME "long long"
        #define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned long long"
        typedef conduit_long_long conduit_int16;
        typedef conduit_ulong_long conduit_uint16;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_LONG
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_NATIVE_LONG_LONG conduit_int16
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint16
        #define CONDUIT_NATIVE_LONG_LONG_DATATYPE_ID CONDUIT_INT16_T
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG_DATATYPE_ID CONDUIT_UINT16_T
    #endif
#elif CONDUIT_BITSOF_LONG_LONG == 32
    #ifndef CONDUIT_INT32
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_INT32 CONDUIT_LONG_LONG
        #define CONDUIT_UINT32 CONDUIT_ULONG_LONG
        #define CONDUIT_INT32_NATIVE_TYPENAME "long long"
        #define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned long long"
        typedef conduit_long_long conduit_int32;
        typedef conduit_ulong_long conduit_uint32;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_LONG
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_NATIVE_LONG_LONG conduit_int32
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint32
        #define CONDUIT_NATIVE_LONG_LONG_DATATYPE_ID CONDUIT_INT32_T
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG_DATATYPE_ID CONDUIT_UINT32_T
    #endif
#elif CONDUIT_BITSOF_LONG_LONG == 64
    #ifndef CONDUIT_INT64
        #define CONDUIT_USE_LONG_LONG
        #define CONDUIT_INT64 CONDUIT_LONG_LONG
        #define CONDUIT_UINT64 CONDUIT_ULONG_LONG
        #define CONDUIT_INT64_NATIVE_TYPENAME "long long"
        #define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned long long"
        typedef conduit_long_long conduit_int64;
        typedef conduit_ulong_long conduit_uint64;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_LONG
        #define CONDUIT_NATIVE_LONG_LONG conduit_int64
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint64
        #define CONDUIT_NATIVE_LONG_LONG_DATATYPE_ID CONDUIT_INT64_T
        #define CONDUIT_NATIVE_UNSIGNED_LONG_LONG_DATATYPE_ID CONDUIT_UINT64_T
    #endif
#endif
#endif 

//-----------------------------------------------------------------------------
// -- int size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_INT == 8
    #ifndef CONDUIT_INT8
        #define CONDUIT_INT8 CONDUIT_INT
        #define CONDUIT_UINT8 CONDUIT_UINT
        #define CONDUIT_INT8_NATIVE_TYPENAME "int"
        #define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned int"
        typedef conduit_int  conduit_int8;
        typedef conduit_uint conduit_uint8;
    #endif
    #ifndef CONDUIT_NATIVE_INT
        #define CONDUIT_NATIVE_INT conduit_int8
        #define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint8
        #define CONDUIT_NATIVE_INT_DATATYPE_ID CONDUIT_INT8_T
        #define CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID CONDUIT_UINT8_T
    #endif
#elif CONDUIT_BITSOF_INT == 16
    #ifndef CONDUIT_INT16
        #define CONDUIT_INT16 CONDUIT_INT
        #define CONDUIT_UINT16 CONDUIT_UINT
        #define CONDUIT_INT16_NATIVE_TYPENAME "int"
        #define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned int"
        typedef conduit_int  conduit_int16;
        typedef conduit_uint conduit_uint16;
    #endif
    #ifndef CONDUIT_NATIVE_INT
        #define CONDUIT_NATIVE_INT conduit_int16
        #define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint16
        #define CONDUIT_NATIVE_INT_DATATYPE_ID CONDUIT_INT16_T
        #define CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID CONDUIT_UINT16_T
    #endif

#elif CONDUIT_BITSOF_INT == 32
    #ifndef CONDUIT_INT32
        #define CONDUIT_INT32 CONDUIT_INT
        #define CONDUIT_UINT32 CONDUIT_UINT
        #define CONDUIT_INT32_NATIVE_TYPENAME "int"
        #define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned int"
        typedef conduit_int  conduit_int32;
        typedef conduit_uint conduit_uint32;
    #endif
    #ifndef CONDUIT_NATIVE_INT
        #define CONDUIT_NATIVE_INT conduit_int32
        #define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint32
        #define CONDUIT_NATIVE_INT_DATATYPE_ID CONDUIT_INT32_T
        #define CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID CONDUIT_UINT32_T
    #endif
                
#elif CONDUIT_BITSOF_INT == 64
    #ifndef CONDUIT_INT64
        #define CONDUIT_INT64 CONDUIT_INT
        #define CONDUIT_UINT64 CONDUIT_UINT
        #define CONDUIT_INT64_NATIVE_TYPENAME "int"
        #define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned int"
        typedef conduit_int  conduit_int64;
        typedef conduit_uint conduit_uint64;
    #endif
    #ifndef CONDUIT_NATIVE_INT
        #define CONDUIT_NATIVE_INT conduit_int64
        #define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint64
        #define CONDUIT_NATIVE_INT_DATATYPE_ID CONDUIT_INT64_T
        #define CONDUIT_NATIVE_UNSIGNED_INT_DATATYPE_ID CONDUIT_UINT64_T
    #endif
#endif

//-----------------------------------------------------------------------------
// -- short size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_SHORT == 8
    #ifndef CONDUIT_INT8
        #define CONDUIT_INT8 CONDUIT_SHORT
        #define CONDUIT_UINT8 CONDUIT_USHORT
        #define CONDUIT_INT8_NATIVE_TYPENAME "short"
        #define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned short"
        typedef conduit_short  conduit_int8;
        typedef conduit_ushort conduit_uint8;
    #endif
    #ifndef CONDUIT_NATIVE_SHORT
        #define CONDUIT_NATIVE_SHORT conduit_int8
        #define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint8
        #define CONDUIT_NATIVE_SHORT_DATATYPE_ID CONDUIT_INT8_T
        #define CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID CONDUIT_UINT8_T
    #endif
#elif CONDUIT_BITSOF_SHORT == 16
    #ifndef CONDUIT_INT16
        #define CONDUIT_INT16 CONDUIT_SHORT
        #define CONDUIT_UINT16 CONDUIT_USHORT
        #define CONDUIT_INT16_NATIVE_TYPENAME "short"
        #define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned short"
        typedef conduit_short  conduit_int16;
        typedef conduit_ushort conduit_uint16;
    #endif
    #ifndef CONDUIT_NATIVE_SHORT
        #define CONDUIT_NATIVE_SHORT conduit_int16
        #define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint16
        #define CONDUIT_NATIVE_SHORT_DATATYPE_ID CONDUIT_INT16_T
        #define CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID CONDUIT_UINT16_T
    #endif
#elif CONDUIT_BITSOF_SHORT == 32
    #ifndef CONDUIT_INT32
        #define CONDUIT_INT32 CONDUIT_SHORT
        #define CONDUIT_UINT32 CONDUIT_USHORT
        #define CONDUIT_INT32_NATIVE_TYPENAME "short"
        #define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned short"
        typedef conduit_short  conduit_int32;
        typedef conduit_ushort conduit_uint32;
    #endif
    #ifndef CONDUIT_NATIVE_SHORT
        #define CONDUIT_NATIVE_SHORT conduit_int32
        #define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint32
        #define CONDUIT_NATIVE_SHORT_DATATYPE_ID CONDUIT_INT32_T
        #define CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID CONDUIT_UINT32_T
    #endif
#elif CONDUIT_BITSOF_SHORT == 64
    #ifndef CONDUIT_INT64
        #define CONDUIT_INT64 CONDUIT_SHORT
        #define CONDUIT_UINT64 CONDUIT_USHORT
        #define CONDUIT_INT64_NATIVE_TYPENAME "short"
        #define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned short"
        typedef conduit_short  conduit_int64;
        typedef conduit_ushort conduit_uint64;
    #endif
    #ifndef CONDUIT_NATIVE_SHORT
        #define CONDUIT_NATIVE_SHORT conduit_int64
        #define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint64
        #define CONDUIT_NATIVE_SHORT_DATATYPE_ID CONDUIT_INT64_T
        #define CONDUIT_NATIVE_UNSIGNED_SHORT_DATATYPE_ID CONDUIT_UINT64_T
    #endif
#endif

//-----------------------------------------------------------------------------
// -- char size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_CHAR == 8
    #ifndef CONDUIT_INT8
        #define CONDUIT_INT8 CONDUIT_BYTE
        #define CONDUIT_UINT8 CONDUIT_UBYTE
        #define CONDUIT_INT8_NATIVE_TYPENAME "signed char"
        #define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned char"
        typedef conduit_byte  conduit_int8;
        typedef conduit_ubyte conduit_uint8;
    #endif
    #ifndef CONDUIT_NATIVE_CHAR
        #define CONDUIT_NATIVE_CHAR conduit_int8
        #define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint8
        #define CONDUIT_NATIVE_CHAR_DATATYPE_ID CONDUIT_INT8_T
        #define CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID CONDUIT_UINT8_T
    #endif
#elif CONDUIT_BITSOF_CHAR == 16
    #ifndef CONDUIT_INT16
        #define CONDUIT_INT16 CONDUIT_BYTE
        #define CONDUIT_UINT16 CONDUIT_UBYTE
        #define CONDUIT_INT16_NATIVE_TYPENAME "signed char"
        #define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned char"
        typedef conduit_byte  conduit_int16;
        typedef conduit_ubyte conduit_uint16;
    #endif 
    #ifndef CONDUIT_NATIVE_CHAR    
        #define CONDUIT_NATIVE_CHAR conduit_int16
        #define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint16
        #define CONDUIT_NATIVE_CHAR_DATATYPE_ID CONDUIT_INT16_T
        #define CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID CONDUIT_UINT16_T
    #endif
#elif CONDUIT_BITSOF_CHAR == 32
    #ifndef CONDUIT_INT32
        #define CONDUIT_INT32 CONDUIT_BYTE
        #define CONDUIT_UINT32 CONDUIT_UBYTE
        #define CONDUIT_INT32_NATIVE_TYPENAME "signed char"
        #define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned char"
        typedef conduit_byte  conduit_int32;
        typedef conduit_ubyte conduit_uint32;
    #endif
    #ifndef CONDUIT_NATIVE_CHAR
        #define CONDUIT_NATIVE_CHAR conduit_int32
        #define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint32
        #define CONDUIT_NATIVE_CHAR_DATATYPE_ID CONDUIT_INT32_T
        #define CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID CONDUIT_UINT32_T
    #endif
#elif CONDUIT_BITSOF_CHAR == 64
    #ifndef CONDUIT_INT64
        #define CONDUIT_INT64 CONDUIT_BYTE
        #define CONDUIT_UINT64 CONDUIT_UBYTE
        #define CONDUIT_INT64_NATIVE_TYPENAME "signed char"
        #define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned char"
        typedef conduit_byte  conduit_int64;
        typedef conduit_ubyte conduit_uint64;
    #endif
    #ifndef CONDUIT_NATIVE_CHAR
        #define CONDUIT_NATIVE_CHAR conduit_int64
        #define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint64
        #define CONDUIT_NATIVE_CHAR_DATATYPE_ID CONDUIT_INT64_T
        #define CONDUIT_NATIVE_UNSIGNED_CHAR_DATATYPE_ID CONDUIT_UINT64_T
    #endif
#endif

//-----------------------------------------------------------------------------
// -- double size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_DOUBLE == 32
    #ifndef CONDUIT_FLOAT32
        #define CONDUIT_FLOAT32 CONDUIT_DOUBLE
        #define CONDUIT_FLOAT32_NATIVE_TYPENAME "double"
        typedef double conduit_float32;
    #endif
    #ifndef CONDUIT_NATIVE_DOUBLE
        #define CONDUIT_NATIVE_DOUBLE conduit_float32
        #define CONDUIT_NATIVE_DOUBLE_DATATYPE_ID CONDUIT_FLOAT32_T        
    #endif
#elif CONDUIT_BITSOF_DOUBLE == 64
    #ifndef CONDUIT_FLOAT64
        #define CONDUIT_FLOAT64 CONDUIT_DOUBLE
        #define CONDUIT_FLOAT64_NATIVE_TYPENAME "double"
        typedef conduit_double conduit_float64;
    #endif
    #ifndef CONDUIT_NATIVE_DOUBLE
        #define CONDUIT_NATIVE_DOUBLE conduit_float64
        #define CONDUIT_NATIVE_DOUBLE_DATATYPE_ID CONDUIT_FLOAT64_T
    #endif
#elif CONDUIT_BITSOF_DOUBLE == 80
    #ifndef CONDUIT_FLOAT80
        #define CONDUIT_FLOAT80 CONDUIT_DOUBLE
        #define CONDUIT_FLOAT80_NATIVE_TYPENAME "double"
        typedef conduit_double conduit_float80;
    #endif
    #ifndef CONDUIT_NATIVE_DOUBLE
        #define CONDUIT_NATIVE_DOUBLE conduit_float80
        // ERROR: no native to conduit mapping
    #endif
#elif CONDUIT_BITSOF_DOUBLE == 96
    #ifndef CONDUIT_FLOAT96
        #define CONDUIT_FLOAT96 CONDUIT_DOUBLE
        #define CONDUIT_FLOAT96_NATIVE_TYPENAME "double"
        typedef conduit_double conduit_float96;
    #endif
    #ifndef CONDUIT_NATIVE_DOUBLE
        #define CONDUIT_NATIVE_DOUBLE conduit_float96
        // ERROR: no native to conduit mapping
    #endif
#elif CONDUIT_BITSOF_DOUBLE == 128
    #ifndef CONDUIT_FLOAT128
        #define CONDUIT_FLOAT128 CONDUIT_DOUBLE
        #define CONDUIT_FLOAT128_NATIVE_TYPENAME "double"
        typedef conduit_double conduit_float128;
    #endif
    #ifndef CONDUIT_NATIVE_DOUBLE
        #define CONDUIT_NATIVE_DOUBLE conduit_float128
        // ERROR: no native to conduit mapping
    #endif
#endif

//-----------------------------------------------------------------------------
// -- float size checks --
//-----------------------------------------------------------------------------

#if CONDUIT_BITSOF_FLOAT == 32
    #ifndef CONDUIT_FLOAT32
        #define CONDUIT_FLOAT32 CONDUIT_FLOAT
        #define CONDUIT_FLOAT32_NATIVE_TYPENAME "float"
        typedef conduit_float conduit_float32;
    #endif
    #ifndef CONDUIT_NATIVE_FLOAT
        #define CONDUIT_NATIVE_FLOAT conduit_float32
        #define CONDUIT_NATIVE_FLOAT_DATATYPE_ID CONDUIT_FLOAT32_T
    #endif
#elif CONDUIT_BITSOF_FLOAT == 64
    #ifndef CONDUIT_FLOAT64
        #define CONDUIT_FLOAT64 CONDUIT_FLOAT
        #define CONDUIT_FLOAT64_NATIVE_TYPENAME "float"
        typedef conduit_float conduit_float64;
    #endif
    #ifndef CONDUIT_NATIVE_FLOAT
        #define CONDUIT_NATIVE_FLOAT conduit_float64
        #define CONDUIT_NATIVE_FLOAT_DATATYPE_ID CONDUIT_FLOAT64_T
    #endif
#elif CONDUIT_BITSOF_FLOAT == 80
    #ifndef CONDUIT_FLOAT80
        #define CONDUIT_FLOAT80 CONDUIT_FLOAT
        #define CONDUIT_FLOAT80_NATIVE_TYPENAME "float"
        typedef conduit_float conduit_float80;
    #endif
    #ifndef CONDUIT_NATIVE_FLOAT
        #define CONDUIT_NATIVE_FLOAT conduit_float80
        // ERROR: no native to conduit mapping
    #endif
#elif CONDUIT_BITSOF_FLOAT == 96
    #ifndef CONDUIT_FLOAT96
        #define CONDUIT_FLOAT96 CONDUIT_FLOAT
        #define CONDUIT_FLOAT96_NATIVE_TYPENAME "float"
        typedef conduit_float conduit_float96;
    #endif
    #ifndef CONDUIT_NATIVE_FLOAT
        #define CONDUIT_NATIVE_FLOAT conduit_float96
        // ERROR: no native to conduit mapping
    #endif
#elif CONDUIT_BITSOF_FLOAT == 128
    #ifndef CONDUIT_FLOAT128
        #define CONDUIT_FLOAT128 CONDUIT_FLOAT
        #define CONDUIT_FLOAT128_NATIVE_TYPENAME "float"
        typedef conduit_float conduit_float128;
    #endif
    #ifndef CONDUIT_NATIVE_FLOAT
        #define CONDUIT_NATIVE_FLOAT conduit_float128
        // ERROR: no native to conduit mapping
    #endif
#endif

//-----------------------------------------------------------------------------
// -- long double size checks --
//-----------------------------------------------------------------------------
#ifdef CONDUIT_HAS_LONG_DOUBLE
#if CONDUIT_BITSOF_LONG_DOUBLE == 32
    #ifndef CONDUIT_FLOAT32
        #define CONDUIT_USE_LONG_DOUBLE
        #define CONDUIT_FLOAT32 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT32_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float32;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
        #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float32
        #define CONDUIT_NATIVE_LONG_DOUBLE_DATATYPE_ID CONDUIT_FLOAT32_T
    #endif
#elif CONDUIT_BITSOF_LONG_DOUBLE == 64
    #ifndef CONDUIT_FLOAT64
        #define CONDUIT_USE_LONG_DOUBLE
        #define CONDUIT_FLOAT64 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT64_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float64;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
        #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float64
        #define CONDUIT_NATIVE_LONG_DOUBLE_DATATYPE_ID CONDUIT_FLOAT64_T
    #endif
#elif CONDUIT_BITSOF_LONG_DOUBLE == 80
    #ifndef CONDUIT_FLOAT80
        #define CONDUIT_FLOAT80 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT80_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float80;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
        #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float80
    #endif
#elif CONDUIT_BITSOF_LONG_DOUBLE == 96
    #ifndef CONDUIT_FLOAT96
        #define CONDUIT_FLOAT96 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT96_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float96;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
        #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float96
    #endif
#elif CONDUIT_BITSOF_LONG_DOUBLE == 128
    #ifndef CONDUIT_FLOAT128
        #define CONDUIT_FLOAT128 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT128_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float128;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
            #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float128
    #endif
#elif CONDUIT_BITSOF_LONG_DOUBLE == 256
    #ifndef CONDUIT_FLOAT256
        #define CONDUIT_FLOAT256 CONDUIT_LONG_DOUBLE
        #define CONDUIT_FLOAT256_NATIVE_TYPENAME "long double"
        typedef conduit_long_double conduit_float256;
    #endif
    #ifndef CONDUIT_NATIVE_LONG_DOUBLE
        #define CONDUIT_NATIVE_LONG_DOUBLE conduit_float256
    #endif
#endif
#endif

//-----------------------------------------------------------------------------
/// End of typedefs for numarray style bit-width names.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Bit-width type map sanity checks
//-----------------------------------------------------------------------------

//
// check that we were able to resolve all of the bitwidth style types we want
// to support

// signed int
#ifndef CONDUIT_INT8_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to int8
#endif

#ifndef CONDUIT_INT16_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to int16
#endif

#ifndef CONDUIT_INT32_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to int32
#endif

#ifndef CONDUIT_INT64_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to int64
#endif

// unsigned ints
#ifndef CONDUIT_UINT8_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to uint8
#endif

#ifndef CONDUIT_UINT16_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to uint16
#endif

#ifndef CONDUIT_UINT32_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to uint32
#endif

#ifndef CONDUIT_UINT64_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to uint64
#endif

// floating points numbers
#ifndef CONDUIT_FLOAT32_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to float32
#endif

#ifndef CONDUIT_FLOAT64_NATIVE_TYPENAME
#error Bitwidth Style Types: no native type found that maps to float64
#endif

//-----------------------------------------------------------------------------
///End Bit-width type map sanity checks
//-----------------------------------------------------------------------------


#endif

