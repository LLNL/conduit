/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: Bithwidth_Style_Types.h
///

#ifndef __CONDUIT_BITWIDTH_STYLE_TYPES_H
#define __CONDUIT_BITWIDTH_STYLE_TYPES_H

#include <limits.h>
#include "Conduit_Config.h"

///
/// Bit width annotated Style Standard Data Types
/// Derived from numpy (which provides very comprehensive support for these types)
///

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * harcoded
 */
#ifdef __APPLE__
    #undef SIZEOF_LONG
    #ifdef __LP64__
        #define SIZEOF_LONG      8
    #else
        #define SIZEOF_LONG      4
    #endif
#endif


#if SIZEOF_LONG_DOUBLE == SIZEOF_DOUBLE
        typedef double conduit_longdouble;
#else
        typedef long double conduit_longdouble;
#endif

typedef bool                conduit_bool;
typedef signed char         conduit_byte;
typedef unsigned char       conduit_ubyte;
typedef unsigned short      conduit_ushort;
typedef unsigned int        conduit_uint;
typedef unsigned long       conduit_ulong;
typedef unsigned long long  conduit_ulonglong;

typedef char                conduit_char;
typedef short               conduit_short;
typedef int                 conduit_int;
typedef long                conduit_long;
typedef long long           conduit_longlong;

typedef float               conduit_float;
typedef double              conduit_double;


#define BITSOF_CHAR CHAR_BIT

#define BITSOF_BOOL (SIZEOF_BYTE * CHAR_BIT)
#define BITSOF_BYTE (SIZEOF_BYTE * CHAR_BIT)
#define BITSOF_SHORT (SIZEOF_SHORT * CHAR_BIT)
#define BITSOF_INT (SIZEOF_INT * CHAR_BIT)
#define BITSOF_LONG (SIZEOF_LONG * CHAR_BIT)
#define BITSOF_LONG_LONG (SIZEOF_LONG_LONG * CHAR_BIT)
#define BITSOF_INTP (SIZEOF_INTP * CHAR_BIT)
#define BITSOF_HALF (SIZEOF_HALF * CHAR_BIT)
#define BITSOF_FLOAT (SIZEOF_FLOAT * CHAR_BIT)
#define BITSOF_DOUBLE (SIZEOF_DOUBLE * CHAR_BIT)
#define BITSOF_LONG_DOUBLE (SIZEOF_LONG_DOUBLE * CHAR_BIT)


#if BITSOF_BOOL == 8
#define CONDUIT_BOOL8 CONDUIT_BOOL
#define CONDUIT_BOOL8_NATIVE_TYPENAME "bool"
#define CONDUIT_NATIVE_BOOL conduit_bool8
        typedef conduit_bool conduit_bool8;
#else
#define CONDUIT_BOOL8 CONDUIT_CHAR
#define CONDUIT_BOOL8_NATIVE_TYPENAME "char"
#define CONDUIT_NATIVE_CHAR conduit_bool8
        typedef conduit_char conduit_bool8;
#endif


#if BITSOF_LONG == 8
#define CONDUIT_INT8 CONDUIT_LONG
#define CONDUIT_UINT8 CONDUIT_ULONG
#define CONDUIT_INT8_NATIVE_TYPENAME "long"
#define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned long"
#define CONDUIT_NATIVE_LONG conduit_int8
#define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint8
        typedef conduit_long  conduit_int8;
        typedef conduit_ulong conduit_uint8;
#elif BITSOF_LONG == 16
#define CONDUIT_INT16 CONDUIT_LONG
#define CONDUIT_UINT16 CONDUIT_ULONG
#define CONDUIT_INT16_NATIVE_TYPENAME "long"
#define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned long"
#define CONDUIT_NATIVE_LONG conduit_int16
#define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint16
        typedef conduit_long  conduit_int16;
        typedef conduit_ulong conduit_uint16;
#elif BITSOF_LONG == 32
#define CONDUIT_INT32 CONDUIT_LONG
#define CONDUIT_UINT32 CONDUIT_ULONG
#define CONDUIT_INT32_NATIVE_TYPENAME "long"
#define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned long"
#define CONDUIT_NATIVE_LONG conduit_int32
#define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint32
        typedef conduit_long  conduit_int32;
        typedef conduit_ulong conduit_uint32;
#elif BITSOF_LONG == 64
#define CONDUIT_INT64 CONDUIT_LONG
#define CONDUIT_UINT64 CONDUIT_ULONG
#define CONDUIT_INT64_NATIVE_TYPENAME "long"
#define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned long"
#define CONDUIT_NATIVE_LONG conduit_int64
#define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint64
        typedef conduit_long  conduit_int64;
        typedef conduit_ulong conduit_uint64;
#elif BITSOF_LONG == 128
#define CONDUIT_INT128 CONDUIT_LONG
#define CONDUIT_UINT128 CONDUIT_ULONG
#define CONDUIT_INT128_NATIVE_TYPENAME "long"
#define CONDUIT_UINT128_NATIVE_TYPENAME "unsigned long"
#define CONDUIT_NATIVE_LONG conduit_int128
#define CONDUIT_NATIVE_UNSIGNED_LONG conduit_uint128
        typedef conduit_long  conduit_int128;
        typedef conduit_ulong conduit_uint128;
#endif

#if BITSOF_LONG_LONG == 8
#ifndef CONDUIT_INT8
#define CONDUIT_INT8 CONDUIT_LONG_LONG
#define CONDUIT_UINT8 CONDUIT_ULONG_LONG
#define CONDUIT_INT8_NATIVE_TYPENAME "long long"
#define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned long long"
#define CONDUIT_NATIVE_LONG_LONG conduit_int8
#define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint8
        typedef conduit_longlong conduit_int8;
        typedef conduit_ulonglong conduit_uint8;
#endif
#elif BITSOF_LONG_LONG == 16
#ifndef CONDUIT_INT16
#define CONDUIT_INT16 CONDUIT_LONG_LONG
#define CONDUIT_UINT16 CONDUIT_ULONG_LONG
#define CONDUIT_INT16_NATIVE_TYPENAME "long long"
#define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned long long"
#define CONDUIT_NATIVE_LONG_LONG conduit_int16
#define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint16
        typedef conduit_longlong conduit_int16;
        typedef conduit_ulonglong conduit_uint16;
#endif
#elif BITSOF_LONG_LONG == 32
#ifndef CONDUIT_INT32
#define CONDUIT_INT32 CONDUIT_LONG_LONG
#define CONDUIT_UINT32 CONDUIT_ULONG_LONG
#define CONDUIT_INT32_NATIVE_TYPENAME "long long"
#define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned long long"
#define CONDUIT_NATIVE_LONG_LONG conduit_int32
#define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint32
        typedef conduit_longlong conduit_int32;
        typedef conduit_ulonglong conduit_uint32;
#endif
#elif BITSOF_LONG_LONG == 64
#ifndef CONDUIT_INT64
#define CONDUIT_INT64 CONDUIT_LONG_LONG
#define CONDUIT_UINT64 CONDUIT_ULONG_LONG
#define CONDUIT_INT64_NATIVE_TYPENAME "long long"
#define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned long long"
#define CONDUIT_NATIVE_LONG_LONG conduit_int64
#define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint64
        typedef conduit_longlong conduit_int64;
        typedef conduit_ulonglong conduit_uint64;
#endif
#elif BITSOF_LONG_LONG == 128
#ifndef CONDUIT_INT128
#define CONDUIT_INT128 CONDUIT_LONG_LONG
#define CONDUIT_UINT128 CONDUIT_ULONG_LONG
#define CONDUIT_INT128_NATIVE_TYPENAME "long long"
#define CONDUIT_UINT128_NATIVE_TYPENAME "unsigned long long"
#define CONDUIT_NATIVE_LONG_LONG conduit_int128
#define CONDUIT_NATIVE_UNSIGNED_LONG_LONG conduit_uint128
        typedef conduit_longlong conduit_int128;
        typedef conduit_ulonglong conduit_uint128;
#endif
#endif

#if BITSOF_INT == 8
#ifndef CONDUIT_INT8
#define CONDUIT_INT8 CONDUIT_INT
#define CONDUIT_UINT8 CONDUIT_UINT
#define CONDUIT_INT8_NATIVE_TYPENAME "int"
#define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned int"
#define CONDUIT_NATIVE_INT conduit_int8
#define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint8
        typedef conduit_int  conduit_int8;
        typedef conduit_uint conduit_uint8;
#endif
#elif CONDUIT_BITSOF_INT == 16
#ifndef CONDUIT_INT16
#define CONDUIT_INT16 CONDUIT_INT
#define CONDUIT_UINT16 CONDUIT_UINT
#define CONDUIT_INT16_NATIVE_TYPENAME "int"
#define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned int"
#define CONDUIT_NATIVE_INT conduit_int16
#define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint16
        typedef conduit_int  conduit_int16;
        typedef conduit_uint conduit_uint16;
#endif
#elif BITSOF_INT == 32
#ifndef CONDUIT_INT32
#define CONDUIT_INT32 CONDUIT_INT
#define CONDUIT_UINT32 CONDUIT_UINT
#define CONDUIT_INT32_NATIVE_TYPENAME "int"
#define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned int"
#define CONDUIT_NATIVE_INT conduit_int32
#define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint32
        typedef conduit_int  conduit_int32;
        typedef conduit_uint conduit_uint32;
#endif
#elif BITSOF_INT == 64
#ifndef CONDUIT_INT64
#define CONDUIT_INT64 CONDUIT_INT
#define CONDUIT_UINT64 CONDUIT_UINT
#define CONDUIT_INT64_NATIVE_TYPENAME "int"
#define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned int"
#define CONDUIT_NATIVE_INT conduit_int64
#define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint64
        typedef conduit_int  conduit_int64;
        typedef conduit_uint conduit_uint64;
#endif
#elif BITSOF_INT == 128
#ifndef CONDUIT_INT128
#define CONDUIT_INT128 CONDUIT_INT
#define CONDUIT_UINT128 CONDUIT_UINT
#define CONDUIT_INT128_NATIVE_TYPENAME "int"
#define CONDUIT_UINT128_NATIVE_TYPENAME "unsigned int"
#define CONDUIT_NATIVE_INT conduit_int128
#define CONDUIT_NATIVE_UNSIGNED_INT conduit_uint128
        typedef conduit_int  conduit_int128;
        typedef conduit_uint conduit_uint128;
#endif
#endif

#if BITSOF_SHORT == 8
#ifndef CONDUIT_INT8
#define CONDUIT_INT8 CONDUIT_SHORT
#define CONDUIT_UINT8 CONDUIT_USHORT
#define CONDUIT_INT8_NATIVE_TYPENAME "short"
#define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned short"
#define CONDUIT_NATIVE_SHORT conduit_int8
#define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint8
        typedef conduit_short  conduit_int8;
        typedef conduit_ushort conduit_uint8;
#endif
#elif BITSOF_SHORT == 16
#ifndef CONDUIT_INT16
#define CONDUIT_INT16 CONDUIT_SHORT
#define CONDUIT_UINT16 CONDUIT_USHORT
#define CONDUIT_INT16_NATIVE_TYPENAME "short"
#define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned short"
#define CONDUIT_NATIVE_SHORT conduit_int16
#define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint16
        typedef conduit_short  conduit_int16;
        typedef conduit_ushort conduit_uint16;
#endif
#elif BITSOF_SHORT == 32
#ifndef CONDUIT_INT32
#define CONDUIT_INT32 CONDUIT_SHORT
#define CONDUIT_UINT32 CONDUIT_USHORT
#define CONDUIT_INT32_NATIVE_TYPENAME "short"
#define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned short"
#define CONDUIT_NATIVE_SHORT conduit_int32
#define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint32
        typedef conduit_short  conduit_int32;
        typedef conduit_ushort conduit_uint32;
#endif
#elif BITSOF_SHORT == 64
#ifndef CONDUIT_INT64
#define CONDUIT_INT64 CONDUIT_SHORT
#define CONDUIT_UINT64 CONDUIT_USHORT
#define CONDUIT_INT64_NATIVE_TYPENAME "short"
#define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned short"
#define CONDUIT_NATIVE_SHORT conduit_int64
#define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint64
        typedef conduit_short  conduit_int64;
        typedef conduit_ushort conduit_uint64;
#endif
#elif BITSOF_SHORT == 128
#ifndef CONDUIT_INT128
#define CONDUIT_INT128 CONDUIT_SHORT
#define CONDUIT_UINT128 CONDUIT_USHORT
#define CONDUIT_INT128_NATIVE_TYPENAME "short"
#define CONDUIT_UINT128_NATIVE_TYPENAME "unsigned short"
#define CONDUIT_NATIVE_SHORT conduit_int128
#define CONDUIT_NATIVE_UNSIGNED_SHORT conduit_uint128
        typedef conduit_short  conduit_int128;
        typedef conduit_ushort conduit_uint128;
#endif
#endif


#if BITSOF_CHAR == 8
#ifndef CONDUIT_INT8
#define CONDUIT_INT8 CONDUIT_BYTE
#define CONDUIT_UINT8 CONDUIT_UBYTE
#define CONDUIT_INT8_NATIVE_TYPENAME "signed char"
#define CONDUIT_UINT8_NATIVE_TYPENAME "unsigned char"
#define CONDUIT_NATIVE_CHAR conduit_int8
#define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint8
        typedef conduit_byte  conduit_int8;
        typedef conduit_ubyte conduit_uint8;
#endif
#elif BITSOF_CHAR == 16
#ifndef CONDUIT_INT16
#define CONDUIT_INT16 CONDUIT_BYTE
#define CONDUIT_UINT16 CONDUIT_UBYTE
#define CONDUIT_INT16_NATIVE_TYPENAME "signed char"
#define CONDUIT_UINT16_NATIVE_TYPENAME "unsigned char"
#define CONDUIT_NATIVE_CHAR conduit_int16
#define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint16
        typedef conduit_byte  conduit_int16;
        typedef conduit_ubyte conduit_uint16;
#endif
#elif BITSOF_CHAR == 32
#ifndef CONDUIT_INT32
#define CONDUIT_INT32 CONDUIT_BYTE
#define CONDUIT_UINT32 CONDUIT_UBYTE
#define CONDUIT_INT32_NATIVE_TYPENAME "signed char"
#define CONDUIT_UINT32_NATIVE_TYPENAME "unsigned char"
#define CONDUIT_NATIVE_CHAR conduit_int32
#define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint32
        typedef conduit_byte  conduit_int32;
        typedef conduit_ubyte conduit_uint32;
#endif
#elif BITSOF_CHAR == 64
#ifndef CONDUIT_INT64
#define CONDUIT_INT64 CONDUIT_BYTE
#define CONDUIT_UINT64 CONDUIT_UBYTE
#define CONDUIT_INT64_NATIVE_TYPENAME "signed char"
#define CONDUIT_UINT64_NATIVE_TYPENAME "unsigned char"
#define CONDUIT_NATIVE_CHAR conduit_int64
#define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint64
        typedef conduit_byte  conduit_int64;
        typedef conduit_ubyte conduit_uint64;
#endif
#elif BITSOF_CHAR == 128
#ifndef CONDUIT_INT128
#define CONDUIT_INT128 CONDUIT_BYTE
#define CONDUIT_UINT128 CONDUIT_UBYTE
#define CONDUIT_INT128_NATIVE_TYPENAME "signed char"
#define CONDUIT_UINT128_NATIVE_TYPENAME "unsigned char"
#define CONDUIT_NATIVE_CHAR conduit_int128
#define CONDUIT_NATIVE_UNSIGNED_CHAR conduit_uint128
        typedef conduit_byte  conduit_int128;
        typedef conduit_ubyte conduit_uint128;
#endif
#endif


#if BITSOF_DOUBLE == 32
#ifndef CONDUIT_FLOAT32
#define CONDUIT_FLOAT32 CONDUIT_DOUBLE
#define CONDUIT_FLOAT32_NATIVE_TYPENAME "double"
#define CONDUIT_NATIVE_DOUBLE conduit_float32
        typedef double conduit_float32;
#endif
#elif BITSOF_DOUBLE == 64
#ifndef CONDUIT_FLOAT64
#define CONDUIT_FLOAT64 CONDUIT_DOUBLE
#define CONDUIT_FLOAT64_NATIVE_TYPENAME "double"
#define CONDUIT_NATIVE_DOUBLE conduit_float64
        typedef conduit_double conduit_float64;
#endif
#elif BITSOF_DOUBLE == 80
#ifndef CONDUIT_FLOAT80
#define CONDUIT_FLOAT80 CONDUIT_DOUBLE
#define CONDUIT_FLOAT80_NATIVE_TYPENAME "double"
#define CONDUIT_NATIVE_DOUBLE conduit_float80
        typedef conduit_double conduit_float80;
#endif
#elif BITSOF_DOUBLE == 96
#ifndef CONDUIT_FLOAT96
#define CONDUIT_FLOAT96 CONDUIT_DOUBLE
#define CONDUIT_FLOAT96_NATIVE_TYPENAME "double"
#define CONDUIT_NATIVE_DOUBLE conduit_float96
        typedef conduit_double conduit_float96;
#endif
#elif BITSOF_DOUBLE == 128
#ifndef CONDUIT_FLOAT128
#define CONDUIT_FLOAT128 CONDUIT_DOUBLE
#define CONDUIT_FLOAT128_NATIVE_TYPENAME "double"
#define CONDUIT_NATIVE_DOUBLE conduit_float128
        typedef conduit_double conduit_float128;
#endif
#endif

#if BITSOF_FLOAT == 32
#ifndef CONDUIT_FLOAT32
#define CONDUIT_FLOAT32 CONDUIT_FLOAT
#define CONDUIT_FLOAT32_NATIVE_TYPENAME "float"
#define CONDUIT_NATIVE_FLOAT conduit_float32
        typedef conduit_float conduit_float32;
#endif
#elif BITSOF_FLOAT == 64
#ifndef CONDUIT_FLOAT64
#define CONDUIT_FLOAT64 CONDUIT_FLOAT
#define CONDUIT_FLOAT64_NATIVE_TYPENAME "float"
#define CONDUIT_NATIVE_FLOAT conduit_float64
        typedef conduit_float conduit_float64;
#endif
#elif CONDUIT_BITSOF_FLOAT == 80
#ifndef CONDUIT_FLOAT80
#define CONDUIT_FLOAT80 CONDUIT_FLOAT
#define CONDUIT_FLOAT80_NATIVE_TYPENAME "float"
#define CONDUIT_NATIVE_FLOAT conduit_float80
        typedef conduit_float conduit_float80;
#endif
#elif BITSOF_FLOAT == 96
#ifndef CONDUIT_FLOAT96
#define CONDUIT_FLOAT96 CONDUIT_FLOAT
#define CONDUIT_FLOAT96_NATIVE_TYPENAME "float"
#define CONDUIT_NATIVE_FLOAT conduit_float96
        typedef conduit_float conduit_float96;
#endif
#elif BITSOF_FLOAT == 128
#ifndef CONDUIT_FLOAT128
#define CONDUIT_FLOAT128 CONDUIT_FLOAT
#define CONDUIT_FLOAT128_NATIVE_TYPENAME "float"
#define CONDUIT_NATIVE_FLOAT conduit_float128
        typedef conduit_float conduit_float128;
#endif
#endif

#if BITSOF_LONG_DOUBLE == 32
#ifndef CONDUIT_FLOAT32
#define CONDUIT_FLOAT32 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT32_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float32
        typedef conduit_longdouble conduit_float32;
#endif
#elif BITSOF_LONG_DOUBLE == 64
#ifndef CONDUIT_FLOAT64
#define CONDUIT_FLOAT64 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT64_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float64
        typedef conduit_longdouble conduit_float64;
#endif
#elif BITSOF_LONG_DOUBLE == 80
#ifndef CONDUIT_FLOAT80
#define CONDUIT_FLOAT80 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT80_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float80
        typedef conduit_longdouble conduit_float80;
#endif
#elif BITSOF_LONG_DOUBLE == 96
#ifndef CONDUIT_FLOAT96
#define CONDUIT_FLOAT96 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT96_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float96
        typedef conduit_longdouble conduit_float96;
#endif
#elif BITSOF_LONG_DOUBLE == 128
#ifndef CONDUIT_FLOAT128
#define CONDUIT_FLOAT128 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT128_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float128
        typedef conduit_longdouble conduit_float128;
#endif
#elif BITSOF_LONG_DOUBLE == 256
#define CONDUIT_FLOAT256 CONDUIT_LONG_DOUBLE
#define CONDUIT_FLOAT256_NATIVE_TYPENAME "long double"
#define CONDUIT_NATIVE_LONG_DOUBLE conduit_float256
        typedef conduit_longdouble conduit_float256;
#endif

/* End of typedefs for numarray style bit-width names */

#endif