###############################################################################
# Copyright (c) Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-666778
#
# All rights reserved.
#
# This file is part of Conduit.
#
# For details, see: http://software.llnl.gov/conduit/.
#
# Please also read conduit/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


###############################################################################
#  Conduit Data Type IDs
###############################################################################

################################################
# generic types
################################################
set(CONDUIT_EMPTY_ID  0)
set(CONDUIT_OBJECT_ID 1)
set(CONDUIT_LIST_ID   2)
################################################
# signed integer types
################################################
set(CONDUIT_INT8_ID   3)
set(CONDUIT_INT16_ID  4)
set(CONDUIT_INT32_ID  5)
set(CONDUIT_INT64_ID  6)
################################################
# unsigned integer types
################################################
set(CONDUIT_UINT8_ID   7)
set(CONDUIT_UINT16_ID  8)
set(CONDUIT_UINT32_ID  9)
set(CONDUIT_UINT64_ID  10)
################################################
# floating point types
################################################
set(CONDUIT_FLOAT32_ID 11)
set(CONDUIT_FLOAT64_ID 12)
################################################
#  string types
################################################
set(CONDUIT_CHAR8_STR_ID  13)


#-----------------------------------------------------------------------------
# Logic to provide bitwidth annotated style standard data types
#-----------------------------------------------------------------------------
# Derived from numpy (which provides very comprehensive support
# for these types)
#-----------------------------------------------------------------------------

#--------------------------------------------------------------------------
# bytes to bits size definitions
#--------------------------------------------------------------------------
# NOTE: we are assuming a char is 8 bits (CHAR_BIT in limits.h will tell us )
# In the future we should do a CMake try_compile check to make sure.
#
set(CONDUIT_BITSOF_CHAR 8)

math(EXPR CONDUIT_BITSOF_SHORT "${CONDUIT_SIZEOF_SHORT} * ${CONDUIT_BITSOF_CHAR}")
math(EXPR CONDUIT_BITSOF_INT   "${CONDUIT_SIZEOF_INT}   * ${CONDUIT_BITSOF_CHAR}")
math(EXPR CONDUIT_BITSOF_LONG  "${CONDUIT_SIZEOF_LONG}  * ${CONDUIT_BITSOF_CHAR}")

if(CONDUIT_HAS_LONG_LONG)
    math(EXPR CONDUIT_BITSOF_LONG_LONG  "${CONDUIT_SIZEOF_LONG_LONG} * ${CONDUIT_BITSOF_CHAR}")
endif()

math(EXPR CONDUIT_BITSOF_FLOAT   "${CONDUIT_SIZEOF_FLOAT}   * ${CONDUIT_BITSOF_CHAR}")
math(EXPR CONDUIT_BITSOF_DOUBLE  "${CONDUIT_SIZEOF_DOUBLE}  * ${CONDUIT_BITSOF_CHAR}")

if(CONDUIT_HAS_LONG_DOUBLE)
    math(EXPR CONDUIT_BITSOF_LONG_DOUBLE "${CONDUIT_SIZEOF_LONG_DOUBLE} * ${CONDUIT_BITSOF_CHAR}")
endif()


#-----------------------------------------------------------------------------
# -- long long type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_long_long)
if(CONDUIT_HAS_LONG_LONG)
    if(${CONDUIT_BITSOF_LONG_LONG} EQUAL 8)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_INT8_TYPE)
            set(CONDUIT_USE_LONG_LONG 1)
            #
            set(CONDUIT_INT8_TYPE   "conduit_signed_long_long")
            set(CONDUIT_UINT8_TYPE  "conduit_unsigned_long_long")
            #
            set(CONDUIT_INT8_NATIVE_TYPE  "signed long long")
            set(CONDUIT_UINT8_NATIVE_TYPE "unsigned long long")
            #
        endif()
        ####
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_SIGNED_LONG_LONG_ID)
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID   ${CONDUIT_INT8_ID})
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID ${CONDUIT_UINT8_ID})
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_TYPE   "conduit_int8")
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_TYPE "conduit_uint8")
        endif()
    elseif(${CONDUIT_BITSOF_LONG_LONG} EQUAL 16)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_INT16_TYPE)
            set(CONDUIT_USE_LONG_LONG 1)
            #
            set(CONDUIT_INT16_TYPE   "conduit_signed_long_long")
            set(CONDUIT_UINT16_TYPE  "conduit_unsigned_long_long")
            #
            set(CONDUIT_INT16_NATIVE_TYPE  "signed long long")
            set(CONDUIT_UINT16_NATIVE_TYPE "unsigned long long")
            #
        endif()
        ####
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_SIGNED_LONG_LONG_ID)
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID   ${CONDUIT_INT16_ID})
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID ${CONDUIT_UINT16_ID})
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID   "conduit_int16")
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID "conduit_uint16")
        endif()
    elseif(${CONDUIT_BITSOF_LONG_LONG} EQUAL 32)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_INT32_TYPE)
            set(CONDUIT_USE_LONG_LONG 1)
            #
            set(CONDUIT_INT32_TYPE   "conduit_signed_long_long")
            set(CONDUIT_UINT32_TYPE  "conduit_unsigned_long_long")
            #
            set(CONDUIT_INT32_NATIVE_TYPE  "signed long long")
            set(CONDUIT_UINT32_NATIVE_TYPE "unsigned long long")
            #
        endif()
        ################################################
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_SIGNED_LONG_LONG_ID)
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID   ${CONDUIT_INT32_ID})
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID ${CONDUIT_UINT32_ID})
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_TYPE   "conduit_int32")
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_TYPE "conduit_uint32")
        endif()
    elseif(${CONDUIT_BITSOF_LONG_LONG} EQUAL 64)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_INT64_TYPE)
            set(CONDUIT_USE_LONG_LONG 1)
            #
            set(CONDUIT_INT64_TYPE   "conduit_signed_long_long")
            set(CONDUIT_UINT64_TYPE  "conduit_unsigned_long_long")
            #
            set(CONDUIT_INT64_NATIVE_TYPE  "signed long long")
            set(CONDUIT_UINT64_NATIVE_TYPE "unsigned long long")
            #
        endif()
        ####
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_SIGNED_LONG_LONG_TYPE)
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID   ${CONDUIT_INT64_ID})
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID ${CONDUIT_UINT64_ID})
            #
            set(CONDUIT_NATIVE_SIGNED_LONG_LONG_TYPE   "conduit_int64")
            set(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_TYPE "conduit_uint64")
        endif()
    endif()
endif()
endmacro()

#------------------------------------------------------------------------------
# -- long type mapping --
#------------------------------------------------------------------------------
macro(bitwidth_map_long)
if(${CONDUIT_BITSOF_LONG} EQUAL 8)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT8_TYPE)
        set(CONDUIT_USE_LONG 1)
        #
        set(CONDUIT_INT8_TYPE   "conduit_signed_long")
        set(CONDUIT_UINT8_TYPE  "conduit_unsigned_long")
        #
        set(CONDUIT_INT8_NATIVE_TYPE  "signed long")
        set(CONDUIT_UINT8_NATIVE_TYPE "unsigned long")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_LONG_TYPE)
        set(CONDUIT_NATIVE_SIGNED_LONG_ID   ${CONDUIT_INT8_ID})
        set(CONDUIT_NATIVE_UNSIGNED_LONG_ID ${CONDUIT_UINT8_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_TYPE   "conduit_int8")
        set(CONDUIT_NATIVE_UNSIGNED_LONG_TYPE "conduit_uint8")
    endif()
elseif(${CONDUIT_BITSOF_LONG} EQUAL 16)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT16_TYPE)
        set(CONDUIT_USE_LONG 1)
        #
        set(CONDUIT_INT16_TYPE   "conduit_signed_long")
        set(CONDUIT_UINT16_TYPE  "conduit_unsigned_long")
        #
        set(CONDUIT_INT16_NATIVE_TYPE  "signed long")
        set(CONDUIT_UINT16_NATIVE_TYPE "unsigned long")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_LONG_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_ID   ${CONDUIT_INT16_ID})
        set(CONDUIT_NATIVE_UNSIGNED_LONG_ID ${CONDUIT_UINT16_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_TYPE   "conduit_int16")
        set(CONDUIT_NATIVE_UNSIGNED_LONG_TYPE "conduit_uint16")
    endif()
elseif(${CONDUIT_BITSOF_LONG} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT32_TYPE)
        set(CONDUIT_USE_LONG 1)
        #
        set(CONDUIT_INT32_TYPE   "conduit_signed_long")
        set(CONDUIT_UINT32_TYPE  "conduit_unsigned_long")
        #
        set(CONDUIT_INT32_NATIVE_TYPE  "signed long")
        set(CONDUIT_UINT32_NATIVE_TYPE "unsigned long")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_LONG_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_ID   ${CONDUIT_INT32_ID})
        set(CONDUIT_NATIVE_UNSIGNED_LONG_ID ${CONDUIT_UINT32_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_TYPE   "conduit_int32")
        set(CONDUIT_NATIVE_UNSIGNED_LONG_TYPE "conduit_uint32")
    endif()
elseif(${CONDUIT_BITSOF_LONG} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT64_TYPE)
        set(CONDUIT_USE_LONG 1)
        #
        set(CONDUIT_INT64_TYPE   "conduit_signed_long")
        set(CONDUIT_UINT64_TYPE  "conduit_unsigned_long")
        #
        set(CONDUIT_INT64_NATIVE_TYPE  "signed long")
        set(CONDUIT_UINT64_NATIVE_TYPE "unsigned long")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_LONG_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_ID   ${CONDUIT_INT64_ID})
        set(CONDUIT_NATIVE_UNSIGNED_LONG_ID ${CONDUIT_UINT64_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_LONG_TYPE   "conduit_int64")
        set(CONDUIT_NATIVE_UNSIGNED_LONG_TYPE "conduit_uint64")
    endif()
endif()
endmacro()

#-----------------------------------------------------------------------------
# -- int size checks --
#-----------------------------------------------------------------------------
macro(bitwidth_map_int)
if(${CONDUIT_BITSOF_INT} EQUAL 8)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT8_TYPE)
        set(CONDUIT_USE_INT 1)
        #
        set(CONDUIT_INT8_TYPE   "conduit_int")
        set(CONDUIT_UINT8_TYPE  "conduit_uint")
        #
        set(CONDUIT_INT8_NATIVE_TYPE  "signed int")
        set(CONDUIT_UINT8_NATIVE_TYPE "unsigned int")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_INT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_INT_ID   ${CONDUIT_INT8_ID})
        set(CONDUIT_NATIVE_UNSIGNED_INT_ID ${CONDUIT_UINT8_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_INT_TYPE   "conduit_int8")
        set(CONDUIT_NATIVE_UNSIGNED_INT_TYPE "conduit_uint8")
    endif()
elseif(${CONDUIT_BITSOF_INT} EQUAL 16)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT16_TYPE)
        set(CONDUIT_USE_INT 1)
        #
        set(CONDUIT_INT16_TYPE   "conduit_signed_int")
        set(CONDUIT_UINT16_TYPE  "conduit_unsigned_int")
        #
        set(CONDUIT_INT16_NATIVE_TYPE  "signed int")
        set(CONDUIT_UINT16_NATIVE_TYPE "unsigned int")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_INT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_INT_ID   ${CONDUIT_INT16_ID})
        set(CONDUIT_NATIVE_UNSIGNED_INT_ID ${CONDUIT_UINT16_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_INT_TYPE   "conduit_int16")
        set(CONDUIT_NATIVE_UNSIGNED_INT_TYPE "conduit_uint16")
    endif()
elseif(${CONDUIT_BITSOF_INT} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT32_TYPE)
        set(CONDUIT_USE_INT 1)
        #
        set(CONDUIT_INT32_TYPE   "conduit_signed_int")
        set(CONDUIT_UINT32_TYPE  "conduit_unsigned_int")
        #
        set(CONDUIT_INT32_NATIVE_TYPE  "signed int")
        set(CONDUIT_UINT32_NATIVE_TYPE "unsigned int")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_INT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_INT_ID   ${CONDUIT_INT32_ID})
        set(CONDUIT_NATIVE_UNSIGNED_INT_ID ${CONDUIT_UINT32_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_INT_TYPE   "conduit_int32")
        set(CONDUIT_NATIVE_UNSIGNED_INT_TYPE "conduit_uint32")
    endif()
elseif(${CONDUIT_BITSOF_INT} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT64_TYPE)
        set(CONDUIT_USE_INT 1)
        #
        set(CONDUIT_INT64_TYPE   "conduit_int")
        set(CONDUIT_UINT64_TYPE  "conduit_uint")
        #
        set(CONDUIT_INT64_NATIVE_TYPE  "signed int")
        set(CONDUIT_UINT64_NATIVE_TYPE "unsigned int")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_INT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_INT_ID   ${CONDUIT_INT64_ID})
        set(CONDUIT_NATIVE_UNSIGNED_INT_ID ${CONDUIT_UINT64_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_INT_TYPE   "conduit_int64")
        set(CONDUIT_NATIVE_UNSIGNED_INT_TYPE "conduit_uint64")
    endif()
endif()
endmacro()


#-----------------------------------------------------------------------------
# -- short type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_short)
if(${CONDUIT_BITSOF_SHORT} EQUAL 8)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT8_TYPE)
        set(CONDUIT_USE_SHORT 1)
        #
        set(CONDUIT_INT8_TYPE   "conduit_signed_short")
        set(CONDUIT_UINT8_TYPE  "conduit_unsigned_short")
        #
        set(CONDUIT_INT8_NATIVE_TYPE  "signed short")
        set(CONDUIT_UINT8_NATIVE_TYPE "unsigned short")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_SHORT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_ID   ${CONDUIT_INT8_ID})
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_ID ${CONDUIT_UINT8_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_TYPE   "conduit_int8")
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_TYPE "conduit_uint8")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 16)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT16_TYPE)
        set(CONDUIT_USE_SHORT 1)
        #
        set(CONDUIT_INT16_TYPE   "conduit_signed_short")
        set(CONDUIT_UINT16_TYPE  "conduit_unsigned_short")
        #
        set(CONDUIT_INT16_NATIVE_TYPE  "signed short")
        set(CONDUIT_UINT16_NATIVE_TYPE "unsigned short")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_SHORT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_ID   ${CONDUIT_INT16_ID})
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_ID ${CONDUIT_UINT16_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_TYPE   "conduit_int16")
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_TYPE "conduit_uint16")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT32_TYPE)
        set(CONDUIT_USE_SHORT 1)
        #
        set(CONDUIT_INT32_TYPE   "conduit_signed_short")
        set(CONDUIT_UINT32_TYPE  "conduit_unsigned_short")
        #
        set(CONDUIT_INT32_NATIVE_TYPE  "signed short")
        set(CONDUIT_UINT32_NATIVE_TYPE "unsigned short")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_SHORT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_ID   ${CONDUIT_INT32_ID})
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_ID ${CONDUIT_UINT32_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_TYPE   "conduit_int32")
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_TYPE "conduit_uint32")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT64_TYPE)
        set(CONDUIT_USE_SHORT 1)
        #
        set(CONDUIT_INT64_TYPE   "conduit_signed_short")
        set(CONDUIT_UINT64_TYPE  "conduit_unsigned_short")
        #
        set(CONDUIT_INT64_NATIVE_TYPE  "signed short")
        set(CONDUIT_UINT64_NATIVE_TYPE "unsigned short")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_INT_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_ID   ${CONDUIT_INT64_ID})
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_ID ${CONDUIT_UINT64_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_SHORT_TYPE   "conduit_int64")
        set(CONDUIT_NATIVE_UNSIGNED_SHORT_TYPE "conduit_uint64")
    endif()
endif()
endmacro()


#-----------------------------------------------------------------------------
# -- char type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_char)
if(${CONDUIT_BITSOF_CHAR} EQUAL 8)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT8_TYPE)
        set(CONDUIT_USE_CHAR 1)
        #
        set(CONDUIT_INT8_TYPE   "conduit_signed_char")
        set(CONDUIT_UINT8_TYPE  "conduit_unsigned_char")
        #
        set(CONDUIT_INT8_NATIVE_TYPE  "signed char")
        set(CONDUIT_UINT8_NATIVE_TYPE "unsigned char")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_CHAR_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_ID   ${CONDUIT_INT8_ID})
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_ID ${CONDUIT_UINT8_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_TYPE   "conduit_int8")
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_TYPE "conduit_uint8")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 16)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT16_TYPE)
        set(CONDUIT_USE_CHAR 1)
        #
        set(CONDUIT_INT16_TYPE   "conduit_signed_char")
        set(CONDUIT_UINT16_TYPE  "conduit_unsigned_char")
        #
        set(CONDUIT_INT16_NATIVE_TYPE  "signed char")
        set(CONDUIT_UINT16_NATIVE_TYPE "unsigned char")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_CHAR_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_ID   ${CONDUIT_INT16_ID})
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_ID ${CONDUIT_UINT16_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_TYPE   "conduit_int16")
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_TYPE "conduit_uint16")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT32_TYPE)
        set(CONDUIT_USE_CHAR 1)
        #
        set(CONDUIT_INT32_TYPE   "conduit_signed_char")
        set(CONDUIT_UINT32_TYPE  "conduit_unsigned_char")
        #
        set(CONDUIT_INT32_NATIVE_TYPE  "signed char")
        set(CONDUIT_UINT32_NATIVE_TYPE "unsigned char")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_CHAR_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_ID   ${CONDUIT_INT32_ID})
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_ID ${CONDUIT_UINT32_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_TYPE   "conduit_int32")
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_TYPE "conduit_uint32")
    endif()
elseif(${CONDUIT_BITSOF_SHORT} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_INT64_TYPE)
        set(CONDUIT_USE_CHAR 1)
        #
        set(CONDUIT_INT64_TYPE   "conduit_signed_char")
        set(CONDUIT_UINT64_TYPE  "conduit_unsigned_char")
        #
        set(CONDUIT_INT64_NATIVE_TYPE  "signed_char")
        set(CONDUIT_UINT64_NATIVE_TYPE "unsigned char")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_SIGNED_CHAR_TYPE)
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_ID   ${CONDUIT_INT64_ID})
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_ID ${CONDUIT_UINT64_ID})
        #
        set(CONDUIT_NATIVE_SIGNED_CHAR_TYPE   "conduit_int64")
        set(CONDUIT_NATIVE_UNSIGNED_CHAR_TYPE "conduit_uint64")
    endif()
endif()
endmacro()


#-----------------------------------------------------------------------------
# -- double type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_double)
if(${CONDUIT_BITSOF_DOUBLE} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_FLOAT32_TYPE)
        set(CONDUIT_USE_DOUBLE 1)
        set(CONDUIT_FLOAT32_TYPE "conduit_double")
        #
        set(CONDUIT_FLOAT32_NATIVE_TYPE "double")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_DOUBLE_TYPE)
        set(CONDUIT_NATIVE_DOUBLE_ID  ${CONDUIT_FLOAT32_ID})
        #
        set(CONDUIT_NATIVE_DOUBLE_TYPE     "conduit_float32")
    endif()
elseif(${CONDUIT_BITSOF_DOUBLE} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_FLOAT64_TYPE)
        set(CONDUIT_USE_DOUBLE 1)
        set(CONDUIT_FLOAT64_TYPE "conduit_double")
        #
        set(CONDUIT_FLOAT64_NATIVE_TYPE "double")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_DOUBLE_TYPE)
        set(CONDUIT_NATIVE_DOUBLE_ID  ${CONDUIT_FLOAT64_ID})
        #
        set(CONDUIT_NATIVE_DOUBLE_TYPE     "conduit_float64")
    endif()
endif()
endmacro()

#-----------------------------------------------------------------------------
#  -- float type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_float)
if(${CONDUIT_BITSOF_FLOAT} EQUAL 32)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_FLOAT32_TYPE)
        set(CONDUIT_USE_FLOAT 1)
        set(CONDUIT_FLOAT32_TYPE "conduit_float")
        #
        set(CONDUIT_FLOAT32_NATIVE_TYPE "float")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_FLOAT_TYPE)
        set(CONDUIT_NATIVE_FLOAT_ID  ${CONDUIT_FLOAT32_ID})
        #
        set(CONDUIT_NATIVE_FLOAT_TYPE     "conduit_float32")
    endif()
elseif(${CONDUIT_BITSOF_FLOAT} EQUAL 64)
    ####
    # conduit to native
    ####
    # make sure we haven't already mapped this type
    if(NOT CONDUIT_FLOAT64_TYPE)
        set(CONDUIT_USE_FLOAT 1)
        set(CONDUIT_FLOAT64_TYPE "conduit_float")
        #
        set(CONDUIT_FLOAT64_NATIVE_TYPE "float")
    endif()
    ####
    # native to conduit
    ####
    # check to see if the native type map has been made
    if(NOT CONDUIT_NATIVE_FLOAT_TYPE)
        set(CONDUIT_NATIVE_FLOAT_ID  ${CONDUIT_FLOAT64_ID})
        #
        set(CONDUIT_NATIVE_FLOAT_TYPE     "conduit_float64")
    endif()
endif()
endmacro()

#-----------------------------------------------------------------------------
#  -- long double type mapping --
#-----------------------------------------------------------------------------
macro(bitwidth_map_long_double)
if(${CONDUIT_HAS_LONG_DOUBLE})
    if(${CONDUIT_BITSOF_LONG_DOUBLE} EQUAL 32)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_FLOAT32_TYPE)
            set(CONDUIT_USE_LONG_DOUBLE 1)
            set(CONDUIT_FLOAT32_TYPE "conduit_long_double")
            #
            set(CONDUIT_FLOAT32_NATIVE_TYPE "long double")
        endif()
        ####
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_FLOAT_TYPE)
            set(CONDUIT_NATIVE_LONG_DOUBLE_ID  ${CONDUIT_FLOAT32_ID})
            #
            set(CONDUIT_NATIVE_LONG_DOUBLE_TYPE     "conduit_float32")
        endif()
    elseif(${CONDUIT_BITSOF_LONG_DOUBLE} EQUAL 64)
        ####
        # conduit to native
        ####
        # make sure we haven't already mapped this type
        if(NOT CONDUIT_FLOAT64_TYPE)
            set(CONDUIT_USE_LONG_DOUBLE 1)
            set(CONDUIT_FLOAT64_TYPE "conduit_long_double")
            #
            set(CONDUIT_FLOAT64_NATIVE_TYPE "long double")
        endif()
        ####
        # native to conduit
        ####
        # check to see if the native type map has been made
        if(NOT CONDUIT_NATIVE_LONG_DOUBLE_TYPE)
            set(CONDUIT_NATIVE_LONG_DOUBLE_ID  ${CONDUIT_FLOAT64_ID})
            #
            set(CONDUIT_NATIVE_LONG_DOUBLE_TYPE     "conduit_float64")
        endif()
    endif()
endif()
endmacro()

#------------------------------------------------------------------------------
# End checks for numarray style bit-width names.
#------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Map c native types to conduit bitwidth style types
#-----------------------------------------------------------------------------
# We prefer `long` by default. c++11 will map either `long` or `long long`
# as int64, the choice is platform and/or compiler dependent.  
# We have use try_compile with c++11 to find if we should prefer `long long` 
# over `long`.
#-----------------------------------------------------------------------------

set(bitwidth_prefer_long_long FALSE)
if(CONDUIT_USE_CXX11)
    # note: this call assumes this file included from libs/conduit/CMakeLists.txt
    try_compile( bitwidth_prefer_long_long   # result var
                 ${CMAKE_CURRENT_BINARY_DIR}/cmake  # build dir
                 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_cxx11_check_if_long_long_is_std_int64.cpp
                 OUTPUT_VARIABLE bitwidth_prefer_long_long_out
                 CXX_STANDARD 11
                 CXX_STANDARD_REQUIRED TRUE)
endif()


# map our types using the macros above
bitwidth_map_int()
if(bitwidth_prefer_long_long)
    message(STATUS "Bitwidth Mapping: preferring `long long` over `long` for c++11 compatibility")
    bitwidth_map_long_long()
    bitwidth_map_long()
else()
    message(STATUS "Bitwidth Mapping: preferring `long` over `long long` for c++11 compatibility")
    bitwidth_map_long()
    bitwidth_map_long_long()
endif()

bitwidth_map_short()
bitwidth_map_char()

# for fp types, use same mapping order for all platforms
bitwidth_map_double()
bitwidth_map_float()
bitwidth_map_long_double()

#-----------------------------------------------------------------------------
# Display Mapping Results during CMake Configure
#-----------------------------------------------------------------------------
message(STATUS "Bitwidth Mapping Results:")
#-----------------------------------------------------------------------------
# -- bitwidth style signed integer types --
#-----------------------------------------------------------------------------
message(STATUS " conduit::int8  native type: ${CONDUIT_INT8_NATIVE_TYPE}")
message(STATUS " conduit::int16 native type: ${CONDUIT_INT16_NATIVE_TYPE}")
message(STATUS " conduit::int32 native type: ${CONDUIT_INT32_NATIVE_TYPE}")
message(STATUS " conduit::int64 native type: ${CONDUIT_INT64_NATIVE_TYPE}")

#-----------------------------------------------------------------------------
# -- bitwidth style unsigned integer types --
#-----------------------------------------------------------------------------
message(STATUS " conduit::uint8  native type: ${CONDUIT_UINT8_NATIVE_TYPE}")
message(STATUS " conduit::uint16 native type: ${CONDUIT_UINT16_NATIVE_TYPE}")
message(STATUS " conduit::uint32 native type: ${CONDUIT_UINT32_NATIVE_TYPE}")
message(STATUS " conduit::uint64 native type: ${CONDUIT_UINT64_NATIVE_TYPE}")

#-----------------------------------------------------------------------------
# -- bitwidth style floating point types
#-----------------------------------------------------------------------------
message(STATUS " conduit::float32 native type: ${CONDUIT_FLOAT32_NATIVE_TYPE}")
message(STATUS " conduit::float64 native type: ${CONDUIT_FLOAT64_NATIVE_TYPE}")
