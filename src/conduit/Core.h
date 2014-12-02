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
/// file: Core.h
///

#ifndef __CONDUIT_CORE_H
#define __CONDUIT_CORE_H

#include "Conduit_Exports.h"

#include <string>
#include <iostream>

#include "Bitwidth_Style_Types.h"


namespace conduit
{

class Node;

typedef conduit_bool8   bool8;

typedef conduit_uint8   uint8;
typedef conduit_uint16  uint16;
typedef conduit_uint32  uint32;
typedef conduit_uint64  uint64;

typedef conduit_int8    int8;
typedef conduit_int16   int16;
typedef conduit_int32   int32;
typedef conduit_int64   int64;

typedef conduit_float32 float32;
typedef conduit_float64 float64;

typedef uint32 index32_t;
typedef uint64 index64_t;

#ifdef CONDUIT_INDEX_32
typedef index32_t index_t;
#else
typedef index64_t index_t;
#endif 

std::string CONDUIT_API about();
void        CONDUIT_API about(Node &);

}

#endif
