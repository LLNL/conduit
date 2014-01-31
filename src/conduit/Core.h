///
/// file: Core.h
///

#ifndef __CONDUIT_CORE_H
#define __CONDUIT_CORE_H

#include <string>
#include <iostream>

#include "Bitwidth_Style_Types.h"

namespace conduit
{


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


#ifdef CONDUIT_INDEX_32
typedef uint32 index_t;
#else
typedef uint64 index_t;
#endif 

std::string  version();

}

#endif
