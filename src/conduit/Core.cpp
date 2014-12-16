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
/// file: Core.cpp
///

#include "Core.h"
#include "Node.h"

namespace conduit
{

//============================================
std::string
about()
{
    Node n;
    about(n);
    return n.to_json(true,2);
}

//============================================
void
about(Node &n)
{
    n.reset();
    n["version"] = "{alpha}";
    n["copyright"] = "\n"
"Copyright (c) 2014, Lawrence Livermore National Security, LLC \n"
"Produced at the Lawrence Livermore National Laboratory.  \n"
"\n"
"All rights reserved. \n"
"\n"
"This source code cannot be distributed without further review from  \n"
"Lawrence Livermore National Laboratory. \n";
    // TODO: include compiler info, license info, etc

    // Type Info Map
    Node &nn = n["native_typemap"];

    // ints
#ifdef CONDUIT_INT8_NATIVE_TYPENAME
    nn["int8"] = CONDUIT_INT8_NATIVE_TYPENAME;
#else
    nn["int8"] = "<unmapped>";
#endif
#ifdef CONDUIT_INT16_NATIVE_TYPENAME
    nn["int16"] = CONDUIT_INT16_NATIVE_TYPENAME;
#else
    nn["int16"] = "<unmapped>";
#endif
#ifdef CONDUIT_INT32_NATIVE_TYPENAME
    nn["int32"] = CONDUIT_INT32_NATIVE_TYPENAME;
#else
    nn["int32"] = "<unmapped>";
#endif
#ifdef CONDUIT_INT64_NATIVE_TYPENAME
    nn["int64"] = CONDUIT_INT64_NATIVE_TYPENAME;
#else
    nn["int64"] = "<unmapped>";
#endif

    // unsigned ints
#ifdef CONDUIT_UINT8_NATIVE_TYPENAME
    nn["uint8"] = CONDUIT_UINT8_NATIVE_TYPENAME;
#else
    nn["uint8"] = "<unmapped>";
#endif
#ifdef CONDUIT_UINT16_NATIVE_TYPENAME
    nn["uint16"] = CONDUIT_UINT16_NATIVE_TYPENAME;
#else
    nn["uint16"] = "<unmapped>";
#endif
#ifdef CONDUIT_UINT32_NATIVE_TYPENAME
    nn["uint32"] = CONDUIT_UINT32_NATIVE_TYPENAME;
#else
    nn["uint32"] = "<unmapped>";
#endif
#ifdef CONDUIT_UINT64_NATIVE_TYPENAME
    nn["uint64"] = CONDUIT_UINT64_NATIVE_TYPENAME;
#else
    nn["uint64"] = "<unmapped>";
#endif

    // floating points numbers
#ifdef CONDUIT_FLOAT32_NATIVE_TYPENAME
    nn["float32"] = CONDUIT_FLOAT32_NATIVE_TYPENAME;
#else
    nn["float32"] = "<unmapped>";
#endif
#ifdef CONDUIT_FLOAT64_NATIVE_TYPENAME
    nn["float64"] = CONDUIT_FLOAT64_NATIVE_TYPENAME;
#else
    nn["float64"] = "<unmapped>";
#endif

    
    
}


}