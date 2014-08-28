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
}


}