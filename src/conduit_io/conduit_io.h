//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_io.h
///
//-----------------------------------------------------------------------------


#ifndef __CONDUIT_IO_H
#define __CONDUIT_IO_H

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms -- 
//-----------------------------------------------------------------------------
#include "Conduit_IO_Exports.h"

#include "conduit.h"
#include "Conduit_IO_Config.h"

// include optional libs
#ifdef CONDUIT_IO_ENABLE_SILO
#include "conduit_silo.h"
#endif

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
void CONDUIT_IO_API save(const  Node &node,
                         const std::string &path);

void CONDUIT_IO_API load(const std::string &path,
                         Node &node);

//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how conduit_io was
/// configured.
//-----------------------------------------------------------------------------
 std::string CONDUIT_IO_API about();
 void        CONDUIT_IO_API about(Node &);

};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif

