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
/// file: conduit_silo.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_SILO_H
#define __CONDUIT_SILO_H


#include "conduit_io.h"
#include <silo.h>

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
void CONDUIT_IO_API silo_save(const  Node &node,
                              const std::string &path);

void CONDUIT_IO_API silo_load(const std::string &path,
                              Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API silo_save(const  Node &node,
                              const std::string &file_path,
                              const std::string &silo_obj_path);

void CONDUIT_IO_API silo_load(const std::string &file_path,
                              const std::string &silo_obj_path,
                              Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_IO_API silo_save(const  Node &node,
                              DBfile *dbfile,
                              const std::string &silo_obj_path);

void CONDUIT_IO_API silo_load(DBfile *dbfile,
                              const std::string &silo_obj_path,
                              Node &node);


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif

