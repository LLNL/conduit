// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.
#ifndef CONDUIT_RELAY_SILO_API_HPP
#define CONDUIT_RELAY_SILO_API_HPP

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_silo_api.hpp
///
//-----------------------------------------------------------------------------

/// NOTE: This file is included from other headers that provide namespaces.
///       Do not directly include this file!

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API silo_write(const Node &node,
                                  const std::string &path);

void CONDUIT_RELAY_API silo_read(const std::string &path,
                                 Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API silo_write(const  Node &node,
                                  const std::string &file_path,
                                  const std::string &silo_obj_path);

void CONDUIT_RELAY_API silo_read(const std::string &file_path,
                                 const std::string &silo_obj_path,
                                 Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API silo_write(const  Node &node,
                                  DBfile *dbfile,
                                  const std::string &silo_obj_path);

void CONDUIT_RELAY_API silo_read(DBfile *dbfile,
                                 const std::string &silo_obj_path,
                                 Node &node);


//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API silo_mesh_write(const Node &mesh,
                                       const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API silo_mesh_write(const Node &mesh,
                                       const std::string &file_path,
                                       const std::string &silo_obj_path);

//-----------------------------------------------------------------------------
// -- begin <>::silo --
//-----------------------------------------------------------------------------
namespace silo
{

void CONDUIT_RELAY_API silo_mesh_write(const Node &n, 
                     DBfile *dbfile,
                     const std::string &silo_obj_path);

//-----------------------------------------------------------------------------
// -- end <>::silo --
//-----------------------------------------------------------------------------
}

#endif
