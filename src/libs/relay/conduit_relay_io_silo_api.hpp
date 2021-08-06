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
// Save a blueprint mesh to silo
//-----------------------------------------------------------------------------
/// These methods assume `mesh` is a valid blueprint mesh.
///
/// Note: These methods use "save" semantics, they will overwrite existing
///       files.
///
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path);

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///      TODO
///
//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
//                                  const std::string &path,
//                                  const conduit::Node &opts);

//-----------------------------------------------------------------------------
// Write a blueprint mesh to silo
//-----------------------------------------------------------------------------
/// These methods assume `mesh` is a valid blueprint mesh.
///
/// Note: These methods use "write" semantics, they will append to existing
///       files.
///
///
//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
//                                   const std::string &path);

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///      TODO
///
//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
//                                   const std::string &path,
//                                   const conduit::Node &opts);


//-----------------------------------------------------------------------------
// Load a blueprint mesh from root + file set
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API read_mesh(const std::string &root_file_path,
//                                  conduit::Node &mesh);


//-----------------------------------------------------------------------------
///
/// opts:
///      TODO
///
//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API read_mesh(const std::string &root_file_path,
//                                  const conduit::Node &opts,
//                                  conduit::Node &mesh);


//-----------------------------------------------------------------------------
// The load semantics, the mesh node is reset before reading.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
                                 conduit::Node &mesh);


//-----------------------------------------------------------------------------
///
/// opts:
///      TODO
///
//-----------------------------------------------------------------------------
// void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
//                                  const conduit::Node &opts,
//                                  conduit::Node &mesh);


//-----------------------------------------------------------------------------
}


#endif
