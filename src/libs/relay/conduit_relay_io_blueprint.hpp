// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_blueprint.hpp
///
//-----------------------------------------------------------------------------

// NOTE: This functionality is a placeholder for more general functionality in
// future versions of Blueprint. That said, the functions in this header are
// subject to change and could be moved with any future iteration of Conduit,
// so use this header with caution!

#ifndef CONDUIT_RELAY_IO_BLUEPRINT_HPP
#define CONDUIT_RELAY_IO_BLUEPRINT_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::io::blueprint
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// Save a blueprint mesh to root + file set
//-----------------------------------------------------------------------------
/// Note: These methods use "save" semantics, they will overwrite existing
///       files. 
///
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const std::string &protocol);

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///      file_style: "default", "root_only", "multi_file"
///            when # of domains == 1,  "default"   ==> "root_only"
///            else,                    "default"   ==> "multi_file"
///
///      suffix: "default", "cycle", "none" 
///            when # of domains == 1,  "default"   ==> "none"
///            else,                    "default"   ==> "cycle"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      number_of_files:  {# of files}
///            when "multi_file":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const std::string &protocol,
                                 const conduit::Node &opts);

//-----------------------------------------------------------------------------
// Write a blueprint mesh to root + file set
//-----------------------------------------------------------------------------
/// Note: These methods use "write" semantics, they will append to existing
///       files. 
///
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path,
                                  const std::string &protocol);

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///      file_style: "default", "root_only", "multi_file"
///            when # of domains == 1,  "default"   ==> "root_only"
///            else,                    "default"   ==> "multi_file"
///
///      suffix: "default", "cycle", "none" 
///            when # of domains == 1,  "default"   ==> "none"
///            else,                    "default"   ==> "cycle"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      number_of_files:  {# of files}
///            when "multi_file":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files
///
///      truncate: "false", "true" (used if present, default ==> "false")
///           when "true" overwrites existing files (relay 'save' semantics)
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path,
                                  const std::string &protocol,
                                  const conduit::Node &opts);


//-----------------------------------------------------------------------------
// Load a blueprint mesh from root + file set
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API read_mesh(const std::string &root_file_path,
                                 conduit::Node &mesh);


//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where bp data includes
///           more than one mesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API read_mesh(const std::string &root_file_path,
                                 const conduit::Node &opts,
                                 conduit::Node &mesh);


//-----------------------------------------------------------------------------
// The load semantics, the mesh node is reset before reading.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
                                 conduit::Node &mesh);


//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where bp data includes
///           more than one mesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_mesh(const std::string &root_file_path,
                                 const conduit::Node &opts,
                                 conduit::Node &mesh);


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io::blueprint --
//-----------------------------------------------------------------------------

}

//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::relay::io_blueprint -- (DEPRECATED!)
//-----------------------------------------------------------------------------
namespace io_blueprint
{

//////////////////////////////////////////////////////////////////////////////
// DEPRECATED FUNCTIONS
//////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const conduit::Node &mesh,
                            const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const conduit::Node &mesh,
                            const std::string &path,
                            const std::string &protocol);

}

//-----------------------------------------------------------------------------
// -- end conduit::relay::io_blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
