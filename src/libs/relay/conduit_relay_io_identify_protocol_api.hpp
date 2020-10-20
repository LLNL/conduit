// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_identify_protocol_api.hpp
///
//-----------------------------------------------------------------------------
#ifndef CONDUIT_RELAY_IO_IDENTIFY_PROTOCOL_API_HPP
#define CONDUIT_RELAY_IO_IDENTIFY_PROTOCOL_API_HPP
#include <string>

#include "conduit_relay_exports.h"

//-----------------------------------------------------------------------------
/// Helper that identifies a relay io protocol from a file path.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API identify_protocol(const std::string &path,
                                         std::string &io_type);

//-----------------------------------------------------------------------------
/// Helper that identifies the underlying file type by opening it 
/// reading a small amount of data and a set of heuristics. 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API identify_file_type(const std::string &file_path,
                                          std::string &file_type);


#endif
