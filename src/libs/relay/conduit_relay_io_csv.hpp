// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_csv.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_IO_CSV_HPP
#define CONDUIT_RELAY_IO_CSV_HPP

//-----------------------------------------------------------------------------
// conduit lib include
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_node.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

//-----------------------------------------------------------------------------
// -- begin conduit --
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
CONDUIT_RELAY_API void read_csv(const std::string &path,
                                const Node &options,
                                Node &table);

//-----------------------------------------------------------------------------
/**
@brief Accepts a blueprint table and writes it out to the given filename.
*/
CONDUIT_RELAY_API void write_csv(const Node &table,
                                 const std::string &path,
                                 const Node &options);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit --
//-----------------------------------------------------------------------------


#endif
