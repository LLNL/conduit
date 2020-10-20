// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_RELAY_HPP
#define CONDUIT_RELAY_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

#include "conduit_relay_io.hpp"
#include "conduit_relay_io_handle.hpp"
#include "conduit_relay_io_blueprint.hpp"
#include "conduit_relay_web.hpp"
#include "conduit_relay_web_node_viewer_server.hpp"


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
/// The about methods construct human readable info about how relay was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_RELAY_API about();
void        CONDUIT_RELAY_API about(conduit::Node &res);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------



#endif

