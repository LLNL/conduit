// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_H
#define CONDUIT_RELAY_H

//-----------------------------------------------------------------------------
// -- includes for the public conduit relay c interface -- 
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_relay_config.h"
#include "conduit_relay_exports.h"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- conduit_relay c interface  --
//-----------------------------------------------------------------------------

CONDUIT_RELAY_API void conduit_relay_about(conduit_node *cnode);

#ifdef __cplusplus
}
#endif
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

#include  "conduit_relay_io.h"

//-----------------------------------------------------------------------------
// -- end header guard ifdef
//-----------------------------------------------------------------------------
#endif
