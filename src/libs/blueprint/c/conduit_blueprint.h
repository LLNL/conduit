// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_H
#define CONDUIT_BLUEPRINT_H

//-----------------------------------------------------------------------------
// -- includes for the public conduit blueprint c interface -- 
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint_exports.h"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- conduit_blueprint c interface  --
//-----------------------------------------------------------------------------

CONDUIT_BLUEPRINT_API void conduit_blueprint_about(conduit_node *cnode);


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint protocol.
/// Messages related to the verification are be placed in the "info" node.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_verify(const char *protocol,
                                                   const conduit_node *cnode,
                                                   conduit_node *cinfo);

#ifdef __cplusplus
}
#endif
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

#include "conduit_blueprint_mcarray.h"
#include "conduit_blueprint_mesh.h"
#include "conduit_blueprint_table.h"


//-----------------------------------------------------------------------------
// -- end header guard ifdef
//-----------------------------------------------------------------------------
#endif
