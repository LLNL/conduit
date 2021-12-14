// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_table.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_TABLE_H
#define CONDUIT_BLUEPRINT_TABLE_H

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
// -- conduit_blueprint_table c interface  --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Verify passed node conforms to the blueprint table protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_table_verify(
                                                    const conduit_node *cnode,
                                                    conduit_node *cinfo);

//-----------------------------------------------------------------------------
/// Verify passed node conforms to given blueprint table sub protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_table_verify_sub_protocol(
                                                    const char *protocol,
                                                    const conduit_node *cnode,
                                                    conduit_node *cinfo);

//-----------------------------------------------------------------------------
/// Interface to generate example table blueprint data.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API void conduit_blueprint_table_examples_basic(
                                                    conduit_index_t nx,
                                                    conduit_index_t ny,
                                                    conduit_index_t nz,
                                                    conduit_node *cres);

#ifdef __cplusplus
}
#endif
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- end header guard ifdef
//-----------------------------------------------------------------------------
#endif
