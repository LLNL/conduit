// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_H
#define CONDUIT_BLUEPRINT_MESH_H

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
// -- conduit_blueprint_mesh c interface  --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mesh protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mesh_verify(const conduit_node *cnode,
                                                        conduit_node *cinfo);


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mesh sub protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mesh_verify_sub_protocol(const char *protocol,
                                                                     const conduit_node *cnode,
                                                                     conduit_node *cinfo);

//-----------------------------------------------------------------------------
/// Generate mesh::index from valid mesh.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API void conduit_blueprint_mesh_generate_index(const conduit_node *cmesh,
                                                                 const char *ref_path,
                                                                 conduit_index_t num_domains,
                                                                 conduit_node *cindex_out);

//-----------------------------------------------------------------------------
/// Interface to generate example mesh blueprint data.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API void conduit_blueprint_mesh_examples_braid(const char *mesh_type,
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
