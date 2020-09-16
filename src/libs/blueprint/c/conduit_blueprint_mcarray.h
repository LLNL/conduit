// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mcarray.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MCARRAY_H
#define CONDUIT_BLUEPRINT_MCARRAY_H

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
// -- conduit_blueprint_mcarray c interface  --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mcarray protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mcarray_verify(const conduit_node *cnode,
                                                           conduit_node *cinfo);


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mcarray sub protocol.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mcarray_verify_sub_protocol(const char *protocol,
                                                                        const conduit_node *cnode,
                                                                        conduit_node *cinfo);

//----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mcarray_is_interleaved(const conduit_node *cnode);

//----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mcarray_to_contiguous(const conduit_node *cnode,
                                                                  conduit_node *cdest);

//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API int conduit_blueprint_mcarray_to_interleaved(const conduit_node *cnode,
                                                                   conduit_node *cdest);

//-----------------------------------------------------------------------------
/// Interface to generate example mesh blueprint data.
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API void conduit_blueprint_mcarray_examples_xyz(const char *mcarray_type,
                                                                  conduit_index_t npts,
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
