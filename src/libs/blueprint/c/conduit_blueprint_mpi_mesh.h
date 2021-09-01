// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MPI_MESH_H
#define CONDUIT_BLUEPRINT_MPI_MESH_H

//-----------------------------------------------------------------------------
// -- includes for the public conduit blueprint c interface -- 
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint_exports.h"
#include <mpi.h>

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- conduit_blueprint_mpi_mesh c interface  --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Partition a mesh
//-----------------------------------------------------------------------------
CONDUIT_BLUEPRINT_API void conduit_blueprint_mpi_mesh_partition(const conduit_node *cmesh,
                                                                const conduit_node *coptions,
                                                                conduit_node *coutput,
                                                                MPI_Comm comm);
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
