// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mpi_mesh.h"
#include "conduit_blueprint_mpi_mesh.hpp"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Partition a mesh
//-----------------------------------------------------------------------------
void
conduit_blueprint_mpi_mesh_partition(const conduit_node *cmesh,
                                     const conduit_node *coptions,
                                     conduit_node *coutput,
                                     MPI_Comm comm)
{
    const Node &mesh = cpp_node_ref(cmesh);
    const Node &options = cpp_node_ref(coptions);
    Node &output  = cpp_node_ref(coutput);
    conduit::blueprint::mpi::mesh::partition(mesh, options, output, comm);
}

//-----------------------------------------------------------------------------
/// Flatten a mesh to a table
//-----------------------------------------------------------------------------
void
conduit_blueprint_mpi_mesh_flatten(const conduit_node *cmesh,
                                   const conduit_node *coptions,
                                   conduit_node *coutput,
                                   MPI_Comm comm)
{
    const Node &mesh = cpp_node_ref(cmesh);
    const Node &options = cpp_node_ref(coptions);
    Node &output  = cpp_node_ref(coutput);
    conduit::blueprint::mpi::mesh::flatten(mesh, options, output, comm);
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
