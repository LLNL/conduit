// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.h"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mesh protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mesh_verify(const conduit_node *cnode,
                              conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mesh::verify(n,info);
}


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mesh sub protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mesh_verify_sub_protocol(const char *protocol,
                                           const conduit_node *cnode,
                                           conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mesh::verify(std::string(protocol),n,info);
}


//-----------------------------------------------------------------------------
/// Generate mesh::index from valid mesh
//-----------------------------------------------------------------------------
void
conduit_blueprint_mesh_generate_index(const conduit_node *cmesh,
                                      const char *ref_path,
                                      conduit_index_t num_domains,
                                      conduit_node *cindex_out)
{
    const Node &mesh = cpp_node_ref(cmesh);
    Node &index_out  = cpp_node_ref(cindex_out);
    blueprint::mesh::generate_index(mesh,
                                    std::string(ref_path),
                                    num_domains,
                                    index_out);
}

//-----------------------------------------------------------------------------
/// Interface to generate example data
//-----------------------------------------------------------------------------
void
conduit_blueprint_mesh_examples_basic(const char *mesh_type,
                                      conduit_index_t nx,
                                      conduit_index_t ny,
                                      conduit_index_t nz,
                                      conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::basic(std::string(mesh_type),
                                     nx,ny,nz,
                                     res);
}

void
conduit_blueprint_mesh_examples_braid(const char *mesh_type,
                                      conduit_index_t nx,
                                      conduit_index_t ny,
                                      conduit_index_t nz,
                                      conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::braid(std::string(mesh_type),
                                     nx,ny,nz,
                                     res);
}

void
conduit_blueprint_mesh_examples_julia(conduit_index_t nx,
                                      conduit_index_t ny,
                                      conduit_float64 x_min,
                                      conduit_float64 x_max,
                                      conduit_float64 y_min,
                                      conduit_float64 y_max,
                                      conduit_float64 c_re,
                                      conduit_float64 c_im,
                                      conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::julia(nx,ny,
                                     x_min,x_max,
                                     y_min,y_max,
                                     c_re,c_im,
                                     res);
}

void
conduit_blueprint_mesh_examples_spiral(conduit_index_t ndomains,
                                       conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::spiral(ndomains,
                                      res);
}

void
conduit_blueprint_mesh_examples_polytess(conduit_index_t nlevels,
                                         conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::polytess(nlevels,
                                        res);
}

void
conduit_blueprint_mesh_examples_misc(const char *mesh_type,
                                     conduit_index_t nx,
                                     conduit_index_t ny,
                                     conduit_index_t nz,
                                     conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::misc(std::string(mesh_type),
                                    nx,ny,nz,
                                    res);
}



}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

