// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mcarray_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.h"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mcarray protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_verify(const conduit_node *cnode,
                                 conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mcarray::verify(n,info);
}


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mcarray sub protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_verify_sub_protocol(const char *protocol,
                                              const conduit_node *cnode,
                                              conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mcarray::verify(std::string(protocol),n,info);
}


//----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_is_interleaved(const conduit_node *cnode)
{
    const Node &n = cpp_node_ref(cnode);
    return (int)blueprint::mcarray::is_interleaved(n);
}

//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_to_contiguous(const conduit_node *cnode,
                                        conduit_node *cdest)
{
    const Node &n = cpp_node_ref(cnode);
    Node &dest    = cpp_node_ref(cdest);
    return (int)blueprint::mcarray::to_contiguous(n,dest);
}

//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_to_interleaved(const conduit_node *cnode,
                                         conduit_node *cdest)
{
    const Node &n = cpp_node_ref(cnode);
    Node &dest    = cpp_node_ref(cdest);
    return (int)blueprint::mcarray::to_interleaved(n,dest);
}


//-----------------------------------------------------------------------------
/// Interface to generate example data
//-----------------------------------------------------------------------------
void
conduit_blueprint_mcarray_examples_xyz(const char *mcarray_type,
                                       conduit_index_t npts,
                                       conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mcarray::examples::xyz(std::string(mcarray_type),
                                      npts,
                                      res);
}



}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

