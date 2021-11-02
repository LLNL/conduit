// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_table_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_table.h"

#include "conduit_node.hpp"
#include "conduit_blueprint_table.hpp"
#include "conduit_blueprint_table_examples.hpp"
#include "conduit_cpp_to_c.hpp"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Verify passed node conforms to the blueprint table protocol.
//-----------------------------------------------------------------------------
int conduit_blueprint_table_verify(const conduit_node *cnode,
                                   conduit_node *cinfo)
{
    const Node &node = cpp_node_ref(cnode);
    Node &info = cpp_node_ref(cinfo);
    return static_cast<int>(blueprint::table::verify(node, info));
}

//-----------------------------------------------------------------------------
/// Verify passed node conforms to given blueprint table sub protocol.
//-----------------------------------------------------------------------------
int conduit_blueprint_table_verify_sub_protocol(const char *protocol,
                                                const conduit_node *cnode,
                                                conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return static_cast<int>(blueprint::table::verify(std::string(protocol), n ,info));
}

//-----------------------------------------------------------------------------
/// Interface to generate example table blueprint data.
//-----------------------------------------------------------------------------
void conduit_blueprint_table_examples_basic(conduit_index_t nx,
                                            conduit_index_t ny,
                                            conduit_index_t nz,
                                            conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::table::examples::basic(nx, ny, nz, res);
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
