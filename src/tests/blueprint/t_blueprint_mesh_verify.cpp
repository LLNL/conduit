//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

std::vector<std::string> get_log_keywords()
{
    Node log_node;
    log::info(log_node,"","");
    log::optional(log_node,"","");
    log::error(log_node,"","");
    log::validation(log_node,false);
    return log_node.child_names();
}

std::vector<std::string> create_coordsys(const std::string& d1,
                                         const std::string& d2,
                                         const std::string& d3="")
{
    std::vector<std::string> dim_vector;

    dim_vector.push_back(d1);
    dim_vector.push_back(d2);

    if(d3 != "")
    {
        dim_vector.push_back(d3);
    }

    return dim_vector;
}

const std::vector<std::string> LOG_KEYWORDS = get_log_keywords();

const std::vector<std::string> LOGICAL_COORDSYS = create_coordsys("i","j","k");
const std::vector<std::string> CARTESIAN_COORDSYS = create_coordsys("x","y","z");
const std::vector<std::string> SPHERICAL_COORDSYS = create_coordsys("r","theta","phi");
const std::vector<std::string> CYLINDRICAL_COORDSYS = create_coordsys("r","z");

const std::vector<std::string> COORDINATE_COORDSYSS[] =
    {CARTESIAN_COORDSYS, CYLINDRICAL_COORDSYS, SPHERICAL_COORDSYS};

typedef bool (*VerifyFun)(const Node&, Node&);

/// Helper Functions ///

bool is_valid_coordsys(bool (*coordsys_valid_fun)(const Node&, Node&),
                       const std::vector<std::string>& coordsys)
{
    Node n, info;

    bool is_valid = true;
    for(size_t ci = 0; ci < coordsys.size(); ci++)
    {
        const std::string& coordsys_dim = coordsys[ci];

        n[coordsys_dim].set("test");
        is_valid &= !coordsys_valid_fun(n,info);

        n[coordsys_dim].set(10);
        is_valid &= coordsys_valid_fun(n,info);

        // FIXME: The coordinate system checking functions shouldn't accept
        // systems such as (y) or (x, z); all successive dimensions should
        // require the existence of previous coordsys dimensions.
        /*
        if( ci > 0 )
        {
            const std::string& prev_dim = coordsys[ci-1];
            n.remove(prev_dim);
            is_valid &= !coordsys_valid_fun(n,info);
            n[coordsys_dim].set(10);
        }
        */
    }

    return is_valid;
}


bool has_consistent_validity(const Node &n)
{
    // TODO(JRC): This function will have problems for given nodes containing
    // nested lists.
    bool is_consistent = !n.dtype().is_object() ||
        (n.has_child("valid") && n["valid"].dtype().is_string() &&
        (n["valid"].as_string() == "true" || n["valid"].as_string() == "false"));

    NodeConstIterator itr = n.children();
    while(itr.has_next())
    {
        const Node &chld= itr.next();
        const std::string chld_name = itr.name();
        if(std::find(LOG_KEYWORDS.begin(), LOG_KEYWORDS.end(), chld_name) ==
            LOG_KEYWORDS.end())
        {
            is_consistent &= has_consistent_validity(chld);
            if(is_consistent)
            {
                bool n_valid = n["valid"].as_string() == "true";
                bool c_valid = chld["valid"].as_string() == "true";
                is_consistent &= !(n_valid && !c_valid);
            }
        }
    }

    return is_consistent;
}

/// Wrapper Functions ///

bool verify_coordset_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("coordset",n,info);
}

bool verify_topology_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("topology",n,info);
}

bool verify_matset_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("matset",n,info);
}

bool verify_field_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("field",n,info);
}

bool verify_adjset_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("adjset",n,info);
}

bool verify_nestset_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("nestset",n,info);
}

bool verify_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("index",n,info);
}

bool verify_coordset_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("coordset/index",n,info);
}

bool verify_topology_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("topology/index",n,info);
}

bool verify_matset_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("matset/index",n,info);
}

bool verify_field_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("field/index",n,info);
}

bool verify_adjset_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("adjset/index",n,info);
}

bool verify_nestset_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("nestset/index",n,info);
}

bool verify_mesh_multi_domain_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::is_multi_domain(n);
}

void CHECK_MESH(VerifyFun verify, const Node &n, Node &info, bool expected)
{
    EXPECT_EQ(verify(n, info), expected);
    EXPECT_TRUE(has_consistent_validity(info));
}

/// Mesh Coordinate Set Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_logical_dims)
{
    VerifyFun verify_coordset_logical = blueprint::mesh::logical_dims::verify;

    Node n, info;
    CHECK_MESH(verify_coordset_logical,n,info,false);

    EXPECT_TRUE(is_valid_coordsys(verify_coordset_logical,LOGICAL_COORDSYS));

    EXPECT_FALSE(is_valid_coordsys(verify_coordset_logical,CARTESIAN_COORDSYS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_origin)
{
    VerifyFun verify_uniform_origin = blueprint::mesh::coordset::uniform::origin::verify;

    Node n, info;
    // FIXME: The origin verification function shouldn't accept an empty node.
    // CHECK_MESH(verify_uniform_origin,n,info,false);

    EXPECT_TRUE(is_valid_coordsys(verify_uniform_origin,CARTESIAN_COORDSYS));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_origin,SPHERICAL_COORDSYS));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_origin,CYLINDRICAL_COORDSYS));

    EXPECT_FALSE(is_valid_coordsys(verify_uniform_origin,LOGICAL_COORDSYS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_spacing)
{
    VerifyFun verify_uniform_spacing = blueprint::mesh::coordset::uniform::spacing::verify;

    Node n, info;
    // FIXME: The spacing verification function shouldn't accept an empty node.
    // CHECK_MESH(verify_uniform_spacing,n,info,false);

    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dx","dy","dz")));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dr","dtheta","dphi")));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dr","dz")));

    EXPECT_FALSE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("di","dj","dk")));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform)
{
    VerifyFun verify_uniform_coordset = blueprint::mesh::coordset::uniform::verify;

    Node n, info;
    CHECK_MESH(verify_uniform_coordset,n,info,false);

    n["type"].set("uniform");
    CHECK_MESH(verify_uniform_coordset,n,info,false);

    n["dims"]["i"].set(1);
    n["dims"]["j"].set(2);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    n["dims"]["k"].set("test");
    CHECK_MESH(verify_uniform_coordset,n,info,false);
    n["dims"]["k"].set(3);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    Node dims = n["dims"];
    n.remove("dims");

    n["origin"]["x"].set(10);
    n["origin"]["y"].set(20);
    CHECK_MESH(verify_uniform_coordset,n,info,false);

    n["dims"].set(dims);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    n["origin"]["z"].set("test");
    CHECK_MESH(verify_uniform_coordset,n,info,false);
    n["origin"]["z"].set(30);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    n["spacing"]["dx"].set(0.1);
    n["spacing"]["dy"].set(0.2);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    n["spacing"]["dz"].set("test");
    CHECK_MESH(verify_uniform_coordset,n,info,false);
    n["spacing"]["dz"].set(0.3);
    CHECK_MESH(verify_uniform_coordset,n,info,true);

    n["type"].set("rectilinear");
    CHECK_MESH(verify_uniform_coordset,n,info,false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_rectilinear)
{
    VerifyFun verify_rectilinear_coordset = blueprint::mesh::coordset::rectilinear::verify;

    Node n, info;
    CHECK_MESH(verify_rectilinear_coordset,n,info,false);

    n["values"].set("test");
    CHECK_MESH(verify_rectilinear_coordset,n,info,false);

    n["type"].set("rectilinear");
    CHECK_MESH(verify_rectilinear_coordset,n,info,false);

    for(size_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(size_t cj = 0; cj < coord_coordsys.size(); cj++)
        {
            n["values"][coord_coordsys[cj]].set(DataType::float64(10));
            CHECK_MESH(verify_rectilinear_coordset,n,info,true);
            // info.print();
        }
    }

    // check case where number of elements for each child doesn't match
    // (rectilinear coordsets use cross product of input coord arrays, they
    //  don't need to be mcarrays)
    for(size_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(size_t cj = 0; cj < coord_coordsys.size(); cj++)
        {
            n["values"][coord_coordsys[cj]].set(DataType::float64(cj + 5));
            CHECK_MESH(verify_rectilinear_coordset,n,info,true);
            // info.print();
        }
    }

    n["type"].set("uniform");
    CHECK_MESH(verify_rectilinear_coordset,n,info,false);


    // FIXME: The logical coordinate system shouldn't be an accepted value
    // for the rectilinear verify function.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_COORDSYS.size(); ci++)
    {
        n["values"][LOGICAL_COORDSYS[ci]].set(DataType::float64(10));
        CHECK_MESH(verify_rectilinear_coordset,n,info,false);
    }
    */
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_explicit)
{
    VerifyFun verify_explicit_coordset = blueprint::mesh::coordset::_explicit::verify;

    Node n, info;
    CHECK_MESH(verify_explicit_coordset,n,info,false);

    n["values"].set("test");
    CHECK_MESH(verify_explicit_coordset,n,info,false);

    n["type"].set("explicit");
    CHECK_MESH(verify_explicit_coordset,n,info,false);

    for(size_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(size_t cj = 0; cj < coord_coordsys.size(); cj++)
        {
            n["values"][coord_coordsys[cj]].set(DataType::float64(10));
            CHECK_MESH(verify_explicit_coordset,n,info,true);
        }
    }

    n["type"].set("uniform");
    CHECK_MESH(verify_explicit_coordset,n,info,false);

    // FIXME: The logical coordinate system shouldn't be an accepted value
    // for the explicit verify function.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_COORDSYS.size(); ci++)
    {
        n["values"][LOGICAL_COORDSYS[ci]].set(DataType::float64(10));
        CHECK_MESH(verify_explicit_coordset,n,info,false);
    }
    */
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_types)
{
    VerifyFun verify_coordset = blueprint::mesh::coordset::verify;

    Node n, info;

    const std::string coordset_types[] = {"uniform", "rectilinear", "explicit"};
    const std::string coordset_fids[] = {"uniform", "rectilinear", "quads"};
    for(index_t ci = 0; ci < 3; ci++)
    {
        n.reset();
        blueprint::mesh::examples::braid(coordset_fids[ci],10,10,1,n);
        Node& coordset_node = n["coordsets/coords"];
        CHECK_MESH(verify_coordset,coordset_node,info,true);

        coordset_node["type"].set(0);
        CHECK_MESH(verify_coordset,coordset_node,info,false);

        coordset_node["type"].set("unstructured");
        CHECK_MESH(verify_coordset,coordset_node,info,false);

        if(ci != 2)
        {
            coordset_node["type"].set(coordset_types[2]);
            EXPECT_FALSE(blueprint::mesh::topology::verify(coordset_node,info));
        }

        coordset_node["type"].set(coordset_types[ci]);
        CHECK_MESH(verify_coordset,coordset_node,info,true);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_coordsys)
{
    VerifyFun verify_coordset_coordsys = blueprint::mesh::coordset::coord_system::verify;

    Node n, info;
    CHECK_MESH(verify_coordset_coordsys,n,info,false);

    const std::string coordsys_types[] = {"cartesian", "cylindrical", "spherical"};
    for(index_t ci = 0; ci < 3; ci++)
    {
        n.reset();
        info.reset();

        n["type"].set(coordsys_types[ci]);
        CHECK_MESH(verify_coordset_coordsys,n,info,false);

        n["axes"].set(0);
        CHECK_MESH(verify_coordset_coordsys,n,info,false);

        n["axes"].reset();
        const std::vector<std::string>& coordsys = COORDINATE_COORDSYSS[ci];
        for(size_t ai = 0; ai < coordsys.size(); ai++)
        {
            n["axes"][coordsys[ai]].set(10);
            CHECK_MESH(verify_coordset_coordsys,n,info,true);
        }

        n["type"].set(coordsys_types[(ci == 0) ? 2 : ci - 1]);
        CHECK_MESH(verify_coordset_coordsys,n,info,false);

        n["type"].set("barycentric");
        CHECK_MESH(verify_coordset_coordsys,n,info,false);

        n["type"].set(10);
        CHECK_MESH(verify_coordset_coordsys,n,info,false);
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_general)
{
    VerifyFun verify_coordset_funs[] = {
        blueprint::mesh::coordset::verify,
        verify_coordset_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_coordset = verify_coordset_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_coordset,mesh,info,false);

        blueprint::mesh::examples::braid("uniform",10,10,10,mesh);
        Node& n = mesh["coordsets"]["coords"];

        n.remove("type");
        CHECK_MESH(verify_coordset,n,info,false);
        n["type"].set("structured");
        CHECK_MESH(verify_coordset,n,info,false);
        n["type"].set("rectilinear");
        CHECK_MESH(verify_coordset,n,info,false);

        n["type"].set("uniform");
        CHECK_MESH(verify_coordset,n,info,true);
    }
}

/// Mesh Topology Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_points)
{
    VerifyFun verify_points_topology = blueprint::mesh::topology::points::verify;
    Node n, info;

    CHECK_MESH(verify_points_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set("points");
    CHECK_MESH(verify_points_topology,n,info,true);

    n["coordset"].set(1);
    CHECK_MESH(verify_points_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set(1);
    CHECK_MESH(verify_points_topology,n,info,false);

    n["type"].set("uniform");
    CHECK_MESH(verify_points_topology,n,info,false);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_uniform)
{
    VerifyFun verify_uniform_topology = blueprint::mesh::topology::uniform::verify;
    Node n, info;

    CHECK_MESH(verify_uniform_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set("uniform");
    CHECK_MESH(verify_uniform_topology,n,info,true);

    n["coordset"].set(1);
    CHECK_MESH(verify_uniform_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set(1);
    CHECK_MESH(verify_uniform_topology,n,info,false);

    n["type"].set("points");
    CHECK_MESH(verify_uniform_topology,n,info,false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_rectilinear)
{
    VerifyFun verify_rectilinear_topology = blueprint::mesh::topology::rectilinear::verify;
    Node n, info;

    CHECK_MESH(verify_rectilinear_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set("rectilinear");
    CHECK_MESH(verify_rectilinear_topology,n,info,true);

    n["coordset"].set(1);
    CHECK_MESH(verify_rectilinear_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set(1);
    CHECK_MESH(verify_rectilinear_topology,n,info,false);

    n["type"].set("points");
    CHECK_MESH(verify_rectilinear_topology,n,info,false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_structured)
{
    VerifyFun verify_structured_topology = blueprint::mesh::topology::structured::verify;

    Node n, info;
    CHECK_MESH(verify_structured_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set("structured");
    CHECK_MESH(verify_structured_topology,n,info,false);

    n["elements"].set(0);
    CHECK_MESH(verify_structured_topology,n,info,false);

    n["elements"].reset();
    n["elements"]["dims"].set(0);
    CHECK_MESH(verify_structured_topology,n,info,false);

    n["elements"]["dims"].reset();
    n["elements"]["dims"]["x"].set(5);
    n["elements"]["dims"]["y"].set(10);
    CHECK_MESH(verify_structured_topology,n,info,false);

    n["elements"]["dims"].reset();
    n["elements"]["dims"]["i"].set(15);
    n["elements"]["dims"]["j"].set(20);
    CHECK_MESH(verify_structured_topology,n,info,true);

    n["elements"]["dims"]["k"].set(25);
    CHECK_MESH(verify_structured_topology,n,info,true);

    n["type"].set("unstructured");
    CHECK_MESH(verify_structured_topology,n,info,false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_unstructured)
{
    VerifyFun verify_unstructured_topology = blueprint::mesh::topology::unstructured::verify;

    Node n, info;
    CHECK_MESH(verify_unstructured_topology,n,info,false);

    n["coordset"].set("coords");
    n["type"].set("unstructured");
    CHECK_MESH(verify_unstructured_topology,n,info,false);

    n["elements"].set(0);
    CHECK_MESH(verify_unstructured_topology,n,info,false);

    { // Single Shape Topology Tests //
        n["elements"].reset();
        n["elements"]["shape"].set("polygon");
        CHECK_MESH(verify_unstructured_topology,n,info,false);

        n["elements"]["shape"].set("quad");
        CHECK_MESH(verify_unstructured_topology,n,info,false);

        n["elements"]["connectivity"].set("quad");
        CHECK_MESH(verify_unstructured_topology,n,info,false);
        n["elements"]["connectivity"].set(DataType::float64(10));
        CHECK_MESH(verify_unstructured_topology,n,info,false);

        n["elements"]["connectivity"].set(DataType::int32(1));
        CHECK_MESH(verify_unstructured_topology,n,info,true);
        n["elements"]["connectivity"].set(DataType::int32(10));
        CHECK_MESH(verify_unstructured_topology,n,info,true);

        n["type"].set("structured");
        CHECK_MESH(verify_unstructured_topology,n,info,false);
        n["type"].set("unstructured");
        CHECK_MESH(verify_unstructured_topology,n,info,true);
    }

    { // Mixed Shape Topology List Tests //
        n["elements"].reset();

        n["elements"]["a"].set(0);
        CHECK_MESH(verify_unstructured_topology,n,info,false);

        n["elements"]["a"]["shape"].set("quad");
        n["elements"]["a"]["connectivity"].set(DataType::int32(5));
        CHECK_MESH(verify_unstructured_topology,n,info,true);

        n["elements"]["b"]["shape"].set("polygon");
        CHECK_MESH(verify_unstructured_topology,n,info,false);
        n["elements"]["b"]["shape"].set("quad");
        CHECK_MESH(verify_unstructured_topology,n,info,false);
        n["elements"]["b"]["connectivity"].set(DataType::float32(3));
        CHECK_MESH(verify_unstructured_topology,n,info,false);
        n["elements"]["b"]["connectivity"].set(DataType::int32(1));
        CHECK_MESH(verify_unstructured_topology,n,info,true);

        n["elements"]["c"]["shape"].set("tri");
        n["elements"]["c"]["connectivity"].set(DataType::int32(5));
        CHECK_MESH(verify_unstructured_topology,n,info,true);

        n["type"].set("structured");
        CHECK_MESH(verify_unstructured_topology,n,info,false);
    }

    { // Multiple Shape Topology Stream Tests //
        // FIXME: Implement once multiple unstructured shape topologies are implemented.
        n["elements"].reset();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_types)
{
    VerifyFun verify_topology = blueprint::mesh::topology::verify;

    Node n, info;

    const std::string topology_types[] = {"points", "uniform", "rectilinear", "structured", "unstructured"};
    const std::string topology_fids[] = {"points_implicit", "uniform", "rectilinear", "structured", "quads"};
    for(index_t ti = 0; ti < 5; ti++)
    {
        n.reset();
        blueprint::mesh::examples::braid(topology_fids[ti],10,10,1,n);
        Node& topology_node = n["topologies/mesh"];
        CHECK_MESH(verify_topology,topology_node,info,true);

        topology_node["type"].set(0);
        CHECK_MESH(verify_topology,topology_node,info,false);

        topology_node["type"].set("explicit");
        CHECK_MESH(verify_topology,topology_node,info,false);

        if(ti != 4)
        {
            topology_node["type"].set(topology_types[4]);
            CHECK_MESH(verify_topology,topology_node,info,false);
        }

        topology_node["type"].set(topology_types[ti]);
        CHECK_MESH(verify_topology,topology_node,info,true);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_shape)
{
    VerifyFun verify_topology = blueprint::mesh::topology::verify;

    Node n, info;

    const std::string topology_shapes[] = {"point", "line", "tri", "quad", "tet", "hex"};
    const std::string topology_fids[] = {"points", "lines", "tris", "quads", "tets", "hexs"};
    for(index_t ti = 0; ti < 6; ti++)
    {
        n.reset();
        blueprint::mesh::examples::braid(topology_fids[ti],10,10,1,n);
        Node& topology_node = n["topologies/mesh"];
        CHECK_MESH(verify_topology,topology_node,info,true);

        topology_node["elements/shape"].set(0);
        CHECK_MESH(verify_topology,topology_node,info,false);

        topology_node["elements/shape"].set("unstructured");
        CHECK_MESH(verify_topology,topology_node,info,false);

        topology_node["elements/shape"].set(topology_shapes[ti]);
        CHECK_MESH(verify_topology,topology_node,info,true);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_general)
{
    VerifyFun verify_topology_funs[] = {
        blueprint::mesh::topology::verify,
        verify_topology_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_topology = verify_topology_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_topology,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        Node& n = mesh["topologies"]["mesh"];

        { // Type Field Tests //
            n.remove("type");
            CHECK_MESH(verify_topology,n,info,false);
            n["type"].set("explicit");
            CHECK_MESH(verify_topology,n,info,false);

            // FIXME: Remove the comments from the following line once the verify functions
            // for uniform and rectilinear topologies have been implemented.
            const std::string topology_types[] = {/*"uniform", "rectilinear", */"structured"};
            for(index_t ti = 0; ti < 1; ti++)
            {
                n["type"].set(topology_types[ti]);
                CHECK_MESH(verify_topology,n,info,false);
            }

            n["type"].set("unstructured");
            CHECK_MESH(verify_topology,n,info,true);
        }

        { // Coordset Field Tests //
            n.remove("coordset");
            CHECK_MESH(verify_topology,n,info,false);

            n["coordset"].set(0);
            CHECK_MESH(verify_topology,n,info,false);

            n["coordset"].set("coords");
            CHECK_MESH(verify_topology,n,info,true);
        }

        { // Grid Function Field Tests //
            n["grid_function"].set(10);
            CHECK_MESH(verify_topology,n,info,false);
            n["grid_function"].set("coords_gf");
            CHECK_MESH(verify_topology,n,info,true);
        }
    }
}

/// Mesh Matsets Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, matset_general)
{
    VerifyFun verify_matset_funs[] = {
        blueprint::mesh::matset::verify,
        verify_matset_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_matset = verify_matset_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_matset,mesh,info,false);

        blueprint::mesh::examples::misc("matsets",10,10,1,mesh);
        Node& n = mesh["matsets"]["mesh"];
        CHECK_MESH(verify_matset,n,info,true);

        { // Topology Field Tests //
            n.remove("topology");
            CHECK_MESH(verify_matset,n,info,false);
            n["topology"].set(10);
            CHECK_MESH(verify_matset,n,info,false);
            n["topology"].set("mesh");
            CHECK_MESH(verify_matset,n,info,true);
        }

        { // Volume Fractions Field Tests //
            Node vfs = n["volume_fractions"];

            n.remove("volume_fractions");
            CHECK_MESH(verify_matset,n,info,false);
            n["volume_fractions"].set("values");
            CHECK_MESH(verify_matset,n,info,false);
            n["volume_fractions"].set(DataType::float64(10));
            CHECK_MESH(verify_matset,n,info,false);

            n["volume_fractions"].reset();
            n["volume_fractions"]["x"].set("Hello, ");
            n["volume_fractions"]["y"].set("World!");
            CHECK_MESH(verify_matset,n,info,false);

            n["volume_fractions"].reset();
            n["volume_fractions"]["m1"].set(DataType::float64(5));
            n["volume_fractions"]["m2"].set(DataType::float64(5));
            CHECK_MESH(verify_matset,n,info,true);

            n["volume_fractions"].reset();
            n["volume_fractions"].set(vfs);
            CHECK_MESH(verify_matset,n,info,true);
        }
    }
}

/// Mesh Field Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_general)
{
    VerifyFun verify_field_funs[] = {
        blueprint::mesh::field::verify,
        verify_field_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_field = verify_field_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_field,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        Node& n = mesh["fields"]["braid"];

        { // Topology Field Tests //
            n.remove("topology");
            CHECK_MESH(verify_field,n,info,false);
            n["topology"].set(10);
            CHECK_MESH(verify_field,n,info,false);
            n["topology"].set("mesh");
            CHECK_MESH(verify_field,n,info,true);
        }

        { // Values Field Tests //
            Node values = n["values"];

            n.remove("values");
            CHECK_MESH(verify_field,n,info,false);
            n["values"].set("values");
            CHECK_MESH(verify_field,n,info,false);
            n["values"].set(DataType::float64(10));
            CHECK_MESH(verify_field,n,info,true);

            n["values"].reset();
            n["values"]["x"].set("Hello, ");
            n["values"]["y"].set("World!");
            CHECK_MESH(verify_field,n,info,false);

            n["values"].reset();
            n["values"]["x"].set(DataType::float64(5));
            n["values"]["y"].set(DataType::float64(5));
            CHECK_MESH(verify_field,n,info,true);

            n["values"].set(values);
            CHECK_MESH(verify_field,n,info,true);
        }

        { // Association/Basis Field Tests //
            n.remove("association");
            CHECK_MESH(verify_field,n,info,false);

            n["association"].set("zone");
            CHECK_MESH(verify_field,n,info,false);
            n["association"].set("vertex");
            CHECK_MESH(verify_field,n,info,true);

            n.remove("association");
            n["basis"].set(0);
            CHECK_MESH(verify_field,n,info,false);
            n["basis"].set("basis");
            CHECK_MESH(verify_field,n,info,true);

            n["association"].set("vertex");
            CHECK_MESH(verify_field,n,info,true);
        }
    }
}

/// Mesh Domain Adjacencies Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, adjset_general)
{
    VerifyFun verify_adjset_funs[] = {
        blueprint::mesh::adjset::verify,
        verify_adjset_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_adjset = verify_adjset_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_adjset,mesh,info,false);

        blueprint::mesh::examples::misc("adjsets",10,10,1,mesh);
        Node& n = mesh.child(0)["adjsets"].child(0);
        CHECK_MESH(verify_adjset,n,info,true);

        { // Topology Field Tests //
            n.remove("topology");
            CHECK_MESH(verify_adjset,n,info,false);
            n["topology"].set(10);
            CHECK_MESH(verify_adjset,n,info,false);
            n["topology"].set("mesh");
            CHECK_MESH(verify_adjset,n,info,true);
        }

        { // Groups Field Tests //
            Node groups = n["groups"];

            n.remove("groups");
            CHECK_MESH(verify_adjset,n,info,false);
            n["groups"].set("groups");
            CHECK_MESH(verify_adjset,n,info,false);
            n["groups"].set(DataType::float64(10));
            CHECK_MESH(verify_adjset,n,info,false);

            n["groups"].reset();
            n["groups"]["g1"].set("Hello, ");
            CHECK_MESH(verify_adjset,n,info,false);
            n["groups"]["g2"].set("World!");
            CHECK_MESH(verify_adjset,n,info,false);

            n["groups"].reset();
            n["groups"]["g1"]["neighbors"].set(DataType::int32(5));
            CHECK_MESH(verify_adjset,n,info,false);
            n["groups"]["g1"]["values"].set(DataType::float32(5));
            CHECK_MESH(verify_adjset,n,info,false);
            n["groups"]["g1"]["values"].set(DataType::int32(5));
            CHECK_MESH(verify_adjset,n,info,true);

            n["groups"].reset();
            n["groups"]["g1"]["neighbors"].set(DataType::int32(5));
            n["groups"]["g1"]["values"].set(DataType::int32(5));
            CHECK_MESH(verify_adjset,n,info,true);
            n["groups"]["g2"]["neighbors"].set(DataType::int32(5));
            n["groups"]["g2"]["values"].set(DataType::int32(5));
            CHECK_MESH(verify_adjset,n,info,true);

            n["groups"].reset();
            n["groups"].set(groups);
            CHECK_MESH(verify_adjset,n,info,true);
        }

        { // Association Field Tests //
            n.remove("association");
            CHECK_MESH(verify_adjset,n,info,false);

            n["association"].set("zone");
            CHECK_MESH(verify_adjset,n,info,false);
            n["association"].set("vertex");
            CHECK_MESH(verify_adjset,n,info,true);
        }
    }
}

/// Mesh Domain Nesting (AMR) Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, nestset_types)
{
    VerifyFun verify_nestset = blueprint::mesh::nestset::verify;

    Node n, info;

    const std::string nestset_types[] = {"parent", "child"};
    for(index_t ti = 0; ti < 2; ti++)
    {
        n.reset();
        blueprint::mesh::examples::misc("nestsets",5,5,1,n);
        Node& nestset_node = n.child(0)["nestsets/mesh_nest"];
        Node& window_node = nestset_node["windows"].child(0);
        CHECK_MESH(verify_nestset,nestset_node,info,true);

        window_node["domain_type"].set(0);
        CHECK_MESH(verify_nestset,nestset_node,info,false);

        window_node["domain_type"].set("ancestor");
        CHECK_MESH(verify_nestset,nestset_node,info,false);

        window_node["domain_type"].set(nestset_types[ti]);
        CHECK_MESH(verify_nestset,nestset_node,info,true);
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, nestset_general)
{
    VerifyFun verify_nestset_funs[] = {
        blueprint::mesh::nestset::verify,
        verify_nestset_protocol};
    const std::string nestset_logical_fields[3] = {"ratio", "origin", "dims"};

    Node logical_template, window_template_slim, window_template_full;
    {
        logical_template["i"].set(DataType::int32(1));
        logical_template["j"].set(DataType::int32(1));

        window_template_slim["domain_id"].set(DataType::int32(1));
        window_template_slim["domain_type"].set("child");
        window_template_slim["ratio"].set(logical_template);

        window_template_full.set(window_template_slim);
        window_template_full["origin"].set(logical_template);
        window_template_full["dims"].set(logical_template);
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_nestset = verify_nestset_funs[fi];

        Node mesh, info;
        CHECK_MESH(verify_nestset,mesh,info,false);

        blueprint::mesh::examples::misc("nestsets",5,5,1,mesh);
        Node& n = mesh.child(0)["nestsets"].child(0);
        CHECK_MESH(verify_nestset,n,info,true);

        { // Topology Field Tests //
            n.remove("topology");
            CHECK_MESH(verify_nestset,n,info,false);
            n["topology"].set(10);
            CHECK_MESH(verify_nestset,n,info,false);
            n["topology"].set("mesh");
            CHECK_MESH(verify_nestset,n,info,true);
        }

        { // Windows Field Tests //
            Node windows = n["windows"];

            n.remove("windows");
            CHECK_MESH(verify_nestset,n,info,false);
            n["windows"].set("windows");
            CHECK_MESH(verify_nestset,n,info,false);
            n["windows"].set(DataType::float64(10));
            CHECK_MESH(verify_nestset,n,info,false);

            n["windows"].reset();
            n["windows"]["w1"].set("Hello, ");
            CHECK_MESH(verify_nestset,n,info,false);
            n["windows"]["w2"].set("World!");
            CHECK_MESH(verify_nestset,n,info,false);

            n["windows"].reset();
            n["windows"]["w1"].set(window_template_slim);
            CHECK_MESH(verify_nestset,n,info,true);
            n["windows"]["w1"]["domain_id"].set(DataType::float32(1));
            CHECK_MESH(verify_nestset,n,info,false);

            n["windows"].reset();
            n["windows"]["w1"].set(window_template_slim);
            CHECK_MESH(verify_nestset,n,info,true);
            n["windows"]["w1"]["domain_type"].set(0);
            CHECK_MESH(verify_nestset,n,info,false);
            n["windows"]["w1"]["domain_type"].set("ancestor");
            CHECK_MESH(verify_nestset,n,info,false);

            for(index_t fi = 0; fi < 3; fi++)
            {
                const std::string& logical_field = nestset_logical_fields[fi];

                n["windows"].reset();
                n["windows"]["w1"].set(window_template_full);
                CHECK_MESH(verify_nestset,n,info,true);
                n["windows"]["w1"][logical_field].set(0);
                CHECK_MESH(verify_nestset,n,info,false);
                n["windows"]["w1"][logical_field].reset();
                n["windows"]["w1"][logical_field]["i"].set(DataType::int32(1));
                CHECK_MESH(verify_nestset,n,info,false);
                n["windows"]["w1"][logical_field]["j"].set(DataType::int32(1));
                CHECK_MESH(verify_nestset,n,info,true);
            }

            n["windows"].reset();
            n["windows"]["w1"].set(window_template_full);
            CHECK_MESH(verify_nestset,n,info,true);
            n["windows"]["w2"].set(window_template_slim);
            CHECK_MESH(verify_nestset,n,info,true);
            n["windows"]["w3"].set(window_template_full);
            CHECK_MESH(verify_nestset,n,info,true);

            n["windows"].reset();
            n["windows"].set(windows);
            CHECK_MESH(verify_nestset,n,info,true);
        }

        { // Association Field Tests //
            n.remove("association");
            CHECK_MESH(verify_nestset,n,info,false);

            n["association"].set("zone");
            CHECK_MESH(verify_nestset,n,info,false);
            n["association"].set("element");
            CHECK_MESH(verify_nestset,n,info,true);
        }
    }
}

/// Mesh Index Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_coordset)
{
    VerifyFun verify_coordset_index_funs[] = {
        blueprint::mesh::coordset::index::verify,
        verify_coordset_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_coordset_index = verify_coordset_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_coordset_index,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& cindex = index["coordsets"]["coords"];
        CHECK_MESH(verify_coordset_index,cindex,info,true);

        { // Type Field Tests //
            cindex.remove("type");
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["type"].set("undefined");
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["type"].set("explicit");
            CHECK_MESH(verify_coordset_index,cindex,info,true);
        }

        { // Coord System Field Tests //
            Node coordsys = cindex["coord_system"];
            cindex.remove("coord_system");

            CHECK_MESH(verify_coordset_index,cindex,info,false);
            cindex["coord_system"].set("invalid");
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["coord_system"].reset();
            cindex["coord_system"]["type"].set("logical");
            cindex["coord_system"]["axes"]["i"].set(10);
            cindex["coord_system"]["axes"]["j"].set(10);
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["coord_system"].reset();
            cindex["coord_system"].set(coordsys);
            CHECK_MESH(verify_coordset_index,cindex,info,true);
        }

        { // Path Field Tests //
            cindex.remove("path");
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["path"].set(5);
            CHECK_MESH(verify_coordset_index,cindex,info,false);

            cindex["path"].set("path");
            CHECK_MESH(verify_coordset_index,cindex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_topology)
{
    VerifyFun verify_topo_index_funs[] = {
        blueprint::mesh::topology::index::verify,
        verify_topology_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_topo_index = verify_topo_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_topo_index,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& tindex = index["topologies"]["mesh"];
        CHECK_MESH(verify_topo_index,tindex,info,true);

        { // Type Field Tests //
            tindex.remove("type");
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["type"].set("undefined");
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["type"].set("unstructured");
            CHECK_MESH(verify_topo_index,tindex,info,true);
        }

        { // Coordset Field Tests //
            tindex.remove("coordset");
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["coordset"].set(0);
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["coordset"].set("path");
            CHECK_MESH(verify_topo_index,tindex,info,true);
        }

        { // Path Field Tests //
            tindex.remove("path");
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["path"].set(5);
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["path"].set("path");
            CHECK_MESH(verify_topo_index,tindex,info,true);
        }

        { // Grid Function Field Tests //
            tindex["grid_function"].set(10);
            CHECK_MESH(verify_topo_index,tindex,info,false);

            tindex["grid_function"].set("path");
            CHECK_MESH(verify_topo_index,tindex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_matset)
{
    VerifyFun verify_matset_index_funs[] = {
        blueprint::mesh::matset::index::verify,
        verify_matset_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_matset_index = verify_matset_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_matset_index,mesh,info,false);

        blueprint::mesh::examples::misc("matsets",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& mindex = index["matsets"]["mesh"];
        CHECK_MESH(verify_matset_index,mindex,info,true);

        { // Topology Field Tests //
            mindex.remove("topology");
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["topology"].set(0);
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["topology"].set("path");
            CHECK_MESH(verify_matset_index,mindex,info,true);
        }

        { // Materials Field Tests //
            mindex.remove("materials");
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["materials"];
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["materials/mat1"].set(1);
            CHECK_MESH(verify_matset_index,mindex,info,true);
            mindex["materials/mat2"].set(2);
            CHECK_MESH(verify_matset_index,mindex,info,true);
        }

        { // Path Field Tests //
            mindex.remove("path");
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["path"].set(5);
            CHECK_MESH(verify_matset_index,mindex,info,false);

            mindex["path"].set("path");
            CHECK_MESH(verify_matset_index,mindex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_field)
{
    VerifyFun verify_field_index_funs[] = {
        blueprint::mesh::field::index::verify,
        verify_field_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_field_index = verify_field_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_field_index,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& findex = index["fields"]["braid"];
        CHECK_MESH(verify_field_index,findex,info,true);

        { // Topology Field Tests //
            Node topo = findex["topology"];
            findex.remove("topology");

            CHECK_MESH(verify_field_index,findex,info,false);
            findex["topology"].set(0);
            CHECK_MESH(verify_field_index,findex,info,false);

            findex["topology"].set("path");
            CHECK_MESH(verify_field_index,findex,info,true);

            findex["topology"].reset();
            findex["topology"].set(topo);
            CHECK_MESH(verify_field_index,findex,info,true);
        }

        { // Component Count Field Tests //
            Node comps = findex["number_of_components"];
            findex.remove("number_of_components");

            CHECK_MESH(verify_field_index,findex,info,false);
            findex["number_of_components"].set("three");
            CHECK_MESH(verify_field_index,findex,info,false);

            findex["number_of_components"].set(3);
            CHECK_MESH(verify_field_index,findex,info,true);

            findex["number_of_components"].reset();
            findex["number_of_components"].set(comps);
            CHECK_MESH(verify_field_index,findex,info,true);
        }

        { // Path Field Tests //
            Node path = findex["path"];
            findex.remove("path");

            CHECK_MESH(verify_field_index,findex,info,false);
            findex["path"].set(0);
            CHECK_MESH(verify_field_index,findex,info,false);

            findex["path"].set("path");
            CHECK_MESH(verify_field_index,findex,info,true);

            findex["path"].reset();
            findex["path"].set(path);
            CHECK_MESH(verify_field_index,findex,info,true);
        }

        { // Association Field Tests //
            findex["association"].set("zone");
            CHECK_MESH(verify_field_index,findex,info,false);
            findex["association"].set("vertex");
            CHECK_MESH(verify_field_index,findex,info,true);

            findex.remove("association");
            findex["basis"].set(0);
            CHECK_MESH(verify_field_index,findex,info,false);
            findex["basis"].set("basis");
            CHECK_MESH(verify_field_index,findex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_adjset)
{
    VerifyFun verify_adjset_index_funs[] = {
        blueprint::mesh::adjset::index::verify,
        verify_adjset_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_adjset_index = verify_adjset_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_adjset_index,mesh,info,false);

        blueprint::mesh::examples::misc("adjsets",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh["domain0"],"quads",1,index);
        Node& aindex = index["adjsets"].child(0);
        CHECK_MESH(verify_adjset_index,aindex,info,true);

        { // Topology Field Tests //
            Node topo = aindex["topology"];
            aindex.remove("topology");

            CHECK_MESH(verify_adjset_index,aindex,info,false);
            aindex["topology"].set(0);
            CHECK_MESH(verify_adjset_index,aindex,info,false);

            aindex["topology"].set("path");
            CHECK_MESH(verify_adjset_index,aindex,info,true);

            aindex["topology"].reset();
            aindex["topology"].set(topo);
            CHECK_MESH(verify_adjset_index,aindex,info,true);
        }

        { // Path Field Tests //
            Node path = aindex["path"];
            aindex.remove("path");

            CHECK_MESH(verify_adjset_index,aindex,info,false);
            aindex["path"].set(0);
            CHECK_MESH(verify_adjset_index,aindex,info,false);

            aindex["path"].set("path");
            CHECK_MESH(verify_adjset_index,aindex,info,true);

            aindex["path"].reset();
            aindex["path"].set(path);
            CHECK_MESH(verify_adjset_index,aindex,info,true);
        }

        { // Association Field Tests //
            Node assoc = aindex["association"];
            aindex.remove("association");

            aindex["association"].set("zone");
            CHECK_MESH(verify_adjset_index,aindex,info,false);
            aindex["association"].set("vertex");
            CHECK_MESH(verify_adjset_index,aindex,info,true);

            aindex["association"].reset();
            aindex["association"].set(assoc);
            CHECK_MESH(verify_adjset_index,aindex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_nestset)
{
    VerifyFun verify_nestset_index_funs[] = {
        blueprint::mesh::nestset::index::verify,
        verify_nestset_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_nestset_index = verify_nestset_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_nestset_index,mesh,info,false);

        blueprint::mesh::examples::misc("nestsets",5,5,1,mesh);
        blueprint::mesh::generate_index(mesh["domain0"],"quads",1,index);
        Node& aindex = index["nestsets"].child(0);
        CHECK_MESH(verify_nestset_index,aindex,info,true);

        { // Topology Field Tests //
            Node topo = aindex["topology"];
            aindex.remove("topology");

            CHECK_MESH(verify_nestset_index,aindex,info,false);
            aindex["topology"].set(0);
            CHECK_MESH(verify_nestset_index,aindex,info,false);

            aindex["topology"].set("path");
            CHECK_MESH(verify_nestset_index,aindex,info,true);

            aindex["topology"].reset();
            aindex["topology"].set(topo);
            CHECK_MESH(verify_nestset_index,aindex,info,true);
        }

        { // Path Field Tests //
            Node path = aindex["path"];
            aindex.remove("path");

            CHECK_MESH(verify_nestset_index,aindex,info,false);
            aindex["path"].set(0);
            CHECK_MESH(verify_nestset_index,aindex,info,false);

            aindex["path"].set("path");
            CHECK_MESH(verify_nestset_index,aindex,info,true);

            aindex["path"].reset();
            aindex["path"].set(path);
            CHECK_MESH(verify_nestset_index,aindex,info,true);
        }

        { // Association Field Tests //
            Node assoc = aindex["association"];
            aindex.remove("association");

            aindex["association"].set("zone");
            CHECK_MESH(verify_nestset_index,aindex,info,false);
            aindex["association"].set("vertex");
            CHECK_MESH(verify_nestset_index,aindex,info,true);

            aindex["association"].reset();
            aindex["association"].set(assoc);
            CHECK_MESH(verify_nestset_index,aindex,info,true);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_general)
{
    VerifyFun verify_index_funs[] = {
        blueprint::mesh::index::verify,
        verify_index_protocol};

    for(index_t fi = 0; fi < 2; fi++)
    {
        VerifyFun verify_index = verify_index_funs[fi];

        Node mesh, index, info;
        CHECK_MESH(verify_index,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        CHECK_MESH(verify_index,index,info,true);

        { // Coordsets Field Tests //
            info.reset();
            Node coords = index["coordsets"];
            index.remove("coordsets");

            CHECK_MESH(verify_index,index,info,false);
            index["coordsets"].set("coords");
            CHECK_MESH(verify_index,index,info,false);

            index["coordsets"].reset();
            index["coordsets"]["coords1"].set("coords");
            index["coordsets"]["coords2"].set("coords");
            CHECK_MESH(verify_index,index,info,false);

            index["coordsets"].reset();
            index["coordsets"].set(coords);
            CHECK_MESH(verify_index,index,info,true);
        }

        { // Topologies Field Tests //
            info.reset();
            Node topos = index["topologies"];
            index.remove("topologies");

            CHECK_MESH(verify_index,index,info,false);
            index["topologies"].set("topo");
            CHECK_MESH(verify_index,index,info,false);

            index["topologies"].reset();
            index["topologies"]["topo1"].set("topo");
            index["topologies"]["topo2"].set("topo");
            CHECK_MESH(verify_index,index,info,false);

            index["topologies"].reset();
            index["topologies"]["mesh"]["type"].set("invalid");
            index["topologies"]["mesh"]["path"].set("quads/topologies/mesh");
            index["topologies"]["mesh"]["coordset"].set("coords");
            CHECK_MESH(verify_index,index,info,false);
            index["topologies"]["mesh"]["type"].set("unstructured");
            CHECK_MESH(verify_index,index,info,true);

            index["topologies"]["mesh"]["coordset"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["topologies"]["mesh"]["coordset"].set("coords");
            CHECK_MESH(verify_index,index,info,true);

            index["coordsets"]["coords"]["type"].set("invalid");
            CHECK_MESH(verify_index,index,info,false);
            index["coordsets"]["coords"]["type"].set("explicit");
            CHECK_MESH(verify_index,index,info,true);

            index["topologies"].reset();
            index["topologies"].set(topos);
            CHECK_MESH(verify_index,index,info,true);
        }

        { // Matsets Field Tests //
            info.reset();
            Node matsets = index["matsets"];
            index.remove("matsets");
            CHECK_MESH(verify_index,index,info,true);

            index["matsets"].set("matset");
            CHECK_MESH(verify_index,index,info,false);

            index["matsets"].reset();
            index["matsets"]["matset1"].set("matset1");
            index["matsets"]["matset1"].set("matset2");
            CHECK_MESH(verify_index,index,info,false);

            index["matsets"].reset();
            index["matsets"]["matset"]["topology"].set("mesh");
            index["matsets"]["matset"]["materials"].set("invalid");
            index["matsets"]["matset"]["path"].set("quads/matsets/matset");
            CHECK_MESH(verify_index,index,info,false);
            index["matsets"]["matset"]["materials"]["mat1"];
            CHECK_MESH(verify_index,index,info,true);
            index["matsets"]["matset"]["materials"]["mat2"];
            CHECK_MESH(verify_index,index,info,true);

            index["matsets"]["matset"]["topology"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["matsets"]["matset"]["topology"].set("mesh");
            CHECK_MESH(verify_index,index,info,true);

            // TODO(JRC): Change this code so that the "matsets" section is
            // re-added once it's included in the test Blueprint mesh.
            index["matsets"].reset();
            index.remove("matsets");
            CHECK_MESH(verify_index,index,info,true);
        }

        { // Fields Field Tests //
            info.reset();
            Node fields = index["fields"];
            index.remove("fields");
            CHECK_MESH(verify_index,index,info,true);

            index["fields"].set("field");
            CHECK_MESH(verify_index,index,info,false);

            index["fields"].reset();
            index["fields"]["field1"].set("field1");
            index["fields"]["field1"].set("field2");
            CHECK_MESH(verify_index,index,info,false);

            index["fields"].reset();
            index["fields"]["field"]["number_of_components"].set("invalid");
            index["fields"]["field"]["association"].set("vertex");
            index["fields"]["field"]["path"].set("quads/fields/braid");
            index["fields"]["field"]["topology"].set("mesh");
            CHECK_MESH(verify_index,index,info,false);
            index["fields"]["field"]["number_of_components"].set(1);
            CHECK_MESH(verify_index,index,info,true);

            index["fields"]["field"]["topology"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["fields"]["field"]["topology"].set("mesh");
            CHECK_MESH(verify_index,index,info,true);

            index["topologies"]["mesh"]["type"].set("invalid");
            CHECK_MESH(verify_index,index,info,false);
            index["topologies"]["mesh"]["type"].set("unstructured");
            CHECK_MESH(verify_index,index,info,true);

            index["fields"].reset();
            index["fields"].set(fields);
            CHECK_MESH(verify_index,index,info,true);
        }

        { // Adjsets Field Tests //
            info.reset();
            Node adjsets = index["adjsets"];
            index.remove("adjsets");
            CHECK_MESH(verify_index,index,info,true);

            index["adjsets"].set("adjset");
            CHECK_MESH(verify_index,index,info,false);

            index["adjsets"].reset();
            index["adjsets"]["adjset1"].set("adjset1");
            index["adjsets"]["adjset1"].set("adjset2");
            CHECK_MESH(verify_index,index,info,false);

            index["adjsets"].reset();
            index["adjsets"]["adjset"]["topology"].set("mesh");
            index["adjsets"]["adjset"]["association"].set("vertex");
            index["adjsets"]["adjset"]["path"].set(0);
            CHECK_MESH(verify_index,index,info,false);
            index["adjsets"]["adjset"]["path"].set("quads/adjsets/adjset");
            CHECK_MESH(verify_index,index,info,true);

            index["adjsets"]["adjset"]["topology"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["adjsets"]["adjset"]["topology"].set("mesh");
            CHECK_MESH(verify_index,index,info,true);

            index["adjsets"]["adjset"]["association"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["adjsets"]["adjset"]["association"].set("element");
            CHECK_MESH(verify_index,index,info,true);

            // TODO(JRC): Change this code so that the "adjsets" section is
            // re-added once it's included in the test Blueprint mesh.
            index["adjsets"].reset();
            index.remove("adjsets");
            CHECK_MESH(verify_index,index,info,true);
        }

        { // Nestsets Field Tests //
            info.reset();
            Node nestsets = index["nestsets"];
            index.remove("nestsets");
            CHECK_MESH(verify_index,index,info,true);

            index["nestsets"].set("nestset");
            CHECK_MESH(verify_index,index,info,false);

            index["nestsets"].reset();
            index["nestsets"]["nestset1"].set("nestset1");
            index["nestsets"]["nestset1"].set("nestset2");
            CHECK_MESH(verify_index,index,info,false);

            index["nestsets"].reset();
            index["nestsets"]["nestset"]["topology"].set("mesh");
            index["nestsets"]["nestset"]["association"].set("vertex");
            index["nestsets"]["nestset"]["path"].set(0);
            CHECK_MESH(verify_index,index,info,false);
            index["nestsets"]["nestset"]["path"].set("quads/nestsets/nestset");
            CHECK_MESH(verify_index,index,info,true);

            index["nestsets"]["nestset"]["topology"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["nestsets"]["nestset"]["topology"].set("mesh");
            CHECK_MESH(verify_index,index,info,true);

            index["nestsets"]["nestset"]["association"].set("nonexistent");
            CHECK_MESH(verify_index,index,info,false);
            index["nestsets"]["nestset"]["association"].set("element");
            CHECK_MESH(verify_index,index,info,true);

            // TODO(JRC): Change this code so that the "nestsets" section is
            // re-added once it's included in the test Blueprint mesh.
            index["nestsets"].reset();
            index.remove("nestsets");
            CHECK_MESH(verify_index,index,info,true);
        }
    }
}

/// Mesh Integration Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, mesh_multi_domain)
{
    Node mesh, info;
    EXPECT_FALSE(blueprint::mesh::is_multi_domain(mesh));

    Node domains[2];
    blueprint::mesh::examples::braid("quads",10,10,1,domains[0]);
    blueprint::mesh::to_multi_domain(domains[0],mesh);
    EXPECT_TRUE(blueprint::mesh::is_multi_domain(mesh));

    { // Redundant "to_multi_domain" Tests //
        Node temp;
        blueprint::mesh::to_multi_domain(mesh,temp);
        EXPECT_TRUE(blueprint::mesh::is_multi_domain(temp));
    }

    blueprint::mesh::examples::braid("quads",5,5,1,domains[1]);
    mesh.append().set_external(domains[1]);
    EXPECT_TRUE(blueprint::mesh::is_multi_domain(mesh));

    for(index_t di = 0; di < 2; di++)
    {
        Node& domain = mesh.child(di);
        EXPECT_FALSE(blueprint::mesh::is_multi_domain(domain));

        Node coordsets = domain["coordsets"];
        domain.remove("coordsets");
        EXPECT_FALSE(blueprint::mesh::is_multi_domain(mesh));

        domain["coordsets"].reset();
        domain["coordsets"].set(coordsets);
        EXPECT_TRUE(blueprint::mesh::is_multi_domain(mesh));
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, mesh_general)
{
    VerifyFun verify_mesh_funs[] = {
        blueprint::mesh::verify, // single_domain verify
        blueprint::mesh::verify, // multi_domain verify
        verify_mesh_multi_domain_protocol};

    for(index_t fi = 0; fi < 3; fi++)
    {
        VerifyFun verify_mesh = verify_mesh_funs[fi];

        Node mesh, mesh_data, info;
        CHECK_MESH(verify_mesh,mesh,info,false);

        blueprint::mesh::examples::braid("quads",10,10,1,mesh_data);

        Node* domain_ptr = NULL;
        if(fi == 0)
        {
            mesh.set_external(mesh_data);
            domain_ptr = &mesh;
        }
        else
        {
            blueprint::mesh::to_multi_domain(mesh_data,mesh);
            domain_ptr = &mesh.child(0);
        }
        Node& domain = *domain_ptr;

        CHECK_MESH(verify_mesh,mesh,info,true);
        // info.print();

        { // Coordsets Field Tests //
            Node coordsets = domain["coordsets"];
            domain.remove("coordsets");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["coordsets"].set("path");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["coordsets"].reset();
            domain["coordsets"]["coords"]["type"].set("invalid");
            domain["coordsets"]["coords"]["values"]["x"].set(DataType::float64(10));
            domain["coordsets"]["coords"]["values"]["y"].set(DataType::float64(10));
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["coordsets"]["coords"]["type"].set("explicit");
            CHECK_MESH(verify_mesh,mesh,info,true);
            domain["coordsets"]["coords2"]["type"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["coordsets"].reset();
            domain["coordsets"].set(coordsets);
            CHECK_MESH(verify_mesh,mesh,info,true);
        }

        { // Topologies Field Tests //
            Node topologies = domain["topologies"];
            domain.remove("topologies");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["topologies"].set("path");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["topologies"].reset();
            domain["topologies"]["mesh"]["type"].set("invalid");
            domain["topologies"]["mesh"]["coordset"].set("coords");
            domain["topologies"]["mesh"]["elements"]["shape"].set("quad");
            domain["topologies"]["mesh"]["elements"]["connectivity"].set(DataType::int32(10));
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["topologies"]["mesh"]["type"].set("unstructured");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["coordsets"]["coords"]["type"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["coordsets"]["coords"]["type"].set("explicit");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["topologies"]["mesh"]["coordset"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["topologies"]["mesh"]["coordset"].set("coords");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["topologies"]["grid"]["type"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["topologies"].reset();
            domain["topologies"].set(topologies);
            CHECK_MESH(verify_mesh,mesh,info,true);
        }

        { // Matsets Field Tests //
            Node matsets = domain["matsets"];
            domain.remove("matsets");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["matsets"].set("path");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["matsets"].reset();
            domain["matsets"]["mesh"]["topology"].set("mesh");
            domain["matsets"]["mesh"]["volume_fractions"];
            CHECK_MESH(verify_mesh,mesh,info,false);

            Node &vfs = domain["matsets"]["mesh"]["volume_fractions"];
            vfs["mat1"].set(DataType::float32(10));
            CHECK_MESH(verify_mesh,mesh,info,true);
            vfs["mat2"].set(DataType::float32(10));
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["matsets"]["mesh"]["topology"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["matsets"]["mesh"]["topology"].set("mesh");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["matsets"]["boundary"]["topology"].set("mesh");
            CHECK_MESH(verify_mesh,mesh,info,false);

            // TODO(JRC): Change this code so that the "matsets" section is
            // re-added once it's included in the test Blueprint mesh.
            domain["matsets"].reset();
            domain.remove("matsets");
            CHECK_MESH(verify_mesh,mesh,info,true);
        }

        { // Fields Field Tests //
            Node fields = domain["fields"];
            domain.remove("fields");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["fields"].set("path");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["fields"].reset();
            domain["fields"]["temp"]["association"].set("invalid");
            domain["fields"]["temp"]["topology"].set("mesh");
            domain["fields"]["temp"]["values"].set(DataType::float64(10));
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["fields"]["temp"]["association"].set("vertex");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["topologies"]["mesh"]["type"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["topologies"]["mesh"]["type"].set("unstructured");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["fields"]["temp"]["topology"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["fields"]["temp"]["topology"].set("mesh");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["fields"]["accel"]["association"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["fields"].reset();
            domain["fields"].set(fields);
            CHECK_MESH(verify_mesh,mesh,info,true);
        }

        { // Grid Function Field Tests //
            Node topologies = domain["topologies"];
            Node fields = domain["fields"];
            domain.remove("fields");

            domain["topologies"]["mesh"]["grid_function"].set("braid");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["fields"].set(fields);
            domain["topologies"]["mesh"]["grid_function"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["topologies"]["mesh"]["grid_function"].set("braid");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["fields"]["braid"]["association"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["fields"]["braid"]["association"].set("vertex");

            domain["topologies"].reset();
            domain["topologies"].set(topologies);
            CHECK_MESH(verify_mesh,mesh,info,true);
        }

        { // Adjsets Field Tests //
            Node adjsets = domain["adjsets"];
            domain.remove("adjsets");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["adjsets"].set("path");
            CHECK_MESH(verify_mesh,mesh,info,false);

            domain["adjsets"].reset();
            domain["adjsets"]["mesh"]["association"].set("vertex");
            domain["adjsets"]["mesh"]["topology"].set("mesh");
            domain["adjsets"]["mesh"]["groups"];
            CHECK_MESH(verify_mesh,mesh,info,false);

            Node &groups = domain["adjsets"]["mesh"]["groups"];
            groups["g1"]["neighbors"].set(DataType::int32(10));
            CHECK_MESH(verify_mesh,mesh,info,false);
            groups["g1"]["values"].set(DataType::float32(10));
            CHECK_MESH(verify_mesh,mesh,info,false);
            groups["g1"]["values"].set(DataType::int32(10));
            CHECK_MESH(verify_mesh,mesh,info,true);
            groups["g2"].set(groups["g1"]);
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["adjsets"]["mesh"]["topology"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["adjsets"]["mesh"]["topology"].set("mesh");
            CHECK_MESH(verify_mesh,mesh,info,true);

            domain["adjsets"]["mesh"]["association"].set("invalid");
            CHECK_MESH(verify_mesh,mesh,info,false);
            domain["adjsets"]["mesh"]["association"].set("element");
            CHECK_MESH(verify_mesh,mesh,info,true);

            // TODO(JRC): Change this code so that the "adjsets" section is
            // re-added once it's included in the test Blueprint mesh.
            domain["adjsets"].reset();
            domain.remove("adjsets");
            CHECK_MESH(verify_mesh,mesh,info,true);
        }
    }
}
