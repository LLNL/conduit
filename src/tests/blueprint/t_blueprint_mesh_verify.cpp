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

#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Helper Functions ///

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

/// Wrapper Functions ///

bool verify_coordset_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("coordset",n,info);
}

bool verify_topology_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("topology",n,info);
}

bool verify_field_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("field",n,info);
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

bool verify_field_index_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::verify("field/index",n,info);
}

bool verify_mesh_multidomain_protocol(const Node &n, Node &info)
{
    return blueprint::mesh::is_multidomain(n);
}

/// Testing Constants ///

const std::vector<std::string> LOGICAL_COORDSYS = create_coordsys("i","j","k");
const std::vector<std::string> CARTESIAN_COORDSYS = create_coordsys("x","y","z");
const std::vector<std::string> SPHERICAL_COORDSYS = create_coordsys("r","theta","phi");
const std::vector<std::string> CYLINDRICAL_COORDSYS = create_coordsys("r","z");

const std::vector<std::string> COORDINATE_COORDSYSS[] =
    {CARTESIAN_COORDSYS, CYLINDRICAL_COORDSYS, SPHERICAL_COORDSYS};

typedef bool (*VerifyFun)(const Node&, Node&);

/// Mesh Coordinate Set Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_logical_dims)
{
    VerifyFun verify_coordset_logical = blueprint::mesh::logical_dims::verify;

    Node n, info;
    EXPECT_FALSE(verify_coordset_logical(n,info));

    EXPECT_TRUE(is_valid_coordsys(verify_coordset_logical,LOGICAL_COORDSYS));

    EXPECT_FALSE(is_valid_coordsys(verify_coordset_logical,CARTESIAN_COORDSYS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_origin)
{
    VerifyFun verify_uniform_origin = blueprint::mesh::coordset::uniform::origin::verify;

    Node n, info;
    // FIXME: The origin verification function shouldn't accept an empty node.
    // EXPECT_FALSE(verify_uniform_origin(n,info));

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
    // EXPECT_FALSE(verify_uniform_spacing(n,info));

    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dx","dy","dz")));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dr","dtheta","dphi")));
    EXPECT_TRUE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("dr","dz")));

    EXPECT_FALSE(is_valid_coordsys(verify_uniform_spacing,create_coordsys("di","dj","dk")));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["dims"]["i"].set(1);
    n["dims"]["j"].set(2);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["dims"]["k"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n,info));
    n["dims"]["k"].set(3);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));

    Node dims = n["dims"];
    n.remove("dims");

    n["origin"]["x"].set(10);
    n["origin"]["y"].set(20);
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["dims"].set(dims);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["origin"]["z"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n,info));
    n["origin"]["z"].set(30);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["spacing"]["dx"].set(0.1);
    n["spacing"]["dy"].set(0.2);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));

    n["spacing"]["dz"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n,info));
    n["spacing"]["dz"].set(0.3);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_rectilinear)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n,info));

    n["values"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n,info));

    for(size_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(size_t cj = 0; cj < coord_coordsys.size(); cj++)
        {
            n["values"][coord_coordsys[cj]].set(DataType::float64(10));
            EXPECT_TRUE(blueprint::mesh::coordset::rectilinear::verify(n,info));
            info.print();
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
            EXPECT_TRUE(blueprint::mesh::coordset::rectilinear::verify(n,info));
            info.print();
        }
    }


    // FIXME: The logical coordinate system shouldn't be an accepted value
    // for the rectilinear verify function.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_COORDSYS.size(); ci++)
    {
        n["values"][LOGICAL_COORDSYS[ci]].set(DataType::float64(10));
        EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n,info));
    }
    */
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_explicit)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n,info));

    n["values"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n,info));

    for(size_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(size_t cj = 0; cj < coord_coordsys.size(); cj++)
        {
            n["values"][coord_coordsys[cj]].set(DataType::float64(10));
            EXPECT_TRUE(blueprint::mesh::coordset::_explicit::verify(n,info));
        }
    }

    // FIXME: The logical coordinate system shouldn't be an accepted value
    // for the explicit verify function.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_COORDSYS.size(); ci++)
    {
        n["values"][LOGICAL_COORDSYS[ci]].set(DataType::float64(10));
        EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n,info));
    }
    */
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_types)
{
    Node n, info;

    const std::string coordset_types[] = {"uniform", "rectilinear", "explicit"};
    for(index_t ti = 0; ti < 3; ti++)
    {
        n.reset();
        n.set(coordset_types[ti]);
        EXPECT_TRUE(blueprint::mesh::coordset::type::verify(n,info));
    }

    n.reset();
    n.set(0);
    EXPECT_FALSE(blueprint::mesh::coordset::type::verify(n,info));

    n.reset();
    n.set("unstructured");
    EXPECT_FALSE(blueprint::mesh::coordset::type::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_coordsys)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));

    const std::string coordsys_types[] = {"cartesian", "cylindrical", "spherical"};
    for(index_t ci = 0; ci < 3; ci++)
    {
        n.reset();
        info.reset();

        n["type"].set(coordsys_types[ci]);
        EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));

        n["axes"].set(0);
        EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));

        n["axes"].reset();
        const std::vector<std::string>& coordsys = COORDINATE_COORDSYSS[ci];
        for(size_t ai = 0; ai < coordsys.size(); ai++)
        {
            n["axes"][coordsys[ai]].set(10);
            EXPECT_TRUE(blueprint::mesh::coordset::coord_system::verify(n,info));
        }

        n["type"].set(coordsys_types[(ci == 0) ? 2 : ci - 1]);
        EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));

        n["type"].set("barycentric");
        EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));

        n["type"].set(10);
        EXPECT_FALSE(blueprint::mesh::coordset::coord_system::verify(n,info));
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
        EXPECT_FALSE(verify_coordset(mesh,info));

        blueprint::mesh::examples::braid("uniform",10,10,10,mesh);
        Node& n = mesh["coordsets"]["coords"];

        n.remove("type");
        EXPECT_FALSE(verify_coordset(n,info));
        n["type"].set("structured");
        EXPECT_FALSE(verify_coordset(n,info));
        n["type"].set("rectilinear");
        EXPECT_FALSE(verify_coordset(n,info));

        n["type"].set("uniform");
        EXPECT_TRUE(verify_coordset(n,info));
    }
}

/// Mesh Topology Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_uniform)
{
    // FIXME: Implement once 'mesh::topology::uniform::verify' is implemented.
    Node n, info;
    EXPECT_TRUE(blueprint::mesh::topology::uniform::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_rectilinear)
{
    // FIXME: Implement once 'mesh::topology::rectilinear::verify' is implemented.
    Node n, info;
    EXPECT_TRUE(blueprint::mesh::topology::rectilinear::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_structured)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::topology::structured::verify(n,info));

    n["elements"].set(0);
    EXPECT_FALSE(blueprint::mesh::topology::structured::verify(n,info));

    n["elements"].reset();
    n["elements"]["dims"].set(0);
    EXPECT_FALSE(blueprint::mesh::topology::structured::verify(n,info));

    n["elements"]["dims"].reset();
    n["elements"]["dims"]["x"].set(5);
    n["elements"]["dims"]["y"].set(10);
    EXPECT_FALSE(blueprint::mesh::topology::structured::verify(n,info));

    n["elements"]["dims"].reset();
    n["elements"]["dims"]["i"].set(15);
    n["elements"]["dims"]["j"].set(20);
    EXPECT_TRUE(blueprint::mesh::topology::structured::verify(n,info));

    n["elements"]["dims"]["k"].set(25);
    EXPECT_TRUE(blueprint::mesh::topology::structured::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_unstructured)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

    n["elements"].set(0);
    EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

    { // Single Shape Topology Tests //
        n["elements"].reset();
        n["elements"]["shape"].set("polygon");
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["shape"].set("quad");
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["connectivity"].set("quad");
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));
        n["elements"]["connectivity"].set(DataType::float64(10));
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["connectivity"].set(DataType::int32(1));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(n,info));
        n["elements"]["connectivity"].set(DataType::int32(10));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(n,info));
    }

    { // Mixed Shape Topology List Tests //
        n["elements"].reset();

        n["elements"]["a"].set(0);
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["a"]["shape"].set("quad");
        n["elements"]["a"]["connectivity"].set(DataType::int32(5));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["b"]["shape"].set("polygon");
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));
        n["elements"]["b"]["shape"].set("quad");
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));
        n["elements"]["b"]["connectivity"].set(DataType::float32(3));
        EXPECT_FALSE(blueprint::mesh::topology::unstructured::verify(n,info));
        n["elements"]["b"]["connectivity"].set(DataType::int32(1));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(n,info));

        n["elements"]["c"]["shape"].set("tri");
        n["elements"]["c"]["connectivity"].set(DataType::int32(5));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(n,info));
    }

    { // Multiple Shape Topology Stream Tests //
        // FIXME: Implement once multiple unstructured shape topologies are implemented.
        n["elements"].reset();
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_types)
{
    Node n, info;

    const std::string topology_types[] = {"uniform", "rectilinear", "structured", "unstructured"};
    for(index_t ti = 0; ti < 4; ti++)
    {
        n.reset();
        n.set(topology_types[ti]);
        EXPECT_TRUE(blueprint::mesh::topology::type::verify(n,info));
    }

    n.reset();
    n.set(0);
    EXPECT_FALSE(blueprint::mesh::topology::type::verify(n,info));

    n.reset();
    n.set("explicit");
    EXPECT_FALSE(blueprint::mesh::topology::type::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, topology_shape)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::topology::shape::verify(n,info));

    const std::string shape_types[] = {"point", "line", "tri", "quad", "tet", "hex"};
    for(index_t ti = 0; ti < 6; ti++)
    {
        n.reset();
        n.set(shape_types[ti]);
        EXPECT_TRUE(blueprint::mesh::topology::shape::verify(n,info));
    }

    n.reset();
    n.set(10);
    EXPECT_FALSE(blueprint::mesh::topology::shape::verify(n,info));

    n.reset();
    n.set("poly");
    EXPECT_FALSE(blueprint::mesh::topology::shape::verify(n,info));
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
        EXPECT_FALSE(verify_topology(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        Node& n = mesh["topologies"]["mesh"];

        { // Type Field Tests //
            n.remove("type");
            EXPECT_FALSE(verify_topology(n,info));
            n["type"].set("explicit");
            EXPECT_FALSE(verify_topology(n,info));

            // FIXME: Remove the comments from the following line once the verify functions
            // for uniform and rectilinear topologies have been implemented.
            const std::string topology_types[] = {/*"uniform", "rectilinear", */"structured"};
            for(index_t ti = 0; ti < 1; ti++)
            {
                n["type"].set(topology_types[ti]);
                EXPECT_FALSE(verify_topology(n,info));
            }

            n["type"].set("unstructured");
            EXPECT_TRUE(verify_topology(n,info));
        }

        { // Coordset Field Tests //
            n.remove("coordset");
            EXPECT_FALSE(verify_topology(n,info));

            n["coordset"].set(0);
            EXPECT_FALSE(verify_topology(n,info));

            n["coordset"].set("coords");
            EXPECT_TRUE(verify_topology(n,info));
        }

        { // Grid Function Field Tests //
            n["grid_function"].set(10);
            EXPECT_FALSE(verify_topology(n,info));
            n["grid_function"].set("coords_gf");
            EXPECT_TRUE(verify_topology(n,info));
        }
    }
}

/// Mesh Field Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_association)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::field::association::verify(n,info));

    const std::string assoc_types[] = {"vertex", "element"};
    for(index_t ti = 0; ti < 2; ti++)
    {
        n.reset();
        n.set(assoc_types[ti]);
        EXPECT_TRUE(blueprint::mesh::field::association::verify(n,info));
    }

    n.reset();
    n.set(0);
    EXPECT_FALSE(blueprint::mesh::field::association::verify(n,info));

    n.reset();
    n.set("zone");
    EXPECT_FALSE(blueprint::mesh::field::association::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_basis)
{
    // FIXME: Does this have to be verified against anything else?  What does
    // this basis refer to if it isn't a different path in the mesh structure?
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::field::basis::verify(n,info));

    n.reset();
    n.set(0);
    EXPECT_FALSE(blueprint::mesh::field::basis::verify(n,info));

    n.reset();
    n.set("basis");
    EXPECT_TRUE(blueprint::mesh::field::basis::verify(n,info));
}


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
        EXPECT_FALSE(verify_field(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        Node& n = mesh["fields"]["braid"];

        { // Topology Field Tests //
            n.remove("topology");
            EXPECT_FALSE(verify_field(n,info));
            n["topology"].set(10);
            EXPECT_FALSE(verify_field(n,info));
            n["topology"].set("mesh");
            EXPECT_TRUE(verify_field(n,info));
        }

        { // Values Field Tests //
            Node values = n["values"];

            n.remove("values");
            EXPECT_FALSE(verify_field(n,info));
            n["values"].set("values");
            EXPECT_FALSE(verify_field(n,info));
            n["values"].set(DataType::float64(10));
            EXPECT_TRUE(verify_field(n,info));

            n["values"].reset();
            n["values"]["x"].set("Hello, ");
            n["values"]["y"].set("World!");
            EXPECT_FALSE(verify_field(n,info));

            n["values"].reset();
            n["values"]["x"].set(DataType::float64(5));
            n["values"]["y"].set(DataType::float64(5));
            EXPECT_TRUE(verify_field(n,info));

            n["values"].set(values);
            EXPECT_TRUE(verify_field(n,info));
        }

        { // Association/Basis Field Tests //
            n.remove("association");
            EXPECT_FALSE(verify_field(n,info));

            n["association"].set("zone");
            EXPECT_FALSE(verify_field(n,info));
            n["association"].set("vertex");
            EXPECT_TRUE(verify_field(n,info));

            n.remove("association");
            n["basis"].set(0);
            EXPECT_FALSE(verify_field(n,info));
            n["basis"].set("basis");
            EXPECT_TRUE(verify_field(n,info));

            n["association"].set("vertex");
            EXPECT_TRUE(verify_field(n,info));
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
        EXPECT_FALSE(verify_coordset_index(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& cindex = index["coordsets"]["coords"];
        EXPECT_TRUE(verify_coordset_index(cindex,info));

        { // Type Field Tests //
            cindex.remove("type");
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["type"].set("undefined");
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["type"].set("explicit");
            EXPECT_TRUE(verify_coordset_index(cindex,info));
        }

        { // Coord System Field Tests //
            Node coordsys = cindex["coord_system"];
            cindex.remove("coord_system");

            EXPECT_FALSE(verify_coordset_index(cindex,info));
            cindex["coord_system"].set("invalid");
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["coord_system"].reset();
            cindex["coord_system"]["type"].set("logical");
            cindex["coord_system"]["axes"]["i"].set(10);
            cindex["coord_system"]["axes"]["j"].set(10);
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["coord_system"].reset();
            cindex["coord_system"].set(coordsys);
            EXPECT_TRUE(verify_coordset_index(cindex,info));
        }

        { // Path Field Tests //
            cindex.remove("path");
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["path"].set(5);
            EXPECT_FALSE(verify_coordset_index(cindex,info));

            cindex["path"].set("path");
            EXPECT_TRUE(verify_coordset_index(cindex,info));
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
        EXPECT_FALSE(verify_topo_index(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& tindex = index["topologies"]["mesh"];
        EXPECT_TRUE(verify_topo_index(tindex,info));

        { // Type Field Tests //
            tindex.remove("type");
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["type"].set("undefined");
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["type"].set("unstructured");
            EXPECT_TRUE(verify_topo_index(tindex,info));
        }

        { // Coordset Field Tests //
            tindex.remove("coordset");
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["coordset"].set(0);
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["coordset"].set("path");
            EXPECT_TRUE(verify_topo_index(tindex,info));
        }

        { // Path Field Tests //
            tindex.remove("path");
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["path"].set(5);
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["path"].set("path");
            EXPECT_TRUE(verify_topo_index(tindex,info));
        }

        { // Grid Function Field Tests //
            tindex["grid_function"].set(10);
            EXPECT_FALSE(verify_topo_index(tindex,info));

            tindex["grid_function"].set("path");
            EXPECT_TRUE(verify_topo_index(tindex,info));
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
        EXPECT_FALSE(verify_field_index(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        Node& findex = index["fields"]["braid"];
        EXPECT_TRUE(verify_field_index(findex,info));

        { // Topology Field Tests //
            Node topo = findex["topology"];
            findex.remove("topology");

            EXPECT_FALSE(verify_field_index(findex,info));
            findex["topology"].set(0);
            EXPECT_FALSE(verify_field_index(findex,info));

            findex["topology"].set("path");
            EXPECT_TRUE(verify_field_index(findex,info));

            findex["topology"].reset();
            findex["topology"].set(topo);
            EXPECT_TRUE(verify_field_index(findex,info));
        }

        { // Component Count Field Tests //
            Node comps = findex["number_of_components"];
            findex.remove("number_of_components");

            EXPECT_FALSE(verify_field_index(findex,info));
            findex["number_of_components"].set("three");
            EXPECT_FALSE(verify_field_index(findex,info));

            findex["number_of_components"].set(3);
            EXPECT_TRUE(verify_field_index(findex,info));

            findex["number_of_components"].reset();
            findex["number_of_components"].set(comps);
            EXPECT_TRUE(verify_field_index(findex,info));
        }

        { // Path Field Tests //
            Node path = findex["path"];
            findex.remove("path");

            EXPECT_FALSE(verify_field_index(findex,info));
            findex["path"].set(0);
            EXPECT_FALSE(verify_field_index(findex,info));

            findex["path"].set("path");
            EXPECT_TRUE(verify_field_index(findex,info));

            findex["path"].reset();
            findex["path"].set(path);
            EXPECT_TRUE(verify_field_index(findex,info));
        }

        { // Association Field Tests //
            findex["association"].set("zone");
            EXPECT_FALSE(verify_field_index(findex,info));
            findex["association"].set("vertex");
            EXPECT_TRUE(verify_field_index(findex,info));

            findex.remove("association");
            findex["basis"].set(0);
            EXPECT_FALSE(verify_field_index(findex,info));
            findex["basis"].set("basis");
            EXPECT_TRUE(verify_field_index(findex,info));
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
        EXPECT_FALSE(verify_index(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh);
        blueprint::mesh::generate_index(mesh,"quads",1,index);
        EXPECT_TRUE(verify_index(index,info));

        { // Topology Field Tests //
            info.reset();
            Node coords = index["coordsets"];
            index.remove("coordsets");

            EXPECT_FALSE(verify_index(index,info));
            index["coordsets"].set("coords");
            EXPECT_FALSE(verify_index(index,info));

            index["coordsets"].reset();
            index["coordsets"]["coords1"].set("coords");
            index["coordsets"]["coords2"].set("coords");
            EXPECT_FALSE(verify_index(index,info));

            index["coordsets"].reset();
            index["coordsets"].set(coords);
            EXPECT_TRUE(verify_index(index,info));
        }

        { // Components Field Tests //
            info.reset();
            Node topos = index["topologies"];
            index.remove("topologies");

            EXPECT_FALSE(verify_index(index,info));
            index["topologies"].set("topo");
            EXPECT_FALSE(verify_index(index,info));

            index["topologies"].reset();
            index["topologies"]["topo1"].set("topo");
            index["topologies"]["topo2"].set("topo");
            EXPECT_FALSE(verify_index(index,info));

            index["topologies"].reset();
            index["topologies"]["mesh"]["type"].set("unstructured");
            index["topologies"]["mesh"]["path"].set("quads/topologies/mesh");
            index["topologies"]["mesh"]["coordset"].set("nonexitent");
            EXPECT_FALSE(verify_index(index,info));

            index["topologies"]["mesh"]["coordset"].set("coords");
            index["coordsets"]["coords"]["type"].set("invalid");
            EXPECT_FALSE(verify_index(index,info));
            index["coordsets"]["coords"]["type"].set("explicit");

            index["topologies"].reset();
            index["topologies"].set(topos);
            EXPECT_TRUE(verify_index(index,info));
        }

        { // Fields Field Tests //
            info.reset();
            Node fields = index["fields"];
            index.remove("fields");
            EXPECT_TRUE(verify_index(index,info));

            index["fields"].set("field");
            EXPECT_FALSE(verify_index(index,info));

            index["fields"].reset();
            index["fields"]["field1"].set("field1");
            index["fields"]["field1"].set("field2");
            EXPECT_FALSE(verify_index(index,info));

            index["fields"].reset();
            index["fields"]["field"]["number_of_components"].set(1);
            index["fields"]["field"]["association"].set("vertex");
            index["fields"]["field"]["path"].set("quads/fields/braid");
            index["fields"]["field"]["topology"].set("nonexitent");
            EXPECT_FALSE(verify_index(index,info));

            index["fields"]["field"]["topology"].set("mesh");
            index["topologies"]["mesh"]["type"].set("invalid");
            EXPECT_FALSE(verify_index(index,info));
            index["topologies"]["mesh"]["type"].set("unstructured");

            index["fields"].reset();
            index["fields"].set(fields);
            EXPECT_TRUE(verify_index(index,info));
        }
    }
}

/// Mesh Integration Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, mesh_multidomain)
{
    Node mesh, info;
    EXPECT_FALSE(blueprint::mesh::is_multidomain(mesh));

    Node domains[2];
    blueprint::mesh::examples::braid("quads",10,10,1,domains[0]);
    blueprint::mesh::to_multidomain(domains[0],mesh);
    EXPECT_TRUE(blueprint::mesh::is_multidomain(mesh));

    blueprint::mesh::examples::braid("quads",5,5,1,domains[1]);
    mesh.append().set_external(domains[1]);
    EXPECT_TRUE(blueprint::mesh::is_multidomain(mesh));

    for(index_t di = 0; di < 2; di++)
    {
        Node& domain = mesh.child(di);
        EXPECT_FALSE(blueprint::mesh::is_multidomain(domain));

        Node coordsets = domain["coordsets"];
        domain.remove("coordsets");
        EXPECT_FALSE(blueprint::mesh::is_multidomain(mesh));

        domain["coordsets"].reset();
        domain["coordsets"].set(coordsets);
        EXPECT_TRUE(blueprint::mesh::is_multidomain(mesh));
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, mesh_general)
{
    VerifyFun verify_mesh_funs[] = {
        blueprint::mesh::verify, // unidomain verify
        blueprint::mesh::verify, // multidomain verify
        verify_mesh_multidomain_protocol};

    for(index_t fi = 0; fi < 3; fi++)
    {
        VerifyFun verify_mesh = verify_mesh_funs[fi];

        Node mesh, mesh_data, info;
        EXPECT_FALSE(verify_mesh(mesh,info));

        blueprint::mesh::examples::braid("quads",10,10,1,mesh_data);

        Node* domain_ptr = NULL;
        if(fi == 0)
        {
            mesh.set_external(mesh_data);
            domain_ptr = &mesh;
        }
        else
        {
            blueprint::mesh::to_multidomain(mesh_data,mesh);
            domain_ptr = &mesh.child(0);
        }
        Node& domain = *domain_ptr;

        EXPECT_TRUE(verify_mesh(mesh,info));
        info.print();

        { // Coordsets Field Tests //
            Node coordsets = domain["coordsets"];
            domain.remove("coordsets");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["coordsets"].set("path");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["coordsets"].reset();
            domain["coordsets"]["coords"]["type"].set("invalid");
            domain["coordsets"]["coords"]["values"]["x"].set(DataType::float64(10));
            domain["coordsets"]["coords"]["values"]["y"].set(DataType::float64(10));
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["coordsets"]["coords"]["type"].set("explicit");
            EXPECT_TRUE(verify_mesh(mesh,info));
            domain["coordsets"]["coords2"]["type"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["coordsets"].reset();
            domain["coordsets"].set(coordsets);
            EXPECT_TRUE(verify_mesh(mesh,info));
        }

        { // Topologies Field Tests //
            Node topologies = domain["topologies"];
            domain.remove("topologies");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["topologies"].set("path");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["topologies"].reset();
            domain["topologies"]["mesh"]["type"].set("invalid");
            domain["topologies"]["mesh"]["coordset"].set("coords");
            domain["topologies"]["mesh"]["elements"]["shape"].set("quad");
            domain["topologies"]["mesh"]["elements"]["connectivity"].set(DataType::int32(10));
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["topologies"]["mesh"]["type"].set("unstructured");
            EXPECT_TRUE(verify_mesh(mesh,info));

            domain["coordsets"]["coords"]["type"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));
            domain["coordsets"]["coords"]["type"].set("explicit");

            domain["topologies"]["grid"]["type"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["topologies"].reset();
            domain["topologies"].set(topologies);
            EXPECT_TRUE(verify_mesh(mesh,info));
        }

        { // Fields Field Tests //
            Node fields = domain["fields"];
            domain.remove("fields");
            EXPECT_TRUE(verify_mesh(mesh,info));

            domain["fields"].set("path");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["fields"].reset();
            domain["fields"]["temp"]["association"].set("invalid");
            domain["fields"]["temp"]["topology"].set("mesh");
            domain["fields"]["temp"]["values"].set(DataType::float64(10));
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["fields"]["temp"]["association"].set("vertex");
            EXPECT_TRUE(verify_mesh(mesh,info));

            domain["topologies"]["mesh"]["type"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));
            domain["topologies"]["mesh"]["type"].set("unstructured");

            domain["fields"]["accel"]["association"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["fields"].reset();
            domain["fields"].set(fields);
            EXPECT_TRUE(verify_mesh(mesh,info));
        }

        { // Grid Function Field Tests //
            Node topologies = domain["topologies"];
            Node fields = domain["fields"];
            domain.remove("fields");

            domain["topologies"]["mesh"]["grid_function"].set("braid");
            EXPECT_FALSE(verify_mesh(mesh,info));

            domain["fields"].set(fields);
            domain["topologies"]["mesh"]["grid_function"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));
            domain["topologies"]["mesh"]["grid_function"].set("braid");
            EXPECT_TRUE(verify_mesh(mesh,info));

            domain["fields"]["braid"]["association"].set("invalid");
            EXPECT_FALSE(verify_mesh(mesh,info));
            domain["fields"]["braid"]["association"].set("vertex");

            domain["topologies"].reset();
            domain["topologies"].set(topologies);
            EXPECT_TRUE(verify_mesh(mesh,info));
        }
    }
}
