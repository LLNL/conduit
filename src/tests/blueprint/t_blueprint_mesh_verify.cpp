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
/// file: conduit_blueprint_mesh_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "blueprint.hpp"
#include "relay.hpp"

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
    for(index_t ci = 0; ci < coordsys.size(); ci++)
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

/// Testing Constants ///

const std::vector<std::string> LOGICAL_COORDSYS = create_coordsys("i","j","k");
const std::vector<std::string> CARTESIAN_COORDSYS = create_coordsys("x","y","z");
const std::vector<std::string> SPHERICAL_COORDSYS = create_coordsys("r","theta","phi");
const std::vector<std::string> CYLINDRICAL_COORDSYS = create_coordsys("r","z");

const std::vector<std::string> COORDINATE_COORDSYSS[] =
    {CARTESIAN_COORDSYS, CYLINDRICAL_COORDSYS, SPHERICAL_COORDSYS};

/// Mesh Coordinate Set Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_logical_dims)
{
    bool (*verify_coordset_logical)(const Node&, Node&) =
        blueprint::mesh::logical_dims::verify;

    Node n, info;
    EXPECT_FALSE(verify_coordset_logical(n,info));

    EXPECT_TRUE(is_valid_coordsys(verify_coordset_logical,LOGICAL_COORDSYS));

    EXPECT_FALSE(is_valid_coordsys(verify_coordset_logical,CARTESIAN_COORDSYS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_origin)
{
    bool (*verify_uniform_origin)(const Node&, Node&) =
        blueprint::mesh::coordset::uniform::origin::verify;

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
    bool (*verify_uniform_spacing)(const Node&, Node&) =
        blueprint::mesh::coordset::uniform::spacing::verify;

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

    for(index_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(index_t ci = 0; ci < coord_coordsys.size(); ci++)
        {
            n["values"][coord_coordsys[ci]].set(DataType::float64(10));
            EXPECT_TRUE(blueprint::mesh::coordset::rectilinear::verify(n,info));
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

    for(index_t ci = 0; ci < 3; ci++)
    {
        const std::vector<std::string>& coord_coordsys = COORDINATE_COORDSYSS[ci];

        n["values"].reset();
        for(index_t ci = 0; ci < coord_coordsys.size(); ci++)
        {
            n["values"][coord_coordsys[ci]].set(DataType::float64(10));
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

    n.set("unstructured");
    EXPECT_FALSE(blueprint::mesh::coordset::type::verify(n,info));
    n.reset();
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

        const std::vector<std::string>& coordsys = COORDINATE_COORDSYSS[ci];
        for(index_t ai = 0; ai < coordsys.size(); ai++)
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
    Node mesh, info;
    EXPECT_FALSE(blueprint::mesh::coordset::verify(mesh,info));

    blueprint::mesh::examples::braid("uniform",10,10,10,mesh);
    Node& n = mesh["coordsets"]["coords"];

    n.remove("type");
    EXPECT_FALSE(blueprint::mesh::coordset::verify(n,info));
    n["type"].set("structured");
    EXPECT_FALSE(blueprint::mesh::coordset::verify(n,info));
    n["type"].set("rectilinear");
    EXPECT_FALSE(blueprint::mesh::coordset::verify(n,info));

    n["type"].set("uniform");
    EXPECT_TRUE(blueprint::mesh::coordset::verify(n,info));
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
    Node mesh, info;
    EXPECT_FALSE(blueprint::mesh::topology::verify(mesh,info));

    blueprint::mesh::examples::braid("quads",10,10,1,mesh);
    Node& n = mesh["topologies"]["mesh"];

    n.remove("type");
    EXPECT_FALSE(blueprint::mesh::topology::verify(n,info));
    n["type"].set("explicit");
    EXPECT_FALSE(blueprint::mesh::topology::verify(n,info));

    // FIXME: Remove the comments from the following line once the verify functions
    // for uniform and rectilinear topologies have been implemented.
    const std::string topology_types[] = {/*"uniform", "rectilinear", */"structured"};
    for(index_t ti = 0; ti < 1; ti++)
    {
        n["type"].set(topology_types[ti]);
        EXPECT_FALSE(blueprint::mesh::topology::verify(n,info));
    }

    n["type"].set("unstructured");
    EXPECT_TRUE(blueprint::mesh::topology::verify(n,info));

    n.remove("coordset");
    EXPECT_FALSE(blueprint::mesh::topology::verify(n,info));
    n["coordset"].set("coords");
    EXPECT_TRUE(blueprint::mesh::topology::verify(n,info));

    n["grid_function"].set(10);
    EXPECT_FALSE(blueprint::mesh::topology::verify(n,info));
    n["grid_function"].set("coords_gf");
    EXPECT_TRUE(blueprint::mesh::topology::verify(n,info));
}

/// Mesh Field Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_association)
{
    // TODO(JRC): Implement this test case and give it a name.
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_basis)
{
    // TODO(JRC): Implement this test case and give it a name.
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, field_general)
{
    // TODO(JRC): Implement this test case and give it a name.
}

/// Mesh Index Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_coordset)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["type"].set("unstructured");
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["type"].set("uniform");
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["coord_system"].set("invalid");
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["coord_system"].reset();
    n["coord_system"]["type"].set("cartesian");
    n["coord_system"]["axes"]["x"].set(10);
    n["coord_system"]["axes"]["y"].set(10);
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["path"].set(10);
    EXPECT_FALSE(blueprint::mesh::coordset::index::verify(n,info));

    n["path"].set("path");
    EXPECT_TRUE(blueprint::mesh::coordset::index::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_topology)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["type"].set("explicit");
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["type"].set("unstructured");
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["coordset"].set(0);
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["coordset"].set("path");
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["path"].set(5);
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["path"].set("path");
    EXPECT_TRUE(blueprint::mesh::topology::index::verify(n,info));

    n["grid_function"].set(10);
    EXPECT_FALSE(blueprint::mesh::topology::index::verify(n,info));

    n["grid_function"].set("path");
    EXPECT_TRUE(blueprint::mesh::topology::index::verify(n,info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_field)
{
    // TODO(JRC): Implement this test case and give it a name.
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, index_general)
{
    // TODO(JRC): Implement this test case.
}
