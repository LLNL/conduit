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

std::vector<std::string> create_basis(const std::string& d1,
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


bool is_valid_basis(bool (*basis_valid_fun)(const Node&, Node&),
                    const std::vector<std::string>& basis)
{
    Node n, info;

    bool is_valid = true;
    for(index_t bi = 0; bi < basis.size(); bi++)
    {
        const std::string& basis_dim = basis[bi];

        n[basis_dim].set("test");
        is_valid &= !basis_valid_fun(n, info);

        n[basis_dim].set(10);
        is_valid &= basis_valid_fun(n, info);

        // TODO(JRC): Determine whether or not the basis (i, k) should be
        // valid for logical coordinates.
        /*
        if( bi > 0 )
        {
            const std::string& prev_dim = basis[bi-1];
            n.remove(prev_dim);
            is_valid &= !basis_valid_fun(n, info);
            n[basis_dim].set(10);
        }
        */
    }

    return is_valid;
}

/// Testing Constants ///

const std::vector<std::string> LOGICAL_BASIS = create_basis("i","j","k");
const std::vector<std::string> CARTESIAN_BASIS = create_basis("x","y","z");
const std::vector<std::string> SPHERICAL_BASIS = create_basis("r","theta","phi");
const std::vector<std::string> CYLINDRICAL_BASIS = create_basis("r","z");

const std::vector<std::string> CONDUIT_BASES[4] = { LOGICAL_BASIS, CARTESIAN_BASIS, SPHERICAL_BASIS, CYLINDRICAL_BASIS };
const std::vector<std::string> COORDINATE_BASES[3] = { CARTESIAN_BASIS, SPHERICAL_BASIS, CYLINDRICAL_BASIS };

/// Mesh Coordinate Set Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_logical_dims)
{
    bool (*verify_coordset_logical)(const Node&, Node&) = blueprint::mesh::logical_dims::verify;

    Node n, info;
    EXPECT_FALSE(verify_coordset_logical(n, info));

    EXPECT_TRUE(is_valid_basis(verify_coordset_logical,LOGICAL_BASIS));

    EXPECT_FALSE(is_valid_basis(verify_coordset_logical,CARTESIAN_BASIS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_origin)
{
    bool (*verify_uniform_origin)(const Node&, Node&) = blueprint::mesh::coordset::uniform::origin::verify;

    Node n, info;
    // FIXME: The origin verification function shouldn't accept an empty node.
    // EXPECT_FALSE(verify_uniform_origin(n, info));

    EXPECT_TRUE(is_valid_basis(verify_uniform_origin,CARTESIAN_BASIS));
    EXPECT_TRUE(is_valid_basis(verify_uniform_origin,SPHERICAL_BASIS));
    EXPECT_TRUE(is_valid_basis(verify_uniform_origin,CYLINDRICAL_BASIS));

    EXPECT_FALSE(is_valid_basis(verify_uniform_origin,LOGICAL_BASIS));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform_spacing)
{
    bool (*verify_uniform_spacing)(const Node&, Node&) = blueprint::mesh::coordset::uniform::spacing::verify;

    Node n, info;
    // FIXME: The spacing verification function shouldn't accept an empty node.
    // EXPECT_FALSE(verify_uniform_spacing(n, info));

    EXPECT_TRUE(is_valid_basis(verify_uniform_spacing,create_basis("dx","dy","dz")));
    EXPECT_TRUE(is_valid_basis(verify_uniform_spacing,create_basis("dr","dtheta","dphi")));
    EXPECT_TRUE(is_valid_basis(verify_uniform_spacing,create_basis("dr","dz")));

    EXPECT_FALSE(is_valid_basis(verify_uniform_spacing,create_basis("di","dj","dk")));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_uniform)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["dims"]["i"].set(1);
    n["dims"]["j"].set(2);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["dims"]["k"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n, info));
    n["dims"]["k"].set(3);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));

    Node dims = n["dims"];
    n.remove("dims");

    n["origin"]["x"].set(10);
    n["origin"]["y"].set(20);
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["dims"].set(dims);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["origin"]["z"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n, info));
    n["origin"]["z"].set(30);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["spacing"]["dx"].set(0.1);
    n["spacing"]["dy"].set(0.2);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));

    n["spacing"]["dz"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::uniform::verify(n, info));
    n["spacing"]["dz"].set(0.3);
    EXPECT_TRUE(blueprint::mesh::coordset::uniform::verify(n, info));
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_rectilinear)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n, info));

    n["values"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n, info));

    for(index_t bi = 0; bi < 3; bi++)
    {
        const std::vector<std::string>& coord_basis = COORDINATE_BASES[bi];

        n["values"].reset();
        for(index_t ci = 0; ci < coord_basis.size(); ci++)
        {
            n["values"][coord_basis[ci]].set(DataType::float64(10));
            EXPECT_TRUE(blueprint::mesh::coordset::rectilinear::verify(n, info));
        }
    }

    // FIXME: The logical basis shouldn't be an accepted value for rectilinear.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_BASIS.size(); ci++)
    {
        n["values"][LOGICAL_BASIS[ci]].set(DataType::float64(10));
        EXPECT_FALSE(blueprint::mesh::coordset::rectilinear::verify(n, info));
    }
    */
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_explicit)
{
    Node n, info;
    EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n, info));

    n["values"].set("test");
    EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n, info));

    for(index_t bi = 0; bi < 3; bi++)
    {
        const std::vector<std::string>& coord_basis = COORDINATE_BASES[bi];

        n["values"].reset();
        for(index_t ci = 0; ci < coord_basis.size(); ci++)
        {
            n["values"][coord_basis[ci]].set(DataType::float64(10));
            EXPECT_TRUE(blueprint::mesh::coordset::_explicit::verify(n, info));
        }
    }

    // FIXME: The logical basis shouldn't be an accepted value for explicit.
    /*
    n["values"].reset();
    for(index_t ci = 0; ci < LOGICAL_BASIS.size(); ci++)
    {
        n["values"][LOGICAL_BASIS[ci]].set(DataType::float64(10));
        EXPECT_FALSE(blueprint::mesh::coordset::_explicit::verify(n, info));
    }
    */
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_types)
{
    Node n, info;

    n.set("uniform");
    EXPECT_TRUE(blueprint::mesh::coordset::type::verify(n, info));
    n.reset();

    n.set("rectilinear");
    EXPECT_TRUE(blueprint::mesh::coordset::type::verify(n, info));
    n.reset();

    n.set("explicit");
    EXPECT_TRUE(blueprint::mesh::coordset::type::verify(n, info));
    n.reset();

    n.set("unstructured");
    EXPECT_FALSE(blueprint::mesh::coordset::type::verify(n, info));
    n.reset();

    n.set(10);
    EXPECT_FALSE(blueprint::mesh::coordset::type::verify(n, info));
    n.reset();
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, coordset_general)
{
    // TODO(JRC): Implement this test case and give it a name.
}

/// Mesh Topology Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, TOPOLOGY)
{
    // TODO(JRC): Implement this test case and give it a name.
}

/// Mesh Field Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, FIELD)
{
    // TODO(JRC): Implement this test case and give it a name.
}

/// Mesh Index Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, INDEX)
{
    // TODO(JRC): Implement this test case and give it a name.
}

/// Mesh Integration Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_verify, INTEGRATION)
{
    // TODO(JRC): Implement this test case and give it a name.
}
