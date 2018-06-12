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
/// file: t_blueprint_mesh_xform.cpp
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

// TODO(JRC): Figure out a better way to share these constants from the
// "conduit_blueprint_mesh.cpp" source file (would also be useful for the
// contents of "t_blueprint_mesh_verify.cpp").

static const std::string COORD_TYPE_LIST[3] = {"uniform", "rectilinear", "explicit"};
static const std::vector<std::string> COORD_TYPES(COORD_TYPE_LIST,
    COORD_TYPE_LIST + sizeof(COORD_TYPE_LIST) / sizeof(COORD_TYPE_LIST[0]));

static const std::string TOPO_TYPE_LIST[5] = {"points", "uniform", "rectilinear", "structured", "unstructured"};
static const std::vector<std::string> TOPO_TYPES(TOPO_TYPE_LIST,
    TOPO_TYPE_LIST + sizeof(TOPO_TYPE_LIST) / sizeof(TOPO_TYPE_LIST[0]));

typedef bool (*XformFun)(const Node&, Node&);
typedef bool (*VerifyFun)(const Node&, Node&);

/// Testing Helpers ///

std::string get_braid_type(const std::string &mesh_type)
{
    std::string braid_type;
    try
    {
        conduit::Node mesh;
        blueprint::mesh::examples::braid(mesh_type,1,1,1,mesh);
        braid_type = mesh_type;
    }
    // TODO(JRC): Make a more specific catch statement for Conduit-thrown
    // errors only.
    catch(...)
    {
        braid_type = "hexs";
    }

    return braid_type;
}

/// Coordset Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, coordset_general_xforms)
{
    XformFun xform_coordset_funs[] = {
        blueprint::mesh::coordset::to_uniform,
        blueprint::mesh::coordset::to_rectilinear,
        blueprint::mesh::coordset::to_explicit};

    VerifyFun verify_coordset_funs[] = {
        blueprint::mesh::coordset::uniform::verify,
        blueprint::mesh::coordset::rectilinear::verify,
        blueprint::mesh::coordset::_explicit::verify};

    for(index_t xi = 0; xi < COORD_TYPES.size(); xi++)
    {
        const std::string coordset_type = COORD_TYPES[xi];
        const std::string coordset_braid = get_braid_type(coordset_type);

        conduit::Node orig_mesh;
        blueprint::mesh::examples::braid(coordset_braid,2,3,4,orig_mesh);
        conduit::Node &orig_coordset = orig_mesh["coordsets"].child(0);

        for(index_t xj = 0; xj < COORD_TYPES.size(); xj++)
        {
            XformFun to_new_coordset = xform_coordset_funs[xj];
            VerifyFun verify_new_coordset = verify_coordset_funs[xj];

            // TODO(JRC): Diff against coordset generated from higher level
            // braid function for more complete testing.
            conduit::Node xform_coordset, info;
            EXPECT_EQ(to_new_coordset(orig_coordset, xform_coordset), xi <= xj);
            EXPECT_EQ(verify_new_coordset(xform_coordset, info), xi <= xj);
            EXPECT_EQ(orig_coordset.diff(xform_coordset, info), xi != xj);
        }
    }
}

/*
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, coordset_uniform_xform)
{
    // verify that we can only up-promote coordinate sets


    conduit::Node mesh, info;

    blueprint::mesh::examples::braid("uniform",10,10,1,mesh);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, coordset_to_rectilinear)
{
    EXPECT_TRUE(false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, coordset_to_explicit)
{
    EXPECT_TRUE(false);
}

/// Topology Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_to_points)
{
    EXPECT_TRUE(false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_to_uniform)
{
    EXPECT_TRUE(false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_to_rectilinear)
{
    EXPECT_TRUE(false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_to_structured)
{
    EXPECT_TRUE(false);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_to_unstructured)
{
    EXPECT_TRUE(false);
}
*/
