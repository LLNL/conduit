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

typedef bool (*XformCoordsFun)(const Node&, Node&);
typedef bool (*XformTopoFun)(const Node&, Node&, Node&);
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
    catch(conduit::Error &e)
    {
        braid_type = "hexs";
    }

    return braid_type;
}

/// Wrapper Functions ///

bool transform_coordset_uniform_protocol(const Node &n, Node &m)
{
    return blueprint::mesh::coordset::transform("uniform",n,m);
}

bool transform_coordset_rectilinear_protocol(const Node &n, Node &m)
{
    return blueprint::mesh::coordset::transform("rectilinear",n,m);
}

bool transform_coordset_explicit_protocol(const Node &n, Node &m)
{
    return blueprint::mesh::coordset::transform("explicit",n,m);
}

bool transform_topology_points_protocol(const Node &n, Node &m, Node &o)
{
    return blueprint::mesh::topology::transform("points",n,m,o);
}

bool transform_topology_uniform_protocol(const Node &n, Node &m, Node &o)
{
    return blueprint::mesh::topology::transform("uniform",n,m,o);
}

bool transform_topology_rectilinear_protocol(const Node &n, Node &m, Node &o)
{
    return blueprint::mesh::topology::transform("rectilinear",n,m,o);
}

bool transform_topology_structured_protocol(const Node &n, Node &m, Node &o)
{
    return blueprint::mesh::topology::transform("structured",n,m,o);
}

bool transform_topology_unstructured_protocol(const Node &n, Node &m, Node &o)
{
    return blueprint::mesh::topology::transform("unstructured",n,m,o);
}

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, coordset_xforms)
{
    XformCoordsFun xform_direct_funs[] = {
        blueprint::mesh::coordset::uniform::transform,
        blueprint::mesh::coordset::rectilinear::transform,
        blueprint::mesh::coordset::_explicit::transform};
    XformCoordsFun xform_protocol_funs[] = {
        transform_coordset_uniform_protocol,
        transform_coordset_rectilinear_protocol,
        transform_coordset_explicit_protocol};

    VerifyFun verify_funs[] = {
        blueprint::mesh::coordset::uniform::verify,
        blueprint::mesh::coordset::rectilinear::verify,
        blueprint::mesh::coordset::_explicit::verify};

    for(index_t fi = 0; fi < 2; fi++)
    {
        XformCoordsFun *xform_funs = (fi == 0) ?
            &xform_direct_funs[0] : &xform_protocol_funs[0];

        for(index_t xi = 0; xi < COORD_TYPES.size(); xi++)
        {
            const std::string icoordset_type = COORD_TYPES[xi];
            const std::string icoordset_braid = get_braid_type(icoordset_type);

            conduit::Node imesh;
            blueprint::mesh::examples::braid(icoordset_braid,2,3,4,imesh);
            const conduit::Node &icoordset = imesh["coordsets"].child(0);

            for(index_t xj = 0; xj < COORD_TYPES.size(); xj++)
            {
                const std::string jcoordset_type = COORD_TYPES[xj];
                const std::string jcoordset_braid = get_braid_type(jcoordset_type);

                conduit::Node jmesh;
                blueprint::mesh::examples::braid(jcoordset_braid,2,3,4,jmesh);
                conduit::Node &jcoordset = jmesh["coordsets"].child(0);

                XformCoordsFun to_new_coordset = xform_funs[xj];
                VerifyFun verify_new_coordset = verify_funs[xj];

                conduit::Node xcoordset, info;
                EXPECT_EQ(to_new_coordset(icoordset, xcoordset), xi <= xj); // upscale only
                EXPECT_EQ(verify_new_coordset(xcoordset, info), xi <= xj);  // to_x verify
                EXPECT_EQ(jcoordset.diff(xcoordset, info), xi >  xj);       // to_x values
                EXPECT_EQ(icoordset.diff(xcoordset, info), xi != xj);       // same no change
            }
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_xform, topology_xforms)
{
    XformTopoFun xform_direct_funs[] = {
        blueprint::mesh::topology::points::transform,
        blueprint::mesh::topology::uniform::transform,
        blueprint::mesh::topology::rectilinear::transform,
        blueprint::mesh::topology::structured::transform,
        blueprint::mesh::topology::unstructured::transform};
    XformTopoFun xform_protocol_funs[] = {
        transform_topology_points_protocol,
        transform_topology_uniform_protocol,
        transform_topology_rectilinear_protocol,
        transform_topology_structured_protocol,
        transform_topology_unstructured_protocol};

    VerifyFun verify_topology_funs[] = {
        blueprint::mesh::topology::points::verify,
        blueprint::mesh::topology::uniform::verify,
        blueprint::mesh::topology::rectilinear::verify,
        blueprint::mesh::topology::structured::verify,
        blueprint::mesh::topology::unstructured::verify};
    VerifyFun verify_coordset_funs[] = {
        blueprint::mesh::coordset::verify,
        blueprint::mesh::coordset::uniform::verify,
        blueprint::mesh::coordset::rectilinear::verify,
        blueprint::mesh::coordset::_explicit::verify,
        blueprint::mesh::coordset::_explicit::verify};

    for(index_t fi = 0; fi < 2; fi++)
    {
        XformTopoFun *xform_funs = (fi == 0) ?
            &xform_direct_funs[0] : &xform_protocol_funs[0];

        // NOTE(JRC): We skip the "points" topology during this general check
        // because its rules are peculiar and specific.
        for(index_t xi = 1; xi < TOPO_TYPES.size(); xi++)
        {
            const std::string itopology_type = TOPO_TYPES[xi];
            const std::string itopology_braid = get_braid_type(itopology_type);

            conduit::Node imesh;
            blueprint::mesh::examples::braid(itopology_braid,2,3,4,imesh);
            const conduit::Node &itopology = imesh["topologies"].child(0);
            const conduit::Node &icoordset = imesh["coordsets"].child(0);

            for(index_t xj = 1; xj < 4/*TOPO_TYPES.size()*/; xj++)
            {
                const std::string jtopology_type = TOPO_TYPES[xj];
                const std::string jtopology_braid = get_braid_type(jtopology_type);

                conduit::Node jmesh;
                blueprint::mesh::examples::braid(jtopology_braid,2,3,4,jmesh);
                conduit::Node &jtopology = jmesh["topologies"].child(0);
                conduit::Node &jcoordset = jmesh["coordsets"].child(0);

                XformTopoFun to_new_topology = xform_funs[xj];
                VerifyFun verify_new_topology = verify_topology_funs[xj];
                VerifyFun verify_new_coordset = verify_coordset_funs[xj];

                conduit::Node info;
                conduit::Node &xtopology = imesh["topologies/test"];
                conduit::Node &xcoordset = imesh["coordsets/test"];

                // transform should only allow transfers to more explicit topologies
                EXPECT_EQ(to_new_topology(itopology, xtopology, xcoordset), xi <= xj);
                // output in valid cases should always pass verify tests
                EXPECT_EQ(verify_new_topology(xtopology, info), xi <= xj);
                EXPECT_EQ(verify_new_coordset(xcoordset, info), xi <= xj);

                // for valid transforms, the transformed topology should point
                // to the transformed coordset
                if(verify_new_topology(xtopology, info))
                {
                    EXPECT_EQ(xtopology["coordset"].as_string(), xcoordset.name());
                }

                {
                    // NOTE(JRC): This is necessary because the 'coordset' value
                    // will be different from the transform topology since it
                    // will always create a unique personal one and reference it.
                    conduit::Node dxtopology = xtopology;
                    dxtopology["coordset"].set(itopology["coordset"].as_string());

                    // differences are only expected when a transform couldn't be done
                    EXPECT_EQ(jtopology.diff(dxtopology, info), xi >  xj);
                    EXPECT_EQ(jcoordset.diff(xcoordset, info), xi >  xj);
                    // for identity transforms, the result should match the input
                    EXPECT_EQ(itopology.diff(dxtopology, info), xi != xj);
                    EXPECT_EQ(icoordset.diff(xcoordset, info), xi != xj);
                }

                imesh["topologies"].remove("test");
                imesh["coordsets"].remove("test");
            }
        }
    }
}
