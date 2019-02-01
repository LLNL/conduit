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
/// file: t_blueprint_mesh_transform.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <set>
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

typedef void (*XformCoordsFun)(const Node&, Node&);
typedef void (*XformTopoFun)(const Node&, Node&, Node&);
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
    catch(conduit::Error &) // actual exception is unused
    {
        braid_type = "hexs";
    }

    return braid_type;
}

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_transform, coordset_transforms)
{
    XformCoordsFun xform_funs[3][3] = {
        {NULL, blueprint::mesh::coordset::uniform::to_rectilinear, blueprint::mesh::coordset::uniform::to_explicit},
        {NULL, NULL, blueprint::mesh::coordset::rectilinear::to_explicit},
        {NULL, NULL, NULL}};

    VerifyFun verify_funs[3] = {
        blueprint::mesh::coordset::uniform::verify,
        blueprint::mesh::coordset::rectilinear::verify,
        blueprint::mesh::coordset::_explicit::verify};

    for(index_t xi = 0; xi < COORD_TYPES.size(); xi++)
    {
        const std::string icoordset_type = COORD_TYPES[xi];
        const std::string icoordset_braid = get_braid_type(icoordset_type);

        conduit::Node imesh;
        blueprint::mesh::examples::braid(icoordset_braid,2,3,4,imesh);
        const conduit::Node &icoordset = imesh["coordsets"].child(0);

        for(index_t xj = xi + 1; xj < COORD_TYPES.size(); xj++)
        {
            const std::string jcoordset_type = COORD_TYPES[xj];
            const std::string jcoordset_braid = get_braid_type(jcoordset_type);

            // NOTE: The following lines are for debugging purposes only.
            std::cout << "Testing coordset " << icoordset_type << " -> " <<
                jcoordset_type << "..." << std::endl;

            conduit::Node jmesh;
            blueprint::mesh::examples::braid(jcoordset_braid,2,3,4,jmesh);
            conduit::Node &jcoordset = jmesh["coordsets"].child(0);

            XformCoordsFun to_new_coordset = xform_funs[xi][xj];
            VerifyFun verify_new_coordset = verify_funs[xj];

            conduit::Node xcoordset, info;
            to_new_coordset(icoordset, xcoordset);

            EXPECT_TRUE(verify_new_coordset(xcoordset, info));
            EXPECT_FALSE(jcoordset.diff(xcoordset, info));
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_transform, topology_transforms)
{
    XformTopoFun xform_funs[5][5] = {
        {NULL, NULL, NULL, NULL, NULL},
        {NULL, NULL, blueprint::mesh::topology::uniform::to_rectilinear, blueprint::mesh::topology::uniform::to_structured, blueprint::mesh::topology::uniform::to_unstructured},
        {NULL, NULL, NULL, blueprint::mesh::topology::rectilinear::to_structured, blueprint::mesh::topology::rectilinear::to_unstructured},
        {NULL, NULL, NULL, NULL, blueprint::mesh::topology::structured::to_unstructured},
        {NULL, NULL, NULL, NULL, NULL}};
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

        for(index_t xj = xi + 1; xj < TOPO_TYPES.size(); xj++)
        {
            const std::string jtopology_type = TOPO_TYPES[xj];
            const std::string jtopology_braid = get_braid_type(jtopology_type);

            // NOTE: The following lines are for debugging purposes only.
            std::cout << "Testing topology " << itopology_type << " -> " <<
                jtopology_type << "..." << std::endl;

            conduit::Node jmesh;
            blueprint::mesh::examples::braid(jtopology_braid,2,3,4,jmesh);
            conduit::Node &jtopology = jmesh["topologies"].child(0);
            conduit::Node &jcoordset = jmesh["coordsets"].child(0);

            XformTopoFun to_new_topology = xform_funs[xi][xj];
            VerifyFun verify_new_topology = verify_topology_funs[xj];
            VerifyFun verify_new_coordset = verify_coordset_funs[xj];

            conduit::Node info;
            conduit::Node &xtopology = imesh["topologies/test"];
            conduit::Node &xcoordset = imesh["coordsets/test"];
            to_new_topology(itopology, xtopology, xcoordset);

            EXPECT_TRUE(verify_new_topology(xtopology, info));
            EXPECT_TRUE(verify_new_coordset(xcoordset, info));
            EXPECT_EQ(xtopology["coordset"].as_string(), xcoordset.name());

            // NOTE(JRC): This is necessary because the 'coordset' value
            // will be different from the transform topology since it
            // will always create a unique personal one and reference it.
            conduit::Node dxtopology = xtopology;
            dxtopology["coordset"].set(itopology["coordset"].as_string());

            EXPECT_FALSE(jtopology.diff(dxtopology, info));
            EXPECT_FALSE(jcoordset.diff(xcoordset, info));

            imesh["topologies"].remove("test");
            imesh["coordsets"].remove("test");
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_transform, polygonal_transforms)
{
    const std::string TOPO_TYPE_LIST[5]      = {"lines", "tris", "quads","tets","hexs"};
    const index_t TOPO_TYPE_INDICES[5]       = {      2,      3,       4,     4,     8};
    const index_t TOPO_TYPE_FACES[5]         = {      1,      1,       1,     4,     6};
    const index_t TOPO_TYPE_FACE_INDICES[5]  = {      2,      3,       4,     3,     4};

    const index_t MESH_DIMS[3] = {3, 3, 3};

    for(index_t ti = 0; ti < 5; ti++)
    {
        const std::string &topo_type = TOPO_TYPE_LIST[ti];
        const index_t &topo_indices = TOPO_TYPE_INDICES[ti];
        const index_t &topo_faces = TOPO_TYPE_FACES[ti];
        const index_t &topo_findices = TOPO_TYPE_FACE_INDICES[ti];
        const bool is_topo_3d = TOPO_TYPE_FACES[ti] > 1;

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing topology type '" << topo_type << "' -> " <<
            "polygonal..." << std::endl;

        conduit::Node topo_mesh;
        blueprint::mesh::examples::braid(topo_type,
            MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],topo_mesh);
        const conduit::Node &topo_node = topo_mesh["topologies"].child(0);

        conduit::Node topo_poly;
        blueprint::mesh::topology::unstructured::to_polygonal(topo_node, topo_poly);

        conduit::Node info;

        { // Verify Non-Elements Components //
            conduit::Node topo_noelem, poly_noelem;
            topo_noelem.set_external(topo_node);
            topo_noelem.remove("elements");
            poly_noelem.set_external(topo_poly);
            poly_noelem.remove("elements");
            ASSERT_FALSE(topo_noelem.diff(poly_noelem, info));
        }

        { // Verify Element Components //
            ASSERT_EQ(topo_poly["elements/shape"].as_string(),
                is_topo_3d ? "polyhedral" : "polygonal");

            const conduit::Node &topo_conn = topo_node["elements/connectivity"];
            conduit::Node &poly_conn = topo_poly["elements/connectivity"];
            ASSERT_EQ(poly_conn.dtype().id(), topo_conn.dtype().id());

            const index_t topo_len = topo_conn.dtype().number_of_elements();
            const index_t poly_len = poly_conn.dtype().number_of_elements();
            const index_t topo_elems = topo_len / topo_indices;
            const index_t poly_stride = poly_len / topo_elems;

            ASSERT_EQ(poly_stride, (is_topo_3d + topo_faces * (1 + topo_findices)));
            ASSERT_EQ(poly_len % topo_elems, 0);

            conduit::Node topo_conn_array, poly_conn_array;
            topo_conn.to_int64_array(topo_conn_array);
            poly_conn.to_int64_array(poly_conn_array);
            const conduit::int64_array topo_data = topo_conn_array.as_int64_array();
            const conduit::int64_array poly_data = poly_conn_array.as_int64_array();
            for(index_t ep = 0, et = 0; ep < poly_len;
                ep += poly_stride, et += topo_indices)
            {
                ASSERT_EQ(poly_data[ep], is_topo_3d ? topo_faces : topo_findices);

                for(index_t efo = ep + is_topo_3d; efo < ep + poly_stride;
                    efo += 1 + topo_findices)
                {
                    ASSERT_EQ(poly_data[efo], topo_findices);

                    const std::set<index_t> topo_index_set(
                        &topo_data[et],
                        &topo_data[et + topo_indices]);
                    const std::set<index_t> poly_index_set(
                        &poly_data[efo + 1],
                        &poly_data[efo + 1 + topo_findices]);
                    // set of face indices is completely unique (no duplicates)
                    ASSERT_EQ(poly_index_set.size(), topo_findices);
                    // all polygonal face indices can be found in the base element
                    ASSERT_TRUE(std::includes(
                        topo_index_set.begin(), topo_index_set.end(),
                        poly_index_set.begin(), poly_index_set.end()));
                }
            }
        }
    }
}
