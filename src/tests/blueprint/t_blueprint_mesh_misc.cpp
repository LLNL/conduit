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
/// file: t_blueprint_mesh_misc.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <cmath>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

static const std::string ELEM_TYPE_LIST[4]      = {"tris", "quads", "tets", "hexs"};
static const index_t ELEM_TYPE_SUBELEMS[4]      = {     2,       1,      6,      1};
static const index_t ELEM_TYPE_INDICES[4]       = {     3,       4,      4,      8};
static const index_t ELEM_TYPE_FACES[4]         = {     1,       1,      4,      6};
static const index_t ELEM_TYPE_FACE_INDICES[4]  = {     3,       4,      3,      4};

/// Test Cases ///

// TODO(JRC): Consider combining the following two test cases in order to remove
// code duplication (though at the cost of test granularity).

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_offsets, generate_offsets_nonpoly)
{
    const index_t MESH_DIMS[3] = {3, 3, 3};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subelems = ELEM_TYPE_SUBELEMS[ti];
        const index_t &elem_indices = ELEM_TYPE_INDICES[ti];
        const bool is_elem_3d = ELEM_TYPE_FACES[ti] > 1;
        const index_t mesh_elems = (MESH_DIMS[0] - 1) * (MESH_DIMS[1] - 1) *
            (is_elem_3d ? (MESH_DIMS[2] - 1) : 1);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing nonpolygonal offets for type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        blueprint::mesh::examples::braid(
            elem_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);

        Node nonpoly_offsets;
        blueprint::mesh::topology::unstructured::generate_offsets(
            nonpoly_topo, nonpoly_offsets);

        EXPECT_EQ(nonpoly_offsets.dtype().id(),
            nonpoly_topo["elements/connectivity"].dtype().id());
        EXPECT_EQ(nonpoly_offsets.dtype().number_of_elements(),
             mesh_elems * elem_subelems);

        Node nonpoly_offsets_int64;
        nonpoly_offsets.to_int64_array(nonpoly_offsets_int64);
        int64_array nonpoly_offset_data = nonpoly_offsets_int64.as_int64_array();
        for(index_t oi = 0; oi < nonpoly_offsets.dtype().number_of_elements(); oi++)
        {
            ASSERT_EQ(nonpoly_offset_data[oi], oi * elem_indices);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_offsets, generate_offsets_poly)
{
    const index_t MESH_DIMS[3] = {3, 3, 3};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subelems = ELEM_TYPE_SUBELEMS[ti];
        const index_t &elem_faces = ELEM_TYPE_FACES[ti];
        const index_t &elem_face_indices = ELEM_TYPE_FACE_INDICES[ti];
        const bool is_elem_3d = ELEM_TYPE_FACES[ti] > 1;
        const index_t mesh_elems = (MESH_DIMS[0] - 1) * (MESH_DIMS[1] - 1) *
            (is_elem_3d ? (MESH_DIMS[2] - 1) : 1);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing polygonal offets for type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        blueprint::mesh::examples::braid(
            elem_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);

        Node poly_topo, poly_offsets;
        blueprint::mesh::topology::unstructured::to_polygonal(nonpoly_topo, poly_topo);
        blueprint::mesh::topology::unstructured::generate_offsets(poly_topo, poly_offsets);

        EXPECT_EQ(poly_offsets.dtype().id(),
            poly_topo["elements/connectivity"].dtype().id());
        EXPECT_EQ(poly_offsets.dtype().number_of_elements(),
            mesh_elems * elem_subelems);

        Node poly_offsets_int64;
        poly_offsets.to_int64_array(poly_offsets_int64);
        int64_array poly_offset_data = poly_offsets_int64.as_int64_array();
        for(index_t oi = 0; oi < poly_offsets.dtype().number_of_elements(); oi++)
        {
            ASSERT_EQ(poly_offset_data[oi],
                oi * (is_elem_3d + elem_faces * (1 + elem_face_indices)));
        }
    }
}
