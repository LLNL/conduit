// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_query.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_util.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <set>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
namespace bputils = conduit::blueprint::mesh::util;

/// Testing Constants ///

static const std::vector<std::string> MESH_TYPES = {"uniform", "rectilinear", "structured", "tris", "quads", "polygons", "tets", "hexs", "polyhedra"};
static const std::vector<index_t> MESH_TYPE_DIMS = {3, 3, 3, 2, 2, 2, 3, 3, 3};
static const std::vector<index_t> MESH_TYPE_ELEMS = {1, 1, 1, 2, 1, 1, 6, 1, 1};
static const std::vector<index_t> MESH_DIMS = {3, 4, 5};

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, coordset_dims_query)
{
    for(size_t mi = 0; mi < MESH_TYPES.size(); mi++)
    {
        const std::string mesh_type = MESH_TYPES[mi];
        const index_t mesh_type_dims = MESH_TYPE_DIMS[mi];

        conduit::Node mesh;
        blueprint::mesh::examples::basic(mesh_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],mesh);

        const conduit::Node &coordset = mesh["coordsets"].child(0);
        ASSERT_EQ(blueprint::mesh::coordset::dims(coordset), mesh_type_dims);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, coordset_length_query)
{
    for(size_t mi = 0; mi < MESH_TYPES.size(); mi++)
    {
        const std::string mesh_type = MESH_TYPES[mi];
        const index_t mesh_type_dims = MESH_TYPE_DIMS[mi];

        index_t mesh_type_length = 1;
        for(size_t di = 0; di < mesh_type_dims; di++)
        {
            mesh_type_length *= MESH_DIMS[di];
        }

        conduit::Node mesh;
        blueprint::mesh::examples::basic(mesh_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],mesh);

        const conduit::Node &coordset = mesh["coordsets"].child(0);
        ASSERT_EQ(blueprint::mesh::coordset::length(coordset), mesh_type_length);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, topology_dims_query)
{
    for(size_t mi = 0; mi < MESH_TYPES.size(); mi++)
    {
        const std::string mesh_type = MESH_TYPES[mi];
        const index_t mesh_type_dims = MESH_TYPE_DIMS[mi];
        const bool is_mesh_unstructured = mi > 2;

        conduit::Node mesh;
        blueprint::mesh::examples::basic(mesh_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],mesh);

        const conduit::Node &topology = mesh["topologies"].child(0);
        ASSERT_EQ(blueprint::mesh::topology::dims(topology), mesh_type_dims);

        // NOTE(JRC): The following clause uses the 2D topology in the 3D case
        // to verify that the topology's dimensionality is used and not that
        // of the underlying coordset.
        if(is_mesh_unstructured && mesh_type_dims == 3)
        {
            const std::string mesh_type_2d = MESH_TYPES[mi - 3];

            conduit::Node mesh_2d;
            blueprint::mesh::examples::basic(mesh_type_2d,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],mesh_2d);

            conduit::Node & topology_2d = mesh["topologies"][topology.name() + "_2d"];
            topology_2d.set_external(mesh_2d["topologies"].child(0));
            ASSERT_EQ(blueprint::mesh::topology::dims(topology_2d), 2);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, topology_length_query)
{
    for(size_t mi = 0; mi < MESH_TYPES.size(); mi++)
    {
        const std::string mesh_type = MESH_TYPES[mi];
        const index_t mesh_type_dims = MESH_TYPE_DIMS[mi];
        const bool is_mesh_unstructured = mi > 2;

        index_t mesh_type_length = MESH_TYPE_ELEMS[mi];
        for(size_t di = 0; di < mesh_type_dims; di++)
        {
            mesh_type_length *= (MESH_DIMS[di] - 1);
        }

        conduit::Node mesh;
        blueprint::mesh::examples::basic(mesh_type,MESH_DIMS[0],MESH_DIMS[1],MESH_DIMS[2],mesh);

        const conduit::Node &topology = mesh["topologies"].child(0);
        ASSERT_EQ(blueprint::mesh::topology::length(topology), mesh_type_length);

        // NOTE(JRC): The following code also checks that a topology that
        // doesn't extend to the entire coordset (or a "subtopology") returns
        // the correct topology length.
        if(is_mesh_unstructured)
        {
            index_t submesh_type_length = MESH_TYPE_ELEMS[mi];
            for(size_t di = 0; di < mesh_type_dims; di++)
            {
                submesh_type_length *= (MESH_DIMS[di] - 1 - 1);
            }

            conduit::Node submesh;
            blueprint::mesh::examples::basic(mesh_type,MESH_DIMS[0]-1,MESH_DIMS[1]-1,MESH_DIMS[2]-1,submesh);

            conduit::Node &subtopology = mesh["topologies"][topology.name() + "_sub"];
            subtopology.set_external(submesh["topologies"].child(0));
            ASSERT_EQ(blueprint::mesh::topology::length(subtopology), submesh_type_length);
        }
    }
}
