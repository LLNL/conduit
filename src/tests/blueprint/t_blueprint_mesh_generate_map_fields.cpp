// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.
//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_generate_map_fields.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "windows.h"
#endif

#include <string>
#include "conduit_blueprint.hpp"
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::blueprint::mesh;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_and_map_fields_2D)
{
    index_t nlevels = 2;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, n);
    EXPECT_TRUE(verify(n, info));

    topology::unstructured::generate_sides_and_map_fields(n, side_mesh, "topo");
    EXPECT_TRUE(verify(side_mesh, info));

    EXPECT_EQ(side_mesh["fields/level/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/level/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/level/volume_dependent"].as_string(), "false");

    index_t num_field_values = 56;
    index_t num_polygons = 9;
    EXPECT_EQ(side_mesh["fields/level/values"].dtype().number_of_elements(), num_field_values);

    uint32 *level_values = side_mesh["fields/level/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < 8)
        {
            EXPECT_EQ(level_values[i], 1);
        }
        else
        {
            EXPECT_EQ(level_values[i], 2);
        }
    }

    EXPECT_EQ(side_mesh["fields/original_element_ids/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/original_element_ids/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/original_element_ids/volume_dependent"].as_string(), "false");

    EXPECT_EQ(side_mesh["fields/original_element_ids/values"].dtype().number_of_elements(), num_field_values);

    uint32 *id_values = side_mesh["fields/original_element_ids/values"].value();
    
    int i = 0;
    for (int j = 0; j < num_polygons; j ++)
    {
        if (j % 2)
        {
            EXPECT_EQ(id_values[i], j);
            EXPECT_EQ(id_values[i + 1], j);
            EXPECT_EQ(id_values[i + 2], j);
            EXPECT_EQ(id_values[i + 3], j);
            i += 4;
        }
        else
        {
            EXPECT_EQ(id_values[i], j);
            EXPECT_EQ(id_values[i + 1], j);
            EXPECT_EQ(id_values[i + 2], j);
            EXPECT_EQ(id_values[i + 3], j);
            EXPECT_EQ(id_values[i + 4], j);
            EXPECT_EQ(id_values[i + 5], j);
            EXPECT_EQ(id_values[i + 6], j);
            EXPECT_EQ(id_values[i + 7], j);
            i += 8;
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_and_map_fields_3D)
{
    index_t length = 1;
    Node n, side_mesh, info;

    // create a polychain of length 1
    examples::polychain(length, n);
    EXPECT_TRUE(verify(n, info));

    topology::unstructured::generate_sides_and_map_fields(n, side_mesh, "topo");
    EXPECT_TRUE(verify(side_mesh, info));

    EXPECT_EQ(side_mesh["fields/chain/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/chain/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/chain/volume_dependent"].as_string(), "false");

    index_t num_tets_in_hex = 24;
    index_t num_tets_in_triprism = 18;

    index_t num_field_values = num_tets_in_hex + 2 * num_tets_in_triprism;
    EXPECT_EQ(side_mesh["fields/chain/values"].dtype().number_of_elements(), num_field_values);

    int64 *chain_values = side_mesh["fields/chain/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < num_tets_in_hex)
        {
            EXPECT_EQ(chain_values[i], 0);
        }
        else
        {
            EXPECT_EQ(chain_values[i], 1);
        }
    }

    EXPECT_EQ(side_mesh["fields/original_element_ids/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/original_element_ids/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/original_element_ids/volume_dependent"].as_string(), "false");

    EXPECT_EQ(side_mesh["fields/original_element_ids/values"].dtype().number_of_elements(), num_field_values);

    uint32 *id_values = side_mesh["fields/original_element_ids/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < num_tets_in_hex)
        {
            EXPECT_EQ(id_values[i], 0);
        }
        else if (i < num_tets_in_hex + num_tets_in_triprism)
        {
            EXPECT_EQ(id_values[i], 1);
        }
        else
        {
            EXPECT_EQ(id_values[i], 2);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_and_map_fields_exceptions)
{
    index_t nlevels = 2;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];

    // catch float64 error
    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"], 
                                                            side_topo, 
                                                            side_coords, 
                                                            s2dmap, 
                                                            d2smap);
    d2smap["values"].set(conduit::DataType::float64(1));
    try
    {
        blueprint::mesh::topology::unstructured::map_fields_to_generated_sides(n, d2smap, side_mesh, "topo");
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "Unsupported field type in dtype: \"float64\"";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }

    // catch if shape is not tet or tri
    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"], 
                                                            side_topo, 
                                                            side_coords, 
                                                            s2dmap, 
                                                            d2smap);
    side_mesh["topologies/topo/elements/shape"] = "strange_shape";
    try
    {
        blueprint::mesh::topology::unstructured::map_fields_to_generated_sides(n, d2smap, side_mesh, "topo");
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "Bad shape in strange_shape";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }

    // catch if field is vertex associated
    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"], 
                                                            side_topo, 
                                                            side_coords, 
                                                            s2dmap, 
                                                            d2smap);
    n["fields/level/association"] = "vertex";
    try
    {
        blueprint::mesh::topology::unstructured::map_fields_to_generated_sides(n, d2smap, side_mesh, "topo");
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "Vertex associated fields are not supported.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
    n["fields/level/association"] = "element";

    // catch if field is volume dependent
    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"], 
                                                            side_topo, 
                                                            side_coords, 
                                                            s2dmap, 
                                                            d2smap);
    n["fields/level/volume_dependent"] = "true";
    try
    {
        blueprint::mesh::topology::unstructured::map_fields_to_generated_sides(n, d2smap, side_mesh, "topo");
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "Volume dependent fields are not supported.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }

    // next error to catch is on line 3072
}
