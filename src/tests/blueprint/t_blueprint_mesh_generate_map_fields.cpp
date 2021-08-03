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
#include "conduit_relay.hpp"
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::blueprint::mesh;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_2D)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"] = "level";

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);

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
TEST(conduit_blueprint_generate_unstructured, generate_sides_2D_skip_bad_field)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    n["fields/level_fake/topology"] = "bad_topo";

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);

    EXPECT_TRUE(verify(side_mesh, info));

    EXPECT_TRUE(! side_mesh["fields"].has_child("level_fake"));

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
TEST(conduit_blueprint_generate_unstructured, generate_sides_2D_options_no_field_names)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);

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
TEST(conduit_blueprint_generate_unstructured, generate_sides_2D_options_field_prefix)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_prefix"] = "my_prefix_";

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);

    EXPECT_TRUE(verify(side_mesh, info));

    EXPECT_EQ(side_mesh["fields/my_prefix_level/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/my_prefix_level/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/my_prefix_level/volume_dependent"].as_string(), "false");

    index_t num_field_values = 56;
    index_t num_polygons = 9;
    EXPECT_EQ(side_mesh["fields/my_prefix_level/values"].dtype().number_of_elements(), num_field_values);

    uint32 *level_values = side_mesh["fields/my_prefix_level/values"].value();

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
TEST(conduit_blueprint_generate_unstructured, generate_sides_3D)
{
    index_t length = 1;
    Node n, side_mesh, info;

    // create a polychain of length 1
    examples::polychain(length, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"] = "chain";

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);
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
TEST(conduit_blueprint_generate_unstructured, generate_sides_2D_vol_dep)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    // make another field
    Node temp;
    temp = n["fields/level"];
    n["fields/level_vol"] = temp;
    n["fields/level_vol/volume_dependent"] = "true";

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);

    EXPECT_TRUE(verify(side_mesh, info));

    // check level field
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

    // check level_vol field
    EXPECT_EQ(side_mesh["fields/level_vol/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/level_vol/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/level_vol/volume_dependent"].as_string(), "true");
    EXPECT_EQ(side_mesh["fields/level_vol/values"].dtype().number_of_elements(), num_field_values);

    float64 *level_vol_values = side_mesh["fields/level_vol/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < 8)
        {
            EXPECT_NEAR(level_vol_values[i], 0.125f, 0.0001f);
        }
        else if (i < 12)
        {
            EXPECT_NEAR(level_vol_values[i], 0.5f, 0.0001f);
        }
        else if (i < 20)
        {
            EXPECT_NEAR(level_vol_values[i], 0.25f, 0.0001f);
        }
        else if (i < 24)
        {
            EXPECT_NEAR(level_vol_values[i], 0.5f, 0.0001f);
        }
        else if (i < 32)
        {
            EXPECT_NEAR(level_vol_values[i], 0.25f, 0.0001f);
        }
        else if (i < 36)
        {
            EXPECT_NEAR(level_vol_values[i], 0.5f, 0.0001f);
        }
        else if (i < 44)
        {
            EXPECT_NEAR(level_vol_values[i], 0.25f, 0.0001f);
        }
        else if (i < 48)
        {
            EXPECT_NEAR(level_vol_values[i], 0.5f, 0.0001f);
        }
        else if (i < 56)
        {
            EXPECT_NEAR(level_vol_values[i], 0.25f, 0.0001f);
        }
    }

    // check volume field
    EXPECT_EQ(side_mesh["fields/volume/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/volume/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/volume/volume_dependent"].as_string(), "true");
    EXPECT_EQ(side_mesh["fields/volume/values"].dtype().number_of_elements(), num_field_values);

    float64 *volume_values = side_mesh["fields/volume/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < 8)
        {
            EXPECT_NEAR(volume_values[i], 0.6036f, 0.0001f);
        }
        else if (i < 12)
        {
            EXPECT_NEAR(volume_values[i], 0.25f, 0.0001f);
        }
        else if (i < 20)
        {
            EXPECT_NEAR(volume_values[i], 0.6036f, 0.0001f);
        }
        else if (i < 24)
        {
            EXPECT_NEAR(volume_values[i], 0.25f, 0.0001f);
        }
        else if (i < 32)
        {
            EXPECT_NEAR(volume_values[i], 0.6036f, 0.0001f);
        }
        else if (i < 36)
        {
            EXPECT_NEAR(volume_values[i], 0.25f, 0.0001f);
        }
        else if (i < 44)
        {
            EXPECT_NEAR(volume_values[i], 0.6036f, 0.0001f);
        }
        else if (i < 48)
        {
            EXPECT_NEAR(volume_values[i], 0.25f, 0.0001f);
        }
        else if (i < 56)
        {
            EXPECT_NEAR(volume_values[i], 0.6036f, 0.0001f);
        }
    }

    // check original element ids
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
TEST(conduit_blueprint_generate_unstructured, generate_sides_3D_vol_dep)
{
    index_t length = 1;
    Node n, side_mesh, info;

    // create a polychain of length 1
    examples::polychain(length, n);
    EXPECT_TRUE(verify(n, info));

    // make another field
    Node temp;
    temp = n["fields/chain"];
    n["fields/chain_vol"] = temp;
    n["fields/chain_vol/volume_dependent"] = "true";

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;

    blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            options);
    EXPECT_TRUE(verify(side_mesh, info));

    // check chain field
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

    // check volume field
    EXPECT_EQ(side_mesh["fields/volume/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/volume/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/volume/volume_dependent"].as_string(), "true");

    EXPECT_EQ(side_mesh["fields/volume/values"].dtype().number_of_elements(), num_field_values);

    float64 *volume_values = side_mesh["fields/volume/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < num_tets_in_hex)
        {
            EXPECT_NEAR(volume_values[i], 0.3333f, 0.0001f);
        }
        else
        {
            EXPECT_NEAR(volume_values[i], 0.2222f, 0.0001f);
        }
    }

    // check chain_vol field
    EXPECT_EQ(side_mesh["fields/chain_vol/topology"].as_string(), "topo");
    EXPECT_EQ(side_mesh["fields/chain_vol/association"].as_string(), "element");
    EXPECT_EQ(side_mesh["fields/chain_vol/volume_dependent"].as_string(), "true");

    EXPECT_EQ(side_mesh["fields/chain_vol/values"].dtype().number_of_elements(), num_field_values);

    float64 *chain_vol_values = side_mesh["fields/chain_vol/values"].value();

    for (int i = 0; i < num_field_values; i ++)
    {
        if (i < num_tets_in_hex)
        {
            EXPECT_NEAR(chain_vol_values[i], 0.0f, 0.0001f);
        }
        else
        {
            EXPECT_NEAR(chain_vol_values[i], 0.0555f, 0.0001f);
        }
    }

    // check original_element_ids field
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
TEST(conduit_blueprint_generate_unstructured, generate_sides_field_datatype_ex)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"] = "level";

    // catch if field has wrong data type
    try
    {
        n["fields/level/values"].set(conduit::DataType::int8(1));
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "Unsupported field type in dtype: \"int8\"";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_options_field_prefix_ex)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_prefix"] = 123;
    options["field_names"] = "level";

    // catch if field_prefix is not a string
    try
    {
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "field_prefix must be a string.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_options_field_name_ex1)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"] = 1;

    // catch if field_names is not a string
    try
    {
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "field_names must be a string or a list of strings.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_options_field_name_ex2)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"].append().set(1);
    options["field_names"].append().set(2);
    options["field_names"].append().set(3);

    // catch if field_names is not a string
    try
    {
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "field_names must be a string or a list of strings.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_options_field_name_ex3)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"].append().set("level");
    options["field_names"].append().set("level2");

    // catch if field_names is not a string
    try
    {
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "field level2 not found in target.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_options_field_name_ex4)
{
    index_t nlevels = 2;
    index_t nz = 1;
    Node n, side_mesh, info;

    // create polytessalation with two levels
    examples::polytess(nlevels, nz, n);
    EXPECT_TRUE(verify(n, info));

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    Node options;
    options["field_names"].append().set("level");
    options["field_names"].append().set("level_fake");
    n["fields/level_fake/topology"] = "bad_topo";

    // catch if a field is requested that does not use the chosen topology
    try
    {
        blueprint::mesh::topology::unstructured::generate_sides(n["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                options);
        FAIL();
    }
    catch(const std::exception& err)
    {
        std::string msg = "field level_fake does not use topo.";
        std::string actual = err.what();
        EXPECT_TRUE(actual.find(msg) != std::string::npos);
    }
}
