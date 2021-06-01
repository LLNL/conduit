// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_transform.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_relay_mpi.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Testing Constants ///

typedef void (*GenDerivedFun)(Node&, const std::string&, const std::string&, const std::string&, Node&, Node&);
typedef void (*GenDecomposedFun)(Node&, const std::string&, const std::string&, const std::string&, const std::string&, Node&, Node&);

static const index_t DERIVED_TYPE_POINT_ID = 0;
static const index_t DERIVED_TYPE_LINE_ID = 1;
static const index_t DERIVED_TYPE_FACE_ID = 2;

static const std::vector<std::string> DERIVED_TYPE_NAMES = {"point", "line", "face"};
static const std::vector<std::string> DERIVED_TYPE_SHAPES = {"point", "line", "quad"};
static const std::vector<GenDerivedFun> DERIVED_TYPE_FUNS = {
    conduit::blueprint::mpi::mesh::generate_points,
    conduit::blueprint::mpi::mesh::generate_lines,
    conduit::blueprint::mpi::mesh::generate_faces};

/// Testing Helpers ///

// Utility Functions //

std::vector<index_t> to_index_vector(const Node &n)
{
    std::vector<index_t> index_vector;

    Node data;
    for(index_t ni = 0; ni < n.dtype().number_of_elements(); ni++)
    {
        data.set_external(DataType(n.dtype().id(), 1), (void*)n.element_ptr(ni));
        index_vector.push_back(data.to_index_t());
    }

    return std::vector<index_t>(std::move(index_vector));
}

std::set<index_t> to_index_set(const Node &n)
{
    const std::vector<index_t> index_vector = to_index_vector(n);
    return std::set<index_t>(index_vector.begin(), index_vector.end());
}

std::map<std::set<index_t>, std::set<index_t>> to_group_map(const Node &groups)
{
    // {<domain ids>: <entity index list>, ...}
    // verify that these match up
    std::map<std::set<index_t>, std::set<index_t>> group_map;

    NodeConstIterator group_itr = groups.children();
    while(group_itr.has_next())
    {
        const Node &group = group_itr.next();
        group_map[to_index_set(group["neighbors"])] = to_index_set(group["values"]);
    }

    return std::map<std::set<index_t>, std::set<index_t>>(std::move(group_map));
}

// Setup Functions //

void setup_test_mesh_paths(const index_t type,
                           const Node &mesh,
                           Node &info)
{
    info.reset();

    info["src/cset"].set(mesh.child(0).fetch("coordsets").child_names()[0]);
    info["src/topo"].set(mesh.child(0).fetch("topologies").child_names()[0]);
    info["src/aset"].set(mesh.child(0).fetch("adjsets").child_names()[0]);
    // TODO(JRC): This will need to be changes to the centroid cset for decomposed types.
    info["dst/cset"].set(info["src/cset"]);
    info["dst/topo"].set(DERIVED_TYPE_NAMES[type]);
    info["dst/aset"].set(DERIVED_TYPE_NAMES[type] + "_adj");
    info["dst/shape"].set(DERIVED_TYPE_SHAPES[type]);
}

void setup_derived_test_mesh(const index_t type,
                             const size_t ndims,
                             const size_t dims,
                             Node &rank_mesh,
                             Node &rank_paths,
                             Node &rank_s2dmap,
                             Node &rank_d2smap,
                             Node &full_mesh)
{
    rank_mesh.reset();
    full_mesh.reset();

    conduit::blueprint::mesh::examples::misc("adjsets", dims, dims, (ndims == 3) ? dims : 0, full_mesh);
    setup_test_mesh_paths(type, full_mesh, rank_paths);

    const int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
    const int par_size = relay::mpi::size(MPI_COMM_WORLD);
    if(par_size == 1)
    {
        rank_mesh.set_external(full_mesh);
    }
    else
    {
        std::ostringstream oss;
        oss << "domain" << par_rank;
        const std::string domain_name = oss.str();
        rank_mesh.set_external(full_mesh[domain_name]);
    }

    DERIVED_TYPE_FUNS[type](rank_mesh,
                            rank_paths["src/aset"].as_string(),
                            rank_paths["dst/aset"].as_string(),
                            rank_paths["dst/topo"].as_string(),
                            rank_s2dmap,
                            rank_d2smap);
}

// Test Functions //

void test_mesh_paths(const Node &rank_mesh,
                     const Node &rank_paths)
{
    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(rank_mesh);

    for(const Node *domain_ptr : domains)
    {
        const Node &domain = *domain_ptr;

        // Verify Paths //
        EXPECT_TRUE(domain["adjsets"].has_path(rank_paths["dst/aset"].as_string()));
        EXPECT_TRUE(domain["topologies"].has_path(rank_paths["dst/topo"].as_string()));

        // Verify Path Pointers //
        EXPECT_EQ(domain["adjsets"][rank_paths["dst/aset"].as_string()]["topology"].as_string(),
                  rank_paths["dst/topo"].as_string());
        EXPECT_EQ(domain["topologies"][rank_paths["dst/topo"].as_string()]["coordset"].as_string(),
                  rank_paths["src/cset"].as_string());

        // Verify Enum Values //
        EXPECT_EQ(domain["adjsets"][rank_paths["dst/aset"].as_string()]["association"].as_string(),
                  "element");
        EXPECT_EQ(domain["topologies"][rank_paths["dst/topo"].as_string()]["elements/shape"].as_string(),
                  rank_paths["dst/shape"].as_string());
    }
}

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_points)
{
    const index_t TEST_TYPE_ID = DERIVED_TYPE_POINT_ID;
    const index_t TEST_MESH_NDIM = 2;
    const index_t TEST_MESH_RES = 3;

    Node rank_mesh, full_mesh, info;
    Node rank_paths, rank_s2dmap, rank_d2smap;

    setup_derived_test_mesh(TEST_TYPE_ID, TEST_MESH_NDIM, TEST_MESH_RES,
        rank_mesh, rank_paths, rank_s2dmap, rank_d2smap, full_mesh);
    rank_mesh.print();
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", rank_mesh, info, MPI_COMM_WORLD));

    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(rank_mesh);

    { // Sanity Tests //
        test_mesh_paths(rank_mesh, rank_paths);
    }

    { // Adjacency Tests //
        for(const Node *domain_ptr : domains)
        {
            const Node &domain = *domain_ptr;

            const Node &src_aset_groups = domain["adjsets"][rank_paths["src/aset"].as_string()]["groups"];
            const Node &dst_aset_groups = domain["adjsets"][rank_paths["dst/aset"].as_string()]["groups"];

            // Verify Group Data //
            const std::map<std::set<index_t>, std::set<index_t>> src_group_map = to_group_map(src_aset_groups);
            const std::map<std::set<index_t>, std::set<index_t>> dst_group_map = to_group_map(dst_aset_groups);
            ASSERT_EQ(dst_group_map.size(), src_group_map.size());
            ASSERT_EQ(dst_group_map, src_group_map);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_lines)
{
    const index_t TEST_TYPE_ID = DERIVED_TYPE_LINE_ID;
    const index_t TEST_MESH_NDIM = 2;
    const index_t TEST_MESH_RES = 3;

    Node rank_mesh, full_mesh, info;
    Node rank_paths, rank_s2dmap, rank_d2smap;

    setup_derived_test_mesh(TEST_TYPE_ID, TEST_MESH_NDIM, TEST_MESH_RES,
        rank_mesh, rank_paths, rank_s2dmap, rank_d2smap, full_mesh);
    rank_mesh.print();
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", rank_mesh, info, MPI_COMM_WORLD));

    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(rank_mesh);

    { // Sanity Tests //
        test_mesh_paths(rank_mesh, rank_paths);
    }

    { // Adjacency Tests //
        for(const Node *domain_ptr : domains)
        {
            const Node &domain = *domain_ptr;

            // const Node &src_aset_groups = domain["adjsets"][rank_paths["src/aset"].as_string()]["groups"];
            const Node &dst_aset_groups = domain["adjsets"][rank_paths["dst/aset"].as_string()]["groups"];

            // Verify Group Count //
            const std::map<std::set<index_t>, std::set<index_t>> dst_group_map = to_group_map(dst_aset_groups);
            ASSERT_EQ(dst_group_map.size(), 2); // NOTE(JRC): 2 interfaces per domain in the grid

            // Verify Group Data //
            for(const auto &group_pair : dst_group_map)
            {
                const std::set<index_t> &group_neighbors = group_pair.first;
                const std::set<index_t> &group_values = group_pair.second;
                ASSERT_EQ(group_neighbors.size(), 1);
                ASSERT_EQ(group_values.size(), TEST_MESH_RES - 1);
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_faces)
{
    const index_t TEST_TYPE_ID = DERIVED_TYPE_FACE_ID;
    const index_t TEST_MESH_NDIM = 3;
    const index_t TEST_MESH_RES = 3;

    Node rank_mesh, full_mesh, info;
    Node rank_paths, rank_s2dmap, rank_d2smap;

    setup_derived_test_mesh(TEST_TYPE_ID, TEST_MESH_NDIM, TEST_MESH_RES,
        rank_mesh, rank_paths, rank_s2dmap, rank_d2smap, full_mesh);
    rank_mesh.print();
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", rank_mesh, info, MPI_COMM_WORLD));

    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(rank_mesh);

    { // Sanity Tests //
        test_mesh_paths(rank_mesh, rank_paths);
    }

    { // Adjacency Tests //
        for(const Node *domain_ptr : domains)
        {
            const Node &domain = *domain_ptr;

            // const Node &src_aset_groups = domain["adjsets"][rank_paths["src/aset"].as_string()]["groups"];
            const Node &dst_aset_groups = domain["adjsets"][rank_paths["dst/aset"].as_string()]["groups"];

            // Verify Group Count //
            const std::map<std::set<index_t>, std::set<index_t>> dst_group_map = to_group_map(dst_aset_groups);
            ASSERT_EQ(dst_group_map.size(), 2); // NOTE(JRC): 2 interfaces per domain in the grid

            // Verify Group Data //
            for(const auto &group_pair : dst_group_map)
            {
                const std::set<index_t> &group_neighbors = group_pair.first;
                const std::set<index_t> &group_values = group_pair.second;
                ASSERT_EQ(group_neighbors.size(), 1);
                ASSERT_EQ(group_values.size(), (TEST_MESH_RES - 1) * (TEST_MESH_RES - 1));
            }
        }
    }
}

/// Test Driver ///

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
