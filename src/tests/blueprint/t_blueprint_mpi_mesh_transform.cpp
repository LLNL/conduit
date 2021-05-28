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
}

void setup_derived_test_mesh(const index_t type,
                             const size_t dims,
                             Node &rank_mesh,
                             Node &rank_paths,
                             Node &rank_s2dmap,
                             Node &rank_d2smap,
                             Node &full_mesh)
{
    rank_mesh.reset();
    full_mesh.reset();

    conduit::blueprint::mesh::examples::misc("adjsets", dims, dims, 0, full_mesh);
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
                  rank_paths["dst/topo"].as_string()); // NOTE(JRC): dst/topo name matches topological type
    }
}

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_points)
{
    Node rank_mesh, full_mesh, info;
    Node rank_paths, rank_s2dmap, rank_d2smap;

    setup_derived_test_mesh(DERIVED_TYPE_POINT_ID, 3, rank_mesh, rank_paths, rank_s2dmap, rank_d2smap, full_mesh);
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

            // Verify Group Names //
            const std::vector<std::string> src_aset_group_names = src_aset_groups.child_names();
            const std::vector<std::string> dst_aset_group_names = dst_aset_groups.child_names();
            const std::set<std::string> src_aset_group_nameset(src_aset_group_names.begin(), src_aset_group_names.end());
            const std::set<std::string> dst_aset_group_nameset(dst_aset_group_names.begin(), dst_aset_group_names.end());
            EXPECT_EQ(dst_aset_group_nameset, src_aset_group_nameset);

            // Verify Group Contents //
            for(const std::string &group_name : src_aset_group_names)
            {
                const Node &src_group = src_aset_groups[group_name];
                const Node &dst_group = dst_aset_groups[group_name];

                const std::set<index_t> src_group_neighbors = to_index_set(src_group["neighbors"]);
                const std::set<index_t> dst_group_neighbors = to_index_set(dst_group["neighbors"]);
                ASSERT_EQ(dst_group_neighbors, src_group_neighbors);

                const std::set<index_t> src_group_values = to_index_set(src_group["values"]);
                const std::set<index_t> dst_group_values = to_index_set(dst_group["values"]);
                ASSERT_EQ(dst_group_values, src_group_values);
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_lines)
{
    Node rank_mesh, full_mesh, info;
    Node rank_paths, rank_s2dmap, rank_d2smap;

    setup_derived_test_mesh(DERIVED_TYPE_LINE_ID, 3, rank_mesh, rank_paths, rank_s2dmap, rank_d2smap, full_mesh);
    rank_mesh.print();
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", rank_mesh, info, MPI_COMM_WORLD));

    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(rank_mesh);

    { // Sanity Tests //
        test_mesh_paths(rank_mesh, rank_paths);
    }

    // { // Adjacency Tests //
    //     for(const Node *domain_ptr : domains)
    //     {
    //         const Node &domain = *domain_ptr;

    //         const Node &src_aset_groups = domain["adjsets"][rank_paths["src/aset"].as_string()]["groups"];
    //         const Node &dst_aset_groups = domain["adjsets"][rank_paths["dst/aset"].as_string()]["groups"];

    //         // Verify Group Names //
    //         const std::vector<std::string> src_aset_group_names = src_aset_groups.child_names();
    //         const std::vector<std::string> dst_aset_group_names = dst_aset_groups.child_names();
    //         const std::set<std::string> src_aset_group_nameset(src_aset_group_names.begin(), src_aset_group_names.end());
    //         const std::set<std::string> dst_aset_group_nameset(dst_aset_group_names.begin(), dst_aset_group_names.end());

    //         std::set<std::string> src_diff_dst, dst_diff_src;
    //         std::set_difference(src_aset_group_nameset.begin(), src_aset_group_nameset.end(),
    //                             dst_aset_group_nameset.begin(), dst_aset_group_nameset.end(),
    //                             std::inserter(src_diff_dst, src_diff_dst.begin()));
    //         std::set_difference(dst_aset_group_nameset.begin(), dst_aset_group_nameset.end(),
    //                             src_aset_group_nameset.begin(), src_aset_group_nameset.end(),
    //                             std::inserter(dst_diff_src, dst_diff_src.begin()));
    //         EXPECT_EQ(dst_diff_src.size(), 0);
    //         EXPECT_EQ(src_diff_dst.size(), 1);

    //         // Verify Group Contents //
    //         for(const std::string &group_name : src_aset_group_names)
    //         {
    //             const Node &src_group = src_aset_groups[group_name];
    //             const Node &dst_group = dst_aset_groups[group_name];

    //             const std::set<index_t> src_group_neighbors = to_index_set(src_group["neighbors"]);
    //             const std::set<index_t> dst_group_neighbors = to_index_set(dst_group["neighbors"]);
    //             ASSERT_EQ(dst_group_neighbors, src_group_neighbors);

    //             // TODO(JRC): Put in a better check here; just need to make sure that the
    //             // data in the adjacency set looks correct (may require manual check).
    //             // EXPECT_FALSE(dst_aset_groups.diff(src_aset_groups, info));
    //         }
    //     }
    // }
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
