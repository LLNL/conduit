// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_tiled.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_examples.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_blueprint_mpi_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_log.hpp"

#include "blueprint_test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <mpi.h>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using namespace generate;

// Uncomment if we want to write the data files.
//#define CONDUIT_WRITE_TEST_DATA

//---------------------------------------------------------------------------
#ifdef CONDUIT_WRITE_TEST_DATA
/**
 @brief Save the node to an HDF5 compatible with VisIt or the
        conduit_adjset_validate tool.
 */
void save_mesh(const conduit::Node &root, const std::string &filebase)
{
    // NOTE: Enable this to write files for debugging.
    const std::string protocol("hdf5");
    conduit::relay::mpi::io::blueprint::save_mesh(root, filebase, protocol, MPI_COMM_WORLD);
}
#endif

//-----------------------------------------------------------------------------
/**
 @brief Make domains for a tiled mesh.

 @param mesh The node that will contain the domains.
 @param dims The number of zones in each domain. dims[2] == 0 for 2D meshes.
 @param domains The number of domains to make. All values > 0.
 @param reorder The reordering method, if any.
 @param domainNumbering An optional vector that can reorder the domains.
 */
void
make_tiled(conduit::Node &mesh, const int dims[3], const int domains[3],
           const std::string &reorder, const std::vector<int> &domainNumbering)
{
    const int ndoms = domains[0] * domains[1] * domains[2];
    const int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
    const int par_size = relay::mpi::size(MPI_COMM_WORLD);
    std::vector<int> domains_per_rank(par_size, 0);
    for(int di = 0; di < ndoms; di++)
        domains_per_rank[di % par_size]++;
    int offset = 0;
    for(int i = 0; i < par_rank; i++)
        offset += domains_per_rank[i];

    // Make domains.
    const double extents[] = {0., 1., 0., 1., 0., 1.};
    int domainIndex = 0;
    for(int k = 0; k < domains[2]; k++)
    for(int j = 0; j < domains[1]; j++)
    for(int i = 0; i < domains[0]; i++, domainIndex++)
    {
        int domain[] = {i,j,k};

        int domainId = domainIndex;
        if(!domainNumbering.empty())
            domainId = domainNumbering[domainIndex];

        // See if we need to make the domain on this rank.
        if(domainId >= offset && domainId < offset + domains_per_rank[par_rank])
        {
            // Determine the size and location of this domain in the whole.
            double sideX = (extents[1] - extents[0]) / static_cast<double>(domains[0]);
            double sideY = (extents[3] - extents[2]) / static_cast<double>(domains[1]);
            double sideZ = (extents[5] - extents[4]) / static_cast<double>(domains[2]);
            double domainExt[] = {extents[0] + domain[0]     * sideX,
                                  extents[0] + (domain[0]+1) * sideX,
                                  extents[2] + domain[1]     * sideY,
                                  extents[2] + (domain[1]+1) * sideY,
                                  extents[4] + domain[2]     * sideZ,
                                  extents[4] + (domain[2]+1) * sideZ};
            conduit::Node opts;
            opts["extents"].set(domainExt, 6);
            opts["domain"].set(domain, 3);
            opts["domains"].set(domains, 3);
            opts["reorder"] = reorder;

            if(ndoms > 1)
            {
                char domainName[64];
                if(!domainNumbering.empty())
                    sprintf(domainName, "domain_%07d", domainNumbering[domainIndex]);
                else
                    sprintf(domainName, "domain_%07d", domainIndex);
                conduit::Node &dom = mesh[domainName];
                conduit::blueprint::mesh::examples::tiled(dims[0], dims[1], dims[2], dom, opts);
            }
            else
            {
                conduit::blueprint::mesh::examples::tiled(dims[0], dims[1], dims[2], mesh, opts);
            }
        }
    }
}

//-----------------------------------------------------------------------------
template <typename Func>
void
foreach_permutation_impl(int level, std::vector<int> &values, Func &&func)
{
    int n = static_cast<int>(values.size());
    if(level < n)
    {
        int n = static_cast<int>(values.size());
        for(int i = 0; i < n; i++)
        {
            const int *start = &values[0];
            const int *end = start + level;
            if(std::find(start, end, i) == end)
            {
                values[level] = i;
                foreach_permutation_impl(level + 1, values, func);
            }
        }
    }
    else
    {
        func(values);
    }
}

template <typename Func>
void
foreach_permutation(int maxval, Func &&func)
{
    std::vector<int> values(maxval, -1);
    foreach_permutation_impl(0, values, func);
}

//-----------------------------------------------------------------------------
void
test_tiled_adjsets(const int dims[3], const std::string &testName)
{
    // Make 4 domains but alter their locations.
    int index = 0;
    foreach_permutation(4, [&](const std::vector<int> &domainNumbering)
    {
        const int domains[] = {2,2,1};
        const int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
        const std::vector<std::string> reorder{"normal", "kdtree"};
        for(const auto &r : reorder)
        {
            // Make the mesh.
            conduit::Node mesh;
            make_tiled(mesh, dims, domains, r, domainNumbering);

            // Now, make a corner mesh
            conduit::Node s2dmap, d2smap;
            conduit::blueprint::mpi::mesh::generate_corners(mesh,
                                                            "mesh_adjset",
                                                            "corner_adjset",
                                                            "corner_mesh",
                                                            "corner_coords",
                                                            s2dmap,
                                                            d2smap,
                                                            MPI_COMM_WORLD);
            // Convert to pairwise adjset.
            std::vector<conduit::Node *> doms = conduit::blueprint::mesh::domains(mesh);
            for(auto dom_ptr : doms)
            {
               conduit::Node &domain = *dom_ptr;
               conduit::blueprint::mesh::adjset::to_pairwise(domain["adjsets/corner_adjset"],
                                                             domain["adjsets/corner_pairwise_adjset"]);
            }

#ifdef CONDUIT_WRITE_TEST_DATA
            // Save the mesh.
            std::stringstream ss;
            ss << "_r" << r;
            for(const auto &value : domainNumbering)
                ss << "_" << value;
            std::string filebase(testName + ss.str());
            save_mesh(mesh, filebase);
#endif

            // Check that its adjset points are the same along the edges.
            conduit::Node info;
            bool same = conduit::blueprint::mpi::mesh::utils::adjset::compare_pointwise(mesh, "mesh_adjset", info, MPI_COMM_WORLD);
            if(!same)
            {
                mesh.print();
            }
            EXPECT_TRUE(same);

            // Check that its adjset points are the same along the edges.
            same = conduit::blueprint::mpi::mesh::utils::adjset::compare_pointwise(mesh, "corner_pairwise_adjset", info, MPI_COMM_WORLD);
            if(!same)
            {
                mesh.print();
            }
            EXPECT_TRUE(same);
        }
    });
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_tiled, two_dimensional)
{
    const int dims[]= {3,3,0};
    test_tiled_adjsets(dims, "two_dimensional");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_tiled, three_dimensional)
{
    const int dims[]= {2,2,2};
    test_tiled_adjsets(dims, "three_dimensional");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_tiled, three_dimensional_12)
{
    // This 12 domain case was found to cause adjset problems.
    const int dims[] = {2,2,2}, domains[] = {3,2,2};
    const int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
    const std::vector<std::string> reorder{"normal", "kdtree"};

    for(const auto &r : reorder)
    {
        // Make the mesh.
        conduit::Node mesh;
        make_tiled(mesh, dims, domains, r, std::vector<int>{});

        // Now, make a corner mesh
        conduit::Node s2dmap, d2smap;
        conduit::blueprint::mpi::mesh::generate_corners(mesh,
                                                        "mesh_adjset",
                                                        "corner_adjset",
                                                        "corner_mesh",
                                                        "corner_coords",
                                                        s2dmap,
                                                        d2smap,
                                                        MPI_COMM_WORLD);
        // Convert to pairwise adjset.
        std::vector<conduit::Node *> doms = conduit::blueprint::mesh::domains(mesh);
        for(auto dom_ptr : doms)
        {
            conduit::Node &domain = *dom_ptr;
            conduit::blueprint::mesh::adjset::to_pairwise(domain["adjsets/corner_adjset"],
                                                          domain["adjsets/corner_pairwise_adjset"]);
        }

#ifdef CONDUIT_WRITE_TEST_DATA
        // Save the mesh.
        const std::string testName("three_dimensional_12");
        std::stringstream ss;
        ss << "_r_" << r;
        std::string filebase(testName + ss.str());
        save_mesh(mesh, filebase);
#endif

        // Check that its adjset points are the same along the edges.
        conduit::Node info;
        bool same = conduit::blueprint::mpi::mesh::utils::adjset::compare_pointwise(mesh, "corner_pairwise_adjset", info, MPI_COMM_WORLD);
        EXPECT_TRUE(same);
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
