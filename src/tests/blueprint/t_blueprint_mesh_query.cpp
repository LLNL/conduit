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
#include "conduit_log.hpp"

#include <set>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

/// Testing Helpers ///

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, mesh_domains)
{
    { // Empty Tests //
        { // Empty Node Test //
            Node mesh;
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }

        { // Empty Object Test //
            Node mesh(DataType::object());
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }

        { // Empty List Test //
            Node mesh(DataType::list());
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }
    }

    { // Non-Empty Tests //
        { // Uni-Domain Test //
            Node mesh;
            blueprint::mesh::examples::braid("quads",10,10,0,mesh);

            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 1);
            ASSERT_EQ(domains.back(), &mesh);
        }

        { // Multi-Domain Test //
            Node mesh;
            blueprint::mesh::examples::grid("quads",10,10,0,2,2,1,mesh);

            std::set<Node *> ref_domains;
            for(const std::string &child_name : mesh.child_names())
            {
                ref_domains.insert(&mesh.child(child_name));
            }

            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            std::set<Node *> acc_domains(domains.begin(), domains.end());
            ASSERT_EQ(domains.size(), ref_domains.size());
            ASSERT_EQ(acc_domains.size(), ref_domains.size());
            ASSERT_EQ(acc_domains, ref_domains);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, adjset_formats)
{
    { // Pairwise Tests //
        { // Empty Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,0,2,2,1,mesh);

            for(Node *domain : blueprint::mesh::domains(mesh))
            {
                Node &domain_adjset = (*domain)["adjsets"].child(0);
                domain_adjset["groups"].reset();
                domain_adjset["groups"].set(DataType::object());

                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }

        { // Positive Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,0,2,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }

            blueprint::mesh::examples::grid("quads",10,10,0,4,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }

        { // Negative Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,0,2,2,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_FALSE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }

            blueprint::mesh::examples::grid("hexs",10,10,10,2,2,2,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_FALSE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }
    }

    { // Max-Share Tests //
        { // Empty Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,0,2,2,1,mesh);

            for(Node *domain : blueprint::mesh::domains(mesh))
            {
                Node &domain_adjset = (*domain)["adjsets"].child(0);
                domain_adjset["groups"].reset();
                domain_adjset["groups"].set(DataType::object());

                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }
        }

        { // Positive Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,0,2,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }

            blueprint::mesh::examples::grid("quads",10,10,0,2,2,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }
        }

        { // Negative Test //
            Node mesh, info;

            // TODO: All tests return adjsets that are natively 'max-share',
            // so testing the negative case will require either using a transform
            // or constructing a simple non-max-share mesh.
            // ASSERT_TRUE(true);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, point_query)
{
    using PointQuery = conduit::blueprint::mesh::utils::query::PointQuery;
    constexpr int nx = 3, ny = 3, nz = 0;
    constexpr int npts = nx * ny;
    constexpr int domain0 = 0;
    constexpr int domain1 = 1;

    auto getcoord = [](const conduit::Node &coordset, int ptid, double pt3[3])
    {
        auto pt = conduit::blueprint::mesh::utils::coordset::_explicit::coords(coordset, ptid);
        pt3[0] = pt[0];
        pt3[1] = (pt.size() > 1) ? pt[1] : 0.;
        pt3[2] = (pt.size() > 2) ? pt[2] : 0.;
    };

    // Make 2 domains.
    Node doms;
    blueprint::mesh::examples::grid("quads",
                                    nx, ny, nz, // Number of points in x,y,z
                                    2, 1, 1,    // 2 domains in X
                                    doms);

    // Run the point query so we ask for all of the points in domain 0 using
    // the points of domain 0. We should hit them all and have 0..npts-1 for
    // the results.
    PointQuery Q(doms);
    auto domains = conduit::blueprint::mesh::domains(doms);
    const std::string coordsetName("coords");
    conduit::Node &coordset = domains[0]->fetch_existing("coordsets/" + coordsetName);
    for(int ptid = 0; ptid < npts; ptid++)
    {
        double pt3[3];
        getcoord(coordset, ptid, pt3);

        Q.Add(domain0, pt3);
    }
    Q.Execute(coordsetName);
    const auto &res = Q.Results(domain0);
    EXPECT_EQ(res.size(), npts);
    for(size_t i = 0; i < res.size(); i++)
       EXPECT_EQ(res[i], i);

    // Now, add some points that do not exist and re-run the query.
    double bad0[] = {1.23,1.23,1.23};
    double bad1[] = {1.34,1.34,1.34};
    double bad2[] = {1.45,1.45,1.45};
    Q.Add(domain0, bad0);
    Q.Add(domain0, bad1);
    Q.Add(domain0, bad2);
    Q.Execute(coordsetName);
    EXPECT_EQ(res.size(), npts + 3);
    for(size_t i = 0; i < res.size(); i++)
    {
        if(i < npts)
        {
            EXPECT_EQ(res[i], i);
        }
        else
        {
            EXPECT_EQ(res[i], PointQuery::NotFound);
        }
    }

    // Now, ask domain1 about domain0's points. There should be 3 that match.
    Q.Reset();
    for(int ptid = 0; ptid < npts; ptid++)
    {
        double pt3[3];
        getcoord(coordset, ptid, pt3);

        Q.Add(domain1, pt3);
    }
    Q.Execute(coordsetName);
    const auto &res1 = Q.Results(domain1);
    EXPECT_EQ(res1.size(), npts);
    for(size_t i = 0; i < res1.size(); i++)
    {
        if(i % nx == (nx - 1))
        {
            EXPECT_EQ(res1[i], i - (nx - 1));
        }
        else
        {
            EXPECT_EQ(res1[i], PointQuery::NotFound);
        }
    }

}
