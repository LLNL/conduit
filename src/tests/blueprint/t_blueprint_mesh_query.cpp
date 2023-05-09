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
#include "conduit_blueprint_mesh_kdtree.hpp"
#include "conduit_log.hpp"

#include <cmath>
#include <set>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using PointQuery = conduit::blueprint::mesh::utils::query::PointQuery;

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

        Q.add(domain0, pt3);
    }
    Q.execute(coordsetName);
    const auto &res = Q.results(domain0);
    EXPECT_EQ(res.size(), npts);
    for(size_t i = 0; i < res.size(); i++)
       EXPECT_EQ(res[i], i);

    // Now, add some points that do not exist and re-run the query.
    double bad0[] = {1.23,1.23,1.23};
    double bad1[] = {1.34,1.34,1.34};
    double bad2[] = {1.45,1.45,1.45};
    Q.add(domain0, bad0);
    Q.add(domain0, bad1);
    Q.add(domain0, bad2);
    Q.execute(coordsetName);
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
    Q.reset();
    for(int ptid = 0; ptid < npts; ptid++)
    {
        double pt3[3];
        getcoord(coordset, ptid, pt3);

        Q.add(domain1, pt3);
    }
    Q.execute(coordsetName);
    const auto &res1 = Q.results(domain1);
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

//---------------------------------------------------------------------------
template <typename T>
void make_coords_3d(T **coords, int dims[3])
{
    // Initialization.
    int npts = dims[0] * dims[1] * dims[2];
    for(int dim = 0; dim < 3; dim++)
        coords[dim] = new T[npts];

    // Make coordinates.
    int idx = 0;
    for(int k = 0; k < dims[2]; k++)
    for(int j = 0; j < dims[1]; j++)
    for(int i = 0; i < dims[0]; i++, idx++)
    {
        coords[0][idx] = static_cast<T>(i);
        coords[1][idx] = static_cast<T>(j);
        coords[2][idx] = static_cast<T>(k);
    } 
}

//---------------------------------------------------------------------------
template <typename T>
void free_coords_3d(T **coords)
{
    delete [] coords[0];
    delete [] coords[1];
    delete [] coords[2];
}

//---------------------------------------------------------------------------
template <typename T>
void make_coords_2d(T **coords, int dims[2])
{
    // Initialization.
    int npts = dims[0] * dims[1];
    for(int dim = 0; dim < 2; dim++)
        coords[dim] = new T[npts];

    // Make coordinates.
    int idx = 0;
    for(int j = 0; j < dims[1]; j++)
    for(int i = 0; i < dims[0]; i++, idx++)
    {
        coords[0][idx] = static_cast<T>(i);
        coords[1][idx] = static_cast<T>(j);
    } 
}

//---------------------------------------------------------------------------
template <typename T>
void free_coords_2d(T **coords)
{
    delete [] coords[0];
    delete [] coords[1];
}

//---------------------------------------------------------------------------
template <typename T>
void kdtree_3d(int dims[3])
{
    // Initialization.
    T *coords[3];
    make_coords_3d(coords, dims);
    int npts = dims[0] * dims[1] * dims[2];
    
    // Make sure we can identify all of the coordinates.
    conduit::blueprint::mesh::utils::kdtree<T *, T, 3> search;
    search.initialize(coords, npts);
    int found, foundCount = 0;
    for(int i = 0; i < npts; i++)
    {
        T pt[3];
        pt[0] = coords[0][i];
        pt[1] = coords[1][i];
        pt[2] = coords[2][i];

        found = search.findPoint(pt);
        EXPECT_EQ(found, i);

        foundCount += (found != search.NotFound) ? 1 : 0;
    }
    EXPECT_EQ(foundCount, npts);

    // Make sure some bad points fail.
    T badpt[] = {-1., -1., -1.};
    found = search.findPoint(badpt);
    EXPECT_EQ(found, search.NotFound);

    free_coords_3d(coords);
}

//---------------------------------------------------------------------------
template <typename T>
void kdtree_2d(int dims[2])
{
    // Initialization.
    T *coords[2];
    make_coords_2d(coords, dims);
    int npts = dims[0] * dims[1];

    conduit::blueprint::mesh::utils::kdtree<T *, T, 2> search;
    search.initialize(coords, npts);
    int found, foundCount = 0;
    for(int i = 0; i < npts; i++)
    {
        T pt[2];
        pt[0] = coords[0][i];
        pt[1] = coords[1][i];

        found = search.findPoint(pt);
        EXPECT_EQ(found, i);

        foundCount += (found != search.NotFound) ? 1 : 0;
    }
    EXPECT_EQ(foundCount, npts);

    // Make sure some bad points fail.
    T badpt[] = {-1., -1.};
    found = search.findPoint(badpt);
    EXPECT_EQ(found, search.NotFound);

    free_coords_2d(coords);
}

//---------------------------------------------------------------------------
template <typename T>
void single_domain_point_query_3d(int dims[3])
{
    // Initialization.
    T *coords[3];
    make_coords_3d(coords, dims);
    int npts = dims[0] * dims[1] * dims[2];

    // NOTE: We don't need the topology.
    conduit::Node mesh;
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(coords[0], npts);
    mesh["coordsets/coords/values/y"].set_external(coords[1], npts);
    mesh["coordsets/coords/values/z"].set_external(coords[2], npts);

    // Make sure we can identify all of the coordinates.
    PointQuery Q(mesh);
    const int domain0 = 0;
    for(int i = 0; i < npts; i++)
    {
        double pt[3];
        pt[0] = coords[0][i];
        pt[1] = coords[1][i];
        pt[2] = coords[2][i];

        Q.add(domain0, pt);
    }

    // Add 1 bad point at the end.
    double badpt[] = {-1., -1., -1.};
    auto badIdx = Q.add(domain0, badpt);

    // Execute
    Q.execute("coords");

    // The first npts query points should exist.
    const auto &res = Q.results(domain0);
    for(int i = 0; i < npts; i++)
    {
        EXPECT_EQ(res[i], i);
    }
    // The last query point should not exist.
    EXPECT_EQ(res[badIdx], Q.NotFound);

    free_coords_3d(coords);
}

//---------------------------------------------------------------------------
template <typename T>
void single_domain_point_query_2d(int dims[2])
{
    // Initialization.
    T *coords[2];
    make_coords_2d(coords, dims);
    int npts = dims[0] * dims[1];

    // NOTE: We don't need the topology.
    conduit::Node mesh;
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(coords[0], npts);
    mesh["coordsets/coords/values/y"].set_external(coords[1], npts);

    // Make sure we can identify all of the coordinates.
    PointQuery Q(mesh);
    const int domain0 = 0;
    for(int i = 0; i < npts; i++)
    {
        double pt[3];
        pt[0] = coords[0][i];
        pt[1] = coords[1][i];
        pt[2] = 0.;

        Q.add(domain0, pt);
    }

    // Add 1 bad point at the end.
    double badpt[] = {-1., -1., -1.};
    auto badIdx = Q.add(domain0, badpt);

    // Execute
    Q.execute("coords");

    // The first npts query points should exist.
    const auto &res = Q.results(domain0);
    for(int i = 0; i < npts; i++)
    {
        EXPECT_EQ(res[i], i);
    }
    // The last query point should not exist.
    EXPECT_EQ(res[badIdx], Q.NotFound);

    free_coords_2d(coords);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, kdtree_3d)
{
    int dims[3]={30,30,30};
    kdtree_3d<double>(dims);
    kdtree_3d<float>(dims);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, kdtree_2d)
{
    int dims[2]={30,30};
    kdtree_2d<double>(dims);
    kdtree_2d<float>(dims);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, point_query_3d)
{
    // Pick dimensions that are larger than the search threshold so it will
    // attempt accelerated search.
    int dims[3];
    dims[0] = 3 + static_cast<int>(pow(PointQuery::SEARCH_THRESHOLD, 1./3.));
    dims[1] = dims[2] = dims[0];

    single_domain_point_query_3d<double>(dims);
    single_domain_point_query_3d<float>(dims);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, point_query_2d)
{
    // Pick dimensions that are larger than the search threshold so it will
    // attempt accelerated search.
    int dims[2];
    dims[0] = 3 + static_cast<int>(pow(PointQuery::SEARCH_THRESHOLD, 1./2.));
    dims[1] = dims[0];

    single_domain_point_query_2d<double>(dims);
    single_domain_point_query_2d<float>(dims);
}
