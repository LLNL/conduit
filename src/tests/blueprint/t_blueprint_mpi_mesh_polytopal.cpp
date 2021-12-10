// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_polytopal.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"

#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Transform Tests ///

// These tests do a polygonal transformation of a structured AMR mesh into
// a polygonal mesh.  The mesh has two domains and looks like this: 
//
//  _____________________________________________
//  |    |    |    |    |__|__|__|__|__|__|__|__|
//  |____|____|____|____|__|__|__|__|__|__|__|__|
//  |    |    |    |    |__|__|__|__|__|__|__|__|
//  |____|____|____|____|__|__|__|__|__|__|__|__|
//  |    |    |    |    |__|__|__|__|__|__|__|__|
//  |____|____|____|____|__|__|__|__|__|__|__|__|
//  |    |    |    |    |__|__|__|__|__|__|__|__|
//  |____|____|____|____|__|__|__|__|__|__|__|__|
//
//        domain 0              domain 1

void test_polytopal_create_coarse_domain(Node& domain)
{
    domain["state/domain_id"] = 0;
    domain["state/level_id"] = 0;

    std::vector<double> xcoords;
    for (int i = 0; i < 5; ++i)
    {
        xcoords.push_back(0.0);
        xcoords.push_back(0.25);
        xcoords.push_back(0.5);
        xcoords.push_back(0.75);
        xcoords.push_back(1.0);
    }
    std::vector<double> ycoords;
    double yval = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        yval += 0.25;
    }

    Node& coords = domain["coordsets/coords"];
    coords["type"] = "explicit";
    coords["values/x"].set(xcoords);
    coords["values/y"].set(ycoords);


    Node& topo = domain["topologies/topo"];
    topo["coordset"] = "coords";
    topo["type"] = "structured";

    topo["elements/origin/i0"] = 0;
    topo["elements/origin/j0"] = 0;
    topo["elements/dims/i"] = 4;
    topo["elements/dims/j"] = 4;

    Node& field = domain["fields/field"];
    field["association"] = "element";
    field["type"] = "scalar";
    field["topology"] = "topo";

    std::vector<double> vals(16, 0.5);
    field["values"].set(vals);

    Node& adjset = domain["adjsets/adjset"];
    adjset["association"] =  "vertex";
    adjset["topology"] =  "topo";

    Node& group = adjset["groups/group_000001"];
    std::vector<int> nbrs(2);
    nbrs[0] = 0;
    nbrs[1] = 1;
    group["neighbors"].set(nbrs);
    group["rank"] = 1;

    group["windows/window_000000/level_id"] = 0;
    group["windows/window_000000/origin/i"] = 4;
    group["windows/window_000000/origin/j"] = 0;
    group["windows/window_000000/dims/i"] = 1;
    group["windows/window_000000/dims/j"] = 5;
    group["windows/window_000000/ratio/i"] = 2;
    group["windows/window_000000/ratio/j"] = 2;
    group["windows/window_000001/level_id"] = 1;
    group["windows/window_000001/origin/i"] = 8;
    group["windows/window_000001/origin/j"] = 0;
    group["windows/window_000001/dims/i"] = 1;
    group["windows/window_000001/dims/j"] = 9;
    group["windows/window_000001/ratio/i"] = 2;
    group["windows/window_000001/ratio/j"] = 2;
}

void test_polytopal_create_fine_domain(Node& domain)
{
    domain["state/domain_id"] = 1;
    domain["state/level_id"] = 1;

    std::vector<double> xcoords;
    for (int i = 0; i < 9; ++i)
    {
        xcoords.push_back(1.0);
        xcoords.push_back(1.125);
        xcoords.push_back(1.25);
        xcoords.push_back(1.375);
        xcoords.push_back(1.5);
        xcoords.push_back(1.625);
        xcoords.push_back(1.75);
        xcoords.push_back(1.875);
        xcoords.push_back(1.0);
    }
    std::vector<double> ycoords;
    double yval = 0.0;
    for (int i = 0; i < 9; ++i)
    {
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        ycoords.push_back(yval);
        yval += 0.125;
    }

    Node& coords = domain["coordsets/coords"];
    coords["type"] = "explicit";
    coords["values/x"].set(xcoords);
    coords["values/y"].set(ycoords);

    Node& topo = domain["topologies/topo"];
    topo["coordset"] = "coords";
    topo["type"] = "structured";

    topo["elements/origin/i0"] = 8;
    topo["elements/origin/j0"] = 0;
    topo["elements/dims/i"] = 8;
    topo["elements/dims/j"] = 8;

    Node& field = domain["fields/field"];
    field["association"] = "element";
    field["type"] = "scalar";
    field["topology"] = "topo";

    std::vector<double> vals(64, 1.5);
    field["values"].set(vals);

    Node& adjset = domain["adjsets/adjset"];
    adjset["association"] =  "vertex";
    adjset["topology"] =  "topo";

    Node& group = adjset["groups/group_000000"];
    std::vector<int> nbrs(2);
    nbrs[0] = 1;
    nbrs[1] = 0;
    group["neighbors"].set(nbrs);
    group["rank"] = 0;

    group["windows/window_000001/level_id"] = 1;
    group["windows/window_000001/origin/i"] = 8;
    group["windows/window_000001/origin/j"] = 0;
    group["windows/window_000001/dims/i"] = 1;
    group["windows/window_000001/dims/j"] = 9;
    group["windows/window_000001/ratio/i"] = 2;
    group["windows/window_000001/ratio/j"] = 2;
    group["windows/window_000000/level_id"] = 0;
    group["windows/window_000000/origin/i"] = 4;
    group["windows/window_000000/origin/j"] = 0;
    group["windows/window_000000/dims/i"] = 1;
    group["windows/window_000000/dims/j"] = 5;
    group["windows/window_000000/ratio/i"] = 2;
    group["windows/window_000000/ratio/j"] = 2;
}

void test_verify_topologies(Node& struct_topo, Node& poly_topo)
{
    Node info;
    EXPECT_TRUE(conduit::blueprint::mesh::topology::unstructured::verify(
        poly_topo,info));
    EXPECT_EQ(poly_topo["elements/shape"].as_string(), "polygonal");

    int num_elems_structured =
        struct_topo["elements/dims/i"].as_int() *
        struct_topo["elements/dims/j"].as_int();
    int num_sizes = poly_topo["elements/sizes"].dtype().number_of_elements();       
    int num_offsets = poly_topo["elements/offsets"].dtype().number_of_elements();
    EXPECT_EQ(num_elems_structured, num_sizes);
    EXPECT_EQ(num_elems_structured, num_offsets);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_polytopal, amr_2d_transform_serial)
{
    int par_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    Node mesh, info;

    if (par_rank > 0)
    {
        Node& domain_0 = mesh["domain_000000"];
        Node& domain_1 = mesh["domain_000001"];
        test_polytopal_create_coarse_domain(domain_0);
        test_polytopal_create_fine_domain(domain_1);
    }

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    Node poly;

    conduit::blueprint::mpi::mesh::to_polygonal(mesh, poly, "topo", MPI_COMM_WORLD);

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",poly,info, MPI_COMM_WORLD));

    if (par_rank > 0)
    {
        Node& mesh_topo = mesh["domain_000000/topologies/topo"];
        Node& poly_topo = poly["domain_000000/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);

        mesh_topo = mesh["domain_000001/topologies/topo"];
        poly_topo = poly["domain_000001/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_polytopal, amr_2d_transform_parallel)
{
    Node mesh, info;
    std::string filename;

    int par_rank;
    int par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    if (par_size == 1) return;

    if (par_rank == 0)
    {
        filename = "transform_parallel_0.json";

        Node& domain_0 = mesh["domain_000000"];
        test_polytopal_create_coarse_domain(domain_0);
    }
    else if (par_rank == 1)
    {
        filename = "transform_parallel_1.json";
        Node& domain_1 = mesh["domain_000001"];
        test_polytopal_create_fine_domain(domain_1);
    }

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    Node poly;

    conduit::blueprint::mpi::mesh::to_polygonal(mesh, poly, "topo", MPI_COMM_WORLD);

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",poly,info, MPI_COMM_WORLD));

    if (par_rank == 0)
    {
        Node& mesh_topo = mesh["domain_000000/topologies/topo"];
        Node& poly_topo = poly["domain_000000/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);
    }
    else if (par_rank == 1)
    {
        Node& mesh_topo = mesh["domain_000001/topologies/topo"];
        Node& poly_topo = poly["domain_000001/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_polytopal, to_polytopal_amr_2d_transform_parallel)
{
    Node mesh, info;
    std::string filename;

    int par_rank;
    int par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    if (par_size == 1) return;

    if (par_rank == 0)
    {
        filename = "transform_parallel_0.json";

        Node& domain_0 = mesh["domain_000000"];
        test_polytopal_create_coarse_domain(domain_0);
    }
    else if (par_rank == 1)
    {
        filename = "transform_parallel_1.json";
        Node& domain_1 = mesh["domain_000001"];
        test_polytopal_create_fine_domain(domain_1);
    }

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    Node poly;
    conduit::blueprint::mpi::mesh::to_polytopal(mesh, poly, "topo", MPI_COMM_WORLD);

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",poly,info, MPI_COMM_WORLD));

    if (par_rank == 0)
    {
        Node& mesh_topo = mesh["domain_000000/topologies/topo"];
        Node& poly_topo = poly["domain_000000/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);
    }
    else if (par_rank == 1)
    {
        Node& mesh_topo = mesh["domain_000001/topologies/topo"];
        Node& poly_topo = poly["domain_000001/topologies/topo"];
        test_verify_topologies(mesh_topo, poly_topo);
    }
}


int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
