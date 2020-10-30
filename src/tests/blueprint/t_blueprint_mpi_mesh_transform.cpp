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

#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_transform, amr_2d_transform)
{
    Node mesh, info;
    Node& domain_0 = mesh["domain_000000"];
    Node& domain_1 = mesh["domain_000001"];

    domain_0["state/domain_id"] = 0;
    domain_0["state/level_id"] = 0;
    domain_1["state/domain_id"] = 1;
    domain_1["state/level_id"] = 1;

    std::vector<double> xcoords_0;
    for (int i = 0; i < 5; ++i)
    {
        xcoords_0.push_back(0.0);
        xcoords_0.push_back(0.25);
        xcoords_0.push_back(0.5);
        xcoords_0.push_back(0.75);
        xcoords_0.push_back(1.0);
    }
    std::vector<double> ycoords_0;
    double yval = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        ycoords_0.push_back(yval);
        ycoords_0.push_back(yval);
        ycoords_0.push_back(yval);
        ycoords_0.push_back(yval);
        ycoords_0.push_back(yval);
        yval += 0.25;
    }

    std::vector<double> xcoords_1;
    for (int i = 0; i < 9; ++i)
    {
        xcoords_1.push_back(1.0);
        xcoords_1.push_back(1.125);
        xcoords_1.push_back(1.25);
        xcoords_1.push_back(1.375);
        xcoords_1.push_back(1.5);
        xcoords_1.push_back(1.625);
        xcoords_1.push_back(1.75);
        xcoords_1.push_back(1.875);
        xcoords_1.push_back(1.0);
    }
    std::vector<double> ycoords_1;
    yval = 0.0;
    for (int i = 0; i < 9; ++i)
    {
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        ycoords_1.push_back(yval);
        yval += 0.125;
    }

    Node& coords_0 = domain_0["coordsets/coords"];
    coords_0["type"] = "explicit";
    coords_0["values/x"].set(xcoords_0);
    coords_0["values/y"].set(ycoords_0);

    Node& coords_1 = domain_1["coordsets/coords"];
    coords_1["type"] = "explicit";
    coords_1["values/x"].set(xcoords_1);
    coords_1["values/y"].set(ycoords_1);

    Node& topo_0 = domain_0["topologies/topo"];
    Node& topo_1 = domain_1["topologies/topo"];
    topo_0["coordset"] = "coords";
    topo_1["coordset"] = "coords";
    topo_0["type"] = "structured";
    topo_1["type"] = "structured";

    topo_0["elements/origin/i0"] = 0;
    topo_0["elements/origin/j0"] = 0;
    topo_1["elements/origin/i0"] = 8;
    topo_1["elements/origin/j0"] = 0;
    topo_0["elements/dims/i"] = 4;
    topo_0["elements/dims/j"] = 4;
    topo_1["elements/dims/i"] = 8;
    topo_1["elements/dims/j"] = 8;

    Node& field_0 = domain_0["fields/field"];
    Node& field_1 = domain_1["fields/field"];
    field_0["association"] = "element";
    field_1["association"] = "element";
    field_0["type"] = "scalar";
    field_1["type"] = "scalar";
    field_0["topology"] = "topo";
    field_1["topology"] = "topo";

    std::vector<double> vals_0(16, 0.5);
    std::vector<double> vals_1(16, 1.5);
    field_0["values"].set(vals_0);
    field_1["values"].set(vals_1);

    Node& adjset_0 = domain_0["adjsets/adjset"];
    Node& adjset_1 = domain_1["adjsets/adjset"];
    adjset_0["association"] =  "vertex";
    adjset_0["topology"] =  "topo";
    adjset_1["association"] =  "vertex";
    adjset_1["topology"] =  "topo";

    Node& group_0_1 = adjset_0["groups/group_000001"];
    std::vector<int> nbrs_1(2);
    nbrs_1[0] = 0;
    nbrs_1[1] = 1;
    group_0_1["neighbors"].set(nbrs_1);
    group_0_1["rank"] = 0;

    group_0_1["windows/window_000000/level_id"] = 0;
    group_0_1["windows/window_000000/origin/i"] = 4;
    group_0_1["windows/window_000000/origin/j"] = 0;
    group_0_1["windows/window_000000/dims/i"] = 1;
    group_0_1["windows/window_000000/dims/j"] = 5;
    group_0_1["windows/window_000000/ratio/i"] = 2;
    group_0_1["windows/window_000000/ratio/j"] = 2;
    group_0_1["windows/window_000001/level_id"] = 1;
    group_0_1["windows/window_000001/origin/i"] = 8;
    group_0_1["windows/window_000001/origin/j"] = 0;
    group_0_1["windows/window_000001/dims/i"] = 1;
    group_0_1["windows/window_000001/dims/j"] = 9;
    group_0_1["windows/window_000001/ratio/i"] = 2;
    group_0_1["windows/window_000001/ratio/j"] = 2;

    Node& group_1_0 = adjset_1["groups/group_000000"];
    std::vector<int> nbrs_0(2);
    nbrs_0[0] = 1;
    nbrs_0[1] = 0;
    group_1_0["neighbors"].set(nbrs_0);
    group_1_0["rank"] = 0;

    group_1_0["windows/window_000001/level_id"] = 1;
    group_1_0["windows/window_000001/origin/i"] = 8;
    group_1_0["windows/window_000001/origin/j"] = 0;
    group_1_0["windows/window_000001/dims/i"] = 1;
    group_1_0["windows/window_000001/dims/j"] = 9;
    group_1_0["windows/window_000001/ratio/i"] = 2;
    group_1_0["windows/window_000001/ratio/j"] = 2;
    group_1_0["windows/window_000000/level_id"] = 0;
    group_1_0["windows/window_000000/origin/i"] = 4;
    group_1_0["windows/window_000000/origin/j"] = 0;
    group_1_0["windows/window_000000/dims/i"] = 1;
    group_1_0["windows/window_000000/dims/j"] = 5;
    group_1_0["windows/window_000000/ratio/i"] = 2;
    group_1_0["windows/window_000000/ratio/j"] = 2;

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    Node poly;

    conduit::blueprint::mpi::mesh::to_polygonal(mesh, poly, "topo");

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",poly,info, MPI_COMM_WORLD));
 
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

