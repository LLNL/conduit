// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_parmetis.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_blueprint_mpi_mesh_parmetis.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_utils.hpp"
#include "conduit_fmt/conduit_fmt.h"

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;
using namespace conduit::relay::mpi;
using namespace conduit::utils;

using namespace std;
//
//
// /*
//     Creates an example mesh that using the following pattern:
//
//     |------------|------------|
//     |            |            |
//     |  domain 0  |            |
//     |            |            |
//     |------------|  domain 2..|
//     |            |            |
//     |  domain 1  |            |
//     |            |            |
//     |------------|------------|
//
//     The base grid i and j dims are used to size domains 0 and 1,
//     domain 2 uses grid i, and j*2 as dims.
//
//     There are two topologies:
//
//     The `main` topology is a structured mesh of quads.
//     The `boundary` topology is an unstructured mesh of lines,
//     the line elements correspond to the boundary faces of main`.
//
//     The `main` and `boundary` topologies are defined using the same
//     explicit coordset. Because if this we can relate the elements
//     between them.
//
//     The field `ele_id` provides globally unique ids for elements
//     of the `main` topology.
//
//     The field `domain_id` provides the domain number for elements
//     of the `main` topology.
//
//     The field `bndry_val` is defined on the `boundary` topology
//     as 1 for elements on the external mesh boundary, and 0 for
//     elements on an internal mesh boundary.
//
//     The field `bndry_id` provides globally unique ids for elements
//     of the `bndry` topology.
//
//     #######
//     TODO
//     #######
//     - Add field `bndry_to_main` shows the relationship between the
//       topologies.
//     - Add an adj set?
// */
//
//
//
// void gen_example_data(index_t base_grid_ele_i,
//                       index_t base_grid_ele_j,
//                       Node &mesh)
// {
//     mesh["domain0/coordsets/coords/type"] = "explicit";
//     index_t num_coords = (base_grid_ele_i +1) * (base_grid_ele_j+1);
//     mesh["domain0/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
//     mesh["domain0/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
//
//     float64_array x_vals = mesh["domain0/coordsets/coords/values/x"].value();
//     float64_array y_vals = mesh["domain0/coordsets/coords/values/y"].value();
//
//     index_t idx=0;
//     for(index_t j = 0; j < base_grid_ele_j +1; j++)
//     {
//         for(index_t i = 0; i < base_grid_ele_i +1; i++)
//         {
//             x_vals[idx] = 1.0 * i;
//             y_vals[idx] = 1.0 * j;
//             idx++;
//         }
//     }
//
//     // main topo
//     mesh["domain0/topologies/main/coordset"] = "coords";
//     mesh["domain0/topologies/main/type"] = "structured";
//     mesh["domain0/topologies/main/elements/dims/i"] = base_grid_ele_i;
//     mesh["domain0/topologies/main/elements/dims/j"] = base_grid_ele_j;
//
//     // boundary
//     mesh["domain0/topologies/boundary/coordset"] = "coords";
//     mesh["domain0/topologies/boundary/type"] = "unstructured";
//     mesh["domain0/topologies/boundary/elements/shape"] = "line";
//     index_t num_edges = (base_grid_ele_i *2 + base_grid_ele_j*2);
//     mesh["domain0/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));
//
//     int64_array bnd_conn = mesh["domain0/topologies/boundary/elements/connectivity"].value();
//
//     // four sides
//     // bottom
//     idx = 0;
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i;
//         bnd_conn[idx+1] = i+1;
//         idx+=2;
//     }
//
//     // top
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i +    (base_grid_ele_i +1) * base_grid_ele_j;
//         bnd_conn[idx+1] = i+1 +  (base_grid_ele_i +1) * base_grid_ele_j;
//         idx+=2;
//     }
//
//     // left
//     for(index_t j = 0; j < base_grid_ele_j; j++)
//     {
//         bnd_conn[idx]   = j * (base_grid_ele_i+1);
//         bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
//         idx+=2;
//     }
//
//     // right
//     for(index_t j = 0; j < base_grid_ele_j; j++)
//     {
//         bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
//         bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1;
//         idx+=2;
//     }
//
//
//     index_t num_eles = base_grid_ele_i * base_grid_ele_j;
//     mesh["domain0/fields/ele_id/association"] = "element";
//     mesh["domain0/fields/ele_id/topology"] = "main";
//     mesh["domain0/fields/ele_id/values"].set( DataType::float64(num_eles) );
//
//
//     mesh["domain0/fields/domain_id/association"] = "element";
//     mesh["domain0/fields/domain_id/topology"] = "main";
//     mesh["domain0/fields/domain_id/values"].set( DataType::float64(num_eles) );
//
//     float64_array ele_vals = mesh["domain0/fields/ele_id/values"].value();
//     float64_array dom_id_vals = mesh["domain0/fields/domain_id/values"].value();
//
//     index_t  main_id_global = 0;
//
//     for(index_t i = 0; i < num_eles; i++)
//     {
//         ele_vals[i] = main_id_global;
//         dom_id_vals[i] = 0.0;
//         main_id_global+=1;
//     }
//
//
//     mesh["domain0/fields/bndry_val/association"] = "element";
//     mesh["domain0/fields/bndry_val/topology"] = "boundary";
//     mesh["domain0/fields/bndry_val/values"].set( DataType::float64(num_edges) );
//
//     float64_array bndry_vals = mesh["domain0/fields/bndry_val/values"].value();
//
//     // bottom
//     idx = 0;
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] =  1;
//         idx+=1;
//     }
//
//     // top
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] =  0;
//         idx+=1;
//     }
//
//     // left
//     for(int i=0;i<base_grid_ele_j;i++)
//     {
//         bndry_vals[idx] =  1;
//         idx+=1;
//     }
//
//     // right
//     for(int i=0;i<base_grid_ele_j;i++)
//     {
//         bndry_vals[idx] =  0;
//         idx+=1;
//     }
//
//     mesh["domain0/fields/bndry_id/association"] = "element";
//     mesh["domain0/fields/bndry_id/topology"] = "boundary";
//     mesh["domain0/fields/bndry_id/values"].set( DataType::int64(num_edges) );
//
//     int64_array bndry_id_vals = mesh["domain0/fields/bndry_id/values"].value();
//
//     index_t  bndry_id_global = 0;
//
//     // unique ids for the boundary
//     for(int i=0;i<num_edges;i++)
//     {
//         bndry_id_vals[i] =  bndry_id_global;
//         bndry_id_global+=1;
//     }
//
//
//     // domain 1:
//     mesh["domain1/coordsets/coords/type"] = "explicit";
//     num_coords = (base_grid_ele_i +1) * (base_grid_ele_j+1);
//     mesh["domain1/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
//     mesh["domain1/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
//
//     x_vals = mesh["domain1/coordsets/coords/values/x"].value();
//     y_vals = mesh["domain1/coordsets/coords/values/y"].value();
//
//     idx=0;
//     for(index_t j = 0; j < base_grid_ele_j +1; j++)
//     {
//         for(index_t i = 0; i < base_grid_ele_i +1; i++)
//         {
//             x_vals[idx] = 1.0 * i;
//             y_vals[idx] = 1.0 * j + base_grid_ele_j;
//             idx++;
//         }
//     }
//
//     mesh["domain1/topologies/main/coordset"] = "coords";
//     mesh["domain1/topologies/main/type"] = "structured";
//     mesh["domain1/topologies/main/elements/dims/i"] = base_grid_ele_i;
//     mesh["domain1/topologies/main/elements/dims/j"] = base_grid_ele_j;
//
//
//     // boundary
//     mesh["domain1/topologies/boundary/coordset"] = "coords";
//     mesh["domain1/topologies/boundary/type"] = "unstructured";
//     mesh["domain1/topologies/boundary/elements/shape"] = "line";
//     num_edges = (base_grid_ele_i *2 + base_grid_ele_j*2);
//     mesh["domain1/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));
//
//     bnd_conn = mesh["domain1/topologies/boundary/elements/connectivity"].value();
//
//     // four sides
//     // bottom
//     idx = 0;
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i;
//         bnd_conn[idx+1] = i+1;
//         idx+=2;
//     }
//
//     // top
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i +    (base_grid_ele_i +1) * base_grid_ele_j;
//         bnd_conn[idx+1] = i+1 +  (base_grid_ele_i +1) * base_grid_ele_j;
//         idx+=2;
//     }
//
//     // left
//     for(index_t j = 0; j < base_grid_ele_j; j++)
//     {
//         bnd_conn[idx]   = j * (base_grid_ele_i+1);
//         bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
//         idx+=2;
//     }
//
//     // right
//     for(index_t j = 0; j < base_grid_ele_j; j++)
//     {
//         bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
//         bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1;
//         idx+=2;
//     }
//
//     num_eles = base_grid_ele_i * base_grid_ele_j;
//     mesh["domain1/fields/ele_id/association"] = "element";
//     mesh["domain1/fields/ele_id/topology"] = "main";
//     mesh["domain1/fields/ele_id/values"].set( DataType::float64(num_eles) );
//
//
//     mesh["domain1/fields/domain_id/association"] = "element";
//     mesh["domain1/fields/domain_id/topology"] = "main";
//     mesh["domain1/fields/domain_id/values"].set( DataType::float64(num_eles) );
//
//     ele_vals = mesh["domain1/fields/ele_id/values"].value();
//     dom_id_vals = mesh["domain1/fields/domain_id/values"].value();
//
//     for(index_t i = 0; i < num_eles; i++)
//     {
//         ele_vals[i] = main_id_global;
//         dom_id_vals[i] = 1.0;
//         main_id_global+=1;
//     }
//
//     mesh["domain1/fields/bndry_val/association"] = "element";
//     mesh["domain1/fields/bndry_val/topology"] = "boundary";
//     mesh["domain1/fields/bndry_val/values"].set( DataType::float64(num_edges) );
//
//     bndry_vals = mesh["domain1/fields/bndry_val/values"].value();
//
//     // bottom
//     idx = 0;
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] =  0;
//         idx+=1;
//     }
//
//     // top
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] =  1;
//         idx+=1;
//     }
//
//     // left
//     for(int i=0;i<base_grid_ele_j;i++)
//     {
//         bndry_vals[idx] =  1;
//         idx+=1;
//     }
//
//     // right
//     for(int i=0;i<base_grid_ele_j;i++)
//     {
//         bndry_vals[idx] =  0;
//         idx+=1;
//     }
//
//     mesh["domain1/fields/bndry_id/association"] = "element";
//     mesh["domain1/fields/bndry_id/topology"] = "boundary";
//     mesh["domain1/fields/bndry_id/values"].set( DataType::int64(num_edges) );
//
//     bndry_id_vals = mesh["domain1/fields/bndry_id/values"].value();
//
//     // unique ids for the boundary
//     for(int i=0;i<num_edges;i++)
//     {
//         bndry_id_vals[i] =  bndry_id_global;
//         bndry_id_global+=1;
//     }
//
//     // domain 1
//     mesh["domain2/coordsets/coords/type"] = "explicit";
//     num_coords = (base_grid_ele_i +1) * ((base_grid_ele_j *2 )+1);
//     mesh["domain2/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
//     mesh["domain2/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
//
//     x_vals = mesh["domain2/coordsets/coords/values/x"].value();
//     y_vals = mesh["domain2/coordsets/coords/values/y"].value();
//
//     idx=0;
//     for(index_t j = 0; j < (base_grid_ele_j * 2) + 1; j++)
//     {
//         for(index_t i = 0; i < base_grid_ele_i +1; i++)
//         {
//             x_vals[idx] = 1.0 * i + base_grid_ele_i;
//             y_vals[idx] = 1.0 * j;
//             idx++;
//         }
//     }
//
//     mesh["domain2/topologies/main/coordset"] = "coords";
//     mesh["domain2/topologies/main/type"] = "structured";
//     mesh["domain2/topologies/main/elements/dims/i"] = base_grid_ele_i;
//     mesh["domain2/topologies/main/elements/dims/j"] = base_grid_ele_j*2;
//
//     // boundary
//     mesh["domain2/topologies/boundary/coordset"] = "coords";
//     mesh["domain2/topologies/boundary/type"] = "unstructured";
//     mesh["domain2/topologies/boundary/elements/shape"] = "line";
//     num_edges = (base_grid_ele_i *2 + (base_grid_ele_j*2)*2);
//     mesh["domain2/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));
//
//     bnd_conn = mesh["domain2/topologies/boundary/elements/connectivity"].value();
//
//     // four sides
//     // bottom
//     idx = 0;
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i;
//         bnd_conn[idx+1] = i+1;
//         idx+=2;
//     }
//
//     // top
//     for(index_t i = 0; i < base_grid_ele_i; i++)
//     {
//         bnd_conn[idx]   = i +    (base_grid_ele_i +1) * base_grid_ele_j*2;
//         bnd_conn[idx+1] = i+1 +  (base_grid_ele_i +1) * base_grid_ele_j*2;
//         idx+=2;
//     }
//
//     // left
//     for(index_t j = 0; j < base_grid_ele_j * 2; j++)
//     {
//         bnd_conn[idx]   = j * (base_grid_ele_i+1);
//         bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
//         idx+=2;
//     }
//
//     // right
//     for(index_t j = 0; j < base_grid_ele_j * 2; j++)
//     {
//         bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
//         bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1;
//         idx+=2;
//     }
//
//     num_eles = base_grid_ele_i * base_grid_ele_j*2;
//     mesh["domain2/fields/ele_id/association"] = "element";
//     mesh["domain2/fields/ele_id/topology"] = "main";
//     mesh["domain2/fields/ele_id/values"].set( DataType::float64(num_eles) );
//
//
//     mesh["domain2/fields/domain_id/association"] = "element";
//     mesh["domain2/fields/domain_id/topology"] = "main";
//     mesh["domain2/fields/domain_id/values"].set( DataType::float64(num_eles) );
//
//     ele_vals = mesh["domain2/fields/ele_id/values"].value();
//     dom_id_vals = mesh["domain2/fields/domain_id/values"].value();
//
//     for(index_t i = 0; i < num_eles; i++)
//     {
//         ele_vals[i] = main_id_global;
//         dom_id_vals[i] = 2.0;
//         main_id_global+=1;
//     }
//
//     mesh["domain2/fields/bndry_val/association"] = "element";
//     mesh["domain2/fields/bndry_val/topology"] = "boundary";
//     mesh["domain2/fields/bndry_val/values"].set( DataType::float64(num_edges) );
//
//     bndry_vals = mesh["domain2/fields/bndry_val/values"].value();
//
//     // bottom
//     idx = 0;
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] = 1;
//         idx+=1;
//     }
//
//     // top
//     for(int i=0;i<base_grid_ele_i;i++)
//     {
//         bndry_vals[idx] = 1;
//         idx+=1;
//     }
//
//     // left
//     for(int i=0;i<base_grid_ele_j*2;i++)
//     {
//         bndry_vals[idx] = 0;
//         idx+=1;
//     }
//
//     // right
//     for(int i=0;i<base_grid_ele_j*2;i++)
//     {
//         bndry_vals[idx] = 1;
//         idx+=1;
//     }
//
//     mesh["domain2/fields/bndry_id/association"] = "element";
//     mesh["domain2/fields/bndry_id/topology"] = "boundary";
//     mesh["domain2/fields/bndry_id/values"].set( DataType::int64(num_edges) );
//
//     bndry_id_vals = mesh["domain2/fields/bndry_id/values"].value();
//
//     // unique ids for the boundary
//     for(int i=0;i<num_edges;i++)
//     {
//         bndry_id_vals[i] =  bndry_id_global;
//         bndry_id_global+=1;
//     }
// }




//-----------------------------------------------------------------------------
TEST(blueprint_mpi_load_bal, basic)
{


    //

    int par_size, par_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    // build test using related_boundary example
    // one way to create unbalanced number of eles across ranks:
    //  domain 0 --> rank 0
    //  domain 1 and 2 --> rank 0

    Node mesh;
    index_t base_grid_ele_i = 3;
    index_t base_grid_ele_j = 3;

    if(par_rank == 0)
    {
        conduit::blueprint::mesh::examples::related_boundary(base_grid_ele_i,
                                                             base_grid_ele_j,
                                                             mesh);
    }// end par_rank - 0

    std::string output_base = "tout_bp_mpi_load_bal_basic";

    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
                                                  output_base,
                                                  "hdf5",
                                                   MPI_COMM_WORLD);
    EXPECT_TRUE(true);
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
