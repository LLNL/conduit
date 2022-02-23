// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_related_boundary.cpp
///
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_examples.hpp"
#include "conduit_blueprint_mesh.hpp"


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------
namespace examples
{


void related_boundary(index_t base_grid_ele_i,
                      index_t base_grid_ele_j,
                      Node &mesh)
{
    // --------------------
    // --------------------
    // domain 0
    // --------------------
    // --------------------

    mesh["domain0/coordsets/coords/type"] = "explicit";
    index_t num_coords = (base_grid_ele_i +1) * (base_grid_ele_j+1);
    mesh["domain0/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
    mesh["domain0/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
    
    float64_array x_vals = mesh["domain0/coordsets/coords/values/x"].value();
    float64_array y_vals = mesh["domain0/coordsets/coords/values/y"].value();

    index_t idx=0;
    for(index_t j = 0; j < base_grid_ele_j +1; j++)
    {
        for(index_t i = 0; i < base_grid_ele_i +1; i++)
        {
            x_vals[idx] = 1.0 * i;
            y_vals[idx] = 1.0 * j;
            idx++;
        }
    }
    
    // main topo
    mesh["domain0/topologies/main/coordset"] = "coords";
    mesh["domain0/topologies/main/type"] = "structured";
    mesh["domain0/topologies/main/elements/dims/i"] = base_grid_ele_i;
    mesh["domain0/topologies/main/elements/dims/j"] = base_grid_ele_j;

    // boundary 
    mesh["domain0/topologies/boundary/coordset"] = "coords";
    mesh["domain0/topologies/boundary/type"] = "unstructured";
    mesh["domain0/topologies/boundary/elements/shape"] = "line";
    index_t num_edges = (base_grid_ele_i *2 + base_grid_ele_j*2);
    mesh["domain0/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));

    int64_array bnd_conn = mesh["domain0/topologies/boundary/elements/connectivity"].value();

    // four sides
    // bottom
    idx = 0;
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i;
        bnd_conn[idx+1] = i+1;
        idx+=2;
    }
    
    // top
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i +    (base_grid_ele_i +1) * base_grid_ele_j;
        bnd_conn[idx+1] = i+1 +  (base_grid_ele_i +1) * base_grid_ele_j;
        idx+=2;
    }
    
    // left
    for(index_t j = 0; j < base_grid_ele_j; j++)
    {   
        bnd_conn[idx]   = j * (base_grid_ele_i+1);
        bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
        idx+=2;
    }
    
    // right
    for(index_t j = 0; j < base_grid_ele_j; j++)
    {   
        bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
        bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1; 
        idx+=2;
    }

    index_t num_eles = base_grid_ele_i * base_grid_ele_j;
    mesh["domain0/fields/ele_local_id/association"] = "element";
    mesh["domain0/fields/ele_local_id/topology"] = "main";
    mesh["domain0/fields/ele_local_id/values"].set( DataType::float64(num_eles) );

    mesh["domain0/fields/ele_global_id/association"] = "element";
    mesh["domain0/fields/ele_global_id/topology"] = "main";
    mesh["domain0/fields/ele_global_id/values"].set( DataType::float64(num_eles) );


    mesh["domain0/fields/domain_id/association"] = "element";
    mesh["domain0/fields/domain_id/topology"] = "main";
    mesh["domain0/fields/domain_id/values"].set( DataType::float64(num_eles) );

    float64_array ele_local_vals  = mesh["domain0/fields/ele_local_id/values"].value();
    float64_array ele_global_vals = mesh["domain0/fields/ele_global_id/values"].value();
    float64_array dom_id_vals = mesh["domain0/fields/domain_id/values"].value();

    index_t main_id_global = 0;

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_local_vals[i] = i;
        ele_global_vals[i] = main_id_global;
        dom_id_vals[i] = 0.0;
        main_id_global+=1;
    }

    mesh["domain0/fields/bndry_val/association"] = "element";
    mesh["domain0/fields/bndry_val/topology"] = "boundary";
    mesh["domain0/fields/bndry_val/values"].set( DataType::float64(num_edges) );

    float64_array bndry_vals = mesh["domain0/fields/bndry_val/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] =  1;
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] =  0;
        idx+=1;
    }

    // left
    for(int i=0;i<base_grid_ele_j;i++)
    {
        bndry_vals[idx] =  1;
        idx+=1;
    }

    // right
    for(int i=0;i<base_grid_ele_j;i++)
    {
        bndry_vals[idx] =  0;
        idx+=1;
    }

    index_t bndry_id_global = 0;
    // local boundary id
    mesh["domain0/fields/bndry_local_id/association"] = "element";
    mesh["domain0/fields/bndry_local_id/topology"] = "boundary";
    mesh["domain0/fields/bndry_local_id/values"].set( DataType::int64(num_edges) );

    int64_array bndry_id_local_vals = mesh["domain0/fields/bndry_local_id/values"].value();

    mesh["domain0/fields/bndry_global_id/association"] = "element";
    mesh["domain0/fields/bndry_global_id/topology"] = "boundary";
    mesh["domain0/fields/bndry_global_id/values"].set( DataType::int64(num_edges) );

    int64_array bndry_id_global_vals = mesh["domain0/fields/bndry_global_id/values"].value();

    // global and local unique ids for the boundary
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_local_vals[i]  = i;
        bndry_id_global_vals[i] = bndry_id_global;
        bndry_id_global++;
    }

    // boundary to main relationship
    // each boundary element is related to one of the main elements
    // provide both local and global map as fields
    mesh["domain0/fields/bndry_to_main_local/association"] = "element";
    mesh["domain0/fields/bndry_to_main_local/topology"] = "boundary";
    mesh["domain0/fields/bndry_to_main_local/values"].set( DataType::int64(num_edges) );

    int64_array bndry_to_main_local_vals = mesh["domain0/fields/bndry_to_main_local/values"].value();


    mesh["domain0/fields/bndry_to_main_global/association"] = "element";
    mesh["domain0/fields/bndry_to_main_global/topology"] = "boundary";
    mesh["domain0/fields/bndry_to_main_global/values"].set( DataType::int64(num_edges) );

    int64_array bndry_to_main_global_vals = mesh["domain0/fields/bndry_to_main_global/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_to_main_local_vals[idx]  = i;
        bndry_to_main_global_vals[idx] = bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_to_main_local_vals[idx]  = i + (base_grid_ele_i) * (base_grid_ele_j-1);
        bndry_to_main_global_vals[idx] = bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // left
    for(int j=0;j<base_grid_ele_j;j++)
    {
        bndry_to_main_local_vals[idx]  = j * (base_grid_ele_i);
        bndry_to_main_global_vals[idx] = bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // right
    for(int j=0;j<base_grid_ele_j;j++)
    {
        bndry_to_main_local_vals[idx]  = (j+1) * (base_grid_ele_i) -1;
        bndry_to_main_global_vals[idx] = bndry_to_main_local_vals[idx];
        idx+=1;
    }


    // --------------------
    // --------------------
    // domain 1
    // --------------------
    // --------------------

    index_t domain1_ele_id_offset = main_id_global;

    mesh["domain1/coordsets/coords/type"] = "explicit";
    num_coords = (base_grid_ele_i +1) * (base_grid_ele_j+1);
    mesh["domain1/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
    mesh["domain1/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
    
    x_vals = mesh["domain1/coordsets/coords/values/x"].value();
    y_vals = mesh["domain1/coordsets/coords/values/y"].value();

    idx=0;
    for(index_t j = 0; j < base_grid_ele_j +1; j++)
    {
        for(index_t i = 0; i < base_grid_ele_i +1; i++)
        {
            x_vals[idx] = 1.0 * i;
            y_vals[idx] = 1.0 * j + base_grid_ele_j;
            idx++;
        }
    }

    mesh["domain1/topologies/main/coordset"] = "coords";
    mesh["domain1/topologies/main/type"] = "structured";
    mesh["domain1/topologies/main/elements/dims/i"] = base_grid_ele_i;
    mesh["domain1/topologies/main/elements/dims/j"] = base_grid_ele_j;


    // boundary 
    mesh["domain1/topologies/boundary/coordset"] = "coords";
    mesh["domain1/topologies/boundary/type"] = "unstructured";
    mesh["domain1/topologies/boundary/elements/shape"] = "line";
    num_edges = (base_grid_ele_i *2 + base_grid_ele_j*2);
    mesh["domain1/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));

    bnd_conn = mesh["domain1/topologies/boundary/elements/connectivity"].value();

    // four sides
    // bottom
    idx = 0;
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i;
        bnd_conn[idx+1] = i+1;
        idx+=2;
    }

    // top
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i +   (base_grid_ele_i +1) * base_grid_ele_j;
        bnd_conn[idx+1] = i + 1 + (base_grid_ele_i +1) * base_grid_ele_j;
        idx+=2;
    }

    // left
    for(index_t j = 0; j < base_grid_ele_j; j++)
    {   
        bnd_conn[idx]   = j * (base_grid_ele_i+1);
        bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
        idx+=2;
    }

    // right
    for(index_t j = 0; j < base_grid_ele_j; j++)
    {   
        bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
        bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1; 
        idx+=2;
    }

    num_eles = base_grid_ele_i * base_grid_ele_j;
    mesh["domain1/fields/ele_local_id/association"] = "element";
    mesh["domain1/fields/ele_local_id/topology"] = "main";
    mesh["domain1/fields/ele_local_id/values"].set( DataType::float64(num_eles) );

    mesh["domain1/fields/ele_global_id/association"] = "element";
    mesh["domain1/fields/ele_global_id/topology"] = "main";
    mesh["domain1/fields/ele_global_id/values"].set( DataType::float64(num_eles) );

    mesh["domain1/fields/domain_id/association"] = "element";
    mesh["domain1/fields/domain_id/topology"] = "main";
    mesh["domain1/fields/domain_id/values"].set( DataType::float64(num_eles) );

    ele_local_vals  = mesh["domain1/fields/ele_local_id/values"].value();
    ele_global_vals = mesh["domain1/fields/ele_global_id/values"].value();
    dom_id_vals = mesh["domain1/fields/domain_id/values"].value();

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_local_vals[i] = i;
        ele_global_vals[i] = main_id_global;
        dom_id_vals[i] = 1.0;
        main_id_global++;
    }

    mesh["domain1/fields/bndry_val/association"] = "element";
    mesh["domain1/fields/bndry_val/topology"] = "boundary";
    mesh["domain1/fields/bndry_val/values"].set( DataType::float64(num_edges) );

    bndry_vals = mesh["domain1/fields/bndry_val/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] = 0;
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] = 1;
        idx+=1;
    }

    // left
    for(int i=0;i<base_grid_ele_j;i++)
    {
        bndry_vals[idx] = 1;
        idx+=1;
    }

    // right
    for(int i=0;i<base_grid_ele_j;i++)
    {
        bndry_vals[idx] = 0;
        idx+=1;
    }

    // local boundary id
    mesh["domain1/fields/bndry_local_id/association"] = "element";
    mesh["domain1/fields/bndry_local_id/topology"] = "boundary";
    mesh["domain1/fields/bndry_local_id/values"].set( DataType::int64(num_edges) );

    bndry_id_local_vals = mesh["domain1/fields/bndry_local_id/values"].value();

    mesh["domain1/fields/bndry_global_id/association"] = "element";
    mesh["domain1/fields/bndry_global_id/topology"] = "boundary";
    mesh["domain1/fields/bndry_global_id/values"].set( DataType::int64(num_edges) );

    bndry_id_global_vals = mesh["domain1/fields/bndry_global_id/values"].value();

    // global and local unique ids for the boundary
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_local_vals[i]  = i;
        bndry_id_global_vals[i] = bndry_id_global;
        bndry_id_global++;
    }

    // boundary to main relationship
    // each boundary element is related to one of the main elements
    // provide both local and global map as fields
    mesh["domain1/fields/bndry_to_main_local/association"] = "element";
    mesh["domain1/fields/bndry_to_main_local/topology"] = "boundary";
    mesh["domain1/fields/bndry_to_main_local/values"].set( DataType::int64(num_edges) );

    bndry_to_main_local_vals = mesh["domain1/fields/bndry_to_main_local/values"].value();


    mesh["domain1/fields/bndry_to_main_global/association"] = "element";
    mesh["domain1/fields/bndry_to_main_global/topology"] = "boundary";
    mesh["domain1/fields/bndry_to_main_global/values"].set( DataType::int64(num_edges) );

    bndry_to_main_global_vals = mesh["domain1/fields/bndry_to_main_global/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_to_main_local_vals[idx]  = i;
        bndry_to_main_global_vals[idx] = domain1_ele_id_offset + bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_to_main_local_vals[idx]  = i + (base_grid_ele_i) * (base_grid_ele_j-1);
        bndry_to_main_global_vals[idx] = domain1_ele_id_offset + bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // left
    for(int j=0;j<base_grid_ele_j;j++)
    {
        bndry_to_main_local_vals[idx]  = j * (base_grid_ele_i);
        bndry_to_main_global_vals[idx] = domain1_ele_id_offset + bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // right
    for(int j=0;j<base_grid_ele_j;j++)
    {
        bndry_to_main_local_vals[idx]  = (j+1) * (base_grid_ele_i) -1;
        bndry_to_main_global_vals[idx] = domain1_ele_id_offset + bndry_to_main_local_vals[idx];
        idx+=1;
    }

    // --------------------
    // --------------------
    // domain 2
    // --------------------
    // --------------------

    index_t domain2_ele_id_offset = main_id_global;
    mesh["domain2/coordsets/coords/type"] = "explicit";
    num_coords = (base_grid_ele_i +1) * ((base_grid_ele_j *2 )+1);
    mesh["domain2/coordsets/coords/values/x"].set( DataType::float64(num_coords) );
    mesh["domain2/coordsets/coords/values/y"].set( DataType::float64(num_coords) );
    
    x_vals = mesh["domain2/coordsets/coords/values/x"].value();
    y_vals = mesh["domain2/coordsets/coords/values/y"].value();

    idx=0;
    for(index_t j = 0; j < (base_grid_ele_j * 2) + 1; j++)
    {
        for(index_t i = 0; i < base_grid_ele_i +1; i++)
        {
            x_vals[idx] = 1.0 * i + base_grid_ele_i;
            y_vals[idx] = 1.0 * j;
            idx++;
        }
    }

    mesh["domain2/topologies/main/coordset"] = "coords";
    mesh["domain2/topologies/main/type"] = "structured";
    mesh["domain2/topologies/main/elements/dims/i"] = base_grid_ele_i;
    mesh["domain2/topologies/main/elements/dims/j"] = base_grid_ele_j*2;

    // boundary 
    mesh["domain2/topologies/boundary/coordset"] = "coords";
    mesh["domain2/topologies/boundary/type"] = "unstructured";
    mesh["domain2/topologies/boundary/elements/shape"] = "line";
    num_edges = (base_grid_ele_i *2 + (base_grid_ele_j*2)*2);
    mesh["domain2/topologies/boundary/elements/connectivity"].set(DataType::int64(num_edges*2));

    bnd_conn = mesh["domain2/topologies/boundary/elements/connectivity"].value();

    // four sides
    // bottom
    idx = 0;
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i;
        bnd_conn[idx+1] = i+1;
        idx+=2;
    }
    
    // top
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bnd_conn[idx]   = i +    (base_grid_ele_i +1) * base_grid_ele_j*2;
        bnd_conn[idx+1] = i+1 +  (base_grid_ele_i +1) * base_grid_ele_j*2;
        idx+=2;
    }
    
    // left
    for(index_t j = 0; j < base_grid_ele_j * 2; j++)
    {   
        bnd_conn[idx]   = j * (base_grid_ele_i+1);
        bnd_conn[idx+1] = (j+1) * (base_grid_ele_i+1);
        idx+=2;
    }
    
    // right
    for(index_t j = 0; j < base_grid_ele_j * 2; j++)
    {   
        bnd_conn[idx]   = (j+1) * (base_grid_ele_i+1) -1;
        bnd_conn[idx+1] = (j+2) * (base_grid_ele_i+1) -1; 
        idx+=2;
    }

    num_eles = base_grid_ele_i * base_grid_ele_j*2;
    mesh["domain2/fields/ele_local_id/association"] = "element";
    mesh["domain2/fields/ele_local_id/topology"] = "main";
    mesh["domain2/fields/ele_local_id/values"].set( DataType::float64(num_eles) );

    mesh["domain2/fields/ele_global_id/association"] = "element";
    mesh["domain2/fields/ele_global_id/topology"] = "main";
    mesh["domain2/fields/ele_global_id/values"].set( DataType::float64(num_eles) );


    mesh["domain2/fields/domain_id/association"] = "element";
    mesh["domain2/fields/domain_id/topology"] = "main";
    mesh["domain2/fields/domain_id/values"].set( DataType::float64(num_eles) );

    ele_local_vals  = mesh["domain2/fields/ele_local_id/values"].value();
    ele_global_vals = mesh["domain2/fields/ele_global_id/values"].value();
    dom_id_vals = mesh["domain2/fields/domain_id/values"].value();

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_local_vals[i] = i;
        ele_global_vals[i] = main_id_global;
        dom_id_vals[i] = 2.0;
        main_id_global+=1;
    }
    
    mesh["domain2/fields/bndry_val/association"] = "element";
    mesh["domain2/fields/bndry_val/topology"] = "boundary";
    mesh["domain2/fields/bndry_val/values"].set( DataType::float64(num_edges) );

    bndry_vals = mesh["domain2/fields/bndry_val/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] = 1;
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] = 1;
        idx+=1;
    }

    // left
    for(int i=0;i<base_grid_ele_j*2;i++)
    {
        bndry_vals[idx] = 0;
        idx+=1;
    }

    // right
    for(int i=0;i<base_grid_ele_j*2;i++)
    {
        bndry_vals[idx] = 1;
        idx+=1;
    }

    // local boundary id
    mesh["domain2/fields/bndry_local_id/association"] = "element";
    mesh["domain2/fields/bndry_local_id/topology"] = "boundary";
    mesh["domain2/fields/bndry_local_id/values"].set( DataType::int64(num_edges) );

    bndry_id_local_vals = mesh["domain2/fields/bndry_local_id/values"].value();

    mesh["domain2/fields/bndry_global_id/association"] = "element";
    mesh["domain2/fields/bndry_global_id/topology"] = "boundary";
    mesh["domain2/fields/bndry_global_id/values"].set( DataType::int64(num_edges) );

    bndry_id_global_vals = mesh["domain2/fields/bndry_global_id/values"].value();

    // global and local unique ids for the boundary
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_local_vals[i]  = i;
        bndry_id_global_vals[i] = bndry_id_global;
        bndry_id_global++;
    }

    // boundary to main relationship
    // each boundary element is related to one of the main elements
    // provide both local and global map as fields
    mesh["domain2/fields/bndry_to_main_local/association"] = "element";
    mesh["domain2/fields/bndry_to_main_local/topology"] = "boundary";
    mesh["domain2/fields/bndry_to_main_local/values"].set( DataType::int64(num_edges) );

    bndry_to_main_local_vals = mesh["domain2/fields/bndry_to_main_local/values"].value();


    mesh["domain2/fields/bndry_to_main_global/association"] = "element";
    mesh["domain2/fields/bndry_to_main_global/topology"] = "boundary";
    mesh["domain2/fields/bndry_to_main_global/values"].set( DataType::int64(num_edges) );

    bndry_to_main_global_vals = mesh["domain2/fields/bndry_to_main_global/values"].value();
    
    idx = 0;
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bndry_to_main_local_vals[idx]  = i;
        bndry_to_main_global_vals[idx] = domain2_ele_id_offset + bndry_to_main_local_vals[idx];
        idx++;
    }
    
    // top
    for(index_t i = 0; i < base_grid_ele_i; i++)
    {   
        bndry_to_main_local_vals[idx]  = i + (base_grid_ele_i-1) * (base_grid_ele_j)*2;
        bndry_to_main_global_vals[idx] = domain2_ele_id_offset + bndry_to_main_local_vals[idx];
        idx++;
    }
    
    // left
    for(index_t j = 0; j < base_grid_ele_j * 2; j++)
    {   
        bndry_to_main_local_vals[idx]  = j * (base_grid_ele_i);
        bndry_to_main_global_vals[idx] = domain2_ele_id_offset + bndry_to_main_local_vals[idx];
        idx++;
    }
    
    // right
    for(index_t j = 0; j < base_grid_ele_j * 2; j++)
    {   
        bndry_to_main_local_vals[idx]  = (j+1) * (base_grid_ele_i)-1;
        bndry_to_main_global_vals[idx] = domain2_ele_id_offset + bndry_to_main_local_vals[idx];
        idx++;
    }

    // add an adjset for the main mesh

    //
    // domain 0
    //
    mesh["domain0/adjsets/main_adjset/association"] = "vertex";
    mesh["domain0/adjsets/main_adjset/topology"] = "main";
    Node &d0_adj_groups = mesh["domain0/adjsets/main_adjset/groups"];
    
    d0_adj_groups["group_0_1_2/neighbors"].set({1,2});
    // one point is shared between all three
    d0_adj_groups["group_0_1_2/values"] = base_grid_ele_i;

    // domain 0's entire bottom face (sans center point) is shared
    // by domains 0 and 1
    d0_adj_groups["group_0_1/neighbors"].set({1});
    d0_adj_groups["group_0_1/values"].set(DataType::int64(base_grid_ele_i));
    int64_array adj_vals = d0_adj_groups["group_0_1/values"].value();
    for(int i=0;i<base_grid_ele_i;i++)
    {
        adj_vals[i] = i;
    }

    // the entire right face (sans center point) is shared
    // by domains 0 and 2
    d0_adj_groups["group_0_2/neighbors"].set({2});
    d0_adj_groups["group_0_2/values"].set(DataType::int64(base_grid_ele_j));
    adj_vals = d0_adj_groups["group_0_2/values"].value();
    for(int j=0;j<base_grid_ele_j;j++)
    {
        adj_vals[j] = (j+2) * (base_grid_ele_i+1) -1;
    }

    //
    // domain 1
    //
    mesh["domain1/adjsets/main_adjset/association"] = "vertex";
    mesh["domain1/adjsets/main_adjset/topology"] = "main";
    Node &d1_adj_groups = mesh["domain1/adjsets/main_adjset/groups"];

    d1_adj_groups["group_0_1_2/neighbors"].set({1,2});
    // one point is shared between all three
    d1_adj_groups["group_0_1_2/values"] = (base_grid_ele_i+1) * (base_grid_ele_j+1)-1;

    // domain 1's entire top face (sans center point) is shared
    // by domains 0 and 1
    d1_adj_groups["group_0_1/neighbors"].set({0});
    d1_adj_groups["group_0_1/values"].set(DataType::int64(base_grid_ele_i));
    adj_vals = d1_adj_groups["group_0_1/values"].value();
    for(int i=0;i<base_grid_ele_i;i++)
    {
        adj_vals[i] = i + (base_grid_ele_i +1) * base_grid_ele_j;
    }

    // domain 1's entire right face (sans center point) is shared
    // by domains 1 and 2
    std::cout << base_grid_ele_j << std::endl;
    d1_adj_groups["group_1_2/neighbors"].set({2});
    d1_adj_groups["group_1_2/values"].set(DataType::int64(base_grid_ele_j));
    adj_vals = d1_adj_groups["group_1_2/values"].value();
    for(int j=0;j<base_grid_ele_j;j++)
    {
        adj_vals[j] = (j+1) * (base_grid_ele_i+1) -1;
    }

    //
    // domain 2
    //
    mesh["domain2/adjsets/main_adjset/association"] = "vertex";
    mesh["domain2/adjsets/main_adjset/topology"] = "main";
    Node &d2_adj_groups = mesh["domain2/adjsets/main_adjset/groups"];

    d2_adj_groups["group_0_1_2/neighbors"].set({0,1});
    // one point is shared between all three
    d2_adj_groups["group_0_1_2/values"] = (base_grid_ele_j) * (base_grid_ele_i+1);

    // 1/2 of domain 2's left face (sans center point) is shared
    // by domains 0 and 2
    d2_adj_groups["group_0_2/neighbors"].set({0});
    d2_adj_groups["group_0_2/values"].set(DataType::int64(base_grid_ele_j));
    adj_vals = d2_adj_groups["group_0_2/values"].value();
    for(int j=0;j<base_grid_ele_j;j++)
    {
        adj_vals[j] = (j) * (base_grid_ele_i+1);
    }

    // 1/2 of domain 2's left face (sans center point) is shared
    // by domains 1 and 2
    std::cout << base_grid_ele_j << std::endl;
    d2_adj_groups["group_1_2/neighbors"].set({1});
    d2_adj_groups["group_1_2/values"].set(DataType::int64(base_grid_ele_j));
    adj_vals = d2_adj_groups["group_1_2/values"].value();
    for(int j=0;j<base_grid_ele_j;j++)
    {
        adj_vals[j] = (base_grid_ele_j + j+1) * (base_grid_ele_i+1);
    }

}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
