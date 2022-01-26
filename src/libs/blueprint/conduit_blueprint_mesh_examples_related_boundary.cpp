// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_related_boundary.hpp
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
    mesh["domain0/fields/ele_id/association"] = "element";
    mesh["domain0/fields/ele_id/topology"] = "main";
    mesh["domain0/fields/ele_id/values"].set( DataType::float64(num_eles) );


    mesh["domain0/fields/domain_id/association"] = "element";
    mesh["domain0/fields/domain_id/topology"] = "main";
    mesh["domain0/fields/domain_id/values"].set( DataType::float64(num_eles) );

    float64_array ele_vals = mesh["domain0/fields/ele_id/values"].value();
    float64_array dom_id_vals = mesh["domain0/fields/domain_id/values"].value();

    index_t  main_id_global = 0;

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_vals[i] = main_id_global;
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

    mesh["domain0/fields/bndry_id/association"] = "element";
    mesh["domain0/fields/bndry_id/topology"] = "boundary";
    mesh["domain0/fields/bndry_id/values"].set( DataType::int64(num_edges) );

    int64_array bndry_id_vals = mesh["domain0/fields/bndry_id/values"].value();
    
    index_t  bndry_id_global = 0;
    
    // unique ids for the boundary 
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_vals[i] =  bndry_id_global;
        bndry_id_global+=1;
    }


    // domain 1:
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

    num_eles = base_grid_ele_i * base_grid_ele_j;
    mesh["domain1/fields/ele_id/association"] = "element";
    mesh["domain1/fields/ele_id/topology"] = "main";
    mesh["domain1/fields/ele_id/values"].set( DataType::float64(num_eles) );


    mesh["domain1/fields/domain_id/association"] = "element";
    mesh["domain1/fields/domain_id/topology"] = "main";
    mesh["domain1/fields/domain_id/values"].set( DataType::float64(num_eles) );

    ele_vals = mesh["domain1/fields/ele_id/values"].value();
    dom_id_vals = mesh["domain1/fields/domain_id/values"].value();

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_vals[i] = main_id_global;
        dom_id_vals[i] = 1.0;
        main_id_global+=1;
    }
    
    mesh["domain1/fields/bndry_val/association"] = "element";
    mesh["domain1/fields/bndry_val/topology"] = "boundary";
    mesh["domain1/fields/bndry_val/values"].set( DataType::float64(num_edges) );

    bndry_vals = mesh["domain1/fields/bndry_val/values"].value();

    // bottom
    idx = 0;
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] =  0;
        idx+=1;
    }

    // top
    for(int i=0;i<base_grid_ele_i;i++)
    {
        bndry_vals[idx] =  1;
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

    mesh["domain1/fields/bndry_id/association"] = "element";
    mesh["domain1/fields/bndry_id/topology"] = "boundary";
    mesh["domain1/fields/bndry_id/values"].set( DataType::int64(num_edges) );

    bndry_id_vals = mesh["domain1/fields/bndry_id/values"].value();

    // unique ids for the boundary 
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_vals[i] =  bndry_id_global;
        bndry_id_global+=1;
    }

    // domain 1
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
    mesh["domain2/fields/ele_id/association"] = "element";
    mesh["domain2/fields/ele_id/topology"] = "main";
    mesh["domain2/fields/ele_id/values"].set( DataType::float64(num_eles) );


    mesh["domain2/fields/domain_id/association"] = "element";
    mesh["domain2/fields/domain_id/topology"] = "main";
    mesh["domain2/fields/domain_id/values"].set( DataType::float64(num_eles) );

    ele_vals = mesh["domain2/fields/ele_id/values"].value();
    dom_id_vals = mesh["domain2/fields/domain_id/values"].value();

    for(index_t i = 0; i < num_eles; i++)
    {
        ele_vals[i] = main_id_global;
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

    mesh["domain2/fields/bndry_id/association"] = "element";
    mesh["domain2/fields/bndry_id/topology"] = "boundary";
    mesh["domain2/fields/bndry_id/values"].set( DataType::int64(num_edges) );

    bndry_id_vals = mesh["domain2/fields/bndry_id/values"].value();

    // unique ids for the boundary 
    for(int i=0;i<num_edges;i++)
    {
        bndry_id_vals[i] =  bndry_id_global;
        bndry_id_global+=1;
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
