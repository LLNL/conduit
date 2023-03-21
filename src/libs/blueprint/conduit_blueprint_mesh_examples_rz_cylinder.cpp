// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_rz_cylinder.cpp
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

//-----------------------------------------------------------------------------
// detail namespace for internal helpers
//-----------------------------------------------------------------------------
namespace detail 
{

//-----------------------------------------------------------------------------
void
create_rz_cyl_explicit_coords(index_t nz,
                              index_t nr,
                              Node &res)
{
    res["type"] = "explicit";
    res["values/z"] = DataType::float64( (nz + 1) * (nr + 1) );
    res["values/r"] = DataType::float64( (nz + 1) * (nr + 1) );

    float64_array z_coords = res["values/z"].value();
    float64_array r_coords = res["values/r"].value();

    index_t idx =0;
    for(index_t j=0; j < nr+1; j++)
    {
        float64 z  = -2.0;
        float64 dz = 4.0 / (nz + 1.0);

        for(index_t i=0; i < nz+1; i++)
        {
            z_coords[idx] = z;
            z+=dz;
            r_coords[idx] = (float64) j;
            idx++;
        }
    }
}

//-----------------------------------------------------------------------------
void
create_rz_cyl_field(index_t nz,
                     index_t nr,
                     Node &res)
{
    res["cyl/association"] = "element";
    res["cyl/topology"] = "topo";
    res["cyl/values"] = DataType::float64( nz * nr );
    float64_array vals = res["cyl/values"].value();

    index_t idx =0;
    for(index_t j=0; j < nr; j++)
    {
        for(index_t i=0; i < nz; i++)
        {
            vals[idx] = (float64) j;
            idx++;
        }
    }
}

//-----------------------------------------------------------------------------
} // end namespace detail
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
rz_cylinder(const std::string &mesh_type,
            index_t nz,
            index_t nr,
            Node &res)
{
    res.reset();

    // check ranges of nz and nr
    if(nz <= 0 || nr <= 0)
    {
        // unsupported
        CONDUIT_ERROR("blueprint::mesh::examples::rz_cylinder nz and nz"
                      << " must be >= 1 (nz = " << nz
                      << " nr = " << nr << ")");

    }

    // create rz example with selected topo
    if(mesh_type == "uniform")
    {
        res["coordsets/coords/type"] = "uniform";
        res["coordsets/coords/dims/i"] = nz + 1;
        res["coordsets/coords/dims/j"] = nr + 1;

        res["coordsets/coords/origin/z"] = -2.0;
        res["coordsets/coords/origin/r"] = 0.0;

        res["coordsets/coords/spacing/dz"] = 4.0 / (nz + 1.0);
        res["coordsets/coords/spacing/dr"] = 1.0;

        res["topologies/topo/type"] = "uniform";
        res["topologies/topo/coordset"] = "coords";

    }
    else if(mesh_type == "rectilinear")
    {
        res["coordsets/coords/type"] = "rectilinear";
        res["coordsets/coords/values/z"] = DataType::float64(nz+1);
        res["coordsets/coords/values/r"] = DataType::float64(nr+1);

        float64_array z_coords = res["coordsets/coords/values/z"].value();
        float64_array r_coords = res["coordsets/coords/values/r"].value();

        float64 z  = -2.0;
        float64  dz = 4.0 / (nz + 1.0);

        for(index_t i=0;i<z_coords.number_of_elements();i++)
        {
            z_coords[i] = z;
            z+=dz;
        }

        for(index_t i=0;i<r_coords.number_of_elements();i++)
        {
            r_coords[i] = (float64)i;
        }

        res["topologies/topo/type"] = "rectilinear";
        res["topologies/topo/coordset"] = "coords";

    }
    else if(mesh_type == "structured")
    {
        detail::create_rz_cyl_explicit_coords(nz,nr,res["coordsets/coords"]);

        res["topologies/topo/type"] = "structured";
        res["topologies/topo/coordset"] = "coords";
        res["topologies/topo/elements/dims/i"] = nz;
        res["topologies/topo/elements/dims/j"] = nr;
    }
    else if(mesh_type == "unstructured")
    {
        detail::create_rz_cyl_explicit_coords(nz,nr,res["coordsets/coords"]);
        res["topologies/topo/type"] = "unstructured";
        res["topologies/topo/coordset"] = "coords";
        res["topologies/topo/elements/shape"] = "quad";
        res["topologies/topo/elements/connectivity"] =  DataType::index_t(nz * nr * 4);

        index_t_array conn_vals = res["topologies/topo/elements/connectivity"].value();

        index_t idx =0;
        for(index_t j=0; j < nr; j++)
        {
            for(index_t i=0; i < nz; i++)
            {
                // four idxs per element
                conn_vals[idx]   = (j*(nz+1)) + i;
                conn_vals[idx+1] = (j*(nz+1)) + i+1;
                conn_vals[idx+2] = ((j+1) * (nz+1)) + i+1;
                conn_vals[idx+3] = ((j+1) * (nz+1)) + i;
                idx+=4;
            }
        }
    }
    else
    {
        // unsupported
        CONDUIT_ERROR("blueprint::mesh::examples::rz_cylinder unsupported"
                      " mesh_type = " << mesh_type);
    }
    // add the cyl field
    detail::create_rz_cyl_field(nz,nr,res["fields"]);

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
