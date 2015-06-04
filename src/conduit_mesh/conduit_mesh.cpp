//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <math.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_mesh.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::mesh --
//-----------------------------------------------------------------------------

namespace mesh
{

//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    mesh::about(n);
    return n.to_pure_json();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    // no info yet
    n.reset();
    n["purpose"] = "experimental transforms for mesh conventions.";
}


void
expand(Node &src, Node &des)
{
    if(src.has_path("topologies") && src.has_path("coordsets"))
    {
        // assume all is well, we already have a multi-topology description
        des.set_external(src);
    }
    else if(src.has_path("topology") && src.has_path("coords"))
    {
        // promote single topology to standard multi-topology description
        if(src.has_path("state"))
        {
            des["state"].set_external(src["state"]);
            
        }
        // mesh is the default name for a topo, coords is the default name for a coordset

        des["coordsets/coords"].set_external(src["coords"]);

        Node &des_topo = des["topologies/mesh"];
        
        NodeIterator itr = src["topology"].iterator();
        itr.next();
        std::string topo_name = itr.path();

        des_topo[topo_name]["coordset"].set("coords");


        if(topo_name == "quads")
        {
            des_topo[topo_name]["connectivity"].set_external(src["topology/quads"]);
        }
        else if(topo_name == "tris")
        {
            des_topo[topo_name]["connectivity"].set_external(src["topology/tris"]);            
        }
        
        if(src.has_path("fields"))
        {
            des["fields"].set_external(src["fields"]);
            
            NodeIterator itr = des["fields"].iterator();
            
            while( itr.has_next() )
            {
                Node &field = itr.next();
                field["topologies"].append().set("mesh");
            }
        }
        des.print();
    }
    else
    {
        CONDUIT_ERROR("Missing topologies and coordsets, or topology and coords");
    }
}


namespace examples
{

//---------------------------------------------------------------------------//
const float64 PI_VALUE = 3.14159265359;


//---------------------------------------------------------------------------//
void braid_init_example_state(Node &res)
{
    res["state/time"]   = (float64)3.1415;
    res["state/domain"] = (uint64) 0;
    res["state/cycle"]  = (uint64) 100;
}


//---------------------------------------------------------------------------//
void braid_init_example_point_scalar_field(index_t nx,
                                           index_t ny,
                                            index_t nz,
                                            Node &res)
{
    index_t npts = (nx+1)*(ny+1);
    
    res["association"] = "point";
    res["type"] = "scalar";
    res["values"].set(DataType::float64(npts));
    
    float64 *vals = res["values"].value();

    float dx = 20.0 / float64(nx);
    float dy = 20.0 / float64(ny);
    
    index_t idx = 0;
    
    for(index_t j = 0; j < ny+1 ; j++)
    {
        float64 cy =  (j * dy) + -10.0;
        for(index_t i = 0; i < nx+1 ; i++)
        {
            
            float64 cx =  (i * dx) + -10.0;
                
            vals[idx] =  sin( (2 * PI_VALUE * cx) / 10.0 ) + 
                         sin( (2 * PI_VALUE * cy) / 20.0 );
            idx++;
        }
    }
}

//---------------------------------------------------------------------------//
void braid_init_example_element_scalar_field(index_t nx,
                                             index_t ny,
                                             index_t nz,
                                             Node &res,
                                             index_t prims_per_ele=1)
{
    index_t nele = nx*ny;

    res["association"] = "element";
    res["type"] = "scalar";
    res["values"].set(DataType::float64(nele*prims_per_ele));

    float64 *vals = res["values"].value();

    float dx = 20.0 / float64(nx-1);
    float dy = 20.0 / float64(ny-1);
    
    index_t idx = 0;
    
    for(index_t j = 0; j < ny ; j++)
    {
        float64 cy =  (j * dy) + -10.0;
        for(index_t i = 0; i < nx ; i++)
        {
            float64 cx =  (i * dx) + -10.0;
            float64 cv = 10.0 * sqrt( cx*cx + cy*cy );
            for(index_t ppe = 0; ppe < prims_per_ele; ppe++ )
            {
                vals[idx] = cv;
                idx++;
            }
        }
    }
}


//---------------------------------------------------------------------------//
void
braid_uniform(index_t nx,
              index_t ny,
              index_t nz,
              Node &res)
{
    res.reset();
    braid_init_example_state(res);
    
    Node &dims = res["coords/uniform/dims"];
    dims["i"] = nx;
    dims["j"] = ny;
        
    // -10 to 10 in each dim, 
    Node &origin = res["coords/uniform/origin"];
    origin["x"] = -10.0;
    origin["y"] = -10.0;
    // skip z for now
    Node &spacing = res["coords/uniform/spacing"];
    spacing["x"] = 20.0 / (float64)(nx);
    spacing["y"] = 20.0 / (float64)(ny);
    // skip z for now
    res["topology/uniform"] = "coords"; // or name?
    
    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(nx,ny,nz,fields["braid_pc"]);
    braid_init_example_element_scalar_field(nx,ny,nz,fields["radial_ec"]);
}


//---------------------------------------------------------------------------//
void
braid_rectilinear(index_t nx,
                  index_t ny,
                  index_t nz,
                  Node &res)
{
    res.reset();
    braid_init_example_state(res);
    
    Node &dims = res["coords/rectilinear/"];
    dims["x"].set(DataType::float64(nx+1));
    dims["y"].set(DataType::float64(ny+1));
    float64 *x_vals = dims["x"].value();
    float64 *y_vals = dims["y"].value();

    float64 dx = 20.0 / (float64)(nx);
    float64 dy = 20.0 / (float64)(ny);

    for(int i=0; i < nx+1; i++)
    {
        x_vals[i] = -10.0 + i * dx;
    }
    
    for(int j=0; j < ny+1; j++)
    {
        y_vals[j] = -10.0 + j * dy;
    }
    
    // skip z for now

    res["topology/rectilinear"] = "coords"; // or name?
    
    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(nx,ny,nz,fields["braid_pc"]);
    braid_init_example_element_scalar_field(nx,ny,nz,fields["radial_ec"]);
}


//---------------------------------------------------------------------------//
void
braid_init_explicit_coords(index_t nx,
                           index_t ny,
                           index_t nz,
                           bool interleaved,
                           Node &res)
{
    Node &coords = res["coords"];
    
    index_t npts = (nx+1)*(ny+1);

    // also support interleaved
    coords["x"].set(DataType::float64(npts));
    coords["y"].set(DataType::float64(npts));

    float64 *x_vals = coords["x"].value();
    float64 *y_vals = coords["y"].value();

    float dx = 20.0 / float64(nx);
    float dy = 20.0 / float64(ny);
    // skip z for now

    index_t idx = 0;
    for(index_t j = 0; j < ny+1 ; j++)
    {
        float64 cy =  -10.0 + j * dy;
        for(index_t i = 0; i < nx+1 ; i++)
        {
            x_vals[idx] = -10.0 + i * dx;
            y_vals[idx] = cy;
            idx++;
        }
    }
}


//---------------------------------------------------------------------------//
void
braid_quads(index_t nx,
            index_t ny,
            index_t nz,
            Node &res)
{
    res.reset();
    braid_init_example_state(res);
    braid_init_explicit_coords(nx,ny,nz,false,res);
  
    res["topology/quads"].set(DataType::int32(nx*ny*4));

    int32 *conn = res["topology/quads"].value();

    index_t idx = 0;
    for(index_t j = 0; j < ny ; j++)
    {
        index_t yoff = j * (nx+1);
        for(index_t i = 0; i < nx; i++)
        {
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nx+1);
            conn[idx+2] = yoff + i + 1 + (nx+1);
            conn[idx+3] = yoff + i + 1;

            idx+=4;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(nx,ny,nz,fields["braid_pc"]);
    braid_init_example_element_scalar_field(nx,ny,nz,fields["radial_ec"]);
}

//---------------------------------------------------------------------------//
void
braid_tris(index_t nx,
           index_t ny,
           index_t nz,
           Node &res)
{
    res.reset();
    braid_init_example_state(res);
    braid_init_explicit_coords(nx,ny,nz,false,res);
  
    res["topology/tris"].set(DataType::int32(nx*ny*6));

    int32 *conn = res["topology/tris"].value();

    index_t idx = 0;
    for(index_t j = 0; j < ny ; j++)
    {
        index_t yoff = j * (nx+1);
        for(index_t i = 0; i < nx; i++)
        {
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nx+1);
            conn[idx+2] = yoff + i + 1 + (nx+1);

            conn[idx+3] = yoff + i;
            conn[idx+4] = yoff + i +1;
            conn[idx+5] = yoff + i + 1 + (nx+1);
            
            idx+=6;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(nx,ny,nz,fields["braid_pc"]);
    braid_init_example_element_scalar_field(nx,ny,nz,fields["radial_ec"],2);
}




//---------------------------------------------------------------------------//
void
braid(const std::string &mesh_type,
      index_t nx,
      index_t ny,
      index_t nz,
      Node &res)
{

    if(mesh_type == "uniform")
    {
        braid_uniform(nx,ny,nz,res);
    }
    else if(mesh_type == "rectilinear")
    {
        braid_rectilinear(nx,ny,nz,res);
    }
    else if(mesh_type == "tris")
    {
        braid_tris(nx,ny,nz,res);
    }
    else if(mesh_type == "quads")
    {
        braid_quads(nx,ny,nz,res);
    }
    else
    {
        CONDUIT_ERROR("unknown mesh_type = " << mesh_type);
    }
}




};
//-----------------------------------------------------------------------------
// -- end conduit::mesh::examples --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit::mesh --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
