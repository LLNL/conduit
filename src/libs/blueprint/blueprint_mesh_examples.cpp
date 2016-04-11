//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
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
/// file: blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <math.h>
#include <cassert>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "blueprint_mesh.hpp"


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
void braid_init_example_point_scalar_field(index_t npts_x,
                                           index_t npts_y,
                                           index_t npts_z,
                                           Node &res)
{
    index_t npts = npts_x * npts_y * npts_z;
    
    res["association"] = "point";
    res["type"] = "scalar";
    res["values"].set(DataType::float64(npts));
    
    float64 *vals = res["values"].value();

    float dx = (4.0 * PI_VALUE) / float64(npts_x-1);
    float dy = (2.0 * PI_VALUE) / float64(npts_y-1);
    float dz = (3.0 * PI_VALUE) / float64(npts_z-1);
    
    index_t idx = 0;

    for(index_t k = 0; k < npts_z ; k++)
    {
        float64 cz =  (k * dz) - (1.5 * PI_VALUE);

        for(index_t j = 0; j < npts_y ; j++)
        {
            float64 cy =  (j * dy) - ( PI_VALUE);
            for(index_t i = 0; i < npts_x ; i++)
            {
            
                float64 cx =  (i * dx) + (2.0 * PI_VALUE);
                
                float64 cv =  sin( cx ) + 
                              sin( cy ) + 
                              2 * cos(sqrt( (cx*cx)/2.0 +cy*cy) / .75) +
                              4 * cos( cx*cy / 4.0);
                                  
                if(npts_z > 1)
                {
                    cv += sin( cz ) + 
                          1.5 * cos(sqrt(cx*cx + cy*cy + cz*cz) / .75);
                }
                
                vals[idx] = cv;
                idx++;
            }
        }
    }
}

//---------------------------------------------------------------------------//
void braid_init_example_element_scalar_field(index_t nele_x,
                                             index_t nele_y,
                                             index_t nele_z,
                                             Node &res,
                                             index_t prims_per_ele=1)
{
    index_t nele = nele_x*nele_y;
    
    if(nele_z > 0)
    {
        nele = nele * nele_z;
    }

    res["association"] = "element";
    res["type"] = "scalar";
    res["values"].set(DataType::float64(nele*prims_per_ele));

    float64 *vals = res["values"].value();

    float dx = 20.0 / float64(nele_x);
    float dy = 20.0 / float64(nele_y);
    float dz = 0.0;
    
    if(nele_z > 0 )
    {
        dz = 20.0 / float64(nele_z);
    }
    
    index_t idx = 0;
    for(index_t k = 0; (idx == 0 || k < nele_z); k++)
    {
        float64 cz =  (k * dz) + -10.0;

        for(index_t j = 0; (idx == 0 || j < nele_y) ; j++)
        {
            float64 cy =  (j * dy) + -10.0;
            
            for(index_t i = 0; (idx == 0 || i < nele_x) ; i++)
            {
                float64 cx =  (i * dx) + -10.0;

                float64 cv = 10.0 * sqrt( cx*cx + cy*cy );
                
                if(nele_z != 0)
                {
                    cv = 10.0 * sqrt( cx*cx + cy*cy +cz*cz );
                }

                for(index_t ppe = 0; ppe < prims_per_ele; ppe++ )
                {
                    vals[idx] = cv;
                    idx++;
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
void braid_init_uniform_coordset(index_t npts_x,
                                 index_t npts_y,
                                 index_t npts_z,
                                 Node &coords)
{
    coords["type"] = "uniform";
    Node &dims = coords["dims"];
    dims["i"] = npts_x;
    dims["j"] = npts_y;

    if(npts_z > 1)
    {
        dims["k"] = npts_z;
    }
        
    // -10 to 10 in each dim, 
    Node &origin = coords["origin"];
    origin["x"] = -10.0;
    origin["y"] = -10.0;
    
    if(npts_z > 1)
    {
        origin["z"] = -10.0;
    }
    
    Node &spacing = coords["spacing"];
    spacing["dx"] = 20.0 / (float64)(npts_x-1);
    spacing["dy"] = 20.0 / (float64)(npts_y-1);

    if(npts_z > 1 )
    {
        spacing["dz"] = 20.0 / (float64)(npts_z-1);
    }
}


//---------------------------------------------------------------------------//
void braid_init_rectilinear_coordset(index_t npts_x,
                                     index_t npts_y,
                                     index_t npts_z,
                                     Node &coords)
{
    coords["type"] = "rectilinear";
    coords["x"].set(DataType::float64(npts_x));
    coords["y"].set(DataType::float64(npts_y));
    
    if(npts_z > 1)
    {
        coords["z"].set(DataType::float64(npts_z));
    }

    float64 *x_vals = coords["x"].value();
    float64 *y_vals = coords["y"].value();
    float64 *z_vals = NULL;

    if(npts_z > 1)
    {
        z_vals = coords["z"].value();
    }


    float64 dx = 20.0 / (float64)(npts_x-1);
    float64 dy = 20.0 / (float64)(npts_y-1);
    float64 dz = 0.0;
    
    if(npts_z > 1)
    {
        dz = 20.0 / (float64)(npts_z-1);
    }

    for(int i=0; i < npts_x; i++)
    {
        x_vals[i] = -10.0 + i * dx;
    }
    
    for(int j=0; j < npts_y; j++)
    {
        y_vals[j] = -10.0 + j * dy;
    }
    
    if(npts_z > 1)
    {
        for(int k=0; k < npts_z; k++)
        {
            z_vals[k] = -10.0 + k * dz;
        }
    }
}

//---------------------------------------------------------------------------//
void
braid_init_explicit_coordset(index_t npts_x,
                             index_t npts_y,
                             index_t npts_z,
                             Node &coords)
{
    coords["type"] = "explicit";
    
    index_t npts = npts_x * npts_y * npts_z;

    // also support interleaved
    coords["x"].set(DataType::float64(npts));
    coords["y"].set(DataType::float64(npts));

    if(npts_z > 1)
    {
        coords["z"].set(DataType::float64(npts));
    }

    float64 *x_vals = coords["x"].value();
    float64 *y_vals = coords["y"].value();
    float64 *z_vals = NULL;
    
    if(npts_z > 1)
    {
        z_vals = coords["z"].value();
    }

    float dx = 20.0 / float64(npts_x-1);
    float dy = 20.0 / float64(npts_y-1);

    float dz = 0.0;

    if(npts_z > 1)
    {
        dz = 20.0 / float64(npts_z-1);
    }

    index_t idx = 0;
    for(index_t k = 0; k < npts_z ; k++)
    {
        float64 cz = -10.0 + k * dz;
        
        for(index_t j = 0; j < npts_y ; j++)
        {
            float64 cy =  -10.0 + j * dy;
            
            for(index_t i = 0; i < npts_x ; i++)
            {
                x_vals[idx] = -10.0 + i * dx;
                y_vals[idx] = cy;
                
                if(npts_z > 1)
                {
                    z_vals[idx] = cz;
                }
                
                idx++;
            }
        
        }
    }
}


//---------------------------------------------------------------------------//
void
braid_uniform(index_t npts_x,
              index_t npts_y,
              index_t npts_z,
              Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_uniform_coordset(npts_x,
                                npts_y,
                                npts_z,
                                res["coords"]);

    res["topology/type"] = "uniform";
    res["topology/coordset"] = "coords"; 
    
    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            nele_z,
                                            fields["radial_ec"]);
}



//---------------------------------------------------------------------------//
void
braid_rectilinear(index_t npts_x,
                  index_t npts_y,
                  index_t npts_z,
                  Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_rectilinear_coordset(npts_x,
                                    npts_y,
                                    npts_z,
                                    res["coords"]);
    
    res["topology/type"] = "rectilinear";
    res["topology/coordset"] = "coords"; 
    
    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            nele_z,
                                            fields["radial_ec"]);
}

//---------------------------------------------------------------------------//
void
braid_structured(index_t npts_x,
                 index_t npts_y,
                 index_t npts_z,
                 Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coords"]);
  
    res["topology/type"] = "structured";
    res["topology/elements/dims/i"] = nele_x;
    res["topology/elements/dims/j"] = nele_y;
    
    if(nele_z > 0)
    {
        res["topology/elements/dims/k"] = nele_z; 
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);
                                          
    braid_init_example_element_scalar_field(nele_x,
                                            nele_y, 
                                            nele_z,
                                            fields["radial_ec"]);
}


//---------------------------------------------------------------------------//
void
braid_points_explicit(index_t npts_x,
                      index_t npts_y,
                      index_t npts_z,
                      Node &res)
{
    res.reset();
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coords"]);
    
    res["topology/type"] = "points";
    res["topology/coordset"] = "coords"; 
    
    Node &fields = res["fields"];
    
    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);
    
    braid_init_example_element_scalar_field(npts_x,
                                            npts_y, 
                                            npts_z,
                                            fields["radial_ec"]);
 }


//---------------------------------------------------------------------------//
void
braid_quads(index_t npts_x,
            index_t npts_y,
            Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x - 1;
    index_t nele_y = npts_y - 1;
    index_t nele = nele_x * nele_y;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coords"]);
  
    res["topology/type"] = "unstructured";
    res["topology/elements/shape"] = "quads";
    res["topology/elements/connectivity"].set(DataType::int32(nele*4));
    int32 *conn = res["topology/elements/connectivity"].value();

    index_t idx = 0;
    for(index_t j = 0; j < nele_x ; j++)
    {
        index_t yoff = j * (nele_x+1);
        for(index_t i = 0; i < nele_y; i++)
        {
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nele_x+1);
            conn[idx+2] = yoff + i + 1 + (nele_x+1);
            conn[idx+3] = yoff + i + 1;

            idx+=4;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid_pc"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            0,
                                            fields["radial_ec"]);
}

//---------------------------------------------------------------------------//
void
braid_quads_and_tris(index_t npts_x,
            index_t npts_y,
            Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x - 1;
    index_t nele_y = npts_y - 1;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coords"]);
  
    res["topology/type"] = "unstructured";

    Node& elems = res["topology/elements"];
    elems["stream_shapes/quads/stream_id"] = 9; // VTK_QUAD
    elems["stream_shapes/quads/shape"]     = "quads";
    elems["stream_shapes/tris/stream_id"]  = 5; // VTK_TRIANGLE
    elems["stream_shapes/tris/shape"]      = "tris";

    // Fill in stream IDs and calculate size of the connectivity array
    int count   = 0;
    int ielem   = 0;
    std::vector< int32 > stream_ids_buffer;
    std::vector< int32 > stream_lengths;

    for(index_t j = 0; j < nele_x ; j++)
    {
        for(index_t i = 0; i < nele_y; i++)
        {
             if ( ielem % 2 == 0 )
             {
                 // QUAD
                 stream_ids_buffer.push_back( 9 );
                 stream_lengths.push_back( 1 );
                 count += 4;
             }
             else
             {
                 // TRIANGLE
                 stream_ids_buffer.push_back( 5 );
                 count += 6;
                 stream_lengths.push_back( 2 );
             }

             ++ielem;

        } // END for all i

    } // END for all j


    elems["stream_index/stream_ids"].set(stream_ids_buffer);
    elems["stream_index/stream_lengths"].set(stream_lengths);

    // Allocate connectivity array
    elems["stream"].set(DataType::int32(count));
    int32* conn = elems["stream"].value();

    // Fill in connectivity array
    index_t idx = 0;
    int32 elem  = 0;
    for(index_t j = 0; j < nele_x ; j++)
    {
        index_t yoff = j * (nele_x+1);

        for(index_t i = 0; i < nele_y; i++)
        {
            index_t n1 = yoff + i;
            index_t n2 = n1 + (nele_x+1);
            index_t n3 = n1 + 1 + (nele_x+1);
            index_t n4 = n1 + 1;

            if ( elem % 2 == 0 )
            {
                conn[idx  ] = n1;
                conn[idx+1] = n2;
                conn[idx+2] = n3;
                conn[idx+3] = n4;
                idx+=4;
            }
            else
            {
               conn[idx   ] = n1;
               conn[idx+1 ] = n2;
               conn[idx+2 ] = n4;
               idx+=3;

               conn[idx   ] = n2;
               conn[idx+1 ] = n3;
               conn[idx+2 ] = n4;
               idx+=3;
            }

            ++elem;

        } // END for all i

    } // END for all j


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid_pc"]);

    // braid_init_example_element_scalar_field(nele_x,
    //                                         nele_y,
    //                                         0,
    //                                         fields["radial_ec"]);
}

//---------------------------------------------------------------------------//
void
braid_quads_and_tris_offsets(index_t npts_x,
                             index_t npts_y,
                             Node &res)
{

    res.reset();

    index_t nele_x = npts_x - 1;
    index_t nele_y = npts_y - 1;

    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coords"]);

    res["topology/type"] = "unstructured";

    Node& elems = res["topology/elements"];
    elems["stream_shapes/quads/stream_id"] = 9; // VTK_QUAD
    elems["stream_shapes/quads/shape"]     = "quads";
    elems["stream_shapes/tris/stream_id"]  = 5; // VTK_TRIANGLE
    elems["stream_shapes/tris/shape"]      = "tris";

    // Fill in stream IDs and calculate size of the connectivity array
    int count   = 0;
    int ielem   = 0;
    std::vector< int32 > stream_ids;
    std::vector< int32 > stream_offsets;
    stream_offsets.push_back( 0 );

    for(index_t j = 0; j < nele_x ; j++)
    {
        for(index_t i = 0; i < nele_y; i++)
        {
             int32 next = stream_offsets.back();

             if ( ielem % 2 == 0 )
             {
                 // QUAD
                 stream_offsets.push_back( next+4 );
                 stream_ids.push_back( 9 );
                 count += 4;
             }
             else
             {
                 // TRIANGLE
                 stream_offsets.push_back( next+3 );
                 stream_offsets.push_back( next+6 );
                 stream_ids.push_back( 5 );
                 stream_ids.push_back( 5 );
                 count += 6;
             }

             ++ielem;

        } // END for all i

    } // END for all j


    elems["stream_index/stream_ids"].set(stream_ids);
    elems["stream_index/stream_offsets"].set(stream_offsets);

    // Allocate connectivity array
    elems["stream"].set(DataType::int32(count));
    int32* conn = elems["stream"].value();

    // Fill in connectivity array
    index_t idx = 0;
    int32 elem  = 0;
    for(index_t j = 0; j < nele_x ; j++)
    {
        index_t yoff = j * (nele_x+1);

        for(index_t i = 0; i < nele_y; i++)
        {
            index_t n1 = yoff + i;
            index_t n2 = n1 + (nele_x+1);
            index_t n3 = n1 + 1 + (nele_x+1);
            index_t n4 = n1 + 1;

            if ( elem % 2 == 0 )
            {
                conn[idx  ] = n1;
                conn[idx+1] = n2;
                conn[idx+2] = n3;
                conn[idx+3] = n4;
                idx+=4;
            }
            else
            {
               conn[idx   ] = n1;
               conn[idx+1 ] = n2;
               conn[idx+2 ] = n4;
               idx+=3;

               conn[idx   ] = n2;
               conn[idx+1 ] = n3;
               conn[idx+2 ] = n4;
               idx+=3;
            }

            ++elem;

        } // END for all i

    } // END for all j


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid_pc"]);
}

//---------------------------------------------------------------------------//
void
braid_tris(index_t npts_x,
           index_t npts_y,
           Node &res)
{
    res.reset();
    
    // require npts_x > 0 && npts_y > 0

    index_t nele_quads_x = npts_x-1;
    index_t nele_quads_y = npts_y-1;
    index_t nele_quads = nele_quads_x * nele_quads_y;
        
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coords"]);
  
    res["topology/type"] = "unstructured";
    res["topology/elements/shape"] = "tris";
    res["topology/elements/connectivity"].set(DataType::int32(nele_quads*6));
    int32 *conn = res["topology/elements/connectivity"].value();

    index_t idx = 0;
    for(index_t j = 0; j < nele_quads_y ; j++)
    {
        index_t yoff = j * (nele_quads_x+1);
        
        for(index_t i = 0; i < nele_quads_x; i++)
        {
            // two tris per quad. 
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nele_quads_x+1);
            conn[idx+2] = yoff + i + 1 + (nele_quads_x+1);

            conn[idx+3] = yoff + i;
            conn[idx+4] = yoff + i +1;
            conn[idx+5] = yoff + i + 1 + (nele_quads_x+1);
            
            idx+=6;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid_pc"]);

    braid_init_example_element_scalar_field(nele_quads_x,
                                            nele_quads_y,
                                            0,
                                            fields["radial_ec"],2);
}


//---------------------------------------------------------------------------//
void
braid_hexs(index_t npts_x,
           index_t npts_y,
           index_t npts_z,
           Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x - 1;
    index_t nele_y = npts_y - 1;
    index_t nele_z = npts_z - 1;
    index_t nele = nele_x * nele_y * nele_z;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coords"]);
  
    res["topology/type"] = "unstructured";
    res["topology/elements/shape"] = "hexs";
    res["topology/elements/connectivity"].set(DataType::int32(nele*8));
    int32 *conn = res["topology/elements/connectivity"].value();

    index_t idx = 0;
    for(index_t k = 0; k < nele_z ; k++)
    {
        index_t zoff = k * (nele_x+1)*(nele_y+1);
        index_t zoff_n = (k+1) * (nele_x+1)*(nele_y+1);
        
        for(index_t j = 0; j < nele_y ; j++)
        {
            index_t yoff = j * (nele_x+1);
            index_t yoff_n = (j+1) * (nele_x+1);


            for(index_t i = 0; i < nele_x; i++)
            {
                // ordering is same as VTK_HEXAHEDRON

                conn[idx+0] = zoff + yoff + i;
                conn[idx+1] = zoff + yoff + i + 1;
                conn[idx+2] = zoff + yoff_n + i + 1;
                conn[idx+3] = zoff + yoff_n + i;

                conn[idx+4] = zoff_n + yoff + i;
                conn[idx+5] = zoff_n + yoff + i + 1;
                conn[idx+6] = zoff_n + yoff_n + i + 1;
                conn[idx+7] = zoff_n + yoff_n + i;

                idx+=8;
            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);

    // skip for now, we need a helper that generates 
    // one value for each quad, then two for each tri
    // we can't do this b/c the current ele_scalar_file function
    // is to simplistic. 
    //
    // braid_init_example_element_scalar_field(nele_x,
    //                                         nele_y,
    //                                         nele_z,
    //                                         fields["radial_ec"]);
}

//---------------------------------------------------------------------------//
void
braid_tets(index_t npts_x,
           index_t npts_y,
           index_t npts_z,
           Node &res)
{
    res.reset();
    
    index_t nele_hexs_x = npts_x - 1;
    index_t nele_hexs_y = npts_y - 1;
    index_t nele_hexs_z = npts_z - 1;
    index_t nele_hexs = nele_hexs_x * nele_hexs_y * nele_hexs_z;
    
    index_t tets_per_hex = 6;
    index_t verts_per_tet = 4;
    index_t n_tets_verts = nele_hexs * tets_per_hex * verts_per_tet;

    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coords"]);
  

    res["topology/type"] = "unstructured";
    res["topology/elements/shape"] = "tets";
    res["topology/elements/connectivity"].set(DataType::int32(n_tets_verts));
    int32 *conn = res["topology/elements/connectivity"].value();


    index_t idx = 0;
    for(index_t k = 0; k < nele_hexs_z ; k++)
    {
        index_t zoff = k * (nele_hexs_x+1)*(nele_hexs_y+1);
        index_t zoff_n = (k+1) * (nele_hexs_x+1)*(nele_hexs_y+1);
        
        for(index_t j = 0; j < nele_hexs_y ; j++)
        {
            index_t yoff = j * (nele_hexs_x+1);
            index_t yoff_n = (j+1) * (nele_hexs_x+1);


            for(index_t i = 0; i < nele_hexs_z; i++)
            {
                // Create a local array of the vertex indices
                // ordering is same as VTK_HEXAHEDRON
                index_t vidx[8] = {zoff + yoff + i
                                  ,zoff + yoff + i + 1
                                  ,zoff + yoff_n + i + 1
                                  ,zoff + yoff_n + i
                                  ,zoff_n + yoff + i
                                  ,zoff_n + yoff + i + 1
                                  ,zoff_n + yoff_n + i + 1
                                  ,zoff_n + yoff_n + i};

                // Create six tets all sharing diagonal from vertex 0 to 6
                // Uses SILO convention for vertex order (normals point in)
                conn[idx++] = vidx[0];
                conn[idx++] = vidx[2];
                conn[idx++] = vidx[1];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[3];
                conn[idx++] = vidx[2];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[7];
                conn[idx++] = vidx[3];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[4];
                conn[idx++] = vidx[7];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[5];
                conn[idx++] = vidx[4];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[1];
                conn[idx++] = vidx[5];
                conn[idx++] = vidx[6];

            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);

    braid_init_example_element_scalar_field(nele_hexs_x,
                                            nele_hexs_y,
                                            nele_hexs_z,
                                            fields["radial_ec"],
                                            tets_per_hex);
}

//---------------------------------------------------------------------------//
void
braid_hexs_and_tets(index_t npts_x,
           index_t npts_y,
           index_t npts_z,
           Node &res)
{

    // WARNING -- The code below is UNTESTED.
    //            The SILO writer is missing an implementation for
    //            unstructured indexed_stream meshes in 3D.

    res.reset();

    index_t nele_hexs_x = npts_x - 1;
    index_t nele_hexs_y = npts_y - 1;
    index_t nele_hexs_z = npts_z - 1;
    index_t nele_hexs = nele_hexs_x * nele_hexs_y * nele_hexs_z;


    // Set the number of voxels containing hexs and tets
    index_t n_hex_hexs = (nele_hexs > 1)? nele_hexs / 2 : nele_hexs;
    index_t n_hex_tets = nele_hexs - n_hex_hexs;

    // Compute the sizes of the connectivity array for each element type
    index_t hexs_per_hex = 1;
    index_t verts_per_hex = 8;
    index_t n_hexs_verts = n_hex_hexs * hexs_per_hex * verts_per_hex;

    index_t tets_per_hex = 6;
    index_t verts_per_tet = 4;
    index_t n_tets_verts = n_hex_tets * tets_per_hex * verts_per_tet;


    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coords"]);

    // Setup mesh as unstructured indexed_stream mesh of hexs and tets
    res["topology/type"] = "unstructured";

    res["topology/elements/stream_shapes/hexs/stream_id"] = 0;
    res["topology/elements/stream_shapes/hexs/shape"] = "hexs";

    res["topology/elements/stream_shapes/tets/stream_id"] = 1;
    res["topology/elements/stream_shapes/tets/shape"] = "tets";

    res["topology/elements/stream_index/stream_ids"].set(DataType::int32(4));
    res["topology/elements/stream_index/stream_lengths"].set(DataType::int32(4));

    int32* sidx_ids = res["topology/elements/stream_index/stream_ids"].value();
    int32* sidx_lengths = res["topology/elements/stream_index/stream_lengths"].value();

    // There are four groups -- alternating between hexs and tets
    sidx_ids[0] = 0;
    sidx_ids[1] = 1;
    sidx_ids[2] = 0;
    sidx_ids[3] = 1;

    // Set the lengths of the groups
    // The first two groups have at most length 1
    // The last two groups have the balance of the elements
    switch(nele_hexs)
    {
    case 0:
        sidx_lengths[0] = 0;  sidx_lengths[1] = 0;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 1:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 0;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 2:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 1;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 3:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 1;
        sidx_lengths[2] = 1;  sidx_lengths[3] = 0;
        break;
    default:
        sidx_lengths[0] = 1;
        sidx_lengths[1] = 1;
        sidx_lengths[2] = n_hex_hexs-1;
        sidx_lengths[3] = n_hex_tets-1;
        break;
    }

    res["topology/elements/stream"].set( DataType::int32(n_hexs_verts + n_tets_verts) );
    int32* conn = res["topology/elements/stream"].value();

    index_t idx = 0;
    index_t elem_count = 0;
    for(index_t k = 0; k < nele_hexs_z ; k++)
    {
        index_t zoff = k * (nele_hexs_x+1)*(nele_hexs_y+1);
        index_t zoff_n = (k+1) * (nele_hexs_x+1)*(nele_hexs_y+1);

        for(index_t j = 0; j < nele_hexs_y ; j++)
        {
            index_t yoff = j * (nele_hexs_x+1);
            index_t yoff_n = (j+1) * (nele_hexs_x+1);


            for(index_t i = 0; i < nele_hexs_z; i++)
            {
                // Create a local array of the vertex indices
                // ordering is same as VTK_HEXAHEDRON
                index_t vidx[8] = {zoff + yoff + i
                                  ,zoff + yoff + i + 1
                                  ,zoff + yoff_n + i + 1
                                  ,zoff + yoff_n + i
                                  ,zoff_n + yoff + i
                                  ,zoff_n + yoff + i + 1
                                  ,zoff_n + yoff_n + i + 1
                                  ,zoff_n + yoff_n + i};

                bool isHex = (elem_count == 0)
                          || (elem_count > 1 && elem_count <= n_hex_hexs);


                if(isHex)
                {
                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[3];

                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[6];
                    conn[idx++] = vidx[7];
                }
                else // it is a tet
                {
                    // Create six tets all sharing diagonal from vertex 0 to 6
                    // Uses SILO convention for vertex order (normals point in)
                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[3];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[7];
                    conn[idx++] = vidx[3];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[7];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[6];
                }

                elem_count++;
            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid_pc"]);

//    // Omit for now -- the function assumes a uniform element type
//    braid_init_example_element_scalar_field(nele_hexs_x,
//                                            nele_hexs_y,
//                                            nele_hexs_z,
//                                            fields["radial_ec"],
//                                            tets_per_hex);
}



//---------------------------------------------------------------------------//
void
braid(const std::string &mesh_type,
      index_t npts_x, // number of points in x
      index_t npts_y, // number of points in y
      index_t npts_z, // number of points in z
      Node &res)
{

    if(mesh_type == "uniform")
    {
        braid_uniform(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "rectilinear")
    {
        braid_rectilinear(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "structured")
    {
        braid_structured(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "tris")
    {
        braid_tris(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads")
    {
        braid_quads(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads_and_tris")
    {
        braid_quads_and_tris(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads_and_tris_offsets")
    {
        braid_quads_and_tris_offsets(npts_x,npts_y,res);
    }
    else if(mesh_type == "tets")
    {
        braid_tets(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "hexs")
    {
        braid_hexs(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "hexs_and_tets")
    {
        braid_hexs_and_tets(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "points_explicit")
    {
        braid_points_explicit(npts_x,npts_y,npts_z,res);
    }
    else
    {
        CONDUIT_ERROR("unknown mesh_type = " << mesh_type);
    }
}



//-----------------------------------------------------------------------------
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
