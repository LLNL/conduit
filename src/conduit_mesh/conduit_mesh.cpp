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
    return n.to_json(true,2);
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    // no info yet
    n.reset();
    n["purpose"] = "experimental transforms for mesh conventions.";
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
void braid_init_example_pt_scalar_field(index_t nx,
                                        index_t ny,
                                        index_t nz,
                                        Node &res)
{
    index_t npts = (nx+1)*(ny+1);
    float64 *vals = new float64[npts];

    float dx = 20.0 / float64(nx+1);
    float dy = 20.0 / float64(ny+1);
    
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
    
    res["association"] = "point";
    res["type"] = "scalar";
    res["values"].set(vals,npts);
    
    delete [] vals;
    
}

//---------------------------------------------------------------------------//
void braid_init_example_ele_scalar_field(index_t nx,
                                         index_t ny,
                                         index_t nz,
                                         Node &res)
{
    index_t nele = nx*ny;
    float64 *vals = new float64[nele];

    float dx = 20.0 / float64(nx);
    float dy = 20.0 / float64(ny);
    
    index_t idx = 0;
    
    for(index_t j = 0; j < ny ; j++)
    {
        float64 cy =  (j * dy) + -10.0;
        for(index_t i = 0; i < nx ; i++)
        {
            float64 cx =  (i * dx) + -10.0;
            vals[idx] = 10.0 * sqrt( cx*cx + cy*cy );
            idx++;
        }
    }
    
    res["association"] = "element";
    res["type"] = "scalar";
    res["values"].set(vals,nele);
    
    delete [] vals;

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
    dims["x"] = nx;
    dims["y"] = ny;
        
    // -10 to 10 in each dim, 
    Node &origin = res["coords/uniform/origin"];
    origin["x"] = -10.0;
    origin["y"] = -10.0;
    // skip z for now
    Node &spacing = res["coords/uniform/spacing"];
    spacing["x"] = 20.0 / (float64)(nx+1);
    spacing["y"] = 20.0 / (float64)(ny+1);
    // skip z for now
    res["topology/logical"] = "coords"; // or name?
    
    Node &fields = res["fields"];

    braid_init_example_pt_scalar_field(nx,ny,nz,fields["braid_pc"]);
    braid_init_example_ele_scalar_field(nx,ny,nz,fields["radial_ec"]);
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
