//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_blueprint_mcarray_examples.cpp
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
#include "conduit_blueprint_mcarray_examples.hpp"


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
// -- begin blueprint::mcarray --
//-----------------------------------------------------------------------------

namespace mcarray
{

//-----------------------------------------------------------------------------
// -- begin blueprint::mcarray::examples --
//-----------------------------------------------------------------------------

namespace examples
{

//---------------------------------------------------------------------------//
void
xyz_interleaved(index_t nvals, // total number of "tuples"
                Node &res)
{
    res.reset();
    
    // create x,y,z
    
    index_t stride = sizeof(conduit::float64) * 3;
    Schema s;
    index_t size = sizeof(conduit::float64);
    s["x"].set(DataType::float64(nvals,0,stride));
    s["y"].set(DataType::float64(nvals,size,stride));
    s["z"].set(DataType::float64(nvals,size*2,stride));
    
    // init the output
    res.set(s);
    
    float64_array x_a = res["x"].value();
    float64_array y_a = res["y"].value();
    float64_array z_a = res["z"].value();
    
    for(index_t i=0;i<nvals;i++)
    {
        x_a[i] = 1.0;
        y_a[i] = 2.0;
        z_a[i] = 3.0;
    }
}

//---------------------------------------------------------------------------//
void
xyz_separate(index_t nvals, // total number of "tuples"
             Node &res)
{
    res.reset();

    res["x"].set(DataType::float64(nvals));
    res["y"].set(DataType::float64(nvals));
    res["z"].set(DataType::float64(nvals));
    
    float64_array x_a = res["x"].value();
    float64_array y_a = res["y"].value();
    float64_array z_a = res["z"].value();
    
    for(index_t i=0;i<nvals;i++)
    {
        x_a[i] = 1.0;
        y_a[i] = 2.0;
        z_a[i] = 3.0;
    }
}

//---------------------------------------------------------------------------//
void
xyz_contiguous(index_t nvals, // total number of "tuples"
               Node &res)
{
    res.reset();
    
    // create x,y,z
    
    index_t offset = 0;
    Schema s;
    s["x"].set(DataType::float64(nvals));
    offset += s["x"].dtype().strided_bytes();
    s["y"].set(DataType::float64(nvals,offset));
    offset += s["y"].dtype().strided_bytes();
    s["z"].set(DataType::float64(nvals,offset));
    
    // init the output
    res.set(s);
    
    float64_array x_a = res["x"].value();
    float64_array y_a = res["y"].value();
    float64_array z_a = res["z"].value();
    
    for(index_t i=0;i<nvals;i++)
    {
        x_a[i] = 1.0;
        y_a[i] = 2.0;
        z_a[i] = 3.0;
    }
}

//---------------------------------------------------------------------------//
void
xyz_interleaved_mixed_type(index_t nvals, // total number of "tuples"
                           Node &res)
{
    res.reset();
    
    // create x,y,z
    
    index_t stride = sizeof(conduit::float32);
    stride += sizeof(conduit::float64);
    stride += sizeof(conduit::uint8);
    Schema s;
    s["x"].set(DataType::float32(nvals,0,stride));
    index_t offset = sizeof(conduit::float32);
    s["y"].set(DataType::float64(nvals,offset,stride));
    offset=  sizeof(conduit::float32) + sizeof(conduit::float64);
    s["z"].set(DataType::uint8(nvals,offset,stride));

    // init the output
    res.set(s);
    
    float32_array x_a = res["x"].value();
    float64_array y_a = res["y"].value();
    uint8_array z_a = res["z"].value();
    
    for(index_t i=0;i<nvals;i++)
    {
        x_a[i] = 1.0;
        y_a[i] = 2.0;
        z_a[i] = 3;
    }
    
}

//---------------------------------------------------------------------------//
void
xyz(const std::string &mcarray_type,
    index_t npts, // total number of points
    Node &res)
{

    if(mcarray_type == "interleaved")
    {
        xyz_interleaved(npts,res);
    }
    else if(mcarray_type == "separate")
    {
        xyz_separate(npts,res);
    }
    else if(mcarray_type == "contiguous")
    {
        xyz_contiguous(npts,res);
    }
    else if(mcarray_type == "interleaved_mixed")
    {
        xyz_interleaved_mixed_type(npts,res); 
    }
    else
    {
        CONDUIT_ERROR("unknown mcarray_type = " << mcarray_type);
    }
}



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mcarray::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mcarray --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
