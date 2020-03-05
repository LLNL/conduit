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
#include "conduit_blueprint_sarray_examples.hpp"


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
// -- begin conduit::blueprint::sarray --
//-----------------------------------------------------------------------------

namespace sarray
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::sarray::examples --
//-----------------------------------------------------------------------------

namespace examples
{

//---------------------------------------------------------------------------//
void
sparse_volfrac_matmajor(index_t x, // length of a side
                        index_t y, // length of the other side
                        Node &out)
{
    index_t quarter_x = x / 4;
    index_t half_x = x / 2;
    index_t rest_x = x - half_x;
    index_t half_y = y / 2;
    index_t rest_y = y - half_y;

    index_t nvals = x*y;
    index_t avals = x*half_y;
    index_t bvals = half_x*rest_y + quarter_x*half_y;
    index_t cvals = rest_x*rest_y;

    out.reset();

    // Material A  ----------------------------------
    // Fills the bottom half of the grid; some mixed with B.
    Node a;

    {
      a["n"].set(avals);
      a["nz"].set(DataType::float64(avals));
      a["idx"].set(DataType::int32(avals));

      float64_array avals_a = a["nz"].value();
      int32_array aidx_a = a["idx"].value();

      index_t i = 0;
      for (index_t ly = 0; ly < half_y; ++ly)
      {
        for (index_t lx = 0; lx < x; ++lx)
        {
          if (lx < quarter_x)
          {
            avals_a[i] = 0.5;
          }
          else
          {
            avals_a[i] = 1.0;
          }
          aidx_a[i] = ly*x + lx;

          i += 1;
        }
      }
    }

    out["a"].set(a);

    // Material B  ----------------------------------
    // Fills the top left quarter of the grid; some mixed
    // with A in the bottom-left corner.
    Node b;

    {
      b["n"].set(bvals);
      b["nz"].set(DataType::float64(bvals));
      b["idx"].set(DataType::int32(bvals));

      float64_array bvals_a = b["nz"].value();
      int32_array bidx_a = b["idx"].value();

      index_t i = 0;
      for (index_t ly = 0; ly < half_y; ++ly)
      {
        for (index_t lx = 0; lx < quarter_x; ++lx)
        {
          bvals_a[i] = 0.5;
          bidx_a[i] = ly*x + lx;

          i += 1;
        }
      }

      for (index_t ly = half_y; ly < y; ++ly)
      {
        for (index_t lx = 0; lx < half_x; ++lx)
        {
          bvals_a[i] = 1.0;
          bidx_a[i] = ly*x + lx;

          i += 1;
        }
      }
    }

    out["b"].set(b);

    // Material C  ----------------------------------
    // Fills the top right quarter of the grid.
    Node c;
    
    {
      c["n"].set(cvals);
      c["nz"].set(DataType::float64(cvals));
      c["idx"].set(DataType::int32(cvals));

      float64_array cvals_a = b["nz"].value();
      int32_array cidx_a = b["idx"].value();

      index_t i = 0;
      for (index_t ly = half_y; ly < y; ++ly)
      {
        for (index_t lx = half_x; lx < x; ++lx)
        {
          cvals_a[i] = 1.0;
          cidx_a[i] = ly*x + lx;

          i += 1;
        }
      }
    }

    out["c"].set(c);
}

//---------------------------------------------------------------------------//
void
full_volfrac_matmajor(index_t x, // length of a side
                      index_t y, // length of the other side
                      Node &out)
{
    index_t quarter_x = x / 4;
    index_t half_x = x / 2;
    index_t rest_x = x - half_x;
    index_t half_y = y / 2;
    index_t rest_y = y - half_y;

    index_t nvals = x*y;
    index_t avals = x*half_y;
    index_t bvals = half_x*rest_y + quarter_x*half_y;
    index_t cvals = rest_x*rest_y;

    out.reset();

    out["a"].set(DataType::float64(nvals));
    out["b"].set(DataType::float64(nvals));
    out["c"].set(DataType::float64(nvals));

    float64_array a_a = out["a"].value();
    float64_array b_a = out["b"].value();
    float64_array c_a = out["c"].value();

    for (index_t i = 0; i < nvals; ++i)
    {
      a_a[i] = 0;
      b_a[i] = 0;
      c_a[i] = 0;
    }

    // Material A ----------------------------------------
    // Fills the bottom half of the grid; some mixed with B.
    for (index_t ly = 0; ly < half_y; ++ly)
    {
      for (index_t lx = 0; lx < x; ++lx)
      {
        if (lx < quarter_x)
        {
          a_a[ly*x + lx] = 0.5;
        }
        else
        {
          a_a[ly*x + lx] = 1.0;
        }
      }
    }

    // Material B ----------------------------------------
    // Fills the top left quarter of the grid; some mixed
    // with A in the bottom-left corner.
    for (index_t ly = 0; ly < half_y; ++ly)
    {
      for (index_t lx = 0; lx < quarter_x; ++lx)
      {
          b_a[ly*x + lx] = 0.5;
      }
    }
    for (index_t ly = half_y; ly < y; ++ly)
    {
      for (index_t lx = 0; lx < half_x; ++lx)
      {
          b_a[ly*x + lx] = 1.0;
      }
    }

    // Material C ----------------------------------------
    // Fills the top right quarter of the grid.
    for (index_t ly = half_y; ly < y; ++ly)
    {
      for (index_t lx = half_x; lx < x; ++lx)
      {
          c_a[ly*x + lx] = 1.0;
      }
    }
}

//---------------------------------------------------------------------------//
void
sparse_eye(index_t x, // length of a side
           Node &out)
{
    out.reset();

    out["n"].set(x*x);
    out["nz"].set(DataType::float64(x));
    out["idx"].set(DataType::int32(x));

    float64_array nz_a = out["nz"].value();
    int32_array idx_a = out["idx"].value();

    for (index_t i = 0; i < x; ++i)
    {
      nz_a[i] = 1.0;
      idx_a[i] = i*x + i;
    }
}

//---------------------------------------------------------------------------//
void
full_eye(index_t x, // length of a side
         Node &out)
{
    out.reset();
    
    // create x-by-x array

    index_t nvals = x * x;
    out.set(DataType::float64(nvals));
    float64_array out_a = out.value();

    for (index_t i = 0; i < nvals; ++i)
    {
      out_a[i] = 0.0;
    }

    for (index_t i = 0; i < x; ++i)
    {
      out_a[i*x + i] = 1.0;
    }
}

//---------------------------------------------------------------------------//
void
sparse(const std::string &array_type,
       index_t x, // X-dimension size
       index_t y, // Y-dimension size
       Node &out)
{

    if(array_type == "eye")
    {
        sparse_eye(x,out);
    }
    else if(array_type == "volfrac_matmajor")
    {
        sparse_volfrac_matmajor(x, y, out);
    }
    else
    {
        CONDUIT_ERROR("unknown array_type = " << array_type);
    }
}


//---------------------------------------------------------------------------//
void
full(const std::string &array_type,
     index_t x, // X-dimension size
     index_t y, // Y-dimension size
     Node &out)
{

    if(array_type == "eye")
    {
        full_eye(x,out);
    }
    else if(array_type == "volfrac_matmajor")
    {
        full_volfrac_matmajor(x, y, out);
    }
    else
    {
        CONDUIT_ERROR("unknown array_type = " << array_type);
    }
}



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::sarray::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::sarray --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
