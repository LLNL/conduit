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
/// file: conduit_blueprint_o2mrelation_examples.cpp
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
#include "conduit_blueprint_o2mrelation_examples.hpp"


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
// -- begin blueprint::o2mrelation --
//-----------------------------------------------------------------------------

namespace o2mrelation
{

//-----------------------------------------------------------------------------
// -- begin blueprint::o2mrelation::examples --
//-----------------------------------------------------------------------------

namespace examples
{

//---------------------------------------------------------------------------//
void
uniform(Node &res,
        conduit::index_t nones,
        conduit::index_t nmany,
        conduit::index_t noffset,
        const std::string &index_type)
{
    res.reset();

    bool use_sizes = true;
    { // Argument Wrangling for Sizes/Offsets //
        if(nmany == 0 && noffset != 0)
        {
            nmany = 1;
        }
        else if(nmany != 0 && noffset == 0)
        {
            noffset = nmany;
        }
        else if(nmany == 0 && noffset == 0)
        {
            nmany = noffset = 1;
            use_sizes = false;
        }

        if(noffset < nmany)
        {
            CONDUIT_ERROR("cannot construct one-to-many w/ noffset < many size");
        }
    }

    bool use_indices = true;
    bool rev_indices = false;
    { // Argument Wrangling for Indices //
        if(index_type == "unspecified")
        {
            use_indices = false;
        }
        else if(index_type == "default")
        {
            rev_indices = false;
        }
        else if(index_type == "reversed")
        {
            rev_indices = true;
        }
        else
        {
            CONDUIT_ERROR("unknown index_type = " << index_type);
        }
    }

    const index_t total_data_count = noffset * nones;
    const index_t valid_data_count = nmany * nones;

    res["data"].set(DataType::float32(total_data_count));
    float32_array res_data = res["data"].value();
    res["indices"].set(DataType::uint32(valid_data_count));
    uint32_array res_indices = res["indices"].value();

    for(index_t datum_idx = 0; datum_idx < total_data_count; datum_idx++)
    {
        res_data[datum_idx] = -1.0f;
    }
    for(index_t one_idx = 0, datum_idx = 0; one_idx < nones; one_idx++)
    {
        for(index_t many_idx = 0; many_idx < nmany; many_idx++, datum_idx++)
        {
            res_data[one_idx * noffset + many_idx] = datum_idx + 1.0f;
            res_indices[datum_idx] = many_idx + noffset *
                ( !rev_indices ? one_idx : nones - one_idx - 1 );
        }
    }

    if(!use_indices)
    {
        res.remove("indices");
    }

    if(use_sizes)
    {
        res["sizes"].set(DataType::uint32(nones));
        uint32_array res_sizes = res["sizes"].value();
        for(index_t one_idx = 0; one_idx < nones; one_idx++)
        {
            res_sizes[one_idx] = nmany;
        }

        res["offsets"].set(DataType::uint32(nones));
        uint32_array res_offsets = res["offsets"].value();
        for(index_t one_idx = 0; one_idx < nones; one_idx++)
        {
            res_offsets[one_idx] = use_indices ? one_idx * nmany : one_idx * noffset;
        }
    }
}



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
