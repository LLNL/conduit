// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
            res_indices[datum_idx] = (uint32)(many_idx + noffset *
                ( !rev_indices ? one_idx : nones - one_idx - 1 ));
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
            res_sizes[one_idx] = (uint32) nmany;
        }

        res["offsets"].set(DataType::uint32(nones));
        uint32_array res_offsets = res["offsets"].value();
        for(index_t one_idx = 0; one_idx < nones; one_idx++)
        {
            res_offsets[one_idx] =(uint32)(use_indices ? one_idx * nmany : one_idx * noffset);
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
