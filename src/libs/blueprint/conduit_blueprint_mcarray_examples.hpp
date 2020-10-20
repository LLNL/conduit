// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mcarray_examples.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MCARRAY_EXAMPLES_HPP
#define CONDUIT_BLUEPRINT_MCARRAY_EXAMPLES_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mcarray --
//-----------------------------------------------------------------------------

namespace mcarray
{

//-----------------------------------------------------------------------------
/// Methods that generate example multi-component arrays.
//-----------------------------------------------------------------------------
namespace examples
{
    //-------------------------------------------------------------------------
    /// creates mcarray with num pts * 3 components. 
    /// with the following layout options (passed via mcarray_type)
    ///  interleaved
    ///  separate
    ///  contiguous
    ///  interleaved_mixed
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API xyz(const std::string &mcarray_type,
                                   conduit::index_t npts, // total # of points
                                   conduit::Node &res);

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
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
// -- end conduit --
//-----------------------------------------------------------------------------


#endif 



