// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2mrelation_examples.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_O2MRELATION_EXAMPLES_HPP
#define CONDUIT_BLUEPRINT_O2MRELATION_EXAMPLES_HPP

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
// -- begin conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------

namespace o2mrelation
{

//-----------------------------------------------------------------------------
/// Methods that generate example multi-component arrays.
//-----------------------------------------------------------------------------
namespace examples
{
    //-------------------------------------------------------------------------
    /// creates a one-to-many relation with a given uniform relationship size,
    /// a given uniform offset, and an index specification, which can be one of:
    ///  unspecified
    ///  default
    ///  reversed
    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API uniform(conduit::Node &res,
                                       conduit::index_t nones,
                                       conduit::index_t nmany = 0,
                                       conduit::index_t noffset = 0,
                                       const std::string &index_type = "unspecified");

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
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
// -- end conduit --
//-----------------------------------------------------------------------------


#endif 



