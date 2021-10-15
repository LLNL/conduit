// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_table_examples.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_TABLE_EXAMPLES_HPP
#define CONDUIT_BLUEPRINT_TABLE_EXAMPLES_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"


//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::table --
//-----------------------------------------------------------------------------

namespace table
{

//-----------------------------------------------------------------------------
/// Methods that generate example tables.
//-----------------------------------------------------------------------------
namespace examples
{

//-------------------------------------------------------------------------
/// Creates a table containing an mcarray of "points" (x,y,z) and a scalar
/// field of "point_data". "point_data" starts from 0 and increments by 1
/// down each row. "points" are generated as if they came from a uniform grid.
//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API basic(conduit::index_t nx,
                                 conduit::index_t ny,
                                 conduit::index_t nz,
                                 conduit::Node &res);

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::table::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::table --
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
