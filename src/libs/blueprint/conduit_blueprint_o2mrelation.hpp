// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2mrelation.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_O2M_HPP
#define CONDUIT_BLUEPRINT_O2M_HPP

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
// blueprint protocol interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);

//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const std::string &protocol,
                                  const conduit::Node &n,
                                  conduit::Node &info);

//-----------------------------------------------------------------------------
/// o2mrelation blueprint property/query/transform methods
///
/// These methods can be called on any verified o2mrelation
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::vector<std::string> CONDUIT_BLUEPRINT_API data_paths(const conduit::Node &o2mrelation);

//-----------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API compact_to(const conduit::Node &o2mrelation,
                                      conduit::Node &res);

//-----------------------------------------------------------------------------
/// o2mrelation blueprint miscellaneous methods
///
/// These methods can be called on unverified o2mrelation to bring them
/// into compliance.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API generate_offsets(conduit::Node &n,
                                            conduit::Node &info);

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif 



