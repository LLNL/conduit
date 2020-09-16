// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_HPP
#define CONDUIT_BLUEPRINT_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "conduit_blueprint_exports.h"

#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_mesh_examples.hpp"
#include "conduit_blueprint_mesh_examples_julia.hpp"
#include "conduit_blueprint_mesh_examples_venn.hpp"

#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_examples.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"

#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_mcarray_examples.hpp"

#include "conduit_blueprint_zfparray.hpp"

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
/// The about methods construct human readable info about how blueprint was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_BLUEPRINT_API about();
void        CONDUIT_BLUEPRINT_API about(conduit::Node &n);

//-----------------------------------------------------------------------------
/// blueprint verify interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint protocol.
/// Messages related to the verification are be placed in the "info" node.
//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const std::string &protocol,
                                  const conduit::Node &n,
                                  conduit::Node &info);

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



