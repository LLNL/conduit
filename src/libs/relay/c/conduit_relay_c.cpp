// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_relay.h"

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//---------------------------------------------------------------------------//
void
conduit_relay_about(conduit_node *cnode)
{
    Node *n = cpp_node(cnode);
    relay::about(*n);
}


}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

