// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_conduit.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_blueprint, about)
{
    conduit_node *n = conduit_node_create();
    
    conduit_about(n);
    
    conduit_node_print(n);
    conduit_node_destroy(n);
}

