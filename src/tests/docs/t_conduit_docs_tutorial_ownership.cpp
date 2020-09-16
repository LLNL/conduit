// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, mem_ownership_external)
{
    BEGIN_EXAMPLE("mem_ownership_external");
    int vsize = 5;
    std::vector<float64> vals(vsize,0.0);
    for(int i=0;i<vsize;i++)
    {
        vals[i] = 3.1415 * i;
    }

    Node n;
    n["v_owned"] = vals;
    n["v_external"].set_external(vals);

    n.info().print();
    
    n.print();

    vals[1] = -1 * vals[1];
    n.print();
    END_EXAMPLE("mem_ownership_external");
}




