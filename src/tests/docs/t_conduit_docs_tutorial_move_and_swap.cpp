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
TEST(conduit_tutorial, move)
{
    BEGIN_EXAMPLE("move");
    
    Node n_a, n_b, info;
    // setup initial data
    n_b["path/to/data"] = 42;

    std::cout << "- Before Move -" << std::endl;
    std::cout << "n_a contents:" << std::endl;
    n_a.print();
    std::cout << "n_a memory details:" << std::endl;
    n_a.info(info);
    info.print();

    std::cout << "n_b contents:" << std::endl;
    n_b.print();
    std::cout << "n_b memory details:" << std::endl;
    n_b.info(info);
    info.print();

    // execute the move
    n_a.move(n_b);

    std::cout << "- After Move -" << std::endl;
    std::cout << "n_a contents:" << std::endl;
    n_a.print();
    std::cout << "n_a memory details:" << std::endl;
    n_a.info(info);
    info.print();

    std::cout << "n_b contents:" << std::endl;
    n_b.print();
    std::cout << "n_b memory details:" << std::endl;
    n_b.info(info);
    info.print();

    END_EXAMPLE("move");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, swap)
{
    BEGIN_EXAMPLE("swap");
    
    Node n_a, n_b, info;
    // setup initial data
    n_a["data"] = 10;
    n_b["path/to/data"] = 20;

    std::cout << "- Before Swap -" << std::endl;
    std::cout << "n_a contents:" << std::endl;
    n_a.print();
    std::cout << "n_a memory details:" << std::endl;
    n_a.info(info);
    info.print();

    std::cout << "n_b contents:" << std::endl;
    n_b.print();
    std::cout << "n_b memory details:" << std::endl;
    n_b.info(info);
    info.print();

    n_a.swap(n_b);

    std::cout << "- After Swap -" << std::endl;
    std::cout << "n_a contents:" << std::endl;
    n_a.print();
    std::cout << "n_a memory details:" << std::endl;
    n_a.info(info);
    info.print();

    std::cout << "n_b contents:" << std::endl;
    n_b.print();
    std::cout << "n_b memory details:" << std::endl;
    n_b.info(info);
    info.print();

    END_EXAMPLE("swap");
}




