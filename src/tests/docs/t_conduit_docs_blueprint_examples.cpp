// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_blueprint_examples.cpp
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
TEST(conduit_docs, blueprint_example_1)
{
    BEGIN_EXAMPLE("blueprint_example_1");
    // setup our candidate and info nodes
    Node n, info;

    //create an example mesh
    conduit::blueprint::mesh::examples::braid("tets",
                                               5,5,5,
                                               n);
    // check if n conforms
    if(conduit::blueprint::verify("mesh",n,info))
        std::cout << "mesh verify succeeded." << std::endl;
    else
        std::cout << "mesh verify failed!" << std::endl;

    // show some of the verify details
    info["coordsets"].print();
    END_EXAMPLE("blueprint_example_1");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_example_2)
{
    BEGIN_EXAMPLE("blueprint_example_2");
    // setup our candidate and info nodes
    Node n, verify_info, mem_info;

    // create an example mcarray
    conduit::blueprint::mcarray::examples::xyz("separate",5,n);

    std::cout << "example 'separate' mcarray " << std::endl;
    n.print();
    n.info(mem_info);
    mem_info.print();
    
    // check if n conforms
    if(conduit::blueprint::verify("mcarray",n,verify_info))
    {
        // check if our mcarray has a specific memory layout 
        if(!conduit::blueprint::mcarray::is_interleaved(n))
        {
            // copy data from n into the desired memory layout
            Node xform;
            conduit::blueprint::mcarray::to_interleaved(n,xform);
            std::cout << "transformed to 'interleaved' mcarray " << std::endl;
            xform.print_detailed();
            xform.info(mem_info);
            mem_info.print();
        }
    }
    END_EXAMPLE("blueprint_example_2");
}
