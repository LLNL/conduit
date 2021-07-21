// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_intro_talk_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(conduit_intro_bp_and_relay, basics)
{
    #include <conduit.hpp>
    #include <conduit_blueprint.hpp>
    #include <conduit_relay.hpp>

    // create a "Node", the primary object in conduit
    conduit::Node n;
    // init our Node hierarchy with a few data arrays
    n["coordsets/coords/values/x"] = {0.0,1.0,2.0};
    n["coordsets/coords/values/y"] = {0.0,1.0,2.0};
    n["fields/density/values"] = {1.0,1.0,1.0,1.0};
    n.print();

    //--------//

    // you can mix external and conduit owned data within a
    // Node hierarchy
    std::vector<conduit::float64> vel_u(9,1.0);
    std::vector<conduit::float64> vel_v(9,1.0);

    // use Node::external to init the "u" and "v" nodes of the
    // tree to point to the same memory location of the source vectors.
    n["fields/velocity/values/u"].set_external(vel_u);
    n["fields/velocity/values/v"].set_external(vel_v);

    // show the elements of the "u" array
    n["fields/velocity/values/u"].print();

    // change the first element of the u array (via the vector)

    vel_u[0] = 3.14159;

    // show the elements of the "u" array again
    n["fields/velocity/values"].print();

    // mixed ownership semantics allow you to organize,
    // extend, and annotate existing data
    n["coordsets/coords/type"]  = "rectilinear";
    n["fields/density/units"]  = "g/cc";
    n["fields/velocity/units"] = "m/s";

    n.print();

    //--------//

    // extend our example to a blueprint compliant mesh
    n["topologies/topo/type"]  = "rectilinear";
    n["topologies/topo/coordset"]  = "coords";

    n["fields/density/association"]  = "element";
    n["fields/density/topology"] = "topo";

    n["fields/velocity/association"] = "vertex";
    n["fields/velocity/topology"] = "topo";

    n.print();

    conduit::Node info;
    if(conduit::blueprint::mesh::verify(n,info))
    {
        std::cout << "Mesh Blueprint Verify Success!" << std::endl;
    }
    else
    {
        std::cout << "Mesh Blueprint Verify Failure!" << std::endl;
        info.print();
    }

    conduit::relay::io::blueprint::save_mesh(n,"my_mesh_yaml","yaml");
}
