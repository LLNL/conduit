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
TEST(conduit_docs, relay_io_example_1_json)
{
    BEGIN_EXAMPLE("relay_io_example_1_json");
    // setup node to save
    Node n;
    n["a/my_data"] = 1.0;
    n["a/b/my_string"] = "value";
    std::cout << "\nNode to write:" << std::endl;
    n.print();
    
    //save to json using save
    conduit::relay::io::save(n,"my_output.json");
    
    //load back from json using load
    Node n_load;
    conduit::relay::io::load("my_output.json",n_load);
    std::cout << "\nLoad result:" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_1_json");
}


#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
//-----------------------------------------------------------------------------
TEST(conduit_docs, relay_io_example_1_hdf5)
{
    BEGIN_EXAMPLE("relay_io_example_1_hdf5");
    // setup node to save
    Node n;
    n["a/my_data"] = 1.0;
    n["a/b/my_string"] = "value";
    std::cout << "\nNode to write:" << std::endl;
    n.print();

    //save to hdf5 using save
    conduit::relay::io::save(n,"my_output.hdf5");

    //load back from hdf5 using load
    Node n_load;
    conduit::relay::io::load("my_output.hdf5",n_load);
    std::cout << "\nLoad result:" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_1_hdf5");
}


//-----------------------------------------------------------------------------
TEST(conduit_docs, relay_io_example_2_hdf5)
{
    BEGIN_EXAMPLE("relay_io_example_2_hdf5");
    // setup node to save
    Node n;
    n["a/my_data"] = 1.0;
    n["a/b/my_string"] = "value";
    std::cout << "\nNode to write:" << std::endl;
    n.print();

    //save to hdf5 using save
    conduit::relay::io::save(n,"my_output.hdf5");

    // append a new path to the hdf5 file using save_merged
    Node n2;
    n2["a/b/new_data"] = 42.0;
    std::cout << "\nNode to append:" << std::endl;
    n2.print();
    conduit::relay::io::save_merged(n2,"my_output.hdf5");

    Node n_load;
    // load back from hdf5 using load:
    conduit::relay::io::load("my_output.hdf5",n_load);
    std::cout << "\nLoad result:" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_2_hdf5");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, relay_io_example_3_hdf5)
{
    BEGIN_EXAMPLE("relay_io_example_3_hdf5");
    // setup node to save
    Node n;
    n["a/my_data"] = 1.0;
    n["a/b/my_string"] = "value";
    std::cout << "\nNode to write:" << std::endl;
    n.print();

    //save to hdf5 using generic i/o save
    conduit::relay::io::save(n,"my_output.hdf5");

    // append to existing node with data from hdf5 file using load_merged
    Node n_load;
    n_load["a/b/new_data"] = 42.0;
    std::cout << "\nNode to load into:" << std::endl;
    n_load.print();
    conduit::relay::io::load_merged("my_output.hdf5",n_load);
    std::cout << "\nLoad result:" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_3_hdf5");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, relay_io_example_4_hdf5)
{
    BEGIN_EXAMPLE("relay_io_example_4_hdf5");
    // setup node to save
    Node n;
    n["path/to/my_data"] = 1.0;
    std::cout << "\nNode to write:" << std::endl;
    n.print();

    //save to hdf5 using generic i/o save
    conduit::relay::io::save(n,"my_output.hdf5");

    // load only a subset of the tree
    Node n_load;
    conduit::relay::io::load("my_output.hdf5:path/to",n_load);
    std::cout << "\nLoad result from 'path/to'" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_4_hdf5");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, relay_io_example_5_hdf5)
{
    BEGIN_EXAMPLE("relay_io_example_5_hdf5");
    // setup node to save
    Node n;
    n["my_data"] = 1.0;
    std::cout << "\nNode to write to 'path/to':" << std::endl;
    n.print();

    //save to hdf5 using generic i/o save
    conduit::relay::io::save(n,"my_output.hdf5:path/to");

    // load only a subset of the tree
    Node n_load;
    conduit::relay::io::load("my_output.hdf5",n_load);
    std::cout << "\nLoad result:" << std::endl;
    n_load.print();
    END_EXAMPLE("relay_io_example_5_hdf5");
}



#endif
