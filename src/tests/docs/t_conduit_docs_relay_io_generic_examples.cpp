//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_blueprint_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
// 65-78
TEST(conduit_docs, relay_io_example_1_json)
{
    CONDUIT_INFO("relay_io_example_1_json");

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
    
    CONDUIT_INFO("relay_io_example_1_json");
}


#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
//-----------------------------------------------------------------------------
// 91-107
TEST(conduit_docs, relay_io_example_1_hdf5)
{
    CONDUIT_INFO("relay_io_example_1_hdf5");

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
    
    CONDUIT_INFO("relay_io_example_1_hdf5");
}


//-----------------------------------------------------------------------------
// 118-140
TEST(conduit_docs, relay_io_example_2_hdf5)
{
    CONDUIT_INFO("relay_io_example_2_hdf5");

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

    CONDUIT_INFO("relay_io_example_2_hdf5");
}

//-----------------------------------------------------------------------------
// 150-168
TEST(conduit_docs, relay_io_example_3_hdf5)
{
    CONDUIT_INFO("relay_io_example_3_hdf5");

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
    
    CONDUIT_INFO("relay_io_example_3_hdf5");
}


//-----------------------------------------------------------------------------
// 179-193
TEST(conduit_docs, relay_io_example_4_hdf5)
{
    CONDUIT_INFO("relay_io_example_4_hdf5");

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
    
    CONDUIT_INFO("relay_io_example_4_hdf5");
}

//-----------------------------------------------------------------------------
// 203-217
TEST(conduit_docs, relay_io_example_5_hdf5)
{
    CONDUIT_INFO("relay_io_example_5_hdf5");

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
    
    CONDUIT_INFO("relay_io_example_4_hdf5");
}



#endif
