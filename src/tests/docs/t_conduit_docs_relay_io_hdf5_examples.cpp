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
/// file: t_conduit_docs_relay_io_hdf5_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_io_hdf5.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
// 91-107
TEST(conduit_docs, relay_io_example_hdf5_interface_1)
{
    CONDUIT_INFO("relay_io_example_hdf5_interface_1");

    // setup node to save
    Node n;
    n["a/my_data"] = 1.0;
    n["a/b/my_string"] = "value";
    std::cout << "\nNode to write:" << std::endl;
    n.print();
    
    // open hdf5 file and obtain a handle
    hid_t h5_id = conduit::relay::io::hdf5_create_file("myoutput.hdf5");
    
    // write data 
    conduit::relay::io::hdf5_write(n,h5_id);

    // close our file
    conduit::relay::io::hdf5_close_file(h5_id);
        
    // open our file to read
    h5_id = conduit::relay::io::hdf5_open_file_for_read_write("myoutput.hdf5");

    // check if a subpath exists
    if(conduit::relay::io::hdf5_has_path(h5_id,"a/my_data"))
        std::cout << "\nPath 'myoutput.hdf5:a/my_data' exists" << std::endl;
        
    Node n_read;
    // read a subpath (Note: read works like `load_merged`)
    conduit::relay::io::hdf5_read(h5_id,"a/my_data",n_read);
    std::cout << "\nData loaded:" << std::endl;
    n_read.print();
    
    // write more data to the file
    n.reset();
    // write data (appends data, works like `save_merged`)
    // the Node tree needs to be compatible with the existing
    // hdf5 state, adding new paths is always fine.  
    n["a/my_data"] = 3.1415;
    n["a/b/c"] = 144;
    conduit::relay::io::hdf5_write(n,h5_id);
    
    // Read the entire tree:
    n_read.reset();
    conduit::relay::io::hdf5_read(h5_id,n_read);
    std::cout << "\nData loaded:" << std::endl;
    n_read.print();
    
    // other helpers:
    
    // check if a path is a hdf5 file:
    if(conduit::relay::io::is_hdf5_file("myoutput.hdf5"))
        std::cout << "File \n'myoutput.hdf5' is a hdf5 file" << std::endl;

    CONDUIT_INFO("relay_io_example_hdf5_interface_1");
}



//-----------------------------------------------------------------------------
// 128-157
TEST(conduit_docs, relay_io_example_hdf5_interface_2)
{
    
    CONDUIT_INFO("relay_io_example_hdf5_interface_opts");
    
    Node io_about;
    conduit::relay::io::about(io_about);
    std::cout << "\nRelay I/O Info and Default Options:" << std::endl;
    io_about.print();
    
    Node &hdf5_opts = io_about["options/hdf5"];
    // change the default chunking threshold to 
    // a smaller number to enable compression for
    // a small array
    hdf5_opts["chunking/threshold"]  = 2000;
    hdf5_opts["chunking/chunk_size"] = 2000;

    std::cout << "\nNew HDF5 I/O Options:" << std::endl;
    hdf5_opts.print();
    // set options
    conduit::relay::io::hdf5_set_options(hdf5_opts);
        
    int num_vals = 5000;
    Node n;
    n["my_values"].set(DataType::float64(num_vals));
    
    float64 *v_ptr = n["my_values"].value();
    for(int i=0; i< num_vals; i++)
    {
        v_ptr[i] = float64(i);
    }
    

    // save using options
    std::cout << "\nsaving data to 'myoutput_chunked.hdf5' " << std::endl;
    
    conduit::relay::io::hdf5_save(n,"myoutput_chunked.hdf5");

    CONDUIT_INFO("relay_io_example_hdf5_interface_opts");
}
