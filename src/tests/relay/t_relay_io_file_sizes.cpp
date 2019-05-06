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
/// file: t_relay_io_file_sizes.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"
#include <fstream>

using namespace conduit;
using namespace conduit::relay;

// optional params that can be passed on the command line
std::string input_file_name;
std::string input_protocol;

//-----------------------------------------------------------------------------
void
node_summary(const Node &n, Node &info)
{
    // capture the :
    //   total # of paths
    //   total # of leaves
    //   total # of bytes
    int64 num_paths  = 1;
    int64 num_leaves = 0;
    
    if(n.schema().dtype().is_object() || n.schema().dtype().is_list() )
    {
        NodeConstIterator itr = n.children();
        while(itr.has_next())
        {
            const Node &n_child = itr.next();
            Node n_child_info;
            node_summary(n_child,n_child_info);
            num_leaves += n_child_info["total_leaves"].to_int64();
            num_paths  += n_child_info["total_paths"].to_int64();
        }
    }
    else
    {
        num_leaves++;
    }
    
    info["total_paths"]  = num_paths;
    info["total_leaves"] = num_leaves;
    
    info["total_bytes"].set_int64(n.schema().total_bytes_compact());
}


//-----------------------------------------------------------------------------
float64
ratio_perc(int64 part, int64 whole)
{
    return 100.0 * ((float64)part) / ((float64)whole);
}
//-----------------------------------------------------------------------------
int64
check_output_size(const std::string &fname)
{
    int64 res = utils::file_size(fname);
    // Note: if we are using conduit_bin, there will also a schema file
    if(utils::is_file(fname + "_json"))
    {
        res += utils::file_size(fname + "_json");
    }
    return res;
}

//-----------------------------------------------------------------------------
void
cleanup_output(const std::string &fname)
{
    if(utils::is_file(fname))
    {
        utils::remove_file(fname);
    }

    // Note: if we are using conduit_bin, there will also a schema file
    if(utils::is_file(fname + "_json"))
    {
        utils::remove_file(fname + "_json");
    }
}


//-----------------------------------------------------------------------------
void
file_size_summary(const Node &n, const std::string &fname, Node &info)
{
    info["file_name"] = fname;
    node_summary(n,info);
    int64 dsize   = info["total_bytes"].value();
    int64 fsize = check_output_size(fname);

    info["total_file_size"] = fsize;

    int64 lsize   = info["total_leaves"].value();
    int64 psize   = info["total_paths"].value();
    info["leaves_percent_of_paths"]   = ratio_perc(lsize,psize);


    info["data_percent_of_file_size"] = ratio_perc(dsize,fsize);
}

//-----------------------------------------------------------------------------
void
detect_protocols(Node &protos)
{
    Node n_about;
    relay::about(n_about);
    NodeIterator itr = n_about["io/protocols"].children();
    while(itr.has_next())
    {
        if(itr.next().as_string() == "enabled")
        {
            protos.append().set(itr.name());
        }
    }
    
    protos.append().set("json");
    protos.append().set("conduit_json");
}


//-----------------------------------------------------------------------------
void
check_protocols(const Node &n,
                const Node &protos,
                const std::string &tag="test")
{
    NodeConstIterator itr = protos.children();
    Node info;
    while(itr.has_next())
    {
        std::string proto_name = itr.next().as_string();
        std::cout << "[writing: " << proto_name;
        std::string ofname = "tout_relay_io_file_sizes_" + tag 
                             + "_proto_" + proto_name + "." + proto_name;
        
        cleanup_output(ofname);
                    
        relay::io::save(n,ofname,proto_name);
        
        EXPECT_TRUE(utils::is_file(ofname));
        
        std::cout << " (file size: " 
                  << check_output_size(ofname) 
                   << ") ]" << std::endl;
        file_size_summary(n,ofname,info);
        info.print();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_file_sizes, test_protos)
{
    Node protos;
    detect_protocols(protos);

    // check if a specific file was passed via command line, if so
    // use that instead of dummy data
    if(input_file_name != "")
    {
        Node n;
        std::cout << "[loading: " << input_file_name << "]" << std::endl;
        if(input_protocol != "")
        {
            relay::io::load(input_file_name,input_protocol,n);
        }
        else
        {
            relay::io::load(input_file_name,n);
        }
        check_protocols(n,protos);
        return;
    }
    
    std::cout << "[creating sample data]" << std::endl;
    Node n;   
    // one float64 
    float64 fval = 42.0;
    n["test_1/value"] = fval;
  
    // array w/ 1000 float64s
    n["test_2/value"].set(DataType::float64(1000));
    
    float64_array t2_vals = n["test_2/value"].value();
    for(index_t i=0; i < t2_vals.number_of_elements(); i++)
    {
        t2_vals[i] = fval * i;
    }
    
    // one object, 10 double leaves
    n["test_3/obj/a"].set_float64(fval);
    n["test_3/obj/b"].set_float64(fval);
    n["test_3/obj/c"].set_float64(fval);
    n["test_3/obj/d"].set_float64(fval);
    n["test_3/obj/e"].set_float64(fval);
    
    n["test_3/obj/f"].set_float64(fval);
    n["test_3/obj/g"].set_float64(fval);
    n["test_3/obj/h"].set_float64(fval);
    n["test_3/obj/i"].set_float64(fval);
    n["test_3/obj/j"].set_float64(fval);
    
  
    n["test_4/obj/a"] = t2_vals;
    n["test_4/obj/b"] = t2_vals;
    n["test_4/obj/c"] = t2_vals;
    n["test_4/obj/d"] = t2_vals;
    n["test_4/obj/e"] = t2_vals;
  
    n["test_4/obj/f"] = t2_vals;
    n["test_4/obj/g"] = t2_vals;
    n["test_4/obj/h"] = t2_vals;
    n["test_4/obj/i"] = t2_vals;
    n["test_4/obj/j"] = t2_vals;
  
    NodeIterator itr = n.children();
    
    while(itr.has_next())
    {
        Node &ntest = itr.next();
        check_protocols(ntest,protos,itr.name());
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    // extended usage: t_relay_io_file_size {input file} {input protocol}
    if(argc > 1)
    {
        input_file_name = std::string(argv[1]);
    }

    if(argc > 2)
    {
        input_protocol = std::string(argv[2]);
    }


    result = RUN_ALL_TESTS();
    return result;
}

