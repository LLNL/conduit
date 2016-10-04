//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: conduit_node.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_very_basic)
{
    CONDUIT_INFO("basics_very_basic");
        
    Node n;
    n["my"] = "data";
    n.print(); 
    
    CONDUIT_INFO("basics_very_basic");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_hierarchial)
{
    CONDUIT_INFO("basics_hierarchial");
    
    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;
    n.print();
    
    std::cout << "total bytes: " << n.total_bytes() << std::endl;

    CONDUIT_INFO("basics_hierarchial");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_object_and_list)
{
    CONDUIT_INFO("basics_object_and_list");
    
    Node n;
    n["object_example/val1"] = "data";
    n["object_example/val2"] = 10u;
    n["object_example/val3"] = 3.1415;
    
    for(int i = 0; i < 5 ; i++ )
    {
        Node &list_entry = n["list_example"].append();
        list_entry.set(i);
    }
    
    n.print();

    CONDUIT_INFO("basics_object_and_list");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_mem_spaces)
{
    CONDUIT_INFO("basics_mem_spaces");

    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;

    Node ninfo;
    n.info(ninfo);
    ninfo.print();

    CONDUIT_INFO("basics_mem_spaces");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_bw_style)
{
    CONDUIT_INFO("basics_bw_style");
    
    Node n;
    uint32 val = 100;
    n["test"] = val;
    n.print();
    n.print_detailed();

    CONDUIT_INFO("basics_bw_style");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_bw_style_from_native)
{
    CONDUIT_INFO("basics_bw_style_from_native");
        
    Node n;
    int val = 100;
    n["test"] = val;
    n.print_detailed();

    CONDUIT_INFO("basics_bw_style_from_native");
}


//-----------------------------------------------------------------------------
// 150-155
TEST(conduit_tutorial, numeric_as_dtype)
{
    CONDUIT_INFO("numeric_as_dtype");
        
    Node n;
    int64 val = 100;
    n = val;
    std::cout << n.as_int64() << std::endl;

    CONDUIT_INFO("numeric_as_dtype");
}

//-----------------------------------------------------------------------------
// 163-168
TEST(conduit_tutorial, numeric_via_value)
{
    CONDUIT_INFO("numeric_via_value");
    
    Node n;
    int64 val = 100;
    n = val;
    int64 my_val = n.value();
    std::cout << my_val << std::endl;

    CONDUIT_INFO("numeric_via_value");
}


//-----------------------------------------------------------------------------
// 177-188
TEST(conduit_tutorial, numeric_ptr_as_dtype)
{
    CONDUIT_INFO("numeric_ptr_as_dtype");
    
    int64 vals[4] = {100,200,300,400};

    Node n;
    n.set(vals,4);

    int64 *my_vals = n.as_int64_ptr();

    for(index_t i=0; i < 4; i++)
    {
        std::cout << "my_vals[" << i << "] = " << my_vals[i] << std::endl;
    }

    CONDUIT_INFO("numeric_ptr_as_dtype");
}

//-----------------------------------------------------------------------------
// 195-207
TEST(conduit_tutorial, numeric_ptr_via_value)
{
    CONDUIT_INFO("numeric_ptr_via_value");
     
    int64 vals[4] = {100,200,300,400};

    Node n;
    n.set(vals,4);

    int64 *my_vals = n.value();

    for(index_t i=0; i < 4; i++)
    {
        std::cout << "my_vals[" << i << "] = " << my_vals[i] << std::endl;
    }
    
    CONDUIT_INFO("numeric_ptr_via_value");
}

//-----------------------------------------------------------------------------
// 195-207
TEST(conduit_tutorial, numeric_strided_data_array)
{ 
    CONDUIT_INFO("numeric_strided_data_array");
    
    int64 vals[4] = {100,200,300,400};

    Node n;
    n.set(vals,2, // # of elements
               0, // offset in bytes
               sizeof(int64)*2); // stride in bytes
    
    int64_array  my_vals = n.value();
    
    for(index_t i=0; i < 2; i++)
    {
        std::cout << "my_vals[" << i << "] = " << my_vals[i] << std::endl;
    }
    
    my_vals.print();
    
    CONDUIT_INFO("numeric_strided_data_array");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_double_conversion_start)
{
    CONDUIT_INFO("numeric_double_conversion_start");
}

//-----------------------------------------------------------------------------
void must_have_doubles_function(double *vals,int num_vals)
{
    for(int i = 0; i < num_vals; i++)
    {
        std::cout << "vals[" << i << "] = " <<  vals[i] << std::endl;
    }
}

//-----------------------------------------------------------------------------
void process_doubles(Node & n)
{
    Node res;
    // We have a node that we are interested in processing with
    // and existing function that only handles doubles.

    if( n.dtype().is_double() && n.dtype().is_compact() )
    {
        std::cout << " using existing buffer" << std::endl;

        // we already have a contiguous double array
        res.set_external(n);
    }
    else
    {
        std::cout << " converting to temporary double array " << std::endl;

        // Create a compact double array with the values of the input.
        // Standard casts are used to convert each source element to
        // a double in the new array.
        n.to_double_array(res);
    }

    res.print();

    double *dbl_vals = res.value();
    int num_vals = res.dtype().number_of_elements();
    must_have_doubles_function(dbl_vals,num_vals);
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_double_conversion)
{
    
    float32 f32_vals[4] = {100.0,200.0,300.0,400.0};
    double  d_vals[4]   = {1000.0,2000.0,3000.0,4000.0};

    Node n;
    n["float32_vals"].set(f32_vals,4);
    n["double_vals"].set(d_vals,4);

    std::cout << "float32 case: " << std::endl;

    process_doubles(n["float32_vals"]);

    std::cout << "double case: " << std::endl;

    process_doubles(n["double_vals"]);
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_double_conversion_end)
{
    CONDUIT_INFO("numeric_double_conversion_end");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_std)
{
    CONDUIT_INFO("json_generator_std");
    
    Generator g("{test: {dtype: float64, value: 100.0}}","conduit_json");
    
    Node n;
    g.walk(n);
    
    std::cout << n["test"].as_float64() <<std::endl;
    n.print();
    n.print_detailed();
    
    CONDUIT_INFO("json_generator_std");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_pure_json)
{
    CONDUIT_INFO("json_generator_pure_json");
    
    Generator g("{test: 100.0}","json");

    Node n;
    g.walk(n);
        
    std::cout << n["test"].as_float64() <<std::endl;
    n.print_detailed();
    n.print();

    CONDUIT_INFO("json_generator_pure_json");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_bind_to_incore)
{
    CONDUIT_INFO("json_generator_bind_to_incore");
    
    float64 vals[2];
    Generator g("{a: {dtype: float64, value: 100.0}, b: {dtype: float64, value: 200.0} }",
                "conduit_json",
                vals);

    Node n;
    g.walk_external(n);
    
    std::cout << n["a"].as_float64() << " vs " << vals[0] << std::endl;
    std::cout << n["b"].as_float64() << " vs " << vals[1] << std::endl;

    n.print();

    Node ninfo;
    n.info(ninfo);
    ninfo.print();
    
    CONDUIT_INFO("json_generator_bind_to_incore");
}

//-----------------------------------------------------------------------------
/// TODO: This doesn't need to be in the generator section.
TEST(conduit_tutorial, json_generator_compact)
{
    CONDUIT_INFO("json_generator_compact");
    
    float64 vals[] = { 100.0,-100.0,
                       200.0,-200.0,
                       300.0,-300.0,
                       400.0,-400.0,
                       500.0,-500.0};

    // stride though the data with two different views. 
    Generator g1("{dtype: float64, length: 5, stride: 16}",
                 "conduit_json",
                 vals);
    Generator g2("{dtype: float64, length: 5, stride: 16, offset:8}",
                 "conduit_json",
                  vals);

    Node n1;
    g1.walk_external(n1);
    n1.print();

    Node n2;
    g2.walk_external(n2);
    n2.print();

    // look at the memory space info for our two views
    Node ninfo;
    n1.info(ninfo);
    ninfo.print();

    n2.info(ninfo);
    ninfo.print();

    // compact data from n1 to a new node
    Node n1c;
    n1.compact_to(n1c);

    // look at the resulting compact data
    n1c.print();
    n1c.schema().print();
    n1c.info(ninfo);
    ninfo.print();

    // compact data from n2 to a new node
    Node n2c;
    n2.compact_to(n2c);

    // look at the resulting compact data
    n2c.print();
    n2c.info(ninfo);
    ninfo.print();

    CONDUIT_INFO("json_generator_compact");
}

//-----------------------------------------------------------------------------
// 383-401
TEST(conduit_tutorial, mem_ownership_external)
{
    CONDUIT_INFO("mem_ownership_external");
     
    index_t vsize = 5;
    std::vector<float64> vals(vsize,0.0);
    for(index_t i=0;i<vsize;i++)
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

    CONDUIT_INFO("mem_ownership_external");
}


