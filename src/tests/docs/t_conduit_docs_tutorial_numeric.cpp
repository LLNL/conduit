//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: t_conduit_docs_tutorial_numeric.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
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

