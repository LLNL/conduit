// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_numeric.cpp
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
TEST(conduit_tutorial, numeric_as_dtype)
{
    BEGIN_EXAMPLE("numeric_as_dtype");
    Node n;
    int64 val = 100;
    n = val;
    std::cout << n.as_int64() << std::endl;
    END_EXAMPLE("numeric_as_dtype");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_via_value)
{
    BEGIN_EXAMPLE("numeric_via_value");
    Node n;
    int64 val = 100;
    n = val;
    int64 my_val = n.value();
    std::cout << my_val << std::endl;
    END_EXAMPLE("numeric_via_value");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_ptr_as_dtype)
{
    BEGIN_EXAMPLE("numeric_ptr_as_dtype");
    int64 vals[4] = {100,200,300,400};

    Node n;
    n.set(vals,4);

    int64 *my_vals = n.as_int64_ptr();

    for(index_t i=0; i < 4; i++)
    {
        std::cout << "my_vals[" << i << "] = " << my_vals[i] << std::endl;
    }
    END_EXAMPLE("numeric_ptr_as_dtype");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_ptr_via_value)
{
    BEGIN_EXAMPLE("numeric_ptr_via_value");
    int64 vals[4] = {100,200,300,400};

    Node n;
    n.set(vals,4);

    int64 *my_vals = n.value();

    for(index_t i=0; i < 4; i++)
    {
        std::cout << "my_vals[" << i << "] = " << my_vals[i] << std::endl;
    }
    END_EXAMPLE("numeric_ptr_via_value");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_strided_data_array)
{ 
    BEGIN_EXAMPLE("numeric_strided_data_array");
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
    END_EXAMPLE("numeric_strided_data_array");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_double_conversion_start)
{
    BEGIN_EXAMPLE("numeric_double_conversion");
}

// _conduit_tutorial_cpp_numeric_introspection_start
//-----------------------------------------------------------------------------
void must_have_doubles_function(double *vals,index_t num_vals)
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
    index_t num_vals = res.dtype().number_of_elements();
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
// _conduit_tutorial_cpp_numeric_introspection_end

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, numeric_double_conversion_end)
{
    END_EXAMPLE("numeric_double_conversion");
}

