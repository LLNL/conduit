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
/// file: conduit_json.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_json, to_json_1)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    n.print_detailed();
}
//-----------------------------------------------------------------------------
TEST(conduit_json, to_json_2)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    uint32   arr[5];
    for(index_t i=0;i<5;i++)
    {
        arr[i] = i*i;
    }

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["arr"].set_external(DataType::uint32(5),arr);


    std::string pure_json = n.to_json();

    n.print_detailed();
    n.print();

    Generator g(pure_json,"json");
    Node n2(g,true);
    n2.print_detailed();
    n2.print();

    //
    // JSON parsing will place values into an int64, 
    // here we use "to_uint32" to do a direct comparsion
    //
    EXPECT_EQ(n["a"].to_int64(),n2["a"].to_int64());
    EXPECT_EQ(n["b"].to_int64(),n2["b"].to_int64());

}

//-----------------------------------------------------------------------------
TEST(conduit_json, to_json_3)
{
    std::string pure_json ="{a:[0,1,2,3,4],b:[0.0,1.1,2.2,3.3]}";
    Generator g(pure_json,"json");
    Node n(g,true);
    n.print_detailed();
}

//-----------------------------------------------------------------------------
TEST(conduit_json, json_inline_value)
{
    uint32   val=0;

    std::string schema ="{dtype: uint32, value:42}";
    // TODO: check for "unit32" , bad spelling!
    Generator g(schema);
    Node n(g,true);
    std::cout << n.as_uint32() << std::endl;
    EXPECT_EQ(42,n.as_uint32());
    
    Generator g2(schema,&val);
    Node n2(g2,true);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(42,val);
    
}

//-----------------------------------------------------------------------------
TEST(conduit_json, json_inline_array)
{
    uint32   arr[5];

    std::string schema ="{dtype:uint32, length:5, value:[0,1,2,3,4]}";
    Node n(schema,arr,true);
    n.print_detailed();
    
    uint32 *ptr = &arr[0];

    for(index_t i=0;i<5;i++)
    {
        //std::cout << arr[i] << " vs " << ptr[i] << std::endl;
        EXPECT_EQ(arr[i],ptr[i]);
    }
    
    std::string schema2 ="{dtype:uint32, value:[10,20,30]}";
    Node n2(schema2,arr,true);
    ptr =n2.as_uint32_ptr();
    n2.print_detailed();
    
    EXPECT_EQ(n2.dtype().number_of_elements(),3);
    for(index_t i=0;i<n2.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],10*(i+1));
    }
    
    std::string schema3 ="{dtype:uint32, value:[100,200,300,400,500]}";
    Node n3(schema3,arr,true);
    ptr =n3.as_uint32_ptr();
    n3.print_detailed();
    
    EXPECT_EQ(n3.dtype().number_of_elements(),5);
    for(index_t i=0;i<n3.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],100*(i+1));
    }
    
    std::string schema4 ="{dtype:uint32, value:[1000,2000]}";
    Node n4(schema4,arr,true);
    ptr =n4.as_uint32_ptr();
    n4.print_detailed();
    
    EXPECT_EQ(n4.dtype().number_of_elements(),2);
    for(index_t i=0;i<n4.dtype().number_of_elements();i++)
    {
        EXPECT_EQ(ptr[i],1000*(i+1));
    }

    // checking to make sure we are using the same memory space
    for(index_t i=2;i<5;i++)
    {
        EXPECT_EQ(ptr[i],100*(i+1));
    }
    
    
}

//-----------------------------------------------------------------------------
TEST(conduit_json, json_bool)
{
    
    std::string pure_json = "{\"value\": true}";
    Generator g(pure_json,"json");
    Node n(g,true);
    n.print_detailed();
    EXPECT_EQ(n["value"].dtype().id(),DataType::UINT8_ID);

}


//-----------------------------------------------------------------------------
TEST(conduit_json, load_from_json)
{
    
    std::string ofname = "test_conduit_schema_load_from_json.conduit_json";

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    n.schema().save(ofname);

    Schema s_dest;    
    s_dest.load(ofname);
    
    EXPECT_EQ(true,s_dest.has_path("a"));
    EXPECT_EQ(DataType::UINT32_ID,s_dest["a"].dtype().id());
    EXPECT_EQ(true,s_dest.has_path("b"));
    EXPECT_EQ(DataType::UINT32_ID,s_dest["b"].dtype().id());
    
}



//-----------------------------------------------------------------------------
TEST(conduit_json, json_explicit_offsets)
{
    
    uint32   vals[100];
    char    *vals_ptr = (char*)&vals;

    std::string schema ="{dtype: uint32, value:42, offset:8}";
    Generator g1(schema,vals_ptr);
    Node n1(g1,true);

    EXPECT_EQ(42,n1.as_uint32());
    EXPECT_EQ((char*)n1.as_uint32_ptr(),vals_ptr+8);
    
    
    schema ="{dtype: uint32, value:52, offset:16}";
    Generator g2(schema,vals_ptr);
    Node n2(g2,true);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(52,n2.as_uint32());
    EXPECT_EQ((char*)n2.as_uint32_ptr(),vals_ptr+16);
    
    
    schema ="{v1 :{dtype: uint32, offset:8}, v2: {dtype: uint32, offset:16}}";
    Generator g3(schema,vals_ptr);
    Node n3(g3,true);

    EXPECT_EQ(42,n3["v1"].as_uint32());
    EXPECT_EQ(52,n3["v2"].as_uint32());
    
    // everything should be wired up to the same pointers
    EXPECT_EQ((char*)n3["v1"].as_uint32_ptr(),vals_ptr+8);
    EXPECT_EQ((char*)n3["v1"].as_uint32_ptr(),(char*)n1.as_uint32_ptr());
    
    EXPECT_EQ((char*)n3["v2"].as_uint32_ptr(),vals_ptr+16);
    EXPECT_EQ((char*)n3["v2"].as_uint32_ptr(),(char*)n2.as_uint32_ptr());
}


//-----------------------------------------------------------------------------
TEST(conduit_json, json_c_type_names)
{
    
    int   vals[100];
    char  *vals_ptr = (char*)&vals;

    std::string schema ="{dtype: int, value:42, offset:8}";
    Generator g1(schema,vals_ptr);
    Node n1(g1,true);

    EXPECT_EQ(42,n1.as_int());
    EXPECT_EQ((char*)n1.as_int_ptr(),vals_ptr+8);
    
    
    schema ="{dtype: int, value:52, offset:16}";
    Generator g2(schema,vals_ptr);
    Node n2(g2,true);
    std::cout << n2.as_int() << std::endl;
    EXPECT_EQ(52,n2.as_int());
    EXPECT_EQ((char*)n2.as_int_ptr(),vals_ptr+16);
    
    
    schema ="{v1 :{dtype: int, offset:8}, v2: {dtype: int, offset:16}}";
    Generator g3(schema,vals_ptr);
    Node n3(g3,true);

    EXPECT_EQ(42,n3["v1"].as_int());
    EXPECT_EQ(52,n3["v2"].as_int());
    
    // everything should be wired up to the same pointers
    EXPECT_EQ((char*)n3["v1"].as_int_ptr(),vals_ptr+8);
    EXPECT_EQ((char*)n3["v1"].as_int_ptr(),(char*)n1.as_int_ptr());
    
    EXPECT_EQ((char*)n3["v2"].as_int_ptr(),vals_ptr+16);
    EXPECT_EQ((char*)n3["v2"].as_int_ptr(),(char*)n2.as_int_ptr());
}

//-----------------------------------------------------------------------------
TEST(conduit_json, json_parse_error)
{
    
    std::string schema ="{dtype: int, value:42, offset:8";
    Node n;
    Generator g(schema);
    ASSERT_THROW(g.walk(n),conduit::Error);

    try
    {
        g.walk(n);
    }
    catch(conduit::Error &e)
    {
        e.print();
    }
}



//-----------------------------------------------------------------------------
TEST(conduit_json, to_base64_json)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    uint32   arr[5];
    for(index_t i=0;i<5;i++)
    {
        arr[i] = i*i;
    }

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["arr"].set_external(DataType::uint32(5),arr);
 
    std::string base64_json = n.to_json("base64_json");
    std::cout << base64_json << std::endl;
    
    Node nparse;
    Generator g(base64_json,"base64_json");
    g.walk(nparse);

    nparse.print();
    
    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    
    uint32 *arr_vals= n["arr"].value();
    
    for(index_t i=0;i<5;i++)
    {
        EXPECT_EQ(arr_vals[i],arr[i]);
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_json, check_empty)
{
    Node n;
    n["path/to/empty"];
    n.print();
    std::string json_txt = n.to_json();
    
    CONDUIT_INFO("json:" << std::endl << json_txt);
    
    Node nparse;
    Generator g(json_txt,"json");
    g.walk(nparse);
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("conduit");
    
    CONDUIT_INFO("conduit:" << std::endl << json_txt);
    
    Generator g2(json_txt,"conduit");
    g2.walk(nparse);
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("base64_json");
    
    CONDUIT_INFO("base64_json:" << std::endl << json_txt);
    
    Generator g3(json_txt,"base64_json");
    g3.walk(nparse);
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());
}

//-----------------------------------------------------------------------------
TEST(conduit_json, check_childless_object)
{
    Node n;
    n["path/to/empty"].set(DataType::object());
    std::string json_txt = n.to_json();
    CONDUIT_INFO("json:(input)" << std::endl << json_txt);
    
    Node nparse;
    Generator g(json_txt,"json");
    g.walk(nparse);
    CONDUIT_INFO("json:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("conduit");
    
    CONDUIT_INFO("conduit:(input)" << std::endl << json_txt);
    
    Generator g2(json_txt,"conduit");
    g2.walk(nparse);
    CONDUIT_INFO("conduit:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("base64_json");
    
    CONDUIT_INFO("base64_json:(input)" << std::endl << json_txt);
    
    Generator g3(json_txt,"base64_json");
    g3.walk(nparse);
    CONDUIT_INFO("base64_json:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());
}


//-----------------------------------------------------------------------------
TEST(conduit_json, check_childless_list)
{
    Node n;
    n["path/to/empty"].set(DataType::list());
    std::string json_txt = n.to_json();
    
    CONDUIT_INFO("json:(input)" << std::endl << json_txt);
    
    Node nparse;
    Generator g(json_txt,"json");
    g.walk(nparse);
    CONDUIT_INFO("json:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("conduit");
    
    CONDUIT_INFO("conduit:(input)" << std::endl << json_txt);
    
    Generator g2(json_txt,"conduit");
    g2.walk(nparse);
    CONDUIT_INFO("conduit:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());


    json_txt = n.to_json("base64_json");
    
    CONDUIT_INFO("base64_json:(input)" << std::endl << json_txt);
    
    Generator g3(json_txt,"base64_json");
    g3.walk(nparse);
    CONDUIT_INFO("base64_json:(output)");
    nparse.print();

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());
}


//-----------------------------------------------------------------------------
TEST(conduit_json, json_string_value_with_escapes)
{
    std::string pure_json = "{\"value\": \"\\\"mystring!\\\"\"}";
    CONDUIT_INFO(pure_json);
    Generator g(pure_json,"json");
    Node n(g,true);
    n.print_detailed();
    EXPECT_EQ(n["value"].dtype().id(),DataType::CHAR8_STR_ID);
    EXPECT_EQ(n["value"].as_string(),"\"mystring!\"");
}



