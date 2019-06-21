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
/// file: t_conduit_yaml.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include <limits>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_yaml, to_yaml)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_yaml() << std::endl;
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_2)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    uint32   arr[5];
    for(uint32 i=0;i<5;i++)
    {
        arr[i] = i*i;
    }

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["arr"].set_external(DataType::uint32(5),arr);


    std::string pure_yaml = n.to_yaml();

    std::cout << n.to_yaml() << std::endl;


    Generator g(pure_yaml,"yaml");
    Node n2(g,true);
    std::cout << n2.to_yaml() << std::endl;

    //
    // YAML parsing will place values into an int64, 
    // here we use "to_int64" to do a direct comparison
    //
    EXPECT_EQ(n["a"].to_int64(),n2["a"].to_int64());
    EXPECT_EQ(n["b"].to_int64(),n2["b"].to_int64());

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_3)
{
    Generator g("{\"a\": [0,1,2,3,4], \"b\":[0.0,1.1,2.2,3.3] }",
                "yaml");
    Node n(g,true);
    std::cout << n.to_yaml() << std::endl;
    
    EXPECT_TRUE(n["a"].dtype().is_int64());
    EXPECT_TRUE(n["b"].dtype().is_float64());

    Generator g2("a: [0,-1,2,-3,4]\nb: [0.0,-1.1,2.2,-3.3]\n",
                 "yaml");
    Node n2(g2,true);
    std::cout << n2.to_yaml() << std::endl;
    
    EXPECT_TRUE(n["a"].dtype().is_int64());
    EXPECT_TRUE(n["b"].dtype().is_float64());
    
    int64_array a_val = n2["a"].value();
    EXPECT_EQ(a_val[0],0);
    EXPECT_EQ(a_val[1],-1);
    EXPECT_EQ(a_val[2],2);
    EXPECT_EQ(a_val[3],-3);
    EXPECT_EQ(a_val[4],4);
    
    float64_array b_val = n2["b"].value();
    EXPECT_EQ(b_val[0],0);
    EXPECT_EQ(b_val[1],-1.1);
    EXPECT_EQ(b_val[2],2.2);
    EXPECT_EQ(b_val[3],-3.3);
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_4)
{
    std::string yaml_txt ="{\"a\": [0,1,2,3,4], \"b\":[0.0,1.1,2.2,3.3] }";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;
    
    EXPECT_TRUE(n["a"].dtype().is_int64());
    EXPECT_TRUE(n["b"].dtype().is_float64());

    yaml_txt = "- here is a string\n- here is another string\n- 10\n";
    g.set_schema(yaml_txt);
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_TRUE(n[0].dtype().is_char8_str());
    EXPECT_TRUE(n[1].dtype().is_char8_str());
    EXPECT_TRUE(n[2].dtype().is_int64());

    yaml_txt  = "- here is a string\n";
    yaml_txt += "-  a: here is another string\n";
    yaml_txt += "   b: more fun\n";
    yaml_txt += "- 42\n";
    g.set_schema(yaml_txt);
    
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;
    
    EXPECT_TRUE(n[0].dtype().is_char8_str());
    EXPECT_TRUE(n[1]["a"].dtype().is_char8_str());
    EXPECT_TRUE(n[1]["b"].dtype().is_char8_str());
    EXPECT_TRUE(n[2].dtype().is_int64());

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_5)
{
    // test promote to float64
    std::string yaml_txt ="a: [0,1,2,3.0,4]";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_TRUE(n["a"].dtype().is_float64());
    EXPECT_EQ(n["a"].dtype().number_of_elements(),5);
}

//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_6)
{
    // test promote to float64
    std::string yaml_txt ="a: myfunc(\"here\")\nb: myfunc('here')";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_TRUE(n["a"].dtype().is_char8_str());
    EXPECT_TRUE(n["b"].dtype().is_char8_str());

    EXPECT_EQ(n["a"].as_string(),"myfunc(\"here\")");
    EXPECT_EQ(n["b"].as_string(),"myfunc('here')");
}

//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_7)
{
    // test promote to float64
    std::string yaml_txt ="a: \"string with an embedded :\"\n";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_TRUE(n["a"].dtype().is_char8_str());

    EXPECT_EQ(n["a"].as_string(),"string with an embedded :");
}

//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_8)
{
    // test promote to float64
    std::string yaml_txt ="a: [ b: [ c,d,e ], f: g]\n";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;


    EXPECT_EQ(n["a"][0]["b"][0].as_string(),"c");
    EXPECT_EQ(n["a"][0]["b"][1].as_string(),"d");
    EXPECT_EQ(n["a"][0]["b"][2].as_string(),"e");
    EXPECT_EQ(n["a"][1]["f"].as_string(),"g");
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_9)
{
    // test promote to float64
    std::string yaml_txt ="a:    leading and trailing space eaten     \n";
    yaml_txt += "b: '      leading and trailing space survives    '\n";
    yaml_txt += "c: \"      leading and trailing space survives    \"\n";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_EQ(n["a"].as_string(),"leading and trailing space eaten");
    EXPECT_EQ(n["b"].as_string(),"      leading and trailing space survives    ");
    EXPECT_EQ(n["c"].as_string(),"      leading and trailing space survives    ");
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_yaml_10)
{
    // test promote to float64
    std::string yaml_txt ="# comment\n# comment 2\na: not a comment";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;
    
    EXPECT_EQ(n["a"].as_string(),"not a comment");
    
}

//-----------------------------------------------------------------------------
TEST(conduit_yaml, suprise_string)
{
    // test promote to float64
    std::string yaml_txt ="a: [0,1,-2,3.0,4, hamburger]";
    Generator g(yaml_txt,
                "yaml");
    Node n;
    g.walk(n);
    std::cout << n.to_yaml() << std::endl;

    EXPECT_TRUE(n["a"].dtype().is_list());
    EXPECT_EQ(n["a"].number_of_children(),6);

    EXPECT_EQ(n["a"][0].as_int64(),0);
    EXPECT_EQ(n["a"][1].as_int64(),1);
    EXPECT_EQ(n["a"][2].as_int64(),-2);
    EXPECT_EQ(n["a"][3].as_float64(),3.0);
    EXPECT_EQ(n["a"][4].as_int64(),4);
    EXPECT_EQ(n["a"][5].as_string(),"hamburger");
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, yaml_bool)
{
    std::string pure_yaml = "{\"value\": true}";
    Generator g(pure_yaml,"yaml");
    Node n(g,true);
    
    std::cout << n.to_yaml() << std::endl;
    
    EXPECT_EQ(n["value"].dtype().id(),DataType::UINT8_ID);

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, yaml_default)
{
    std::string yaml_txt ="{\"a\": [0,1,2,3,4], \"b\":[0.0,1.1,2.2,3.3] }";
    Generator g(yaml_txt,"yaml");
    Node n;
    g.walk(n);

    EXPECT_EQ(n.to_yaml_default(), n.to_yaml());
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, check_empty)
{
    Node n;
    n["path/to/empty"];
    std::cout << n.to_yaml() << std::endl;
    std::string yaml_txt = n.to_yaml();
    
    CONDUIT_INFO("yaml:" << std::endl << yaml_txt);
    
    Node nparse;
    Generator g(yaml_txt,"yaml");
    g.walk(nparse);
    std::cout << nparse.to_yaml() << std::endl;

    EXPECT_EQ(n["path/to/empty"].dtype().id(),
              nparse["path/to/empty"].dtype().id());

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, check_childless_object)
{
    Node n;
    n["path/to/empty"].set(DataType::object());
    std::string yaml_txt = n.to_yaml();
    CONDUIT_INFO("yaml:(input)" << std::endl << yaml_txt);
    
    n.print_detailed();
    
    Node nparse;
    Generator g(yaml_txt,"yaml");
    g.walk(nparse);
    CONDUIT_INFO("yaml:(output)");
    std::cout << nparse.to_yaml() << std::endl;
    
    nparse.print_detailed();

    // we can recover that this was an object from JSON, but 
    // we simply can't recover from yaml
    EXPECT_EQ(nparse["path/to/empty"].dtype().id(),
              DataType::EMPTY_ID);
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, check_childless_list)
{
    Node n;
    n["path/to/empty"].set(DataType::list());
    std::string yaml_txt = n.to_yaml();
    
    CONDUIT_INFO("yaml:(input)" << std::endl << yaml_txt);
    
    Node nparse;
    Generator g(yaml_txt,"yaml");
    g.walk(nparse);
    CONDUIT_INFO("yaml:(output)");
    nparse.print();

    // we can recover that this was a list from JSON, but 
    // we simply can't recover from yaml
    EXPECT_EQ(nparse["path/to/empty"].dtype().id(),
               DataType::EMPTY_ID);

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, json_string_value_with_escapes)
{
    std::string pure_json = "{\"value\": \"\\\"mystring!\\\"\"}";
    CONDUIT_INFO(pure_json);
    Generator g(pure_json,"yaml");
    Node n(g,true);

    std::cout << n.to_yaml() << std::endl;

    EXPECT_EQ(n["value"].dtype().id(),DataType::CHAR8_STR_ID);
    EXPECT_EQ(n["value"].as_string(),"\"mystring!\"");
    
    // this tests a specific bug conduit 0.2.1 json parsing logic
    std::string pure_json_2 = "{ \"testing\" : \"space_before_colon\"}";
    CONDUIT_INFO(pure_json_2);
    Generator g2(pure_json_2,"yaml");
    Node n2(g2,true);

    std::cout << n2.to_yaml() << std::endl;
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, yaml_schema_preserve_floats)
{
    Node n;
    n["i"].set_int64(10);
    n["f"].set_float64(20.0);

    std::string source_yaml= n.to_yaml();
    
    Generator g(source_yaml,"yaml");
    Node n_parse;
    g.walk(n_parse);
    
    std::string parsed_yaml = n.to_yaml();

    CONDUIT_INFO(parsed_yaml);

    EXPECT_TRUE(n_parse["i"].dtype().is_int64());
    EXPECT_TRUE(n_parse["f"].dtype().is_float64());
}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, yaml_inf_and_nan)
{
    Node n;
    n["pos_inf"] =  std::numeric_limits<float64>::infinity();
    n["neg_inf"] = -std::numeric_limits<float64>::infinity();
    n["nan"]     =  std::numeric_limits<float64>::quiet_NaN();
 
    CONDUIT_INFO(n.to_yaml());
    CONDUIT_INFO(n.to_json());
    
    std::string yaml_str = n.to_yaml();
    CONDUIT_INFO(yaml_str);
    
    Generator g(yaml_str,"yaml");
    
    Node n_res;
    g.walk(n_res);

    CONDUIT_INFO(n_res.to_yaml());
    CONDUIT_INFO(n_res.to_json());

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, parse_error_detailed)
{
    {
        bool throw_occured = false;
        try
        {
            std::string pure_json = "{\"value\": \n \n \n \n \"\\\"mystring!\\\"\" \n :}";
            Generator g(pure_json,"yaml");

            Node n_res;
            g.walk(n_res);
        }
        catch(conduit::Error e)
        {
            CONDUIT_INFO(e.message());
             throw_occured = true;
        }

        EXPECT_TRUE(throw_occured);
    }

    {
        bool throw_occured = false;
        try
        {
            std::string pure_json = "{\"value\":\"\\\"mystring!\\\"\" :}";
            Generator g(pure_json,"yaml");

            Node n_res;
            g.walk(n_res);
        }
        catch(conduit::Error e)
        {
            CONDUIT_INFO(e.message());
            throw_occured = true;
        }

        EXPECT_TRUE(throw_occured);
    }

    {
        bool throw_occured = false;

        try
        {
            std::string pure_json = "\n\n\n\n\n\n{\"value\":\"\\\"mystring!\\\"\" :}";
            Generator g(pure_json,"yaml");

            Node n_res;
            g.walk(n_res);
        }
        catch(conduit::Error e)
        {
            CONDUIT_INFO(e.message());
            throw_occured = true;
        }

        EXPECT_TRUE(throw_occured);
    }

    {
        bool throw_occured = false;

        try
        {
            std::string pure_yaml = "a: here is a string with a colon : to test";;
            Generator g(pure_yaml,"yaml");

            Node n_res;
            g.walk(n_res);
        }
        catch(conduit::Error e)
        {
            CONDUIT_INFO(e.message());
            throw_occured = true;
        }

        EXPECT_TRUE(throw_occured);
    }

}


//-----------------------------------------------------------------------------
TEST(conduit_yaml, dup_object_name_error)
{
    std::ostringstream oss;

    // test pure json case for dup child keys
    oss << "{\n";
    oss << "\"t1_key\" : { \"sub\": 10 }, \n";
    oss << "\"t1_key\" : { \"sub\": \"my_string\"}\n";
    oss << "}"; 
    
    CONDUIT_INFO(oss.str());
    
    Generator g(oss.str(),"yaml");

    Node n;
    ASSERT_THROW(g.walk(n),conduit::Error);
    EXPECT_TRUE(n.dtype().is_empty());
    
    
    // dup keys at path
    oss << "{\n";
    oss << "\"a\" : { \"sub\": 10  , \"sub\": \"my_string\"}\n";
    oss << "}"; 
    
    CONDUIT_INFO(oss.str());
    
    Generator g3(oss.str(),"yaml");

    Node n3;
    ASSERT_THROW(g3.walk(n),conduit::Error);
    EXPECT_TRUE(n3.dtype().is_empty());
}
