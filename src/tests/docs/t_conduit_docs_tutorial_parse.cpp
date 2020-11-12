// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_parse.cpp
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
TEST(conduit_tutorial, t_conduit_docs_tutorial_yaml)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml");
    std::string yaml_txt("mykey: 42.0");

    Node n;
    n.parse(yaml_txt,"yaml");

    std::cout << n["mykey"].as_float64() <<std::endl;

    n.print_detailed();
    END_EXAMPLE("t_conduit_docs_tutorial_yaml");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, t_conduit_docs_tutorial_json)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_json");
    std::string json_txt("{\"mykey\": 42.0}");

    Node n;
    n.parse(json_txt,"json");

    std::cout << n["mykey"].as_float64() <<std::endl;

    n.print_detailed();
    END_EXAMPLE("t_conduit_docs_tutorial_json");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, t_conduit_docs_tutorial_yaml_inline_array)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array");
    std::string yaml_txt("myarray: [0.0, 10.0, 20.0, 30.0]");

    Node n;
    n.parse(yaml_txt,"yaml");

    n["myarray"].print();

    n.print_detailed();
    END_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_std)
{
    BEGIN_EXAMPLE("t_json_generator_std");
    Generator g("{test: {dtype: float64, value: 100.0}}","conduit_json");

    Node n;
    g.walk(n);

    std::cout << n["test"].as_float64() <<std::endl;
    n.print();
    n.print_detailed();
    END_EXAMPLE("t_json_generator_std");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_pure_json)
{
    BEGIN_EXAMPLE("t_json_generator_pure_json");
    
    Generator g("{test: 100.0}","json");

    Node n;
    g.walk(n);
        
    std::cout << n["test"].as_float64() <<std::endl;
    n.print_detailed();
    n.print();

    END_EXAMPLE("t_json_generator_pure_json");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_pure_yaml)
{
    BEGIN_EXAMPLE("t_json_generator_pure_yaml");
    Generator g("test: 100.0","yaml");

    Node n;
    g.walk(n);
        
    std::cout << n["test"].as_float64() <<std::endl;
    n.print_detailed();
    n.print();
    END_EXAMPLE("t_json_generator_pure_yaml");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, json_generator_bind_to_incore)
{
    BEGIN_EXAMPLE("t_json_generator_bind_to_incore");
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
    END_EXAMPLE("t_json_generator_bind_to_incore");
}

//-----------------------------------------------------------------------------
/// TODO: This doesn't need to be in the generator section.
TEST(conduit_tutorial, json_generator_compact)
{
    BEGIN_EXAMPLE("t_json_generator_compact");
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
    END_EXAMPLE("t_json_generator_compact");
}

