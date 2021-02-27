// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_utils.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include "conduit_fmt/conduit_fmt.h"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_tutorial_utils , using_fmt)
{
    BEGIN_EXAMPLE("using_fmt");
    // conduit_fmt is installed along with conduit
    #include "conduit_fmt/conduit_fmt.h"

    // fmt features are in the conduit_fmt namespace
    std::string res = conduit_fmt::format("The answer is {}.", 42);
    std::cout << res << std::endl;

    res = conduit_fmt::format("The answer is {answer:0.4f}.",
                              conduit_fmt::arg("answer",3.1415));
    std::cout << res << std::endl;
    END_EXAMPLE("using_fmt");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial_utils , using_utils_fmt_args_obj)
{
    BEGIN_EXAMPLE("using_utils_fmt_args_obj");
    // conduit::utils::format w/ args + object
    // processes named args passed via a conduit Node
    Node args;
    args["answer"] = 42;
    std::string res = conduit::utils::format("The answer is {answer:04}.",
                                            args);
    std::cout << res << std::endl;

    args.reset();
    args["adjective"] =  "other";
    args["answer"] = 3.1415;

    res = conduit::utils::format("The {adjective} answer is {answer:0.4f}.",
                                 args);

    std::cout << res << std::endl;
    END_EXAMPLE("using_utils_fmt_args_obj");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial_utils , using_utils_fmt_args_list)
{
    BEGIN_EXAMPLE("using_utils_fmt_args_list");
    // conduit::utils::format w/ args + list
    // processes ordered args passed via a conduit Node
    Node args;
    args.append() = 42;
    std::string res = conduit::utils::format("The answer is {}.", args);
    std::cout << res << std::endl;

    args.reset();
    args.append() = "other";
    args.append() = 3.1415;

    res = conduit::utils::format("The {} answer is {:0.4f}.", args);

    std::cout << res << std::endl;
    END_EXAMPLE("using_utils_fmt_args_list");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial_utils , using_utils_fmt_maps_obj)
{
    BEGIN_EXAMPLE("using_utils_fmt_maps_obj");
    // conduit::utils::format w/ maps + object
    // processing named args passed via a conduit Node, indexed by map_index
    Node maps;
    maps["answer"].set({ 42.0, 3.1415});

    std::string res = conduit::utils::format("The answer is {answer:04}.",
                                             maps, 0);
    std::cout << res << std::endl;

    res = conduit::utils::format("The answer is {answer:04}.", maps, 1);
    std::cout << res << std::endl << std::endl;


    maps.reset();
    maps["answer"].set({ 42.0, 3.1415});
    Node &slist = maps["position"];
    slist.append() = "first";
    slist.append() = "second";


    res = conduit::utils::format("The {position} answer is {answer:0.4f}.",
                                 maps, 0);

    std::cout << res << std::endl;

    res = conduit::utils::format("The {position} answer is {answer:0.4f}.",
                                 maps, 1);

    std::cout << res << std::endl;
    END_EXAMPLE("using_utils_fmt_maps_obj");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial_utils , using_utils_fmt_maps_list)
{
    BEGIN_EXAMPLE("using_utils_fmt_maps_list");
    // conduit::utils::format w/ maps + list
    // processing ordered args passed via a conduit Node, indexed by map_index
    Node maps;
    maps.append() = { 42.0, 3.1415};
    std::string res = conduit::utils::format("The answer is {}.",
                                             maps, 0);
    std::cout << res << std::endl;

    res = conduit::utils::format("The answer is {}.", maps, 1);
    std::cout << res << std::endl << std::endl;

    maps.reset();

    // first arg
    Node &slist = maps.append();
    slist.append() = "first";
    slist.append() = "second";

    // second arg
    maps.append() = { 42.0, 3.1415};

    res = conduit::utils::format("The {} answer is {:0.4f}.", maps, 0);
    std::cout << res << std::endl;

    res = conduit::utils::format("The {} answer is {:0.4f}.", maps, 1);
    std::cout << res << std::endl;
    END_EXAMPLE("using_utils_fmt_maps_list");
}

