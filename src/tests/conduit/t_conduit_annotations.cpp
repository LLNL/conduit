// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_annotations.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"
#include "conduit_annotations.hpp"

#include <iostream>
#include <limits>
#include "gtest/gtest.h"

#include "t_config.hpp"

//-----------------------------------------------------------------------------
void
annotate_test_func()
{
    CONDUIT_ANNOTATE_MARK_FUNCTION;
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, annotations_support)
{
    conduit::Node about;
    conduit::about(about);

    if( conduit::annotations::supported() )
    {
        EXPECT_EQ(about["annotations"].as_string(),"enabled");
    }
    else
    {
        EXPECT_EQ(about["annotations"].as_string(),"disabled");
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, annotations)
{
    std::string tout_file = "tout_annotations_file.txt";
    if(conduit::annotations::supported())
    {
        // clean up output file if it exists
        conduit::utils::remove_path_if_exists(tout_file);
    }
    
    conduit::Node opts;
    opts["config"] = "runtime-report";
    opts["output_file"] = tout_file;
    conduit::annotations::initialize(opts);
    CONDUIT_ANNOTATE_MARK_BEGIN("test_region");
    annotate_test_func();
    {
        CONDUIT_ANNOTATE_MARK_SCOPE("test_scope");
        conduit::utils::sleep(100);
    }
    CONDUIT_ANNOTATE_MARK_END("test_region");

    conduit::annotations::flush();
    conduit::annotations::finalize();
  
    if(conduit::annotations::supported())
    {
        // make sure perf output file exists
        EXPECT_TRUE(conduit::utils::is_file(tout_file));
    }
}


