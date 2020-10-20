// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_relay_io_adios.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_io.h"
#include "conduit_error.hpp"
#include "conduit_cpp_to_c.hpp"
#include "gtest/gtest.h"

using namespace conduit;

// Include some utility functions
#include "conduit.hpp"
#include "../adios_test_utils.hpp"


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_c, test_io_c_save_and_load)
{
    const float64 pi = 3.141592653589793;
    conduit_int8    a[] = {1,2,3,4,5};
    conduit_int16   b[] = {2,3,4,5,6};
    conduit_int32   c[] = {3,4,5,6,7};
    conduit_int64   d[] = {4,5,6,7,8};
    conduit_float32 e[] = {1.23456, 2.3456, 3.4567, 4.5678, 5.6789};
    conduit_float64 f[] = {pi, 2.*pi, 3.*pi, 4*pi, 5*pi};
    conduit_uint8   g[] = {5,6,7,8,9};
    conduit_uint16  h[] = {6,7,8,9,10};
    conduit_uint32  i[] = {7,8,9,10,11};
    conduit_uint64  j[] = {8,9,10,11,12};
    const char *k = "ADIOS";
    const char *path = "test_io_c_save_and_load.bp";
    conduit_node *out, *in, *info;

    /* Use the C API to make a node and save to ADIOS. */
    out = conduit_node_create();
    conduit_node_set_int8_ptr(conduit_node_fetch(out, "a"), a,
        sizeof(a) / sizeof(conduit_int8));
    conduit_node_set_int16_ptr(conduit_node_fetch(out, "b"), b,
        sizeof(b) / sizeof(conduit_int16));
    conduit_node_set_int32_ptr(conduit_node_fetch(out, "c"), c,
        sizeof(c) / sizeof(conduit_int32));
    conduit_node_set_int64_ptr(conduit_node_fetch(out, "d"), d,
        sizeof(d) / sizeof(conduit_int64));
    conduit_node_set_float32_ptr(conduit_node_fetch(out, "e"), e,
        sizeof(e) / sizeof(conduit_float32));
    conduit_node_set_float64_ptr(conduit_node_fetch(out, "f"), f,
        sizeof(f) / sizeof(conduit_float64));
    conduit_node_set_uint8_ptr(conduit_node_fetch(out, "g"), g,
        sizeof(g) / sizeof(conduit_uint8));
    conduit_node_set_uint16_ptr(conduit_node_fetch(out, "h"), h,
        sizeof(h) / sizeof(conduit_uint16));
    conduit_node_set_uint32_ptr(conduit_node_fetch(out, "i"), i,
        sizeof(i) / sizeof(conduit_uint32));
    conduit_node_set_uint64_ptr(conduit_node_fetch(out, "j"), j,
        sizeof(j) / sizeof(conduit_uint64));
    conduit_node_set_char8_str(conduit_node_fetch(out, "k"), k);

    /* Save the node */
    conduit_relay_io_save(out, path, NULL, NULL);

    /* Read the data back in. */
    in = conduit_node_create();
    conduit_relay_io_load(path, NULL, NULL, in);
    info = conduit_node_create();
    
    EXPECT_EQ(conduit_node_diff(out, in, info,0.0), 0);

    /* Cleanup */
    conduit_node_destroy(out);
    conduit_node_destroy(in);
    conduit_node_destroy(info);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_c, test_io_c_time_series)
{
    const char *path = "test_io_c_time_series.bp";
    const char *protocol = "adios";
    int i, ts, nts = 5;
    conduit_node **out = (conduit_node **)malloc(nts * sizeof(conduit_node *));

    // Write multiple time steps to the same file.
    for(ts = 0; ts < nts; ++ts)
    {
        int idx = ts*100;
        out[ts] = conduit_node_create();
        conduit_node_set_int(conduit_node_fetch(out[ts], "a"), idx + 1);
        conduit_node_set_int(conduit_node_fetch(out[ts], "b"), idx + 2);
        conduit_node_set_int(conduit_node_fetch(out[ts], "c/d"), idx + 3);
        conduit_node_set_int(conduit_node_fetch(out[ts], "c/e"), idx + 4);
        conduit_node_set_float(conduit_node_fetch(out[ts], "f"), 3.14159f * (float)ts);

        if(ts == 0)
            conduit_relay_io_save(out[ts], path, NULL, NULL);
        else
            conduit_relay_io_add_step(out[ts], path, NULL, NULL);

        // Make sure the file has the new  step.
        int qnts = conduit_relay_io_query_number_of_steps(path);
        EXPECT_EQ(qnts, ts+1);
    }
    
    conduit_node *info = conduit_node_create();

    // read back all steps.
    for(int ts = 0; ts < nts; ++ts)
    {
        conduit_node *in = conduit_node_create();
        conduit_relay_io_load_step_and_domain(path, protocol, ts, 0, NULL, in);

        EXPECT_EQ(conduit_node_diff(in, out[ts], info, 0.0), 0);
        conduit_node_destroy(in);
    }
    
    conduit_node_destroy(info);

    for(i = 0; i < nts; ++i)
        conduit_node_destroy(out[i]);
    free(out);
}
