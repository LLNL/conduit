//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: t_relay_io_adios.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_error.hpp"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"

using namespace conduit;

// Include some utility functions
#include "adios_test_utils.hpp"

#if 1
//-----------------------------------------------------------------------------
TEST(conduit_relay_io_adios, test_options_contain_adios)
{
    int has_adios_protocol = 0;
    int has_adios_options = 0;
    Node opts;
    relay::about(opts);
    if(opts.has_child("io"))
    {
        const Node &io = opts["io"];
        if(io.has_child("protocols"))
        {
            const Node &protocols = io["protocols"];
            if(protocols.has_child("adios"))
            {
                if(protocols["adios"].as_string() == std::string("enabled"))
                    has_adios_protocol = 1;
            }
        }

        if(io.has_child("options"))
        {
            const Node &options = io["options"];
            has_adios_options = options.has_child("adios") ? 1 : 0;
        }
    }
    EXPECT_EQ(has_adios_protocol + has_adios_options, 2);
}
#endif

#if 1
//-----------------------------------------------------------------------------
TEST(conduit_relay_io_adios, test_read_badfile)
{
    int caught_error = 0;
    std::string path("does_not_exist.bp");
    try
    {
        Node in;
        relay::io::load(path, in);
    }
    catch(conduit::Error)
    {
        caught_error = 1;
    }

    EXPECT_EQ(caught_error, 1);
}
#endif

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_io_adios, test_scalar_types)
{
    const float64 pi = 3.141592653589793;

    int8    a(1);
    int16   b(2);
    int32   c(3);
    int64   d(4);
    float32 e(1.23456);
    float64 f(pi);
    uint8   g(5);
    uint16  h(6);
    uint32  i(7);
    uint64  j(8);
    char    k(9);
    short   l(10);
    int     m(11);
    long    n(12);
    float   o(1.23456);
    double  p(pi);
    unsigned char  q(13);
    unsigned short r(14);
    unsigned int   s(15);
    unsigned long  t(16);
    std::string    u("Hello from ADIOS");

    Node out;
    out["a"] = a;
    out["b"] = b;
    out["c"] = c;
    out["d"] = d;
    out["e"] = e;
    out["f"] = f;
    out["g"] = g;
    out["h"] = h;
    out["i"] = i;
    out["j"] = j;
    out["k"] = k;
    out["l"] = l;
    out["m"] = m;
    out["n"] = n;
    out["o"] = o;
    out["p"] = p;
    out["q"] = q;
    out["r"] = r;
    out["s"] = s;
    out["t"] = t;
    out["u"] = u;

    std::string path("test_scalar_types.bp");
    relay::io::save(out, path);

    CONDUIT_INFO("Reading " << path);
    Node in;
    relay::io::load(path, in);

    EXPECT_EQ(compare_nodes(out, in, out), true);
}
#endif

#if 0
//-----------------------------------------------------------------------------

//
// NOTE: when we run this test after the scalar test, we get a runtime error in ADIOS.
//       I wonder whether we have to init/finalize to clear out the group definition...
//
//       when I bpls the file, it contains a-p as scalars so it's like the previous
//       scalar definition of "conduit" is in effect.

TEST(conduit_relay_io_adios, test_array_types)
{
    const float64 pi = 3.141592653589793;

    int8    a[] = {1,2,3,4,5};
    int16   b[] = {2,3,4};
    int32   c[] = {3,4,5,6,7,8,9};
    int64   d[] = {4,5,6,7,8};
    float32 e[] = {1.23456, 2.3456, 3.4567};
    float64 f[] = {pi, 2.*pi, 3.*pi};
    uint8   g[] = {5,6,7,8,9};
    uint16  h[] = {6,7,8,9};
    uint32  i[] = {7,8,9,10,11};
    uint64  j[] = {8,9,10,11,12,13,14,15};
    char    k[] = {'A','D','I','O','S'};
    short   l[] = {9,10,11,12,13,14,15,16};
    int     m[] = {10,11,12};
    long    n[] = {11,12,13};
    float   o[] = {1.23456, 2.3456, 3.4567};
    double  p[] = {pi, 2.*pi, 3.*pi};
    unsigned char  q[] = {0,1,2,3};
    unsigned short r[] = {1,2,3,4,5};
    unsigned int   s[] = {2,3,4,5,6,7,8};
    unsigned long  t[] = {3,4,5,6,7,8};

    int32 sequence[] = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

    Node out;
    out["a"].set_int8_ptr(a, sizeof(a)/sizeof(int8));
    out["b"].set_int16_ptr(b, sizeof(b) / sizeof(int16));
    out["c"].set_int32_ptr(c, sizeof(c) / sizeof(int32));
    out["d"].set_int64_ptr(d, sizeof(d) / sizeof(int64));
    out["e"].set_float32_ptr(e, sizeof(e) / sizeof(float32));
    out["f"].set_float64_ptr(f, sizeof(f) / sizeof(float64));
    out["g"].set_uint8_ptr(g, sizeof(g) / sizeof(uint8));
    out["h"].set_uint16_ptr(h, sizeof(h) / sizeof(uint16));
    out["i"].set_uint32_ptr(i, sizeof(i) / sizeof(uint32));
    out["j"].set_uint64_ptr(j, sizeof(j) / sizeof(uint64));
#ifndef CONDUIT_USE_CHAR
    out["k"].set(k, sizeof(k) / sizeof(char));
    out["q"].set(q, sizeof(q) / sizeof(unsigned char));
#endif
#ifndef CONDUIT_USE_SHORT
    out["l"].set(l, sizeof(l) / sizeof(short));
    out["r"].set(r, sizeof(r) / sizeof(unsigned short));
#endif
#ifndef CONDUIT_USE_INT
    out["m"].set(m, sizeof(m) / sizeof(int));
    out["s"].set(s, sizeof(s) / sizeof(unsigned int));
#endif
#ifndef CONDUIT_USE_LONG
    out["n"].set(n, sizeof(n) / sizeof(long));
    out["t"].set(t, sizeof(t) / sizeof(unsigned long));
#endif
    out["o"].set(o, sizeof(o) / sizeof(float));
    out["p"].set(p, sizeof(p) / sizeof(double));

    out["one_elem"].set_int32_ptr(c, 1);

    // Add some variables with stride/offset.
    out["evens"].set(sequence, 5, 0, 2 * sizeof(int32));
    out["odds"].set(sequence, 5, sizeof(int32), 2 * sizeof(int32));
    out["evens_positive"].set(sequence, 4, 2*sizeof(int32), 2 * sizeof(int32));

    std::cout << out.to_json() << std::endl;

    std::string path("test_array_types.bp");
    relay::io::save(out, path);

    CONDUIT_INFO("Reading " << path);
    Node in;
    relay::io::load(path, in);

    std::cout << in.to_json() << std::endl;

    EXPECT_EQ(compare_nodes(out, in, out), true);
}
#endif

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_io_adios, test_vector_types)
{
    std::vector<int8>    a;
    std::vector<int16>   b;
    std::vector<int32>   c;
    std::vector<int64>   d;
    std::vector<float32> e;
    std::vector<float64> f;
    for(int i = 0; i < 10; ++i)
    {
        a.push_back(static_cast<int8>(i));
        b.push_back(static_cast<int16>(i));
        c.push_back(static_cast<int32>(i));
        d.push_back(static_cast<int64>(i));
        e.push_back(static_cast<float32>(i) * 0.1);
        f.push_back(static_cast<float64>(i) * 0.2);
    }

    Node out;
    out["a"] = a;
    out["b"] = b;
    out["c"] = c;
    out["d"] = d;
    out["e"] = e;
    out["f"] = f;

    std::string path("test_vector_types.bp");
    relay::io::save(out, path);

    CONDUIT_INFO("Reading " << path);
    Node in;
    relay::io::load(path, in);

    //std::cout << in.to_json() << std::endl;

    EXPECT_EQ(compare_nodes(out, in, out), true);
}
#endif

/**
Issues: When the ADIOS test cases are all run in the same program,
        it seems like the definition of the ADIOS group might be an issue.
        Or, maybe it's just the problem I have with reading the data.

        There are various vector and int8_array set methods on Conduit.
        Am I saving that stuff out right.

        What happens if I make a list?
**/

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_io_adios, test_list_types)
{

    Node out;
    std::string key("path/to/a_list");
    out["path/to/a"] = 1;
    // Add an empty list to the node.
    out[key].set(DataType::list());
    out["path/to/b"] = 2;
    // Add some elements to the list node.
    out[key].append().set(1.1234);
    out[key].append().set(2.3456);
    out[key].append().set(3.4567);

    // Add another list node
    std::string key2("/path/to/list2");
    out[key2].set(DataType::list());
    // Add various data types to the 2nd list.
    out[key2].append().set(1);
    out[key2].append().set("Hi there");
    const float64 pi = 3.141592653589793;
    const float64 v[] = {pi, 2*pi, 4*pi, 6*pi, 8*pi};
    out[key2].append().set(v, 5);

    // Add another list node and let's put mesh objects in it.
    std::string key3("multiple_domains");
    out[key3].set(DataType::list());
    Node &domain0 = out[key3].append();
    domain0["domain_id"] = 0;
    float64 origin[3] = {0., 0., 0.};
    float64 size[3]   = {3., 4., 5.};
    int     dims[3]   = {4,5,6};
    add_rectilinear_mesh(domain0, origin, size, dims);
    Node &domain1 = out[key3].append();
    domain1["domain_id"] = 1;
    origin[0] += size[0];
    int     dims2[3]   = {7, 9, 11};
    add_rectilinear_mesh(domain1, origin, size, dims2);

    out.print_detailed();
    std::string path("test_list_types.bp");
    relay::io::save(out, path);

    CONDUIT_INFO("Reading " << path);
    Node in;
    relay::io::load(path, in);

    //std::cout << in.to_json() << std::endl;

    EXPECT_EQ(compare_nodes(out, in, out), true);
}
#endif

#if 0
TEST(conduit_relay_io_adios_save_collective, save)
{
    //std::cout << relay::about() << std::endl;
}

TEST(conduit_relay_io_adios, opts_collective)
{
// TODO: Use options to select transforms that compress the data.

    //std::cout << relay::about() << std::endl;
}

TEST(conduit_relay_io_adios, opts_non_collective)
{
// TODO: Use options to select transforms that compress the data.

    //std::cout << relay::about() << std::endl;
}

TEST(conduit_relay_io_adios, opts_transports)
{
// TODO: Use options to select transports. POSIX, hdf5, etc.

    //std::cout << relay::about() << std::endl;
}

TEST(conduit_relay_io_adios, opts_transforms)
{
// TODO: Use options to select transforms that compress the data.

    //std::cout << relay::about() << std::endl;
}

TEST(conduit_relay_io_adios, load)
{
    //std::cout << relay::about() << std::endl;
}
#endif

/* Other test ideas:
   1. time varying data
   2. something that looks like a staging loop
   3. querying how many pieces of data there are
   4. Reading just a few pieces of data from the overall SIF file.
   5. 
 */
