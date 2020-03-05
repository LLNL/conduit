//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: t_blueprint_mcarray_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

typedef bool (*VerifyFun)(const Node&, Node&);

/// Helper Functions ///

bool has_consistent_validity(const Node &n)
{
    // TODO(JRC): This function will have problems for given nodes containing
    // nested lists.
    bool is_consistent = !n.dtype().is_object() ||
        (n.has_child("valid") && n["valid"].dtype().is_string() &&
        (n["valid"].as_string() == "true" || n["valid"].as_string() == "false"));

    NodeConstIterator itr = n.children();
    while(itr.has_next())
    {
        const Node &chld= itr.next();
        const std::string chld_name = itr.name();
        if(std::find(LOG_KEYWORDS.begin(), LOG_KEYWORDS.end(), chld_name) ==
            LOG_KEYWORDS.end())
        {
            is_consistent &= has_consistent_validity(chld);
            if(is_consistent)
            {
                bool n_valid = n["valid"].as_string() == "true";
                bool c_valid = chld["valid"].as_string() == "true";
                is_consistent &= !(n_valid && !c_valid);
            }
        }
    }

    return is_consistent;
}

/// Helper for array verify checks ///

#define CHECK_SARRAY(verify, n, info, expected)  \
{                                                \
    EXPECT_EQ(verify(n, info), expected);        \
    EXPECT_TRUE(has_consistent_validity(info));  \
}                                                \



// sarray:
// - integer child n
// - numeric child nz
// - child idx
//   - integer (array)
//   - list of integer (array)
// - length(nz) == length(idx[0])

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_sarray_verify, sarray_general)
{
    VerifyFun verify_sarray = blueprint::sarray::verify;
    Node n, info;
    const int nnz = 10;
    const int too_low_nnz = 8;
    const int too_high_nnz = 11;

    Node bogus_object;
    bogus_object["alarming"].set(0);
    bogus_object["quality"].set("high");
    bogus_object["values"].set(DataType::int32(nnz));

    CHECK_SARRAY(verify_sarray,n,info,false);

    blueprint::sarray::examples::sparse("eye", nnz, nnz, n);
    CHECK_SARRAY(verify_sarray,n,info,true);

    // integer child n must be present
    n.remove("n");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["n"].set(100.0);
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["n"].set("hundred");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["n"].set(bogus_object);
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["n"].set(100);
    CHECK_SARRAY(verify_sarray,n,info,true);

    // numeric child nz must be present, same length as idx
    Node nz = n["nz"];
    n.remove("nz");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set("not a numeric array");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set(bogus_object);
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set(too_low_nnz);     // A single numeric value won't work
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set(DataType::float64(too_low_nnz));
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set(DataType::float64(too_high_nnz));
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["nz"].set(nz);
    CHECK_SARRAY(verify_sarray,n,info,true);

    // integer child idx must be present, same length as nz
    Node idx = n["idx"];
    n.remove("idx");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set("not a numeric array");
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(bogus_object);
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(too_low_nnz);     // A single numeric value won't work
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(DataType::int32(too_low_nnz));
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(DataType::int64(too_high_nnz));
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(DataType::float64(nnz));  // has to be an integer
    CHECK_SARRAY(verify_sarray,n,info,false);
    n["idx"].set(idx);
    CHECK_SARRAY(verify_sarray,n,info,true);
}

// TODO: test multiple-indirect sarray (idx is a list of integer arrays)
