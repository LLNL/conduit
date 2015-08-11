//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://scalability-llnl.github.io/conduit/.
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
/// file: conduit_node_convert.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_convert, to_arrays)
{
    uint8 data_vals[] = {1,2,3,4,5,6,7,8};
    
    Node n;
    n.set(data_vals,8);

    n.schema().print();
    
    Node nconv;
    
    // signed bw-style
    n.to_int8_array(nconv);
    nconv.print();

    n.to_int16_array(nconv);
    nconv.print();

    n.to_int32_array(nconv);
    nconv.print();
    
    n.to_int64_array(nconv);
    nconv.print();

    // unsigned bw-style
    n.to_uint8_array(nconv);
    nconv.print();

    n.to_uint16_array(nconv);
    nconv.print();

    n.to_uint32_array(nconv);
    nconv.print();
    
    n.to_uint64_array(nconv);
    nconv.print();
    
    // float bw-style
    n.to_float32_array(nconv);
    nconv.print();
    
    n.to_float64_array(nconv);
    nconv.print();

    // signed native c
    n.to_char_array(nconv);
    nconv.print();

    n.to_short_array(nconv);
    nconv.print();

    n.to_int_array(nconv);
    nconv.print();
    
    n.to_long_array(nconv);
    nconv.print();

    // unsigned native c
    n.to_unsigned_char_array(nconv);
    nconv.print();

    n.to_unsigned_short_array(nconv);
    nconv.print();

    n.to_unsigned_int_array(nconv);
    nconv.print();
    
    n.to_unsigned_long_array(nconv);
    nconv.print();

    // float native c
    n.to_float_array(nconv);
    nconv.print();
    
    n.to_double_array(nconv);
    nconv.print();


}

