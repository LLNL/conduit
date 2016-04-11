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
/// file: conduit_utils.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

bool info_occured    = false;
bool warning_occured = false;
bool error_occured   = false;
    

//-----------------------------------------------------------------------------
void 
print_msg(const std::string &msg,
          const std::string &file,
          int line)
{
    std::cout << "File:"    << file << std::endl;
    std::cout << "Line:"    << line << std::endl;
    std::cout << "Message:" << msg  << std::endl;
}

//-----------------------------------------------------------------------------
void 
my_info_handler(const std::string &msg,
                const std::string &file,
                int line)
{
    print_msg(msg,file,line);
    info_occured = true;
}


//-----------------------------------------------------------------------------
void 
my_warning_handler(const std::string &msg,
                   const std::string &file,
                   int line)
{
    print_msg(msg,file,line);
    warning_occured = true;
}

//-----------------------------------------------------------------------------
void 
my_error_handler(const std::string &msg,
                 const std::string &file,
                 int line)
{
    print_msg(msg,file,line);
    error_occured = true;
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, override_info)
{
    utils::handle_info("INFO!",__FILE__,__LINE__);

    EXPECT_FALSE(info_occured);
    
    conduit::utils::set_info_handler(my_info_handler);
    utils::handle_info("INFO!",__FILE__,__LINE__);
    EXPECT_TRUE(info_occured);
    
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
    
    utils::handle_info("INFO!",__FILE__,__LINE__);
}


//-----------------------------------------------------------------------------
TEST(conduit_utils, override_warning)
{
    EXPECT_THROW(utils::handle_warning("WARNING!",__FILE__,__LINE__),
                 conduit::Error);
                 
    EXPECT_FALSE(warning_occured);
    conduit::utils::set_warning_handler(my_warning_handler);
    utils::handle_warning("WARNING!",__FILE__,__LINE__);
    EXPECT_TRUE(warning_occured);
    
    conduit::utils::set_warning_handler(conduit::utils::default_warning_handler);
    
    EXPECT_THROW(utils::handle_warning("WARNING!",__FILE__,__LINE__),
                 conduit::Error);
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, override_error)
{
    EXPECT_THROW(utils::handle_warning("ERROR!",__FILE__,__LINE__),
                 conduit::Error);

    EXPECT_FALSE(error_occured);
    conduit::utils::set_error_handler(my_error_handler);
    utils::handle_error("ERROR!",__FILE__,__LINE__);
    EXPECT_TRUE(error_occured);
    
    conduit::utils::set_error_handler(conduit::utils::default_error_handler);
    
    EXPECT_THROW(utils::handle_warning("ERROR!",__FILE__,__LINE__),
                 conduit::Error);
    
}


