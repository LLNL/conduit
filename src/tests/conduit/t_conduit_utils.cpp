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
/// file: conduit_utils.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

#include "t_config.hpp"

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
TEST(conduit_utils, error_constructors)
{
    conduit::Error e("mymessage","myfile",10);
    CONDUIT_INFO(e.message());
    CONDUIT_INFO(e.what());
        
    try
    {
        utils::handle_warning("ERROR!",__FILE__,__LINE__);
    }
    catch(conduit::Error e)
    {
        conduit::Error ecpy(e);
        CONDUIT_INFO(ecpy.message());
    }
    
    try
    {
        utils::handle_warning("ERROR!",__FILE__,__LINE__);
    }
    catch(conduit::Error e)
    {
        conduit::Error ecpy;
        ecpy = e;
        CONDUIT_INFO(ecpy.message());
    }
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


//-----------------------------------------------------------------------------
TEST(conduit_utils, escape_special_chars)
{
    std::string test = "\"myvalue\":10";
    std::string test_escaped   = utils::escape_special_chars(test);
    std::string test_unescaped = utils::unescape_special_chars(test_escaped);

    CONDUIT_INFO( test << " vs " << test_escaped);

    EXPECT_EQ(test_escaped, "\\\"myvalue\\\":10");
    EXPECT_EQ(test,test_unescaped);


    test = "\" \\ \n \t \b \f \r /";
    test_escaped   = utils::escape_special_chars(test);
    test_unescaped = utils::unescape_special_chars(test_escaped);

    CONDUIT_INFO( test << "\nvs\n" << test_escaped);

    EXPECT_EQ(test_escaped, "\\\" \\\\ \\n \\t \\b \\f \\r /");
    EXPECT_EQ(test,test_unescaped);
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, float64_to_string)
{
    

    float64 v = 10.0;
    
    EXPECT_EQ("10.0",utils::float64_to_string(v));
    
    v = 10000000000000000;
    EXPECT_EQ("1e+16",utils::float64_to_string(v));

}



//-----------------------------------------------------------------------------
TEST(conduit_utils, is_dir)
{
    EXPECT_TRUE(utils::is_directory(CONDUIT_T_SRC_DIR));
    EXPECT_TRUE(utils::is_directory(CONDUIT_T_BIN_DIR));
    
    EXPECT_FALSE(utils::is_directory("asdasdasdasd"));
}


//-----------------------------------------------------------------------------
TEST(conduit_utils, is_file)
{

    std::string tf_path = utils::join_file_path(CONDUIT_T_SRC_DIR,
                                                "conduit");

    tf_path = utils::join_file_path(tf_path,"t_conduit_utils.cpp");

    EXPECT_TRUE(utils::is_file(tf_path));
    
    EXPECT_FALSE(utils::is_file(CONDUIT_T_SRC_DIR));
    EXPECT_FALSE(utils::is_file(CONDUIT_T_BIN_DIR));
    
    EXPECT_FALSE(utils::is_file("asdasdasdasd"));
}



//-----------------------------------------------------------------------------
TEST(conduit_utils, remove_file)
{
    std::ofstream ofs;
    
    ofs.open("t_remove_file.txt");
    ofs << "here" << std::endl;
    ofs.close();

    EXPECT_TRUE(utils::is_file("t_remove_file.txt"));
    
    utils::remove_file("t_remove_file.txt");
    
    EXPECT_FALSE(utils::is_file("t_remove_file.txt"));
}

//-----------------------------------------------------------------------------
TEST(conduit_utils, system_exec)
{
    // TODO: windows test ... 
    EXPECT_EQ(utils::system_execute("pwd"),0);
}



//-----------------------------------------------------------------------------
TEST(conduit_utils, base64_enc_dec)
{
    Node n_src;
    n_src["a"].set_int32(10);
    n_src["b"].set_int32(20);
    n_src["c"].set_int32(30);
    
    // we need compact data for base64
    Node n;
    n_src.compact_to(n);
    
    // use libb64 to encode the data
    index_t nbytes = n.schema().total_bytes();
    Node bb64_data;
    bb64_data.set(DataType::char8_str(nbytes*2+1));
    
    const char *src_ptr = (const char*)n.data_ptr();
    char *dest_ptr      = (char*)bb64_data.data_ptr();
    memset(dest_ptr,0,nbytes*2+1);
    
    utils::base64_encode(src_ptr,nbytes,dest_ptr);

    // use libb64 to decode the data
    std::string base64_str = bb64_data.as_string();
    Node n_res(n.schema());
    const char *bb64_src_ptr = base64_str.c_str();
    int bb64_src_len = base64_str.length();
    void *bb64_dest_ptr = n_res.data_ptr();
    
    utils::base64_decode(bb64_src_ptr,bb64_src_len,bb64_dest_ptr);

    // check we have the same values
    EXPECT_EQ(n_src["a"].as_int32(), n_res["a"].as_int32());
    EXPECT_EQ(n_src["b"].as_int32(), n_res["b"].as_int32());
    EXPECT_EQ(n_src["c"].as_int32(), n_res["c"].as_int32());
}






