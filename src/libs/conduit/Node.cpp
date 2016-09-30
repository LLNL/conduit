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
/// file: Node.cpp
///
//-----------------------------------------------------------------------------
#include "Node.hpp"

//-----------------------------------------------------------------------------
// -- standard cpp lib includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
#include <map>

//-----------------------------------------------------------------------------
// -- standard c lib includes -- 
//-----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#if !defined(CONDUIT_PLATFORM_WINDOWS)
//
// mmap interface not available on windows
// 
#include <sys/mman.h>
#include <unistd.h>
#else
#include "Windows.h"
#endif

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "Utils.hpp"
#include "Generator.hpp"

//-----------------------------------------------------------------------------
//
/// The CONDUIT_CHECK_DTYPE macro is used to check the dtype for leaf access
/// methods. If a type mismatch occurs, the message provides the full path to
/// the node.
// 
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_DTYPE( dtype_node, dtype_id_expect, method, rtn ) \
{                                                                       \
    CONDUIT_CHECK( (dtype_node->dtype().id() == dtype_id_expect) ,      \
                    "Node::" << method << " -- DataType "               \
                    << DataType::id_to_name(dtype_node->dtype().id())   \
                    << " at path " << dtype_node->path()                \
                    << " does not equal expected DataType "             \
                    << DataType::id_to_name(dtype_id_expect));          \
                                                                        \
    if(dtype_node->dtype().id() != dtype_id_expect)                     \
    {                                                                   \
        return rtn;                                                     \
    }                                                                   \
}                                                                       \

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- begin conduit::Node public methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================


//-----------------------------------------------------------------------------
//
// -- begin definition of Node constructors + destructor --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- basic constructor and destruction -- 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node::Node()
{
    init_defaults();
}

//---------------------------------------------------------------------------//
Node::Node(const Node &node)
{
    init_defaults();
    set(node);
}

//---------------------------------------------------------------------------//
Node::~Node()
{
    cleanup();
}

//---------------------------------------------------------------------------//
void
Node::reset()
{
    release();
    m_schema->set(DataType::EMPTY_ID);
}

//-----------------------------------------------------------------------------
// -- constructors for generic types --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Node::Node(const Schema &schema)

{
    init_defaults();
    set(schema);
}

//---------------------------------------------------------------------------//
Node::Node(const Generator &gen,
           bool external)
{
    init_defaults();
    if(external)
    {
        gen.walk_external(*this);
    }
    else
    {
        gen.walk(*this);
    }

}

//---------------------------------------------------------------------------//
Node::Node(const std::string &json_schema,
           void *data,
           bool external)
{
    init_defaults(); 
    Generator g(json_schema,"conduit_json",data);

    if(external)
    {
        g.walk_external(*this);
    }
    else
    {
        g.walk(*this);
    }

}

//---------------------------------------------------------------------------//
Node::Node(const DataType &dtype)
{
    init_defaults();
    set(dtype);
}

//---------------------------------------------------------------------------//
Node::Node(const Schema &schema,
           void *data,
           bool external)
{
    init_defaults();
    std::string json_schema =schema.to_json(); 
    Generator g(json_schema,"conduit_json",data);
    if(external)
    {
        g.walk_external(*this);
    }
    else
    {
        g.walk(*this);
    }
}


//---------------------------------------------------------------------------//
Node::Node(const DataType &dtype,
           void *data,
           bool external)
{    
    init_defaults();
    if(external)
    {
        set_external(dtype,data);
    }
    else
    {
        set(dtype,data);
    }
}

//-----------------------------------------------------------------------------
//
// -- end definition of Node constructors + destructor --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node generate methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::generate(const Generator &gen)
{
    gen.walk(*this);
}

//---------------------------------------------------------------------------//
void
Node::generate_external(const Generator &gen)
{
    gen.walk_external(*this);
}

//---------------------------------------------------------------------------//
void
Node::generate(const std::string &json_schema,
               const std::string &protocol,
               void *data)
               
{
    Generator g(json_schema,protocol,data);
    generate(g);
}
    
//---------------------------------------------------------------------------//
void
Node::generate_external(const std::string &json_schema,
                        const std::string &protocol,
                        void *data)
               
{
    Generator g(json_schema,protocol,data);
    generate_external(g);
}

/// TODO: missing def of several Node::generator methods


//-----------------------------------------------------------------------------
//
// -- end definition of Node generate methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Node basic i/o methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::load(const std::string &stream_path,
           const Schema &schema)
{
    // clear out any existing structure
    reset();
    index_t dsize = schema.total_bytes();

    allocate(dsize);
    std::ifstream ifs;
    ifs.open(stream_path.c_str());
    if(!ifs.is_open())
        CONDUIT_ERROR("<Node::load> failed to open: " << stream_path);
    ifs.read((char *)m_data,dsize);
    ifs.close();

    //
    // See Below
    //
    m_alloced = false;
    
    m_schema->set(schema);
    walk_schema(this,m_schema,m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_alloced = true;
}

//---------------------------------------------------------------------------//
void
Node::load(const std::string &ibase,
           const std::string &protocol)
{
    if(protocol == "conduit_bin")
    {
        // TODO: use generator?
        Schema s;
        std::string ifschema = ibase + "_json";

        s.load(ifschema);
        load(ibase,s);
    }
    // single file json cases
    else
    {
        std::ifstream ifile;
        ifile.open(ibase.c_str());
        if(!ifile.is_open())
            CONDUIT_ERROR("<Node::load> failed to open: " << ibase);
        std::string json_data((std::istreambuf_iterator<char>(ifile)),
                               std::istreambuf_iterator<char>());
        
        Generator g(json_data,protocol);
        g.walk(*this);
    }
        
}

//---------------------------------------------------------------------------//
void
Node::save(const std::string &obase,
           const std::string &protocol) const
{
    if(protocol == "conduit_bin")
    {
        Node res;
        compact_to(res);
        std::string ofschema = obase + "_json";

        res.schema().save(ofschema);
        res.serialize(obase);
    }
    // single file json cases
    else
    {
        to_json_stream(obase,protocol);
    }
}

//---------------------------------------------------------------------------//
void
Node::mmap(const std::string &stream_path)
{
    std::string ifschema = stream_path + "_json";


    Schema s;
    s.load(ifschema);
    mmap(stream_path,s);
}


//---------------------------------------------------------------------------//
void 
Node::mmap(const std::string &stream_path,
           const Schema &schema)
{
    reset();
    index_t dsize = schema.total_bytes();
    Node::mmap(stream_path,dsize);

    //
    // See Below
    //
    m_mmaped = false;
    
    m_schema->set(schema);
    walk_schema(this,m_schema,m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_mmaped = true;
}

//-----------------------------------------------------------------------------
//
// -- end definition of Node basic i/o methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node set methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_node(const Node &node)
{
    if(node.dtype().id() == DataType::OBJECT_ID)
    {
        init(DataType::object());
        std::vector<std::string> paths;
        node.paths(paths);

        for (std::vector<std::string>::const_iterator itr = paths.begin();
             itr < paths.end(); ++itr)
        {
            Schema *curr_schema = this->m_schema->fetch_ptr(*itr);
            index_t idx = this->m_schema->child_index(*itr);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(this);
            curr_node->set(*node.m_children[idx]);
            this->append_node_ptr(curr_node);       
        }        
    }
    else if(node.dtype().id() == DataType::LIST_ID)       
    {   
        init(DataType::list());
        for(index_t i=0;i< (index_t)node.m_children.size(); i++)
        {
            this->m_schema->append();
            Schema *curr_schema = this->m_schema->child_ptr(i);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(this);
            curr_node->set(*node.m_children[i]);
            this->append_node_ptr(curr_node);
        }
    }
    else if (node.dtype().id() != DataType::EMPTY_ID)
    {
        node.compact_to(*this);
    }
    else
    {
        // if passed node is empty -- reset this.
        reset();
    }    
}

//---------------------------------------------------------------------------//
void 
Node::set(const Node &node)
{
    set_node(node);
}

//---------------------------------------------------------------------------//
void 
Node::set_dtype(const DataType &dtype)
{
    init(dtype);
}

//---------------------------------------------------------------------------//
void 
Node::set(const DataType &dtype)
{
    set_dtype(dtype);
}

//---------------------------------------------------------------------------//
void
Node::set_schema(const Schema &schema)
{
    release();
    m_schema->set(schema);
    // allocate data 
    // for this case, we need the total bytes spanned by the schema
    index_t nbytes = m_schema->spanned_bytes();
    allocate(nbytes);
    memset(m_data,0,nbytes);
    // call walk w/ internal data pointer
    walk_schema(this,m_schema,m_data);
}

//---------------------------------------------------------------------------//
void
Node::set(const Schema &schema)
{
    set_schema(schema);
}


//---------------------------------------------------------------------------//
void
Node::set_data_using_schema(const Schema &schema,
                            void *data)
{
    release();
    m_schema->set(schema);   
    // for this case, we need the total bytes spanned by the schema
    index_t nbytes = m_schema->spanned_bytes();
    allocate(nbytes);
    memcpy(m_data, data, nbytes);
    walk_schema(this,m_schema,m_data);
}

//---------------------------------------------------------------------------//
void
Node::set(const Schema &schema,
          void *data)
{
    set_data_using_schema(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_data_using_dtype(const DataType &dtype,
                           void *data)
{
    release();
    m_schema->set(dtype);
    allocate(m_schema->total_bytes());
    memcpy(m_data, data, m_schema->total_bytes());
    walk_schema(this,m_schema,m_data);
}

//---------------------------------------------------------------------------//
void
Node::set(const DataType &dtype, void *data)
{
    set_data_using_dtype(dtype,data);
}

//-----------------------------------------------------------------------------
// -- set for scalar types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_int8(int8 data)
{
    init(DataType::int8());
    // TODO IMP: use element_ptr() ?
    *(int8*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void
Node::set(int8 data)
{
    set_int8(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int16(int16 data)
{
    init(DataType::int16());
    *(int16*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(int16 data)
{
    set_int16(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int32(int32 data)
{
    init(DataType::int32());
    *(int32*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(int32 data)
{
    set_int32(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int64(int64 data)
{
    init(DataType::int64());
    *(int64*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(int64 data)
{
    set_int64(data);
}

//-----------------------------------------------------------------------------
// unsigned integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_uint8(uint8 data)
{
    init(DataType::uint8());
    *(uint8*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(uint8 data)
{
    set_uint8(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint16(uint16 data)
{
    init(DataType::uint16());
    *(uint16*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(uint16 data)
{
    set_uint16(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint32(uint32 data)
{
    init(DataType::uint32());
    *(uint32*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(uint32 data)
{
    set_uint32(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint64(uint64 data)
{
    init(DataType::uint64());
    *(uint64*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(uint64 data)
{
    set_uint64(data);
}

//-----------------------------------------------------------------------------
// floating point scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_float32(float32 data)
{
    init(DataType::float32());
    *(float32*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(float32 data)
{
    set_float32(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_float64(float64 data)
{
    init(DataType::float64());
    *(float64*)((char*)m_data + schema().element_index(0)) = data;
}

//---------------------------------------------------------------------------//
void 
Node::set(float64 data)
{
    set_float64(data);
}


//-----------------------------------------------------------------------------
// set gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set(signed char data)
{
    set((CONDUIT_NATIVE_CHAR)data);
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned char data)
{
    set((CONDUIT_NATIVE_UNSIGNED_CHAR)data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set(short data)
{
    set((CONDUIT_NATIVE_SHORT)data);
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned short data)
{
    set((CONDUIT_NATIVE_UNSIGNED_SHORT)data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set(int data)
{
    set((CONDUIT_NATIVE_INT)data);
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned int data)
{
    set((CONDUIT_NATIVE_UNSIGNED_INT)data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set(long data)
{
    set((CONDUIT_NATIVE_LONG)data);
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned long data)
{
    set((CONDUIT_NATIVE_UNSIGNED_LONG)data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set(float data)
{
    set((CONDUIT_NATIVE_FLOAT)data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set(double data)
{
    set((CONDUIT_NATIVE_DOUBLE)data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_int8_array(const int8_array &data)
{
    init(DataType::int8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int8_array &data)
{
    set_int8_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int16_array(const int16_array &data)
{
    init(DataType::int16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int16_array &data)
{
    set_int16_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int32_array(const int32_array &data)
{
    init(DataType::int32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int32_array &data)
{
    set_int32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int64_array(const int64_array &data)
{
    init(DataType::int64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int64_array &data)
{
    set_int64_array(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_uint8_array(const uint8_array &data)
{
    init(DataType::uint8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint8_array &data)
{
    set_uint8_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint16_array(const uint16_array &data)
{
    init(DataType::uint16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint16_array &data)
{
    set_uint16_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint32_array(const uint32_array  &data)
{
    init(DataType::uint32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint32_array &data)
{
    set_uint32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint64_array(const uint64_array &data)
{
    init(DataType::uint64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//---------------------------------------------------------------------------//
void 
Node::set(const uint64_array  &data)
{
    set_uint64_array(data);
}


//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_float32_array(const float32_array &data)
{
    init(DataType::float32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const float32_array &data)
{
    set_float32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_float64_array(const float64_array &data)
{
    init(DataType::float64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const float64_array &data)
{
    set_float64_array(data);
}


//-----------------------------------------------------------------------------
// set array gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set(const char_array &data)
{
    init(DataType::c_char(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
void
Node::set(const unsigned_char_array &data)
{
    init(DataType::c_unsigned_char(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set(const short_array &data)
{
    init(DataType::c_short(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
void
Node::set(const unsigned_short_array&data)
{
    init(DataType::c_unsigned_short(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set(const int_array &data)
{
    init(DataType::c_int(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
void
Node::set(const unsigned_int_array &data)
{
    init(DataType::c_unsigned_int(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set(const long_array &data)
{
    init(DataType::c_long(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
void
Node::set(const unsigned_long_array &data)
{
    init(DataType::c_unsigned_long(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set(const float_array &data)
{
    init(DataType::c_float(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set(const double_array &data)
{
    init(DataType::c_double(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for string types -- 
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// char8_str use cases
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void 
Node::set_string(const std::string &data)
{
    // size including the null term
    index_t str_size_with_term = data.length()+1;
    DataType str_t(DataType::CHAR8_STR_ID,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_ID);
    init(str_t);
    memcpy(m_data,data.c_str(),sizeof(char)*str_size_with_term);
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::string &data)
{
    set_string(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_char8_str(const char *data)
{
    // size including the null term
    index_t str_size_with_term = strlen(data)+1;
    DataType str_t(DataType::CHAR8_STR_ID,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_ID);
    init(str_t);
    memcpy(m_data,data,sizeof(char)*str_size_with_term);
}


//-----------------------------------------------------------------------------
// -- set for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_int8_vector(const std::vector<int8> &data)
{
    init(DataType::int8(data.size()));
    memcpy(m_data,&data[0],sizeof(int8)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int8> &data)
{
    set_int8_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int16_vector(const std::vector<int16> &data)
{
    init(DataType::int16(data.size()));
    memcpy(m_data,&data[0],sizeof(int16)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int16> &data)
{
    set_int16_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int32_vector(const std::vector<int32> &data)
{
    init(DataType::int32(data.size()));
    memcpy(m_data,&data[0],sizeof(int32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int32> &data)
{
    set_int32_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_int64_vector(const std::vector<int64> &data)
{
    init(DataType::int64(data.size()));
    memcpy(m_data,&data[0],sizeof(int64)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int64> &data)
{
    set_int64_vector(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_uint8_vector(const std::vector<uint8> &data)
{
    init(DataType::uint8(data.size()));
    memcpy(m_data,&data[0],sizeof(uint8)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint8> &data)
{
    set_uint8_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint16_vector(const std::vector<uint16> &data)
{
    init(DataType::uint16(data.size()));
    memcpy(m_data,&data[0],sizeof(uint16)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint16> &data)
{
    set_uint16_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint32_vector(const std::vector<uint32> &data)
{
    init(DataType::uint32(data.size()));
    memcpy(m_data,&data[0],sizeof(uint32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint32> &data)
{
    set_uint32_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint64_vector(const std::vector<uint64> &data)
{
    init(DataType::uint64(data.size()));
    memcpy(m_data,&data[0],sizeof(uint64)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint64> &data)
{
     set_uint64_vector(data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_float32_vector(const std::vector<float32> &data)
{
    init(DataType::float32(data.size()));
    memcpy(m_data,&data[0],sizeof(float32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<float32> &data)
{
    set_float32_vector(data);
}


//---------------------------------------------------------------------------//
void 
Node::set_float64_vector(const std::vector<float64> &data)
{
    init(DataType::float64(data.size()));
    memcpy(m_data,&data[0],sizeof(float64)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<float64> &data)
{
    set_float64_vector(data);
}


//-----------------------------------------------------------------------------
// set vector gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<char> &data)
{
    init(DataType::c_char(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}

//-----------------------------------------------------------------------------
void
Node::set(const std::vector<unsigned char> &data)
{
    init(DataType::c_unsigned_char(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<short> &data)
{
    init(DataType::c_short(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}

//-----------------------------------------------------------------------------
void
Node::set(const std::vector<unsigned short> &data)
{
    init(DataType::c_unsigned_short(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<int> &data)
{
    init(DataType::c_int(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}

//-----------------------------------------------------------------------------
void
Node::set(const std::vector<unsigned int> &data)
{
    init(DataType::c_unsigned_int(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<long> &data)
{
    init(DataType::c_long(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}

//-----------------------------------------------------------------------------
void
Node::set(const std::vector<unsigned long> &data)
{
    init(DataType::c_unsigned_long(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<float> &data)
{
    init(DataType::c_float(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set(const std::vector<double> &data)
{
    init(DataType::c_double(data.size()));
    memcpy(m_data,&data[0],sizeof(char)*data.size());
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// -- set via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_int8_ptr(int8  *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set(int8_array(data,DataType::int8(num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(int8  *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set_int8_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_int16_ptr(int16 *data, 
                    index_t num_elements,
                    index_t offset,
                    index_t stride,
                    index_t element_bytes,
                    index_t endianness)
{
    set(int16_array(data,DataType::int16(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));    
}

//---------------------------------------------------------------------------//
void 
Node::set(int16 *data, 
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set_int16_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_int32_ptr(int32 *data,
                    index_t num_elements,
                    index_t offset,
                    index_t stride,
                    index_t element_bytes,
                    index_t endianness)
{
    set(int32_array(data,DataType::int32(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));        
}

//---------------------------------------------------------------------------//
void 
Node::set(int32 *data,
         index_t num_elements,
         index_t offset,
         index_t stride,
         index_t element_bytes,
         index_t endianness)
{
    set_int32_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_int64_ptr(int64 *data,
                    index_t num_elements,
                    index_t offset,
                    index_t stride,
                    index_t element_bytes,
                    index_t endianness)
{
    set(int64_array(data,DataType::int64(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(int64 *data,
         index_t num_elements,
         index_t offset,
         index_t stride,
         index_t element_bytes,
         index_t endianness)
{
    set_int64_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//-----------------------------------------------------------------------------
// unsigned integer pointer cases
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void 
Node::set_uint8_ptr(uint8  *data,
                    index_t num_elements,
                    index_t offset,
                    index_t stride,
                    index_t element_bytes,
                    index_t endianness)
{
    set(uint8_array(data,DataType::uint8(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(uint8  *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set_uint8_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_uint16_ptr(uint16 *data,
                     index_t num_elements,
                     index_t offset,
                     index_t stride,
                     index_t element_bytes,
                     index_t endianness)
{
    set(uint16_array(data,DataType::uint16(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(uint16 *data,
         index_t num_elements,
         index_t offset,
         index_t stride,
         index_t element_bytes,
         index_t endianness)
{
    set_uint16_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}
//---------------------------------------------------------------------------//
void 
Node::set_uint32_ptr(uint32 *data, 
                     index_t num_elements,
                     index_t offset,
                     index_t stride,
                     index_t element_bytes,
                     index_t endianness)
{
    set(uint32_array(data,DataType::uint32(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness))); 
}

//---------------------------------------------------------------------------//
void 
Node::set(uint32 *data, 
         index_t num_elements,
         index_t offset,
         index_t stride,
         index_t element_bytes,
         index_t endianness)
{
    set_uint32_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}
               
//---------------------------------------------------------------------------//   
void 
Node::set_uint64_ptr(uint64 *data,
                     index_t num_elements,
                     index_t offset,
                     index_t stride,
                     index_t element_bytes,
                     index_t endianness)
{
    set(uint64_array(data,DataType::uint64(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness)));     
    
}

//---------------------------------------------------------------------------//   
void 
Node::set(uint64 *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set_uint64_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//-----------------------------------------------------------------------------
// floating point pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_float32_ptr(float32 *data,
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes,
                      index_t endianness)
{
    set(float32_array(data,DataType::float32(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(float32 *data,
         index_t num_elements,
         index_t offset,
         index_t stride,
         index_t element_bytes,
         index_t endianness)
{
    set_float32_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_float64_ptr(float64 *data, 
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes,
                      index_t endianness)
{
    set(float64_array(data,DataType::float64(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness)));
}

//---------------------------------------------------------------------------//
void 
Node::set(float64 *data, 
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set_float64_ptr(data,num_elements,offset,stride,element_bytes,endianness);
}


//-----------------------------------------------------------------------------
// set pointer gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set(signed char *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(char_array(data,DataType::c_char(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned char *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(unsigned_char_array(data,DataType::c_unsigned_char(num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set(short *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(short_array(data,DataType::c_short(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness)));
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned short *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(unsigned_short_array(data,DataType::c_unsigned_short(num_elements,
                                                             offset,
                                                             stride,
                                                             element_bytes,
                                                             endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set(int *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(int_array(data,DataType::c_int(num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness)));
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned int *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(unsigned_int_array(data,DataType::c_unsigned_int(num_elements,
                                                         offset,
                                                         stride,
                                                         element_bytes,
                                                         endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set(long *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(long_array(data,DataType::c_long(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness)));
}

//-----------------------------------------------------------------------------
void
Node::set(unsigned long *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(unsigned_long_array(data,DataType::c_unsigned_long(num_elements,
                                                           offset,
                                                           stride,
                                                           element_bytes,
                                                           endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set(float *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(float_array(data,DataType::c_float(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set(double *data,
          index_t num_elements,
          index_t offset,
          index_t stride,
          index_t element_bytes,
          index_t endianness)
{
    set(double_array(data,DataType::c_double(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness)));
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- end definition of Node set methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node set_path methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set_path for generic types --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_node(const std::string &path,
                    const Node& data) 
{
    fetch(path).set_node(data);    
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const Node& data) 
{
    set_path_node(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_dtype(const std::string &path,
                     const DataType& dtype)
{
    fetch(path).set_dtype(dtype);
}


//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const DataType& dtype)
{
    set_path_dtype(path,dtype);
}

//---------------------------------------------------------------------------//
void
Node::set_path_schema(const std::string &path,
                      const Schema &schema)
{
    fetch(path).set_schema(schema);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const Schema &schema)
{
    set_path_schema(path,schema);
}

//---------------------------------------------------------------------------//
void
Node::set_path_data_using_schema(const std::string &path,
                                 const Schema &schema,
                                 void *data)
{
    fetch(path).set_data_using_schema(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const Schema &schema,
               void *data)
{
    set_path_data_using_schema(path,schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_data_using_dtype(const std::string &path,
                                const DataType &dtype,
                                void *data)
{
    fetch(path).set_data_using_dtype(dtype,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const DataType &dtype,
               void *data)
{
    set_path_data_using_dtype(path,dtype,data);
}

//-----------------------------------------------------------------------------
// -- set_path for scalar types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_int8(const std::string &path,
                    int8 data)
{
    fetch(path).set_int8(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int8 data)
{
    set_path_int8(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int16(const std::string &path,
                     int16 data)
{
    fetch(path).set_int16(data);
}
 
//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int16 data)
{
    set_path_int16(path,data);
}


//---------------------------------------------------------------------------//
void
Node::set_path_int32(const std::string &path,
                     int32 data)
{
    fetch(path).set_int32(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int32 data)
{
    set_path_int32(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int64(const std::string &path,
                     int64 data)
{
    fetch(path).set_int64(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int64 data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// unsigned integer scalar types 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_uint8(const std::string &path,
                     uint8 data)
{
    fetch(path).set_uint8(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint8 data)
{
    set_path_uint8(path,data);
}
 
//---------------------------------------------------------------------------//
void
Node::set_path_uint16(const std::string &path,
                      uint16 data)
{
    fetch(path).set_uint16(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint16 data)
{
    set_path_uint16(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint32(const std::string &path,
                      uint32 data)
{
    fetch(path).set_uint32(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint32 data)
{
    set_path_uint32(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint64(const std::string &path,
                      uint64 data)
{
    fetch(path).set_uint64(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint64 data)
{
    set_path_uint64(path,data);
}

//-----------------------------------------------------------------------------
// floating point scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_float32(const std::string &path,
                       float32 data)
{
    fetch(path).set_float32(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               float32 data)
{
    set_path_float32(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_float64(const std::string &path,
                       float64 data)
{
    fetch(path).set_float64(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               float64 data)
{
    set_path_float64(path,data);
}

//-----------------------------------------------------------------------------
// set_path gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, signed char data)
{
    set_path(path,(CONDUIT_NATIVE_CHAR)data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, unsigned char data)
{
    set_path(path,(CONDUIT_NATIVE_UNSIGNED_CHAR)data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, short data)
{
    set_path(path,(CONDUIT_NATIVE_SHORT)data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, unsigned short data)
{
    set_path(path,(CONDUIT_NATIVE_UNSIGNED_SHORT)data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, int data)
{
    set_path(path,(CONDUIT_NATIVE_INT)data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, unsigned int data)
{
    set_path(path,(CONDUIT_NATIVE_UNSIGNED_INT)data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, long data)
{
    set_path(path,(CONDUIT_NATIVE_LONG)data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, unsigned long data)
{
    set_path(path,(CONDUIT_NATIVE_UNSIGNED_LONG)data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, float data)
{
    set_path(path,(CONDUIT_NATIVE_FLOAT)data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path, double data)
{
    set_path(path, (CONDUIT_NATIVE_DOUBLE)data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set_path for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_int8_array(const std::string &path,
                          const int8_array &data)
{
    fetch(path).set_int8_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const int8_array &data)
{
    set_path_int8_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int16_array(const std::string &path,
                           const int16_array &data)
{
    fetch(path).set_int16_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const int16_array &data)
{
    set_path_int16_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int32_array(const std::string &path,
                           const int32_array &data)
{
    fetch(path).set_int32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const int32_array &data)
{
    set_path_int32_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int64_array(const std::string &path,
                           const int64_array &data)
{
    fetch(path).set_int64_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const int64_array &data)
{
    set_path_int64_array(path,data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_uint8_array(const std::string &path,
                           const uint8_array &data)
{
    fetch(path).set_uint8_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const uint8_array &data)
{
    set_path_uint8_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint16_array(const std::string &path,
                            const uint16_array &data)
{
    fetch(path).set_uint16_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const uint16_array &data)
{
    set_path_uint16_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint32_array(const std::string &path,
                            const uint32_array &data)
{
    fetch(path).set_uint32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const uint32_array &data)
{
    set_path_uint32_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint64_array(const std::string &path,
                            const uint64_array &data)
{
    fetch(path).set_uint64_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const uint64_array &data)
{
    set_path_uint64_array(path,data);    
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_float32_array(const std::string &path,
                             const float32_array &data)
{
    fetch(path).set_float32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const float32_array &data)
{
    set_path_float32_array(path,data);    
}

//---------------------------------------------------------------------------//
void
Node::set_path_float64_array(const std::string &path,
                             const float64_array &data)
{
    fetch(path).set_float64_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const float64_array &data)
{
    set_path_float64_array(path,data);    
}

//-----------------------------------------------------------------------------
// set_path array gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const char_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const unsigned_char_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const short_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const unsigned_short_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const int_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const unsigned_int_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const long_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const unsigned_long_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const float_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const double_array &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// -- set_path for string types -- 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_string(const std::string &path,
                      const std::string &data)
{
    fetch(path).set_string(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::string &data)
{
    set_path_string(path,data);    
}
//---------------------------------------------------------------------------//
void
Node::set_path_char8_str(const std::string &path,
                         const char* data)
{
    fetch(path).set_char8_str(data);
}


//-----------------------------------------------------------------------------
// -- set_path for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_int8_vector(const std::string &path,
                           const std::vector<int8> &data)
{
    fetch(path).set_int8_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<int8> &data)
{
    set_path_int8_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int16_vector(const std::string &path,
                            const std::vector<int16> &data)
{
    fetch(path).set_int16_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<int16> &data)
{
    set_path_int16_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int32_vector(const std::string &path,
                            const std::vector<int32> &data)
{
    fetch(path).set_int32_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<int32> &data)
{
    set_path_int32_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int64_vector(const std::string &path,
                            const std::vector<int64> &data)
{
    fetch(path).set_int64_vector(data);    
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<int64> &data)
{
    set_path_int64_vector(path,data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_uint8_vector(const std::string &path,
                            const std::vector<uint8> &data)
{
    fetch(path).set_uint8_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<uint8> &data)
{
    set_path_uint8_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint16_vector(const std::string &path,
                             const std::vector<uint16> &data)
{
    fetch(path).set_uint16_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<uint16>  &data)
{
    set_path_uint16_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint32_vector(const std::string &path,
                             const std::vector<uint32> &data)
{
    fetch(path).set_uint32_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<uint32>  &data)
{
    set_path_uint32_vector(path,data);
}


//---------------------------------------------------------------------------//
void
Node::set_path_uint64_vector(const std::string &path,
                             const std::vector<uint64> &data)
{
    fetch(path).set_uint64_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::vector<uint64>  &data)
{
    set_path_uint64_vector(path,data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_float32_vector(const std::string &path,
                              const std::vector<float32> &data)
{
    fetch(path).set_float32_vector(data);
}


//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<float32> &data)
{
    set_path_float32_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_float64_vector(const std::string &path,
                              const std::vector<float64> &data)
{
    fetch(path).set_float64_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<float64> &data)
{
    set_path_float64_vector(path,data);
}


//-----------------------------------------------------------------------------
// set_path vector gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<char> &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<unsigned char> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<short> &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<unsigned short> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<int> &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<unsigned int> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<long> &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<unsigned long> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<float> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               const std::vector<double> &data)
{
    fetch(path).set(data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set_path via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_int8_ptr(const std::string &path,
                       int8  *data,
                       index_t num_elements,
                       index_t offset,
                       index_t stride,
                       index_t element_bytes,
                       index_t endianness)
{
    fetch(path).set_int8_ptr(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int8  *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_int8_ptr(path,
                      data,
                      num_elements,
                      offset,
                      stride,
                      element_bytes,
                      endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int16_ptr(const std::string &path,
                         int16 *data, 
                         index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    fetch(path).set_int16_ptr(data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int16 *data, 
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_int16_ptr(path,
                       data,
                       num_elements,
                       offset,
                       stride,
                       element_bytes,
                       endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int32_ptr(const std::string &path,
                         int32 *data,
                         index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    fetch(path).set_int32_ptr(data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int32 *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_int32_ptr(path,
                       data,
                       num_elements,
                       offset,
                       stride,
                       element_bytes,
                       endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_int64_ptr(const std::string &path,
                         int64 *data,
                         index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    fetch(path).set_int64_ptr(data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               int64 *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_int64_ptr(path,
                       data,
                       num_elements,
                       offset,
                       stride,
                       element_bytes,
                       endianness);
}

//----------------------------------------------------------------------------- 
// unsigned integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint8  *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint8_ptr(const std::string &path,
                         uint8  *data,
                         index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    fetch(path).set_uint8_ptr(data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint16_ptr(const std::string &path,
                          uint16 *data,
                          index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    fetch(path).set_uint16_ptr(data,
                               num_elements,
                               offset,
                               stride,
                               element_bytes,
                               endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint16 *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_uint16_ptr(path,
                        data,
                        num_elements,
                        offset,
                        stride,
                        element_bytes,
                        endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint32_ptr(const std::string &path,
                          uint32 *data, 
                          index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    fetch(path).set_uint32_ptr(data,
                               num_elements,
                               offset,
                               stride,
                               element_bytes,
                               endianness);
}
//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint32 *data, 
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_uint32_ptr(path,
                        data,
                        num_elements,
                        offset,
                        stride,
                        element_bytes,
                        endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_uint64_ptr(const std::string &path,
                          uint64 *data,
                          index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    fetch(path).set_uint64_ptr(data,
                               num_elements,
                               offset,
                               stride,
                               element_bytes,
                               endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               uint64 *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_uint64_ptr(path,
                        data,
                        num_elements,
                        offset,
                        stride,
                        element_bytes,
                        endianness);
}

//-----------------------------------------------------------------------------
// floating point integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_float32_ptr(const std::string &path,
                           float32 *data,
                           index_t num_elements,
                           index_t offset,
                           index_t stride,
                           index_t element_bytes,
                           index_t endianness)
{   
    fetch(path).set_float32_ptr(data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               float32 *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{   
    set_path_float32_ptr(path,
                         data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_float64_ptr(const std::string &path,
                           float64 *data, 
                           index_t num_elements,
                           index_t offset,
                           index_t stride,
                           index_t element_bytes,
                           index_t endianness)
{
    fetch(path).set_float64_ptr(data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}
//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               float64 *data, 
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    set_path_float64_ptr(path,
                         data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-----------------------------------------------------------------------------
// set_path pointer gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               signed char *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               unsigned char *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               short *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               unsigned short *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               int *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               unsigned int *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               long *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               unsigned long *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               float *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path(const std::string &path,
               double *data,
               index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- end definition of Node set_path methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Node set_external methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set_external for generic types --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_external_node(Node &node)
{
    reset();
    m_schema->set(node.schema());
    mirror_node(this,m_schema,&node);
}

//---------------------------------------------------------------------------//
void
Node::set_external(Node &node)
{
    set_external_node(node);
}

//---------------------------------------------------------------------------//
void
Node::set_external_data_using_schema(const Schema &schema,
                                     void *data)
{
    reset();
    m_schema->set(schema);
    walk_schema(this,m_schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_external(const Schema &schema,
                   void *data)
{
    set_external_data_using_schema(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_external_data_using_dtype(const DataType &dtype,
                                    void *data)
{
    reset();
    m_data    = data;
    m_schema->set(dtype);
}

//---------------------------------------------------------------------------//
void
Node::set_external(const DataType &dtype,
                   void *data)
{
    set_external_data_using_dtype(dtype,data);
}

//-----------------------------------------------------------------------------
// -- set_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_int8_ptr(int8 *data,
                            index_t num_elements,
                            index_t offset,
                            index_t stride,
                            index_t element_bytes,
                            index_t endianness)
{
    release();
    m_schema->set(DataType::int8(num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(int8 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_int8_ptr(data,
                          num_elements,
                          offset,
                          stride,
                          element_bytes,
                          endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int16_ptr(int16 *data,
                             index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    release();
    m_schema->set(DataType::int16(num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(int16 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_int16_ptr(data,
                           num_elements,
                           offset,
                           stride,
                           element_bytes,
                           endianness);
}


//---------------------------------------------------------------------------//
void 
Node::set_external_int32_ptr(int32 *data,
                             index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    release();
    m_schema->set(DataType::int32(num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(int32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_int32_ptr(data,
                           num_elements,
                           offset,
                           stride,
                           element_bytes,
                           endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int64_ptr(int64 *data,
                             index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    release();
    m_schema->set(DataType::int64(num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(int64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_int64_ptr(data,
                           num_elements,
                           offset,
                           stride,
                           element_bytes,
                           endianness);
}


//-----------------------------------------------------------------------------
// unsigned integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_uint8_ptr(uint8 *data,
                             index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    release();
    m_schema->set(DataType::uint8(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(uint8 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_uint8_ptr(data,
                           num_elements,
                           offset,
                           stride,
                           element_bytes,
                           endianness);
}



//---------------------------------------------------------------------------//
void 
Node::set_external_uint16_ptr(uint16 *data,
                              index_t num_elements,
                              index_t offset,
                              index_t stride,
                              index_t element_bytes,
                              index_t endianness)
{
    release();
    m_schema->set(DataType::uint16(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness));
    m_data  = data;
}


//---------------------------------------------------------------------------//
void 
Node::set_external(uint16 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_uint16_ptr(data,
                            num_elements,
                            offset,
                            stride,
                            element_bytes,
                            endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint32_ptr(uint32 *data,
                             index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    release();
    m_schema->set(DataType::uint32(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    m_data  = data;
}


//---------------------------------------------------------------------------//
void 
Node::set_external(uint32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_uint32_ptr(data,
                            num_elements,
                            offset,
                            stride,
                            element_bytes,
                            endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint64_ptr(uint64 *data,
                              index_t num_elements,
                              index_t offset,
                              index_t stride,
                              index_t element_bytes,
                              index_t endianness)
{
    release();
    m_schema->set(DataType::uint64(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    m_data  = data;
}


//---------------------------------------------------------------------------//
void 
Node::set_external(uint64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_uint64_ptr(data,
                            num_elements,
                            offset,
                            stride,
                            element_bytes,
                            endianness);
}
//-----------------------------------------------------------------------------
// floating point pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_float32_ptr(float32 *data,
                               index_t num_elements,
                               index_t offset,
                               index_t stride,
                               index_t element_bytes,
                               index_t endianness)
{
    release();
    m_schema->set(DataType::float32(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    m_data  = data;
}

//---------------------------------------------------------------------------//
void 
Node::set_external(float32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_float32_ptr(data,
                            num_elements,
                            offset,
                            stride,
                            element_bytes,
                            endianness);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_float64_ptr(float64 *data,
                               index_t num_elements,
                               index_t offset,
                               index_t stride,
                               index_t element_bytes,
                               index_t endianness)
{
    release();
    m_schema->set(DataType::float64(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    m_data  = data;
}
    
//---------------------------------------------------------------------------//
void 
Node::set_external(float64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    set_external_float64_ptr(data,
                            num_elements,
                            offset,
                            stride,
                            element_bytes,
                            endianness);
}

//-----------------------------------------------------------------------------
// set pointer gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_external(signed char *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_char(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    m_data  = data;
}

//-----------------------------------------------------------------------------
void
Node::set_external(unsigned char *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_unsigned_char(num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_external(short *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_short(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    m_data  = data;
}

//-----------------------------------------------------------------------------
void
Node::set_external(unsigned short *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_unsigned_short(num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_external(int *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_int(num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness));
    m_data  = data;
}

//-----------------------------------------------------------------------------
void
Node::set_external(unsigned int *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_unsigned_int(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_external(long *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_long(num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness));
    m_data  = data;
}

//-----------------------------------------------------------------------------
void
Node::set_external(unsigned long *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_unsigned_long(num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_external(float *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_float(num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_external(double *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::c_double(num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness));
    m_data  = data;
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_int8_array(const int8_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data   = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int8_array &data)
{
    set_external_int8_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int16_array(const int16_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int16_array &data)
{
    set_external_int16_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int32_array(const int32_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int32_array &data)
{
    set_external_int32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int64_array(const int64_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int64_array &data)
{
    set_external_int64_array(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_uint8_array(const uint8_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint8_array &data)
{
    set_external_uint8_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint16_array(const uint16_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint16_array &data)
{
    set_external_uint16_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint32_array(const uint32_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint32_array &data)
{
    set_external_uint32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint64_array(const uint64_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint64_array &data)
{
    set_external_uint64_array(data);
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_float32_array(const float32_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const float32_array &data)
{
    set_external_float32_array(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_float64_array(const float64_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const float64_array &data)
{
    set_external_float64_array(data);
}


//---------------------------------------------------------------------------//
void 
Node::set_external_char8_str(char *data)
{
    release();
    
    // size including the null term
    index_t str_size_with_term = strlen(data)+1;
    DataType str_t(DataType::CHAR8_STR_ID,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_ID);

    m_schema->set(str_t);
    m_data  = data;
}


//-----------------------------------------------------------------------------
// set_external array gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_external(const char_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//-----------------------------------------------------------------------------
void
Node::set_external(const unsigned_char_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_external(const short_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//-----------------------------------------------------------------------------
void
Node::set_external(const unsigned_short_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_external(const int_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//-----------------------------------------------------------------------------
void
Node::set_external(const unsigned_int_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_external(const long_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//-----------------------------------------------------------------------------
void
Node::set_external(const unsigned_long_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_external(const float_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_external(const double_array &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- set_external for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_int8_vector(std::vector<int8> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::int8(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int8> &data)
{
    set_external_int8_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int16_vector(std::vector<int16> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::int16(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int16> &data)
{
    set_external_int16_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int32_vector(std::vector<int32> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::int32(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int32> &data)
{
    set_external_int32_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_int64_vector(std::vector<int64> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::int64(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int64> &data)
{
    set_external_int64_vector(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_uint8_vector(std::vector<uint8> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::uint8(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint8> &data)
{
    set_external_uint8_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint16_vector(std::vector<uint16> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::uint16(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint16> &data)
{
    set_external_uint16_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint32_vector(std::vector<uint32> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::uint32(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint32> &data)
{
    set_external_uint32_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_uint64_vector(std::vector<uint64> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::uint64(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint64> &data)
{
    set_external_uint64_vector(data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external_float32_vector(std::vector<float32> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::float32(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<float32> &data)
{
    set_external_float32_vector(data);
}

//---------------------------------------------------------------------------//
void 
Node::set_external_float64_vector(std::vector<float64> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::float64(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<float64> &data)
{
    set_external_float64_vector(data);
}

//-----------------------------------------------------------------------------
// set_external vector gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<char> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_char(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<unsigned char> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_unsigned_char(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<short> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_short(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<unsigned short> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_unsigned_short(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<int> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_int(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<unsigned int> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_unsigned_int(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<long> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_long(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}

//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<unsigned long> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_unsigned_long(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<float> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_float(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_external(const std::vector<double> &data)
{
    release();
    index_t data_num_ele = (index_t)data.size();
    m_schema->set(DataType::c_double(data_num_ele));
    if(data_num_ele > 0)
        m_data  = (void*)&data[0];
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- end definition of Node set_external methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node set_path_external methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set_external for generic types --
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void
Node::set_path_external_node(const std::string &path,
                             Node &node)
{
    fetch(path).set_external_node(node);
}


//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        Node &node)
{
    set_path_external_node(path,node);
}


//---------------------------------------------------------------------------//
void
Node::set_path_external_data_using_schema(const std::string &path,
                                          const Schema &schema,
                                          void *data)
{
    fetch(path).set_external_data_using_schema(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const Schema &schema,
                        void *data)
{
    set_path_external_data_using_schema(path,schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_data_using_dtype(const std::string &path,
                                         const DataType &dtype,
                                         void *data)
{
    fetch(path).set_external_data_using_dtype(dtype,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const DataType &dtype,
                        void *data)
{
    set_path_external_data_using_dtype(path,dtype,data);
}

//-----------------------------------------------------------------------------
// -- set_path_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_int8_ptr(const std::string &path,
                                 int8  *data,
                                 index_t num_elements,
                                 index_t offset,
                                 index_t stride,
                                 index_t element_bytes,
                                 index_t endianness)
{
    fetch(path).set_external_int8_ptr(data,
                                      num_elements,
                                      offset,
                                      stride,
                                      element_bytes,
                                      endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        int8  *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_int8_ptr(path,
                               data,
                               num_elements,
                               offset,
                               stride,
                               element_bytes,
                               endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int16_ptr(const std::string &path,
                                  int16 *data, 
                                  index_t num_elements,
                                  index_t offset,
                                  index_t stride,
                                  index_t element_bytes,
                                  index_t endianness)
{
    fetch(path).set_external_int16_ptr(data,
                                       num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        int16 *data, 
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_int16_ptr(path,
                                data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int32_ptr(const std::string &path,
                                  int32 *data,
                                  index_t num_elements,
                                  index_t offset,
                                  index_t stride,
                                  index_t element_bytes,
                                  index_t endianness)
{
    fetch(path).set_external_int32_ptr(data,
                                       num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
           int32 *data,
           index_t num_elements,
           index_t offset,
           index_t stride,
           index_t element_bytes,
           index_t endianness)
{
    set_path_external_int32_ptr(path,
                                data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int64_ptr(const std::string &path,
                                  int64 *data,
                                  index_t num_elements,
                                  index_t offset,
                                  index_t stride,
                                  index_t element_bytes,
                                  index_t endianness)
{
    fetch(path).set_external_int64_ptr(data,
                                       num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        int64 *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_int64_ptr(path,
                                data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}

//-----------------------------------------------------------------------------
// unsigned integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint8_ptr(const std::string &path,
                                  uint8  *data,
                                  index_t num_elements,
                                  index_t offset,
                                  index_t stride,
                                  index_t element_bytes,
                                  index_t endianness)
{
    fetch(path).set_external_uint8_ptr(data,
                                       num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness);
}
//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        uint8  *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_uint8_ptr(path,
                                data,
                                num_elements,
                                offset,
                                stride,
                                element_bytes,
                                endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint16_ptr(const std::string &path,
                                   uint16 *data,
                                   index_t num_elements,
                                   index_t offset,
                                   index_t stride,
                                   index_t element_bytes,
                                   index_t endianness)
{
    fetch(path).set_external_uint16_ptr(data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        uint16 *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_uint16_ptr(path,
                                 data,
                                 num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint32_ptr(const std::string &path,
                                   uint32 *data, 
                                   index_t num_elements,
                                   index_t offset,
                                   index_t stride,
                                   index_t element_bytes,
                                   index_t endianness)
{
    fetch(path).set_external_uint32_ptr(data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        uint32 *data, 
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_uint32_ptr(path,
                                 data,
                                 num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness);
}


//---------------------------------------------------------------------------//
void
Node::set_path_external_uint64_ptr(const std::string &path,
                                   uint64 *data,
                                   index_t num_elements,
                                   index_t offset,
                                   index_t stride,
                                   index_t element_bytes,
                                   index_t endianness)
{
    fetch(path).set_external_uint64_ptr(data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        uint64 *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_uint64_ptr(path,
                                 data,
                                 num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness);
}

//-----------------------------------------------------------------------------
// floating point pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_float32_ptr(const std::string &path,
                                    float32 *data,
                                    index_t num_elements,
                                    index_t offset,
                                    index_t stride,
                                    index_t element_bytes,
                                    index_t endianness)
{
    fetch(path).set_external_float32_ptr(data,
                                         num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       float32 *data,
                       index_t num_elements,
                       index_t offset,
                       index_t stride,
                       index_t element_bytes,
                       index_t endianness)
{
    set_path_external_float32_ptr(path,
                                 data,
                                 num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness);
}



//---------------------------------------------------------------------------//
void
Node::set_path_external_float64_ptr(const std::string &path,
                                    float64 *data, 
                                    index_t num_elements,
                                    index_t offset,
                                    index_t stride,
                                    index_t element_bytes,
                                    index_t endianness)
{
    fetch(path).set_external_float64_ptr(data,
                                         num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness);
}


//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        float64 *data, 
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    set_path_external_float64_ptr(path,
                                 data,
                                 num_elements,
                                 offset,
                                 stride,
                                 element_bytes,
                                 endianness);
}

//-----------------------------------------------------------------------------
// set_path_external pointer gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        signed char *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        unsigned char *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        short *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        unsigned short *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        int *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        unsigned int *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        long *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        unsigned long *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        float *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        double *data,
                        index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// -- set_path_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_int8_array(const std::string &path,
                                   const int8_array  &data)
{
    fetch(path).set_external_int8_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const int8_array  &data)
{
    set_path_external_int8_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int16_array(const std::string &path,
                                    const int16_array &data)
{
    fetch(path).set_external_int16_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const int16_array &data)
{
    set_path_external_int16_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int32_array(const std::string &path,
                                    const int32_array &data)
{
    fetch(path).set_external_int32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const int32_array &data)
{
    set_path_external_int32_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int64_array(const std::string &path,
                                    const int64_array &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const int64_array &data)
{
    set_path_external_int64_array(path,data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint8_array(const std::string &path,
                                    const uint8_array  &data)
{
    fetch(path).set_external_uint8_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const uint8_array  &data)
{
    set_path_external_uint8_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint16_array(const std::string &path,
                                     const uint16_array &data)
{
    fetch(path).set_external_uint16_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const uint16_array &data)
{
    set_path_external_uint16_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint32_array(const std::string &path,
                        const uint32_array &data)
{
    fetch(path).set_external_uint32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const uint32_array &data)
{
    set_path_external_uint32_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint64_array(const std::string &path,
                                     const uint64_array &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const uint64_array &data)
{
    set_path_external_uint64_array(path,data);
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_float32_array(const std::string &path,
                                      const float32_array &data)
{
    fetch(path).set_external_float32_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const float32_array &data)
{
    set_path_external_float32_array(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_float64_array(const std::string &path,
                                      const float64_array &data)
{
    fetch(path).set_external_float64_array(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const float64_array &data)
{
    set_path_external_float64_array(path,data);
}


//-----------------------------------------------------------------------------
// set_path_external array gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const char_array &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const unsigned_char_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const short_array &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const unsigned_short_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const int_array &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const unsigned_int_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const long_array &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const unsigned_long_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const float_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const double_array &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// -- set_external for string types ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_char8_str(const std::string &path,
                                 char *data)
{
    fetch(path).set_external_char8_str(data);
}

//-----------------------------------------------------------------------------
// -- set_path_external for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_int8_vector(const std::string &path,
                                   std::vector<int8> &data)
{
    fetch(path).set_external_int8_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int8> &data)
{
    set_path_external_int8_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int16_vector(const std::string &path,
                                     std::vector<int16> &data)
{
    fetch(path).set_external_int16_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int16> &data)
{
    set_path_external_int16_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int32_vector(const std::string &path,
                                     std::vector<int32> &data)
{
    fetch(path).set_external_int32_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int32> &data)
{
    set_path_external_int32_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_int64_vector(const std::string &path,
                        std::vector<int64> &data)
{
    fetch(path).set_external_int64_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int64> &data)
{
    set_path_external_int64_vector(path,data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint8_vector(const std::string &path,
                                     std::vector<uint8> &data)
{
    fetch(path).set_external_uint8_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint8> &data)
{
    set_path_external_uint8_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint16_vector(const std::string &path,
                                      std::vector<uint16> &data)
{
    fetch(path).set_external_uint16_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint16> &data)
{
    set_path_external_uint16_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_uint32_vector(const std::string &path,
                                      std::vector<uint32> &data)
{
    fetch(path).set_external_uint32_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint32> &data)
{
    set_path_external_uint32_vector(path,data);
}


//---------------------------------------------------------------------------//
void
Node::set_path_external_uint64_vector(const std::string &path,
                                      std::vector<uint64> &data)
{
    fetch(path).set_external_uint64_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint64> &data)
{
    set_path_external_uint64_vector(path,data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external_float32_vector(const std::string &path,
                                       std::vector<float32> &data)
{
    fetch(path).set_external_float32_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<float32> &data)
{
    set_path_external_float32_vector(path,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external_float64_vector(const std::string &path,
                                       std::vector<float64> &data)
{
    fetch(path).set_external_float64_vector(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<float64> &data)
{
    set_path_external_float64_vector(path,data);
}


//-----------------------------------------------------------------------------
// set_path_external vector gap methods for c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<char> &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<unsigned char> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<short> &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<unsigned short> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<int> &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<unsigned int> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<long> &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<unsigned long> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<float> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
void
Node::set_path_external(const std::string &path,
                        const std::vector<double> &data)
{
    fetch(path).set_external(data);
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
//
// -- end definition of Node set_path_external methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Node assignment operators --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- assignment operators for generic types --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const Node &node)
{
    if(this != &node)
    {
        set(node);
    }
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const DataType &dtype)
{
    set(dtype);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const Schema &schema)
{
    set(schema);
    return *this;
}

//-----------------------------------------------------------------------------
// --  assignment operators for scalar types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(int8 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(int16 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(int32 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(int64 data)
{
    set(data);
    return *this;
}


//-----------------------------------------------------------------------------
// unsigned integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(uint8 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(uint16 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(uint32 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(uint64 data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// floating point scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(float32 data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(float64 data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// assignment operator gap methods for scalar c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
Node &
Node::operator=(signed char data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(unsigned char data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
Node &
Node::operator=(short data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(unsigned short data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
Node &
Node::operator=(int data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(unsigned int data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
Node &
Node::operator=(long data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(unsigned long data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
Node &
Node::operator=(float data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
Node &
Node::operator=(double data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- assignment operators for string types -- 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::string &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const char *data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// -- assignment operators for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const int8_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const int16_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const int32_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const int64_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const uint8_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const uint16_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const uint32_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const uint64_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const float32_array &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const float64_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// assignment operator gap methods for data array c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
Node &
Node::operator=(const char_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const unsigned_char_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
Node &
Node::operator=(const short_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const unsigned_short_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

 //-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
 //-----------------------------------------------------------------------------
Node &
Node::operator=(const int_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const unsigned_int_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
Node &
Node::operator=(const long_array &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const unsigned_long_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
Node &
Node::operator=(const float_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
Node &
Node::operator=(const double_array &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- assignment operators for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<int8> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<int16> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<int32> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<int64> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<uint8> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<uint16> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<uint32> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<uint64> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<float32> &data)
{
    set(data);
    return *this;
}

//---------------------------------------------------------------------------//
Node &
Node::operator=(const std::vector<float64> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
// assignment operator gap methods for vector c-native types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_CHAR
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<char> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<unsigned char> &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use char check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_SHORT
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<short> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<unsigned short> &data)
{
    set(data);
    return *this;
}//-----------------------------------------------------------------------------
#endif // end use short check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_INT
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<int> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<unsigned int> &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use int check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_LONG
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<long> &data)
{
    set(data);
    return *this;
}

//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<unsigned long> &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use long check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_FLOAT
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<float> &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use float check
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#ifndef CONDUIT_USE_DOUBLE
//-----------------------------------------------------------------------------
Node &
Node::operator=(const std::vector<double> &data)
{
    set(data);
    return *this;
}
//-----------------------------------------------------------------------------
#endif // end use double check
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- end definition of Node assignment operators --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Node transform methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- serialization methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::serialize(std::vector<uint8> &data) const
{
    data = std::vector<uint8>(total_bytes_compact(),0);
    serialize(&data[0],0);
}

//---------------------------------------------------------------------------//
void
Node::serialize(const std::string &stream_path) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    if(!ofs.is_open())
        CONDUIT_ERROR("<Node::serialize> failed to open: " << stream_path);
    serialize(ofs);
    ofs.close();
}


//---------------------------------------------------------------------------//
void
Node::serialize(std::ofstream &ofs) const
{
    index_t dtype_id = dtype().id();
    if( dtype_id == DataType::OBJECT_ID ||
        dtype_id == DataType::LIST_ID)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin(); itr < m_children.end(); ++itr)
        {
            (*itr)->serialize(ofs);
        }
    }
    else if( dtype_id != DataType::EMPTY_ID)
    {
        if(is_compact())
        {
            // ser as is. This copies stride * num_ele bytes
            {
                ofs.write((const char*)element_ptr(0),
                          total_bytes());
            }
        }
        else
        {
            // copy all elements 
            index_t c_num_bytes = total_bytes_compact();
            uint8 *buffer = new uint8[c_num_bytes];
            compact_elements_to(buffer);
            ofs.write((const char*)buffer,c_num_bytes);
            delete [] buffer;
        }
    }
}

//-----------------------------------------------------------------------------
// -- compaction methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::compact_to(Node &n_dest) const
{
    n_dest.reset();
    index_t c_size = total_bytes_compact();
    m_schema->compact_to(*n_dest.schema_ptr());
    n_dest.allocate(c_size);
    
    uint8 *n_dest_data = (uint8*)n_dest.m_data;
    compact_to(n_dest_data,0);
    // need node structure
    walk_schema(&n_dest,n_dest.m_schema,n_dest_data);

}

//-----------------------------------------------------------------------------
// -- update methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::update(const Node &n_src)
{
    // walk src and add it contents to this node
    /// TODO:
    /// arrays and non empty leafs will simply overwrite the current
    /// node, these semantics seem sensible, but we could revisit this
    index_t dtype_id = n_src.dtype().id();
    if( dtype_id == DataType::OBJECT_ID)
    {
        std::vector<std::string> src_paths;
        n_src.paths(src_paths);

        for (std::vector<std::string>::const_iterator itr = src_paths.begin();
             itr < src_paths.end(); ++itr)
        {
            std::string ent_name = *itr;
            fetch(ent_name).update(n_src.fetch(ent_name));
        }
    }
    else if( dtype_id == DataType::LIST_ID)
    {
        // if we are already a list type, then call update on the children
        //  in the list
        index_t src_idx = 0;
        index_t src_num_children = n_src.number_of_children();
        if( dtype().id() == DataType::LIST_ID)
        {
            index_t num_children = number_of_children();
            for(index_t idx=0; 
                (idx < num_children && idx < src_num_children); 
                idx++)
            {
                child(idx).update(n_src.child(idx));
                src_idx++;
            }
        }
        // if the current node is not a list, or if the src has more children
        // than the current node, use append to capture the nodes
        for(index_t idx = src_idx; idx < src_num_children;idx++)
        {
            append().update(n_src.child(idx));
        }
    }
    else if(dtype_id != DataType::EMPTY_ID) // TODO: Empty nodes not propagated?
    {
        // TODO: isn't this the same as a set?
        
        // don't use mem copy b/c we want to preserve striding holes
        
        // if you have the same type dtype, but less elements in the
        // src, it will copy them
        if( (this->dtype().id() == n_src.dtype().id()) &&
                 (this->dtype().number_of_elements() >=  
                   n_src.dtype().number_of_elements())) 
        {
            for(index_t idx = 0;
                idx < n_src.dtype().number_of_elements();
                idx++)
            {
                memcpy(element_ptr(idx),
                       n_src.element_ptr(idx), 
                       this->dtype().element_bytes());
            }
        }
        else // not compatible
        {
            n_src.compact_to(*this);
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::update_compatible(const Node &n_src)
{
    // walk src and copy contents to this node if their entries match
    index_t dtype_id = n_src.dtype().id();
    if( dtype_id == DataType::OBJECT_ID)
    {
        std::vector<std::string> src_paths;
        n_src.paths(src_paths);

        for (std::vector<std::string>::const_iterator itr = src_paths.begin();
             itr < src_paths.end(); ++itr)
        {
            std::string ent_name = *itr;
            if(has_path(ent_name))
            {
                fetch(ent_name).update_compatible(n_src.fetch(ent_name));
                }
        }
    }
    else if( dtype_id == DataType::LIST_ID)
    {
        // if we are already a list type, then call update_compatible on 
        // the children in the list
        index_t src_idx = 0;
        index_t src_num_children = n_src.number_of_children();
        if( dtype().id() == DataType::LIST_ID)
        {
            index_t num_children = number_of_children();
            for(index_t idx=0; 
                (idx < num_children && idx < src_num_children); 
                 idx++)
            {
                child(idx).update_compatible(n_src.child(idx));
                src_idx++;
            }
        }
    }
    else if(dtype_id != DataType::EMPTY_ID) // TODO: Empty nodes not propagated?
    {   
        // don't use mem copy b/c we want to preserve striding holes
        
        // if you have the same type dtype, but less elements in the
        // src, it will copy them 
        if( (this->dtype().id() == n_src.dtype().id()) &&
                 (this->dtype().number_of_elements() >=  
                   n_src.dtype().number_of_elements())) 
        {
            for(index_t idx = 0;
                idx < n_src.dtype().number_of_elements();
                idx++)
            {
                memcpy(element_ptr(idx),
                       n_src.element_ptr(idx), 
                       this->dtype().element_bytes());
            }
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::update_external(Node &n_src)
{
    // walk src and add it contents externally to this node
    index_t dtype_id = n_src.dtype().id();
    if( dtype_id == DataType::OBJECT_ID)
    {
        std::vector<std::string> src_paths;
        n_src.paths(src_paths);

        for (std::vector<std::string>::const_iterator itr = src_paths.begin();
             itr < src_paths.end(); ++itr)
        {
            std::string ent_name = *itr;
            fetch(ent_name).update_external(n_src.fetch(ent_name));
        }
    }
    else if( dtype_id == DataType::LIST_ID)
    {
        // if we are already a list type, then call update_external on the children
        //  in the list
        index_t src_idx = 0;
        index_t src_num_children = n_src.number_of_children();
        if( dtype().id() == DataType::LIST_ID)
        {
            index_t num_children = number_of_children();
            for(index_t idx=0; 
                (idx < num_children && idx < src_num_children); 
                idx++)
            {
                child(idx).update_external(n_src.child(idx));
                src_idx++;
            }
        }
        // if the current node is not a list, or if the src has more children
        // than the current node, use append to capture the nodes
        for(index_t idx = src_idx; idx < src_num_children;idx++)
        {
            append().update_external(n_src.child(idx));
        }
    }
    else if(dtype_id != DataType::EMPTY_ID) // TODO: Empty nodes not propagated?
    {
        // for leaf types, use update_external
        this->set_external(n_src);
    }
}


//-----------------------------------------------------------------------------
// -- endian related --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::endian_swap(index_t endianness)
{
    index_t dtype_id = dtype().id();

    if (dtype_id == DataType::OBJECT_ID || dtype_id == DataType::LIST_ID )
    {
        for(index_t i=0;i<number_of_children();i++)
        {
            child(i).endian_swap(endianness);
        }
    }
    else
    {
        index_t num_ele   = dtype().number_of_elements();
        //note: we always use the default bytes type for endian swap
        index_t ele_bytes = DataType::default_bytes(dtype_id);

        index_t src_endian  = dtype().endianness();
        index_t dest_endian = endianness;
    
        if(src_endian == Endianness::DEFAULT_ID)
        {
            src_endian = Endianness::machine_default();
        }
    
        if(dest_endian == Endianness::DEFAULT_ID)
        {
            dest_endian = Endianness::machine_default();
        }
        
        if(src_endian != dest_endian)
        {
            if(ele_bytes == 2)
            {
                for(index_t i=0;i<num_ele;i++)
                    Endianness::swap16(element_ptr(i));
            }
            else if(ele_bytes == 4)
            {
                for(index_t i=0;i<num_ele;i++)
                    Endianness::swap32(element_ptr(i));
            }
            else if(ele_bytes == 8)
            {
                for(index_t i=0;i<num_ele;i++)
                    Endianness::swap64(element_ptr(i));
            }
        }

        m_schema->dtype().set_endianness(dest_endian);
    }
}

//-----------------------------------------------------------------------------
// -- leaf coercion methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
int8
Node::to_int8() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return as_int8();
        case DataType::INT16_ID: return (int8)as_int16();
        case DataType::INT32_ID: return (int8)as_int32();
        case DataType::INT64_ID: return (int8)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (int8)as_uint8();
        case DataType::UINT16_ID: return (int8)as_uint16();
        case DataType::UINT32_ID: return (int8)as_uint32();
        case DataType::UINT64_ID: return (int8)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (int8)as_float32();
        case DataType::FLOAT64_ID: return (int8)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return (int8)res;
        }
        
    }
    return 0;
}


//---------------------------------------------------------------------------//
int16
Node::to_int16() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (int16)as_int8();
        case DataType::INT16_ID: return as_int16();
        case DataType::INT32_ID: return (int16)as_int32();
        case DataType::INT64_ID: return (int16)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (int16)as_uint8();
        case DataType::UINT16_ID: return (int16)as_uint16();
        case DataType::UINT32_ID: return (int16)as_uint32();
        case DataType::UINT64_ID: return (int16)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (int16)as_float32();
        case DataType::FLOAT64_ID: return (int16)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
int32
Node::to_int32() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (int32)as_int8();
        case DataType::INT16_ID: return (int32)as_int16();
        case DataType::INT32_ID: return as_int32();
        case DataType::INT64_ID: return (int32)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (int32)as_uint8();
        case DataType::UINT16_ID: return (int32)as_uint16();
        case DataType::UINT32_ID: return (int32)as_uint32();
        case DataType::UINT64_ID: return (int32)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (int32)as_float32();
        case DataType::FLOAT64_ID: return (int32)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int32 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
int64
Node::to_int64() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (int64)as_int8();
        case DataType::INT16_ID: return (int64)as_int16();
        case DataType::INT32_ID: return (int64)as_int32();
        case DataType::INT64_ID: return as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (int64)as_uint8();
        case DataType::UINT16_ID: return (int64)as_uint16();
        case DataType::UINT32_ID: return (int64)as_uint32();
        case DataType::UINT64_ID: return (int64)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (int64)as_float32();
        case DataType::FLOAT64_ID: return (int64)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int64 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
uint8
Node::to_uint8() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (uint8)as_int8();
        case DataType::INT16_ID: return (uint8)as_int16();
        case DataType::INT32_ID: return (uint8)as_int32();
        case DataType::INT64_ID: return (uint8)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return as_uint8();
        case DataType::UINT16_ID: return (uint8)as_uint16();
        case DataType::UINT32_ID: return (uint8)as_uint32();
        case DataType::UINT64_ID: return (uint8)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (uint8)as_float32();
        case DataType::FLOAT64_ID: return (uint8)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            uint16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return (uint8)res;

        }
    }
    return 0;
}


//---------------------------------------------------------------------------//
uint16
Node::to_uint16() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (uint16)as_int8();
        case DataType::INT16_ID: return (uint16)as_int16();
        case DataType::INT32_ID: return (uint16)as_int32();
        case DataType::INT64_ID: return (uint16)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (uint16)as_uint8();
        case DataType::UINT16_ID: return as_uint16();
        case DataType::UINT32_ID: return (uint16)as_uint32();
        case DataType::UINT64_ID: return (uint16)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (uint16)as_float32();
        case DataType::FLOAT64_ID: return (uint16)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            uint16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
uint32
Node::to_uint32() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (uint32)as_int8();
        case DataType::INT16_ID: return (uint32)as_int16();
        case DataType::INT32_ID: return (uint32)as_int32();
        case DataType::INT64_ID: return (uint32)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (uint32)as_uint8();
        case DataType::UINT16_ID: return (uint32)as_uint16();
        case DataType::UINT32_ID: return as_uint32();
        case DataType::UINT64_ID: return (uint32)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (uint32)as_float32();
        case DataType::FLOAT64_ID: return (uint32)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            uint32 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
uint64
Node::to_uint64() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (uint64)as_int8();
        case DataType::INT16_ID: return (uint64)as_int16();
        case DataType::INT32_ID: return (uint64)as_int32();
        case DataType::INT64_ID: return (uint64)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (uint64)as_uint8();
        case DataType::UINT16_ID: return (uint64)as_uint16();
        case DataType::UINT32_ID: return (uint64)as_uint32();
        case DataType::UINT64_ID: return as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (uint64)as_float32();
        case DataType::FLOAT64_ID: return (uint64)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            uint64 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}


//---------------------------------------------------------------------------//
float32
Node::to_float32() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (float32)as_int8();
        case DataType::INT16_ID: return (float32)as_int16();
        case DataType::INT32_ID: return (float32)as_int32();
        case DataType::INT64_ID: return (float32)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (float32)as_uint8();
        case DataType::UINT16_ID: return (float32)as_uint16();
        case DataType::UINT32_ID: return (float32)as_uint32();
        case DataType::UINT64_ID: return (float32)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return as_float32();
        case DataType::FLOAT64_ID: return (float32)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            float32 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0.0;
}


//---------------------------------------------------------------------------//
float64
Node::to_float64() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (float64)as_int8();
        case DataType::INT16_ID: return (float64)as_int16();
        case DataType::INT32_ID: return (float64)as_int32();
        case DataType::INT64_ID: return (float64)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (float64)as_uint8();
        case DataType::UINT16_ID: return (float64)as_uint16();
        case DataType::UINT32_ID: return (float64)as_uint32();
        case DataType::UINT64_ID: return (float64)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (float64)as_float32();
        case DataType::FLOAT64_ID: return as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            float64 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0.0;
}


//---------------------------------------------------------------------------//
index_t
Node::to_index_t() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (index_t)as_int8();
        case DataType::INT16_ID: return (index_t)as_int16();
        case DataType::INT32_ID: return (index_t)as_int32();
        case DataType::INT64_ID: return (index_t)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (index_t)as_uint8();
        case DataType::UINT16_ID: return (index_t)as_uint16();
        case DataType::UINT32_ID: return (index_t)as_uint32();
        case DataType::UINT64_ID: return (index_t)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (index_t)as_float32();
        case DataType::FLOAT64_ID: return (index_t)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            index_t res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
// -- std signed types -- //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
char
Node::to_char() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (char)as_int8();
        case DataType::INT16_ID: return (char)as_int16();
        case DataType::INT32_ID: return (char)as_int32();
        case DataType::INT64_ID: return (char)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (char)as_uint8();
        case DataType::UINT16_ID: return (char)as_uint16();
        case DataType::UINT32_ID: return (char)as_uint32();
        case DataType::UINT64_ID: return (char)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (char)as_float32();
        case DataType::FLOAT64_ID: return (char)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return (char)res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
short
Node::to_short() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (short)as_int8();
        case DataType::INT16_ID: return (short)as_int16();
        case DataType::INT32_ID: return (short)as_int32();
        case DataType::INT64_ID: return (short)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (short)as_uint8();
        case DataType::UINT16_ID: return (short)as_uint16();
        case DataType::UINT32_ID: return (short)as_uint32();
        case DataType::UINT64_ID: return (short)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (short)as_float32();
        case DataType::FLOAT64_ID: return (short)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            short res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
int
Node::to_int() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (int)as_int8();
        case DataType::INT16_ID: return (int)as_int16();
        case DataType::INT32_ID: return (int)as_int32();
        case DataType::INT64_ID: return (int)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (int)as_uint8();
        case DataType::UINT16_ID: return (int)as_uint16();
        case DataType::UINT32_ID: return (int)as_uint32();
        case DataType::UINT64_ID: return (int)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (int)as_float32();
        case DataType::FLOAT64_ID: return (int)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            int res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
    
}

//---------------------------------------------------------------------------//
long
Node::to_long() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (long)as_int8();
        case DataType::INT16_ID: return (long)as_int16();
        case DataType::INT32_ID: return (long)as_int32();
        case DataType::INT64_ID: return (long)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (long)as_uint8();
        case DataType::UINT16_ID: return (long)as_uint16();
        case DataType::UINT32_ID: return (long)as_uint32();
        case DataType::UINT64_ID: return (long)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (long)as_float32();
        case DataType::FLOAT64_ID: return (long)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            long res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
long long
Node::to_long_long() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (long long)as_int8();
        case DataType::INT16_ID: return (long long)as_int16();
        case DataType::INT32_ID: return (long long)as_int32();
        case DataType::INT64_ID: return (long long)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (long long)as_uint8();
        case DataType::UINT16_ID: return (long long)as_uint16();
        case DataType::UINT32_ID: return (long long)as_uint32();
        case DataType::UINT64_ID: return (long long)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (long long)as_float32();
        case DataType::FLOAT64_ID: return (long long)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            long long res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// -- std signed types -- //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
unsigned char
Node::to_unsigned_char() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (unsigned char)as_int8();
        case DataType::INT16_ID: return (unsigned char)as_int16();
        case DataType::INT32_ID: return (unsigned char)as_int32();
        case DataType::INT64_ID: return (unsigned char)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (unsigned char)as_uint8();
        case DataType::UINT16_ID: return (unsigned char)as_uint16();
        case DataType::UINT32_ID: return (unsigned char)as_uint32();
        case DataType::UINT64_ID: return (unsigned char)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (unsigned char)as_float32();
        case DataType::FLOAT64_ID: return (unsigned char)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            uint16 res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return (unsigned char)res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
unsigned short
Node::to_unsigned_short() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (unsigned short)as_int8();
        case DataType::INT16_ID: return (unsigned short)as_int16();
        case DataType::INT32_ID: return (unsigned short)as_int32();
        case DataType::INT64_ID: return (unsigned short)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (unsigned short)as_uint8();
        case DataType::UINT16_ID: return (unsigned short)as_uint16();
        case DataType::UINT32_ID: return (unsigned short)as_uint32();
        case DataType::UINT64_ID: return (unsigned short)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (unsigned short)as_float32();
        case DataType::FLOAT64_ID: return (unsigned short)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            unsigned short res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
unsigned int
Node::to_unsigned_int() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (unsigned int)as_int8();
        case DataType::INT16_ID: return (unsigned int)as_int16();
        case DataType::INT32_ID: return (unsigned int)as_int32();
        case DataType::INT64_ID: return (unsigned int)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (unsigned int)as_uint8();
        case DataType::UINT16_ID: return (unsigned int)as_uint16();
        case DataType::UINT32_ID: return (unsigned int)as_uint32();
        case DataType::UINT64_ID: return (unsigned int)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (unsigned int)as_float32();
        case DataType::FLOAT64_ID: return (unsigned int)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            unsigned int res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
unsigned long
Node::to_unsigned_long() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (unsigned long)as_int8();
        case DataType::INT16_ID: return (unsigned long)as_int16();
        case DataType::INT32_ID: return (unsigned long)as_int32();
        case DataType::INT64_ID: return (unsigned long)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (unsigned long)as_uint8();
        case DataType::UINT16_ID: return (unsigned long)as_uint16();
        case DataType::UINT32_ID: return (unsigned long)as_uint32();
        case DataType::UINT64_ID: return (unsigned long)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (unsigned long)as_float32();
        case DataType::FLOAT64_ID: return (unsigned long)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            unsigned long res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
        
    }
    return 0;
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
unsigned long long
Node::to_unsigned_long_long() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (unsigned long long)as_int8();
        case DataType::INT16_ID: return (unsigned long long)as_int16();
        case DataType::INT32_ID: return (unsigned long long)as_int32();
        case DataType::INT64_ID: return (unsigned long long)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (unsigned long long)as_uint8();
        case DataType::UINT16_ID: return (unsigned long long)as_uint16();
        case DataType::UINT32_ID: return (unsigned long long)as_uint32();
        case DataType::UINT64_ID: return (unsigned long long)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (unsigned long long)as_float32();
        case DataType::FLOAT64_ID: return (unsigned long long)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            unsigned long long res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
float
Node::to_float() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (float)as_int8();
        case DataType::INT16_ID: return (float)as_int16();
        case DataType::INT32_ID: return (float)as_int32();
        case DataType::INT64_ID: return (float)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (float)as_uint8();
        case DataType::UINT16_ID: return (float)as_uint16();
        case DataType::UINT32_ID: return (float)as_uint32();
        case DataType::UINT64_ID: return (float)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (float)as_float32();
        case DataType::FLOAT64_ID: return (float)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            float res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0;

}

//---------------------------------------------------------------------------//
double
Node::to_double() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (double)as_int8();
        case DataType::INT16_ID: return (double)as_int16();
        case DataType::INT32_ID: return (double)as_int32();
        case DataType::INT64_ID: return (double)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (double)as_uint8();
        case DataType::UINT16_ID: return (double)as_uint16();
        case DataType::UINT32_ID: return (double)as_uint32();
        case DataType::UINT64_ID: return (double)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (double)as_float32();
        case DataType::FLOAT64_ID: return (double)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            double res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }

    return 0; // TODO:: Error for Obj or list?
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
long double
Node::to_long_double() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:  return (long double)as_int8();
        case DataType::INT16_ID: return (long double)as_int16();
        case DataType::INT32_ID: return (long double)as_int32();
        case DataType::INT64_ID: return (long double)as_int64();
        /* uints */
        case DataType::UINT8_ID:  return (long double)as_uint8();
        case DataType::UINT16_ID: return (long double)as_uint16();
        case DataType::UINT32_ID: return (long double)as_uint32();
        case DataType::UINT64_ID: return (long double)as_uint64();
        /* floats */
        case DataType::FLOAT32_ID: return (long double)as_float32();
        case DataType::FLOAT64_ID: return (long double)as_float64();
        // string case
        case DataType::CHAR8_STR_ID:
        {
            long double res;
            std::stringstream ss(as_char8_str());
            if(ss >> res)
                return res;
        }
    }
    return 0; // TODO:: Error for Obj or list?
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/// convert array to a signed integer arrays
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Node::to_int8_array(Node &res)  const
{
    res.set(DataType::int8(dtype().number_of_elements()));
    int8_array res_array = res.as_int8_array();

    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to int8_array.");
        }
    }

}

//---------------------------------------------------------------------------//
void    
Node::to_int16_array(Node &res) const
{
    res.set(DataType::int16(dtype().number_of_elements()));
 
    int16_array res_array = res.as_int16_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to int16_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_int32_array(Node &res) const
{
    res.set(DataType::int32(dtype().number_of_elements()));
 
    int32_array res_array = res.as_int32_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to int32_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_int64_array(Node &res) const
{
    res.set(DataType::int64(dtype().number_of_elements()));
 
    int64_array res_array = res.as_int64_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to int64_array.");
        }
    }
}

//---------------------------------------------------------------------------//
/// convert array to a unsigned integer arrays
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Node::to_uint8_array(Node &res)  const
{
    res.set(DataType::uint8(dtype().number_of_elements()));
 
    uint8_array res_array = res.as_uint8_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to uint8_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_uint16_array(Node &res) const
{
    res.set(DataType::uint16(dtype().number_of_elements()));
 
    uint16_array res_array = res.as_uint16_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to uint16_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_uint32_array(Node &res) const
{
    res.set(DataType::uint32(dtype().number_of_elements()));
 
    uint32_array res_array = res.as_uint32_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to uint32_array.");
        }
    }

}

//---------------------------------------------------------------------------//
void
Node::to_uint64_array(Node &res) const
{
    res.set(DataType::uint64(dtype().number_of_elements()));
 
    uint64_array res_array = res.as_uint64_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to uint64_array.");
        }
    }

}

//---------------------------------------------------------------------------//
/// convert array to floating point arrays
//---------------------------------------------------------------------------//
void
Node::to_float32_array(Node &res) const
{
    res.set(DataType::float32(dtype().number_of_elements()));
 
    float32_array res_array = res.as_float32_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to float32_array.");
        }
    }

}

//---------------------------------------------------------------------------//
void
Node::to_float64_array(Node &res) const
{
    res.set(DataType::float64(dtype().number_of_elements()));
 
    float64_array res_array = res.as_float64_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to float64_array.");
        }
    }
}

//---------------------------------------------------------------------------//
/// convert array to c signed integer arrays
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
Node::to_char_array(Node &res) const
{
    res.set(DataType::c_char(dtype().number_of_elements()));
 
    char_array res_array = res.as_char_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to char_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_short_array(Node &res) const
{
    res.set(DataType::c_short(dtype().number_of_elements()));
 
    short_array res_array = res.as_short_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to short_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void    
Node::to_int_array(Node &res) const
{
    res.set(DataType::c_int(dtype().number_of_elements()));
 
    int_array res_array = res.as_int_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to int_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_long_array(Node &res) const
{
    res.set(DataType::c_long(dtype().number_of_elements()));
 
    long_array res_array = res.as_long_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to long_array.");
        }
    }
}


//---------------------------------------------------------------------------//
/// convert array to c unsigned integer arrays
//---------------------------------------------------------------------------//
void
Node::to_unsigned_char_array(Node &res) const
{
    res.set(DataType::c_unsigned_char(dtype().number_of_elements()));
 
    unsigned_char_array res_array = res.as_unsigned_char_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to unsigned_char_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_unsigned_short_array(Node &res) const
{
    res.set(DataType::c_unsigned_short(dtype().number_of_elements()));
 
    unsigned_short_array res_array = res.as_unsigned_short_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to unsigned_short_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_unsigned_int_array(Node &res) const
{
    res.set(DataType::c_unsigned_int(dtype().number_of_elements()));
 
    unsigned_int_array res_array = res.as_unsigned_int_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to unsigned_int_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_unsigned_long_array(Node &res) const
{
    res.set(DataType::c_unsigned_long(dtype().number_of_elements()));
 
    unsigned_long_array res_array = res.as_unsigned_long_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to unsigned_long_array.");
        }
    }
}

/// convert array to c floating point arrays
//---------------------------------------------------------------------------//
void
Node::to_float_array(Node &res) const
{
    res.set(DataType::c_float(dtype().number_of_elements()));
 
    float_array res_array = res.as_float_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to float_array.");
        }
    }
}

//---------------------------------------------------------------------------//
void
Node::to_double_array(Node &res) const
{
    res.set(DataType::c_double(dtype().number_of_elements()));
 
    double_array res_array = res.as_double_array();
 
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_ID:
        {   
            res_array.set(this->as_int8_array());
            break;
        }
        case DataType::INT16_ID: 
        {
            res_array.set(this->as_int16_array());
            break;
        }
        case DataType::INT32_ID:
        {
            res_array.set(this->as_int32_array());
            break;
        }
        case DataType::INT64_ID:
        {
            res_array.set(this->as_int64_array());
            break;
        }
        /* uints */
        case DataType::UINT8_ID:
        {
            res_array.set(this->as_uint8_array());
            break;
        }
        case DataType::UINT16_ID: 
        {
            res_array.set(this->as_uint16_array());
            break;
        }
        case DataType::UINT32_ID:
        {
            res_array.set(this->as_uint32_array());
            break;
        }
        case DataType::UINT64_ID:
        {
            res_array.set(this->as_uint64_array());
            break;
        }
        /* floats */
        case DataType::FLOAT32_ID:
        {
            res_array.set(this->as_float32_array());
            break;
        }
        case DataType::FLOAT64_ID: 
        {
            res_array.set(this->as_float64_array());
            break;
        }
        default:
        {
            // error
            CONDUIT_ERROR("Cannot convert non numeric " 
                        << dtype().name() 
                        << " type to double_array.");
        }
    }

}


//-----------------------------------------------------------------------------
// -- Value Helper class ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node::Value::Value(Node *node, bool coerse)
:m_node(node),
 m_coerse(coerse)
{
    
}

//---------------------------------------------------------------------------//
Node::Value::Value(const Value &value)
:m_node(value.m_node), 
 m_coerse(value.m_coerse)
{
    
}

//---------------------------------------------------------------------------//
Node::Value::~Value()
{
    
}

// TODO: why do we have to use a signed char here, does it have to do with
// how we danced around setting string types?
//---------------------------------------------------------------------------//
Node::Value::operator signed char() const
{
    if(m_coerse)
        return m_node->to_char();
    else
        return m_node->as_char();
}

////---------------------------------------------------------------------------//
//Node::Value::operator char() const
//{
//    if(m_coerse)
//        return m_node->to_char();
//    else
//        return m_node->as_char();
//}

//---------------------------------------------------------------------------//
Node::Value::operator short() const
{
    if(m_coerse)
        return m_node->to_short();
    else
        return m_node->as_short();
}

//---------------------------------------------------------------------------//
Node::Value::operator int() const
{
    if(m_coerse)
        return m_node->to_int();
    else
        return m_node->as_int();
}

//---------------------------------------------------------------------------//
Node::Value::operator long() const
{
    if(m_coerse)
        return m_node->to_long();
    else
        return m_node->as_long();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator long long() const
{
    if(m_coerse)
        return m_node->to_long_long();
    else
        return m_node->as_long_long();
}
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator unsigned char() const
{
    if(m_coerse)
        return m_node->to_unsigned_char();
    else
        return m_node->as_unsigned_char();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned short() const
{
    if(m_coerse)
        return m_node->to_unsigned_short();
    else
        return m_node->as_unsigned_short();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned int() const
{
    if(m_coerse)
        return m_node->to_unsigned_int();
    else
        return m_node->as_unsigned_int();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned long() const
{
    if(m_coerse)
        return m_node->to_unsigned_long();
    else
        return m_node->as_unsigned_long();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator unsigned long long() const
{
    if(m_coerse)
        return m_node->to_unsigned_long_long();
    else
        return m_node->as_unsigned_long_long();
}
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
Node::Value::operator float() const
{
    if(m_coerse)
        return m_node->to_float();
    else
        return m_node->as_float();
}

//---------------------------------------------------------------------------//
Node::Value::operator double() const
{
    if(m_coerse)
        return m_node->to_double();
    else
        return m_node->as_double();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::Value::operator long double() const
{
    if(m_coerse)
        return m_node->to_long_double();
    else
        return m_node->as_long_double();
}
#endif

//---------------------------------------------------------------------------//
// -- pointer casts -- 
// (no coercion)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator char*() const
{
    return m_node->as_char_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator signed char*() const
{
    return (signed char*)m_node->as_char_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator short*() const
{
    return m_node->as_short_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator int*() const
{
    return m_node->as_int_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator long*() const
{
    return m_node->as_long_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator long long*() const
{
    return m_node->as_long_long_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator unsigned char*() const
{
    return m_node->as_unsigned_char_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned short*() const
{
    return m_node->as_unsigned_short_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned int*() const
{
    return m_node->as_unsigned_int_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned long*() const
{
    return m_node->as_unsigned_long_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator unsigned long long*() const
{
    return m_node->as_unsigned_long_long_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator float*() const
{
    return m_node->as_float_ptr();
}

//---------------------------------------------------------------------------//
Node::Value::operator double*() const
{
    return m_node->as_double_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::Value::operator long double*() const
{
    return m_node->as_long_double_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// -- array casts -- 
// (no coercion)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator char_array() const
{
    return m_node->as_char_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator short_array() const
{
    return m_node->as_short_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator int_array() const
{
    return m_node->as_int_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator long_array() const
{
    return m_node->as_long_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator long_long_array() const
{
    return m_node->as_long_long_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::Value::operator unsigned_char_array() const
{
    return m_node->as_unsigned_char_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned_short_array() const
{
    return m_node->as_unsigned_short_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned_int_array() const
{
    return m_node->as_unsigned_int_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator unsigned_long_array() const
{
    return m_node->as_unsigned_long_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::Value::operator unsigned_long_long_array() const
{
    return m_node->as_unsigned_long_long_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
Node::Value::operator float_array() const
{
    return m_node->as_float_array();
}

//---------------------------------------------------------------------------//
Node::Value::operator double_array() const
{
    return m_node->as_double_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::Value::operator long_double_array() const
{
    return m_node->as_long_double_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//-----------------------------------------------------------------------------
// -- ConstValue Helper class ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Node::ConstValue::ConstValue(const Node *node, bool coerse)
:m_node(node),
 m_coerse(coerse)
{

}

//---------------------------------------------------------------------------//
Node::ConstValue::ConstValue(const ConstValue &value)
:m_node(value.m_node), 
 m_coerse(value.m_coerse)
{

}

//---------------------------------------------------------------------------//
Node::ConstValue::ConstValue(const Value &value)
:m_node(value.m_node), 
 m_coerse(value.m_coerse)
{

}

//---------------------------------------------------------------------------//
Node::ConstValue::~ConstValue()
{

}

// TODO: why do we have to use a signed char here, does it have to do with
// how we danced around setting string types?
//---------------------------------------------------------------------------//
Node::ConstValue::operator signed char() const
{
    if(m_coerse)
        return m_node->to_char();
    else
        return m_node->as_char();
}

////---------------------------------------------------------------------------//
//Node::ConstValue::operator char() const
//{
//    if(m_coerse)
//        return m_node->to_char();
//    else
//        return m_node->as_char();
//}

//---------------------------------------------------------------------------//
Node::ConstValue::operator short() const
{
    if(m_coerse)
        return m_node->to_short();
    else
        return m_node->as_short();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator int() const
{
    if(m_coerse)
        return m_node->to_int();
    else
        return m_node->as_int();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator long() const
{
    if(m_coerse)
        return m_node->to_long();
    else
        return m_node->as_long();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator long long() const
{
    if(m_coerse)
        return m_node->to_long_long();
    else
        return m_node->as_long_long();
}
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator unsigned char() const
{
    if(m_coerse)
        return m_node->to_unsigned_char();
    else
        return m_node->as_unsigned_char();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator unsigned short() const
{
    if(m_coerse)
        return m_node->to_unsigned_short();
    else
        return m_node->as_unsigned_short();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator unsigned int() const
{
    if(m_coerse)
        return m_node->to_unsigned_int();
    else
        return m_node->as_unsigned_int();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator unsigned long() const
{
    if(m_coerse)
        return m_node->to_unsigned_long();
    else
        return m_node->as_unsigned_long();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator unsigned long long() const
{
    if(m_coerse)
        return m_node->to_unsigned_long_long();
    else
        return m_node->as_unsigned_long_long();
}
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
Node::ConstValue::operator float() const
{
    if(m_coerse)
        return m_node->to_float();
    else
        return m_node->as_float();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator double() const
{
    if(m_coerse)
        return m_node->to_double();
    else
        return m_node->as_double();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::ConstValue::operator long double() const
{
    if(m_coerse)
        return m_node->to_long_double();
    else
        return m_node->as_long_double();
}
#endif

//---------------------------------------------------------------------------//
// -- pointer casts -- 
// (no coercion)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator const char*() const
{
    return m_node->as_char_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const signed char*() const
{
    return (signed char*)m_node->as_char_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const short*() const
{
    return m_node->as_short_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const int*() const
{
    return m_node->as_int_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const long*() const
{
    return m_node->as_long_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator const long long*() const
{
    return m_node->as_long_long_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned char*() const
{
    return m_node->as_unsigned_char_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned short*() const
{
    return m_node->as_unsigned_short_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned int*() const
{
    return m_node->as_unsigned_int_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned long*() const
{
    return m_node->as_unsigned_long_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned long long*() const
{
    return m_node->as_unsigned_long_long_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator const float*() const
{
    return m_node->as_float_ptr();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const double*() const
{
    return m_node->as_double_ptr();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::ConstValue::operator const long double*() const
{
    return m_node->as_long_double_ptr();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// -- array casts -- 
// (no coercion)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator const char_array() const
{
    return m_node->as_char_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const short_array() const
{
    return m_node->as_short_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const int_array() const
{
    return m_node->as_int_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const long_array() const
{
    return m_node->as_long_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator const long_long_array() const
{
    return m_node->as_long_long_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned_char_array() const
{
    return m_node->as_unsigned_char_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned_short_array() const
{
    return m_node->as_unsigned_short_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned_int_array() const
{
    return m_node->as_unsigned_int_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned_long_array() const
{
    return m_node->as_unsigned_long_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
Node::ConstValue::operator const unsigned_long_long_array() const
{
    return m_node->as_unsigned_long_long_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
Node::ConstValue::operator const float_array() const
{
    return m_node->as_float_array();
}

//---------------------------------------------------------------------------//
Node::ConstValue::operator const double_array() const
{
    return m_node->as_double_array();
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
Node::ConstValue::operator const long_double_array() const
{
    return m_node->as_long_double_array();
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//-----------------------------------------------------------------------------
// End Node::ConstValue
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- JSON construction methods ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::string
Node::to_json(const std::string &protocol, 
              index_t indent, 
              index_t depth,
              const std::string &pad,
              const std::string &eoe) const
{
    if(protocol == "json")
    {
        return to_pure_json(indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_json")
    {
        return to_detailed_json(indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_base64_json")
    {
        return to_base64_json(indent,depth,pad,eoe);
    }
    else
    {
        CONDUIT_ERROR("Unknown to_json protocol:" << protocol);
    }

    return "{}";
}

//-----------------------------------------------------------------------------
void
Node::to_json_stream(const std::string &stream_path,
                     const std::string &protocol,
                     index_t indent, 
                     index_t depth,
                     const std::string &pad,
                     const std::string &eoe) const
{
    if(protocol == "json")
    {
        return to_pure_json(stream_path,indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_json")
    {
        return to_detailed_json(stream_path,indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_base64_json")
    {
        return to_base64_json(stream_path,indent,depth,pad,eoe);        
    }
    else
    {
        CONDUIT_ERROR("Unknown to_json protocol:" << protocol);
    }
}
//-----------------------------------------------------------------------------
void
Node::to_json_stream(std::ostream &os,
                     const std::string &protocol,
                     index_t indent, 
                     index_t depth,
                     const std::string &pad,
                     const std::string &eoe) const
{
    if(protocol == "json")
    {
        return to_pure_json(os,indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_json")
    {
        return to_detailed_json(os,indent,depth,pad,eoe);
    }
    else if(protocol == "conduit_base64_json")
    {
        return to_base64_json(os,indent,depth,pad,eoe);        
    }
    else
    {
        CONDUIT_ERROR("Unknown to_json protocol:" << protocol);
    }
}

//---------------------------------------------------------------------------//
// Private to_json helpers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
std::string 
Node::to_json_generic(bool detailed,
                      index_t indent, 
                      index_t depth,
                      const std::string &pad,
                      const std::string &eoe) const
{
    std::ostringstream oss;
    to_json_generic(oss,detailed,indent,depth,pad,eoe);
    return oss.str();
}


//---------------------------------------------------------------------------//
void
Node::to_json_generic(const std::string &stream_path,
                      bool detailed, 
                      index_t indent, 
                      index_t depth,
                      const std::string &pad,
                      const std::string &eoe) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    if(!ofs.is_open())
        CONDUIT_ERROR("<Node::to_json> failed to open: " << stream_path);
    to_json_generic(ofs,detailed,indent,depth,pad,eoe);
    ofs.close();
}


//---------------------------------------------------------------------------//
void
Node::to_json_generic(std::ostream &os,
                      bool detailed, 
                      index_t indent, 
                      index_t depth,
                      const std::string &pad,
                      const std::string &eoe) const
{
    std::ios_base::fmtflags prev_stream_flags(os.flags());
    os.precision(15);
    if(dtype().id() == DataType::OBJECT_ID)
    {
        os << eoe;
        utils::indent(os,indent,depth,pad);
        os << "{" << eoe;
    
        index_t nchildren = (index_t) m_children.size();
        for(index_t i=0; i <  nchildren;i++)
        {
            utils::indent(os,indent,depth+1,pad);
            os << "\""<< m_schema->object_order()[i] << "\": ";
            m_children[i]->to_json_generic(os,
                                           detailed,
                                           indent,
                                           depth+1,
                                           pad,
                                           eoe);
            if(i < nchildren-1)
                os << ",";
            os << eoe;
        }
        utils::indent(os,indent,depth,pad);
        os << "}";
    }
    else if(dtype().id() == DataType::LIST_ID)
    {
        os << eoe;
        utils::indent(os,indent,depth,pad);
        os << "[" << eoe;
        
        index_t nchildren = m_children.size();
        for(index_t i=0; i < nchildren;i++)
        {
            utils::indent(os,indent,depth+1,pad);
            m_children[i]->to_json_generic(os,
                                           detailed,
                                           indent,
                                           depth+1,
                                           pad,
                                           eoe);
            if(i < nchildren-1)
                os << ",";
            os << eoe;
        }
        utils::indent(os,indent,depth,pad);
        os << "]";      
    }
    else // assume leaf data type
    {
        if(detailed)
        {
            std::string dtype_json = dtype().to_json();
            std::string dtype_open;
            std::string dtype_rest;

            // trim the last "}"
            utils::split_string(dtype_json,
                                "}",
                                dtype_open,
                                dtype_rest);
            os<< dtype_open;
            os << ", \"value\": ";
        }

        switch(dtype().id())
        {
            // ints 
            case DataType::INT8_ID:
                as_int8_array().to_json(os);
                break;
            case DataType::INT16_ID:
                as_int16_array().to_json(os);
                break;
            case DataType::INT32_ID:
                as_int32_array().to_json(os);
                break;
            case DataType::INT64_ID:
                as_int64_array().to_json(os);
                break;
            // uints 
            case DataType::UINT8_ID:
                as_uint8_array().to_json(os);
                break;
            case DataType::UINT16_ID: 
                as_uint16_array().to_json(os);
                break;
            case DataType::UINT32_ID:
                as_uint32_array().to_json(os);
                break;
            case DataType::UINT64_ID:
                as_uint64_array().to_json(os);
                break;
            // floats 
            case DataType::FLOAT32_ID:
                as_float32_array().to_json(os);
                break;
            case DataType::FLOAT64_ID:
                as_float64_array().to_json(os);
                break;
            // char8_str
            case DataType::CHAR8_STR_ID: 
                os << "\"" 
                   << utils::escape_special_chars(as_string())
                   << "\""; 
                break;
            // empty
            case DataType::EMPTY_ID: 
                os << "null";
                break;

        }

        if(detailed)
        {
            // complete json entry 
            os << "}";
        }
    }  
    
    os.flags(prev_stream_flags);
}

//---------------------------------------------------------------------------//
std::string
Node::to_pure_json(index_t indent,
                   index_t depth,
                   const std::string &pad,
                   const std::string &eoe) const
{
    return to_json_generic(false,indent,depth,pad,eoe);
}

//---------------------------------------------------------------------------//
void
Node::to_pure_json(const std::string &stream_path,
                   index_t indent,
                   index_t depth,
                   const std::string &pad,
                   const std::string &eoe) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    if(!ofs.is_open())
        CONDUIT_ERROR("<Node::to_pure_json> failed to open: " << stream_path);
    to_json_generic(ofs,false,indent,depth,pad,eoe);
    ofs.close();
}

//---------------------------------------------------------------------------//
void
Node::to_pure_json(std::ostream &os,
                   index_t indent,
                   index_t depth,
                   const std::string &pad,
                   const std::string &eoe) const
{
    to_json_generic(os,false,indent,depth,pad,eoe);
}

//---------------------------------------------------------------------------//
std::string
Node::to_detailed_json(index_t indent, 
                       index_t depth,
                       const std::string &pad,
                       const std::string &eoe) const
{
    return to_json_generic(true,indent,depth,pad,eoe);
}

//---------------------------------------------------------------------------//
void
Node::to_detailed_json(const std::string &stream_path,
                       index_t indent, 
                       index_t depth,
                       const std::string &pad,
                       const std::string &eoe) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    if(!ofs.is_open())
        CONDUIT_ERROR("<Node::to_pure_json> failed to open: " << stream_path);
    to_json_generic(ofs,true,indent,depth,pad,eoe);
    ofs.close();
}


//---------------------------------------------------------------------------//
void
Node::to_detailed_json(std::ostream &os,
                       index_t indent, 
                       index_t depth,
                       const std::string &pad,
                       const std::string &eoe) const
{
    to_json_generic(os,true,indent,depth,pad,eoe);
}

//---------------------------------------------------------------------------//
std::string
Node::to_base64_json(index_t indent,
                     index_t depth,
                     const std::string &pad,
                     const std::string &eoe) const
{
   std::ostringstream oss;
   to_base64_json(oss,indent,depth,pad,eoe);
   return oss.str();
}

//---------------------------------------------------------------------------//
void
Node::to_base64_json(const std::string &stream_path,
                     index_t indent,
                     index_t depth,
                     const std::string &pad,
                     const std::string &eoe) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    if(!ofs.is_open())
        CONDUIT_ERROR("<Node::to_base64_json> failed to open: " << stream_path);
    to_base64_json(ofs,indent,depth,pad,eoe);
    ofs.close();
}


//---------------------------------------------------------------------------//
void
Node::to_base64_json(std::ostream &os,
                     index_t indent,
                     index_t depth,
                     const std::string &pad,
                     const std::string &eoe) const
{
    std::ios_base::fmtflags prev_stream_flags(os.flags());
    os.precision(15);
        
    // we need compact data
    Node n;
    compact_to(n);
    
    // use libb64 to encode the data
    index_t nbytes = n.schema().total_bytes();
    index_t enc_buff_size =  utils::base64_encode_buffer_size(nbytes);
    Node bb64_data;
    bb64_data.set(DataType::char8_str(enc_buff_size));
    
    const char *src_ptr = (const char*)n.data_ptr();
    char *dest_ptr       = (char*)bb64_data.data_ptr();
    memset(dest_ptr,0,enc_buff_size);

    utils::base64_encode(src_ptr,nbytes,dest_ptr);
    
    // create the resulting json
    
    os << eoe;
    utils::indent(os,indent,depth,pad);
    os << "{" << eoe;
    utils::indent(os,indent,depth+1,pad);
    os << "\"schema\": ";

    n.schema().to_json_stream(os,true,indent,depth+1,pad,eoe);

    os  << "," << eoe;
    
    utils::indent(os,indent,depth+1,pad);
    os << "\"data\": " << eoe;
    utils::indent(os,indent,depth+1,pad);
    os << "{" << eoe;
    utils::indent(os,indent,depth+2,pad);
    os << "\"base64\": ";
    bb64_data.to_pure_json(os,0,0,"","");
    os << eoe;
    utils::indent(os,indent,depth+1,pad);
    os << "}" << eoe;
    utils::indent(os,indent,depth,pad);
    os << "}";
    
    os.flags(prev_stream_flags);
}



//-----------------------------------------------------------------------------
//
// -- end definition of Node transform methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node information methods --
//
//-----------------------------------------------------------------------------

// NOTE: several other Node information methods are inlined in Node.h

//---------------------------------------------------------------------------//
void
Node::info(Node &res) const
{
    res.reset();
    info(res,std::string());

    
    // update summary
    index_t tb_alloc = 0;
    index_t tb_mmap  = 0;

    // for each in mem_spaces:
    res["total_bytes"]         = total_bytes();
    res["total_bytes_compact"] = total_bytes_compact();
    
    std::vector<std::string> mchildren;
    Node &mspaces = res["mem_spaces"];
    
    NodeIterator itr = mspaces.children();
    
    while(itr.has_next())
    {
        Node &mspace = itr.next();
        std::string mtype  = mspace["type"].as_string();
        if( mtype == "alloced")
        {
            tb_alloc += mspace["bytes"].to_index_t();
        }
        else if(mtype == "mmaped")
        {
            tb_mmap  += mspace["bytes"].to_index_t();
        }
    }
    res["total_bytes_alloced"] = tb_alloc;
    res["total_bytes_mmaped"]  = tb_mmap;
}

//---------------------------------------------------------------------------//
Node
Node::info()const
{
    // NOTE: this very inefficient w/o move semantics!
    Node res;
    info(res);
    return res;
}

//-----------------------------------------------------------------------------
// -- stdout print methods ---
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void
Node::print() const
{
    to_json_stream(std::cout);
}

//-----------------------------------------------------------------------------
void
Node::print_detailed() const
{
    to_json_stream(std::cout,"conduit_json");
}


//-----------------------------------------------------------------------------
//
// -- end definition of Node information methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin definition of Node entry access methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
NodeIterator
Node::children()
{
    return NodeIterator(this,0);
}

//---------------------------------------------------------------------------//
NodeConstIterator
Node::children() const
{
    return NodeConstIterator(this);
}


//---------------------------------------------------------------------------//
const Node&
Node::fetch_child(const std::string &path) const
{
    // const fetch w/ path requires object role
    if(!dtype().is_object())
    {
        CONDUIT_ERROR("Cannot const fetch_child, Node(" << this->path()
                      << ") is not an object");
    }
    
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // check for parent
    if(p_curr == "..")
    {
        if(m_parent == NULL)
        {
            CONDUIT_ERROR("Cannot const fetch from NULL parent" << path);
        }
        else
        {
            return m_parent->fetch(p_next);
        }
    }

    index_t idx;
    if(m_schema->has_child(p_curr))
    {
        idx = m_schema->child_index(p_curr);
    }
    else
    {
        CONDUIT_ERROR("Cannot const fetch non-existent " 
                      << "child " << p_curr << " from Node("
                      << this->path()
                      << ")");
    }

    if(p_next.empty())
    {
        return  *m_children[idx];
    }
    else
    {
        return m_children[idx]->fetch(p_next);
    }
}


//---------------------------------------------------------------------------//
Node&
Node::fetch_child(const std::string &path)
{
    // const fetch w/ path requires object role
    if(!dtype().is_object())
    {
        CONDUIT_ERROR("Cannot fetch_child, Node(" << this->path()
                      << ") is not an object");
    }
    
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // check for parent
    if(p_curr == "..")
    {
        if(m_parent == NULL)
        {
            CONDUIT_ERROR("Cannot const fetch from NULL parent" << path);
        }
        else
        {
            return m_parent->fetch(p_next);
        }
    }

    index_t idx;
    if(m_schema->has_child(p_curr))
    {
        idx = m_schema->child_index(p_curr);
    }
    else
    {
        CONDUIT_ERROR("Cannot const fetch non-existent " 
                      << "child " << p_curr << " from Node("
                      << this->path()
                      << ")");
    }

    if(p_next.empty())
    {
        return  *m_children[idx];
    }
    else
    {
        return m_children[idx]->fetch(p_next);
    }
}

//---------------------------------------------------------------------------//
Node&
Node::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_ID
    if(dtype().is_object())
    {
        init(DataType::object());
    }
    
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // check for parent
    if(p_curr == "..")
    {
        if(m_parent == NULL)
        {
            CONDUIT_ERROR("Cannot fetch from NULL parent" << path);
        }
        else
        {
            return m_parent->fetch(p_next);
        }
    }

    // if this node doesn't exist yet, we need to create it and
    // link it to a schema
        
    index_t idx;
    if(!m_schema->has_child(p_curr))
    {
        Schema *schema_ptr = m_schema->fetch_ptr(p_curr);
        Node *curr_node = new Node();
        curr_node->set_schema_ptr(schema_ptr);
        curr_node->m_parent = this;
        m_children.push_back(curr_node);
        idx = m_children.size() - 1;
    }
    else
    {
        idx = m_schema->child_index(p_curr);
    }

    if(p_next.empty())
    {
        return  *m_children[idx];
    }
    else
    {
        return m_children[idx]->fetch(p_next);
    }

}

//---------------------------------------------------------------------------//
const Node&
Node::fetch(const std::string &path) const
{
    return fetch_child(path);
}


//---------------------------------------------------------------------------//
Node&
Node::child(index_t idx)
{
    return *m_children[idx];
}


//---------------------------------------------------------------------------//
const Node&
Node::child(index_t idx) const
{
    return *m_children[idx];
}

//---------------------------------------------------------------------------//
Node *
Node::fetch_ptr(const std::string &path)
{
    return &fetch(path);
}

//---------------------------------------------------------------------------//
const Node *
Node::fetch_ptr(const std::string &path) const
{
    // TODO, this could be more efficient with sep imp that doens't
    // use has_path and fetch (two traversals)
    if(has_path(path))
    {
        return &fetch(path);
    }
    else
    {
        return NULL;
    }
}

//---------------------------------------------------------------------------//
Node *
Node::child_ptr(index_t idx)
{
    return &child(idx);
}

//---------------------------------------------------------------------------//
const Node *
Node::child_ptr(index_t idx) const
{
    return &child(idx);
}

//---------------------------------------------------------------------------//
Node&
Node::operator[](const std::string &path)
{
    return fetch(path);
}

//---------------------------------------------------------------------------//
const Node&
Node::operator[](const std::string &path) const
{
    return fetch(path);
}

//---------------------------------------------------------------------------//
Node&
Node::operator[](index_t idx)
{
    return child(idx);
}

//---------------------------------------------------------------------------//
const Node&
Node::operator[](index_t idx) const
{
    return child(idx);
}

//---------------------------------------------------------------------------//
index_t 
Node::number_of_children() const 
{
    return m_schema->number_of_children();
}

//---------------------------------------------------------------------------//
std::string 
Node::path() const
{
    return m_schema->path();
}


//---------------------------------------------------------------------------//
bool
Node::has_path(const std::string &path) const
{
    return m_schema->has_path(path);
}

//---------------------------------------------------------------------------//
void
Node::paths(std::vector<std::string> &paths) const
{
    m_schema->paths(paths);
}

//---------------------------------------------------------------------------//
const std::vector<std::string>&
Node::paths() const
{
    return m_schema->paths();
}

//---------------------------------------------------------------------------//
Node &
Node::append()
{
    init_list();
    index_t idx = m_children.size();
    //
    // This makes a proper copy of the schema for us to use
    //
    m_schema->append();
    Schema *schema_ptr = m_schema->child_ptr(idx);

    Node *res_node = new Node();
    res_node->set_schema_ptr(schema_ptr);
    res_node->m_parent=this;
    m_children.push_back(res_node);
    return *res_node;
}

//---------------------------------------------------------------------------//
void    
Node::remove(index_t idx)
{
    // note: we must remove the child pointer before the
    // schema. b/c the child pointer uses the schema
    // to cleanup
    
    // remove the proper list entry
    delete m_children[idx];
    m_schema->remove(idx);
    m_children.erase(m_children.begin() + idx);
}

//---------------------------------------------------------------------------//
void
Node::remove(const std::string &path)
{
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    index_t idx=m_schema->child_index(p_curr);
    
    if(!p_next.empty())
    {
        m_children[idx]->remove(p_next);
    }
    else
    {
        // note: we must remove the child pointer before the
        // schema. b/c the child pointer uses the schema
        // to cleanup
        
        delete m_children[idx];
        m_schema->remove(p_curr);
        m_children.erase(m_children.begin() + idx);
    }
}


//---------------------------------------------------------------------------//
// helper to create a Schema the describes a list of a homogenous type
void
Node::list_of(const Schema &schema, 
              index_t num_entries)
{
    init_list();

    Schema s_compact;
    schema.compact_to(s_compact);
    
    index_t entry_bytes = s_compact.total_bytes();
    index_t total_bytes = entry_bytes * num_entries;

    // allocate what we need
    allocate(DataType::uint8(total_bytes));
    
    uint8 *ptr =(uint8*)data_ptr();
    
    for(index_t i=0; i <  num_entries ; i++)
    {
        append().set_external(s_compact,ptr);
        ptr += entry_bytes;
    }
}

//---------------------------------------------------------------------------//
// helper to create a Schema the describes a list of a homogenous type
// where the data is held externally.
void
Node::list_of_external(void *data,
                       const Schema &schema, 
                       index_t num_entries)
{
    release();
    init_list();

    Schema s_compact;
    schema.compact_to(s_compact);
    
    index_t entry_bytes = s_compact.total_bytes();

    m_data = data;
    uint8 *ptr = (uint8*) data;
    
    for(index_t i=0; i <  num_entries ; i++)
    {
        append().set_external(s_compact,ptr);
        ptr += entry_bytes;
    }
}

//-----------------------------------------------------------------------------
//
// -- end definition of Node entry access methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Node value access methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// signed integer scalars 
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
int8
Node::as_int8()  const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT8_ID,
                        "as_int8() const",
                        0);
    return *((int8*)element_ptr(0));
}

//---------------------------------------------------------------------------//
int16
Node::as_int16() const
{ 
    CONDUIT_CHECK_DTYPE(this, 
                        DataType::INT16_ID,
                        "as_int16() const",
                        0);
    return *((int16*)element_ptr(0));
}

//---------------------------------------------------------------------------//
int32
Node::as_int32() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT32_ID,
                        "as_int32() const",
                        0);
    return *((int32*)element_ptr(0));
}

//---------------------------------------------------------------------------//
int64
Node::as_int64() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT64_ID,
                        "as_int64() const",
                        0);
    return *((int64*)element_ptr(0));
}

//---------------------------------------------------------------------------//
// unsigned integer scalar
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
uint8
Node::as_uint8() const
{ 
    CONDUIT_CHECK_DTYPE(this, 
                        DataType::UINT8_ID,
                        "as_uint8() const",
                        0);
    return *((uint8*)element_ptr(0));
}

//---------------------------------------------------------------------------//
uint16
Node::as_uint16() const
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT16_ID,
                        "as_uint16() const",
                        0);
    return *((uint16*)element_ptr(0));
}

//---------------------------------------------------------------------------//
uint32
Node::as_uint32() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT32_ID,
                        "as_uint32() const",
                        0);
    return *((uint32*)element_ptr(0));
}

//---------------------------------------------------------------------------//
uint64
Node::as_uint64() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT64_ID,
                        "as_uint64() const",
                        0);
    return *((uint64*)element_ptr(0));
}

//---------------------------------------------------------------------------//
// floating point scalars
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float32
Node::as_float32() const
{ 
    CONDUIT_CHECK_DTYPE(this, 
                        DataType::FLOAT32_ID,
                        "as_float32() const",
                        0);
    return *((float32*)element_ptr(0));
}

//---------------------------------------------------------------------------//
float64
Node::as_float64() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT64_ID,
                        "as_float64() const",
                        0);
    return *((float64*)element_ptr(0));
}

//---------------------------------------------------------------------------//
// signed integers via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
int8 *
Node::as_int8_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this, 
                        DataType::INT8_ID,
                        "as_int8_ptr()",
                        NULL);
    return (int8*)element_ptr(0);
}

//---------------------------------------------------------------------------//
int16 *
Node::as_int16_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT16_ID,
                        "as_int16_ptr()",
                        NULL);
    return (int16*)element_ptr(0);
}

//---------------------------------------------------------------------------//
int32 *
Node::as_int32_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT32_ID,
                        "as_int32_ptr()",
                        NULL);
    return (int32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
int64 *
Node::as_int64_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT64_ID,
                        "as_int64_ptr()",
                        NULL);
    return (int64*)element_ptr(0);
}

//---------------------------------------------------------------------------//
// unsigned integers via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
uint8 *
Node::as_uint8_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT8_ID,
                        "as_uint8_ptr()",
                        NULL);
    return (uint8*)element_ptr(0);
}

//---------------------------------------------------------------------------//
uint16 *
Node::as_uint16_ptr()   
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT16_ID,
                        "as_uint16_ptr()",
                        NULL);
    return (uint16*)element_ptr(0);
}

//---------------------------------------------------------------------------//
uint32 *
Node::as_uint32_ptr()   
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT32_ID,
                        "as_uint32_ptr()",
                        NULL);
    return (uint32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
uint64 *
Node::as_uint64_ptr()   
{     
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT64_ID,
                        "as_uint64_ptr()",
                        NULL);
    return (uint64*)element_ptr(0);
}

//---------------------------------------------------------------------------//
// floating point via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float32 *
Node::as_float32_ptr()  
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT32_ID,
                        "as_float32_ptr()",
                        NULL);
    return (float32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
float64 *
Node::as_float64_ptr()  
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT64_ID,
                        "as_float64_ptr()",
                        NULL);
    return (float64*)element_ptr(0);
}


//---------------------------------------------------------------------------//
// signed integers via pointers (const cases)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const int8 *
Node::as_int8_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT8_ID,
                        "as_int8_ptr() const",
                        NULL);
    return (int8*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const int16 *
Node::as_int16_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT16_ID,
                        "as_int16_ptr() const",
                        NULL);
    return (int16*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const int32 *
Node::as_int32_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT32_ID,
                        "as_int32_ptr() const",
                        NULL);
    return (int32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const int64 *
Node::as_int64_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT64_ID,
                        "as_int64_ptr() const",
                        NULL);
    return (int64*)element_ptr(0);
}

//---------------------------------------------------------------------------//
// unsigned integers via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const uint8 *
Node::as_uint8_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT8_ID,
                        "as_uint8_ptr() const",
                        NULL);
    return (uint8*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const uint16 *
Node::as_uint16_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT16_ID,
                        "as_uint16_ptr() const",
                        NULL);
    return (uint16*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const uint32 *
Node::as_uint32_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT32_ID,
                        "as_uint32_ptr() const",
                        NULL);
    return (uint32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const uint64 *
Node::as_uint64_ptr() const
{     
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT64_ID,
                        "as_uint64_ptr() const",
                        NULL);
    return (uint64*)element_ptr(0);
}

//---------------------------------------------------------------------------//
// floating point via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const float32 *
Node::as_float32_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT32_ID,
                        "as_float32_ptr() const",
                        NULL);
    return (float32*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const float64 *
Node::as_float64_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT64_ID,
                        "as_float64_ptr() const",
                        NULL);
    return (float64*)element_ptr(0);
}


//---------------------------------------------------------------------------//
// signed integer array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
int8_array
Node::as_int8_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT8_ID,
                        "as_int8_array()",
                        int8_array());
    return int8_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
int16_array
Node::as_int16_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT16_ID,
                        "as_int16_array()",
                        int16_array());
    return int16_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
int32_array
Node::as_int32_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT32_ID,
                        "as_int32_array()",
                        int32_array());
    return int32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
int64_array
Node::as_int64_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT64_ID,
                        "as_int64_array()",
                        int64_array());
    return int64_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
// unsigned integer array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
uint8_array
Node::as_uint8_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT8_ID,
                        "as_uint8_array()",
                        uint8_array());
    return uint8_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
uint16_array
Node::as_uint16_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT16_ID,
                        "as_uint16_array()",
                        uint16_array());
    return uint16_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
uint32_array
Node::as_uint32_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT32_ID,
                        "as_uint32_array()",
                        uint32_array());
    return uint32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
uint64_array
Node::as_uint64_array() 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT64_ID,
                        "as_uint64_array()",
                        uint64_array());
    return uint64_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
// floating point array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float32_array
Node::as_float32_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT32_ID,
                        "as_float32_array()",
                        float32_array());
    return float32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
float64_array
Node::as_float64_array()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT64_ID,
                        "as_float64_array()",
                        float64_array());
    return float64_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
// signed integer array types via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const int8_array
Node::as_int8_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT8_ID,
                        "as_int8_array() const",
                        int8_array());
    return int8_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const int16_array
Node::as_int16_array() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT16_ID,
                        "as_int16_array() const",
                        int16_array());
    return int16_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const int32_array
Node::as_int32_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT32_ID,
                        "as_int32_array() const",
                        int32_array());
    return int32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const int64_array
Node::as_int64_array() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::INT64_ID,
                        "as_int64_array() const",
                        int64_array());
    return int64_array(m_data,dtype());
}


//---------------------------------------------------------------------------//
// unsigned integer array types via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const uint8_array
Node::as_uint8_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT8_ID,
                        "as_uint8_array() const",
                        uint8_array());
    return uint8_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const uint16_array
Node::as_uint16_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT16_ID,
                        "as_uint16_array() const",
                        uint16_array());
    return uint16_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const uint32_array
Node::as_uint32_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT32_ID,
                        "as_uint32_array() const",
                        uint32_array());
    return uint32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const uint64_array
Node::as_uint64_array() const 
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::UINT64_ID,
                        "as_uint64_array() const",
                        uint64_array());
    return uint64_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
// floating point array value via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const float32_array
Node::as_float32_array() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT32_ID,
                        "as_float32_array() const",
                        float32_array());
    return float32_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const float64_array
Node::as_float64_array() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        DataType::FLOAT64_ID,
                        "as_float64_array() const",
                        float64_array());
    return float64_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
// char8_str cases
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
char *
Node::as_char8_str()
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::CHAR8_STR_ID,
                        "as_char8_str()",
                        NULL);
    return (char *)element_ptr(0);
}

//---------------------------------------------------------------------------//
const char *
Node::as_char8_str() const
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::CHAR8_STR_ID,
                        "as_char8_str() const",
                        NULL);
    return (const char *)element_ptr(0);
}

//---------------------------------------------------------------------------//
std::string
Node::as_string() const
{
    CONDUIT_CHECK_DTYPE(this,
                        DataType::CHAR8_STR_ID,
                        "as_string() const",
                        std::string());
    return std::string(as_char8_str());
}

//---------------------------------------------------------------------------//
// direct data pointer access 
void *
Node::data_ptr() 
{
    return m_data;
}

//---------------------------------------------------------------------------//
const void *
Node::data_ptr() const
{
    return m_data;
}


//-----------------------------------------------------------------------------
///  Direct access to data at leaf types (native c++ types)
//-----------------------------------------------------------------------------
     
//---------------------------------------------------------------------------//
// signed integer scalars
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
char
Node::as_char() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_CHAR_ID,
                        "as_char() const",
                        0);
    return *((char*)element_ptr(0));
}

//---------------------------------------------------------------------------//
short
Node::as_short() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_SHORT_ID,
                        "as_short() const",
                        0);
    return *((short*)element_ptr(0));
}

//---------------------------------------------------------------------------//
int
Node::as_int() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_INT_ID,
                        "as_int() const",
                        0);
    return *((int*)element_ptr(0));
}

//---------------------------------------------------------------------------//
long
Node::as_long()  const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_ID,
                        "as_long() const",
                        0);
    return *((long*)element_ptr(0));
}


//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
long long
Node::as_long_long()  const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_LONG_ID,
                        "as_long_long() const",
                        0);
    return *((long long*)element_ptr(0));
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// unsigned integer scalars
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
unsigned char
Node::as_unsigned_char() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                        "as_unsigned_char() const",
                        0);
    return *((unsigned char*)element_ptr(0));
}

//---------------------------------------------------------------------------//
unsigned short 
Node::as_unsigned_short() const 
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                        "as_unsigned_short() const",
                        0);
    return *((unsigned short*)element_ptr(0));
}

//---------------------------------------------------------------------------//
unsigned int
Node::as_unsigned_int()const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_INT_ID,
                        "as_unsigned_int() const",
                        0);
    return *((unsigned int*)element_ptr(0));
}

//---------------------------------------------------------------------------//
unsigned long
Node::as_unsigned_long() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                        "as_unsigned_long() const",
                        0);
    return *(( unsigned long*)element_ptr(0));
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
unsigned long long
Node::as_unsigned_long_long() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                        "as_unsigned_long_long() const",
                        0);
    return *(( unsigned long long*)element_ptr(0));
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// floating point scalars
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float
Node::as_float() const 
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_FLOAT_ID,
                        "as_float() const",
                        0);
    return *((float*)element_ptr(0));
}

//---------------------------------------------------------------------------//
double
Node::as_double() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_DOUBLE_ID,
                        "as_double() const",
                        0);
    return *((double*)element_ptr(0));
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
long double
Node::as_long_double() const 
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double() const",
                        0);
    return *((long double*)element_ptr(0));
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// signed integers via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
char *
Node::as_char_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_CHAR_ID,
                        "as_char_ptr()",
                        NULL);
    return (char*)element_ptr(0);
}

//---------------------------------------------------------------------------//
short *
Node::as_short_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_SHORT_ID,
                        "as_short_ptr()",
                        NULL);
    return (short*)element_ptr(0);
}

//---------------------------------------------------------------------------//
int *
Node::as_int_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_INT_ID,
                        "as_int_ptr()",
                        NULL);
    return (int*)element_ptr(0);
}

//---------------------------------------------------------------------------//
long *
Node::as_long_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_ID,
                        "as_long_ptr()",
                        NULL);
    return (long*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
long long *
Node::as_long_long_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_LONG_ID,
                        "as_long_long_ptr()",
                        NULL);
    return (long long*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// unsigned integers via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
unsigned char *
Node::as_unsigned_char_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                        "as_unsigned_char_ptr()",
                        NULL);
    return (unsigned char*)element_ptr(0);
}

//---------------------------------------------------------------------------//
unsigned short *
Node::as_unsigned_short_ptr()
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                        "as_unsigned_short_ptr()",
                        NULL);
    return (unsigned short*)element_ptr(0);
}

//---------------------------------------------------------------------------//
unsigned int *
Node::as_unsigned_int_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_INT_ID,
                        "as_unsigned_int_ptr()",
                        NULL);
    return (unsigned int*)element_ptr(0);
}

//---------------------------------------------------------------------------//
unsigned long *
Node::as_unsigned_long_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                        "as_unsigned_long_ptr()",
                        NULL);
    return (unsigned long*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
unsigned long long *
Node::as_unsigned_long_long_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                        "as_unsigned_long_long_ptr()",
                        NULL);
    return (unsigned long long*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// floating point via pointers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float *
Node::as_float_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_FLOAT_ID,
                        "as_float_ptr()",
                        NULL);
    return (float*)element_ptr(0);
}

//---------------------------------------------------------------------------//
double *
Node::as_double_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_DOUBLE_ID,
                        "as_double_ptr()",
                        NULL);
    return (double*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
long double *
Node::as_long_double_ptr()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double_ptr()",
                        NULL);
    return (long double*)element_ptr(0);
}

//---------------------------------------------------------------------------//
long double *
Node::as_long_double_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double_ptr() const",
                        NULL);
    return (long double*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// signed integers via pointers (const)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const char *
Node::as_char_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_CHAR_ID,
                        "as_char_ptr() const",
                        NULL);
    return (char*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const short *
Node::as_short_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_SHORT_ID,
                        "as_short_ptr() const",
                        NULL);
    return (short*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const int *
Node::as_int_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_INT_ID,
                        "as_int_ptr() const",
                        NULL);
    return (int*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const long *
Node::as_long_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_ID,
                        "as_long_ptr() const",
                        NULL);
    return (long*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
const long long *
Node::as_long_long_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_LONG_ID,
                        "as_long_long_ptr() const",
                        NULL);
    return (long long*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// unsigned integers via pointers (const)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const unsigned char *
Node::as_unsigned_char_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                        "as_unsigned_char_ptr() const",
                        NULL);
    return (unsigned char*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const unsigned short *
Node::as_unsigned_short_ptr() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                        "as_unsigned_short_ptr() const",
                        NULL);
    return (unsigned short*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const unsigned int *
Node::as_unsigned_int_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_INT_ID,
                        "as_unsigned_int_ptr() const",
                        NULL);
    return (unsigned int*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const unsigned long *
Node::as_unsigned_long_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                        "as_unsigned_long_ptr() const",
                        NULL);
    return (unsigned long*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const unsigned long long *
Node::as_unsigned_long_long_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                        "as_unsigned_long_long_ptr() const",
                        NULL);
    return (unsigned long long*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//




//---------------------------------------------------------------------------//
// floating point via pointers (const)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const float *
Node::as_float_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_FLOAT_ID,
                        "as_float_ptr() const",
                        NULL);
    return (float*)element_ptr(0);
}

//---------------------------------------------------------------------------//
const double *
Node::as_double_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_DOUBLE_ID,
                        "as_double_ptr() const",
                        NULL);
    return (double*)element_ptr(0);
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
const long double *
Node::as_long_double_ptr() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double_ptr() const",
                        NULL);
    return (long double*)element_ptr(0);
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// signed integer array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
char_array
Node::as_char_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_CHAR_ID,
                        "as_char_array()",
                        char_array());
    return char_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
short_array
Node::as_short_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_SHORT_ID,
                        "as_short_array()",
                        short_array());
    return short_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
int_array
Node::as_int_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_INT_ID,
                        "as_int_array()",
                        int_array());
    return int_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
long_array
Node::as_long_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_ID,
                        "as_long_array()",
                        long_array());
    return long_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
long_long_array
Node::as_long_long_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_LONG_ID,
                        "as_long_long_array()",
                        long_long_array());
    return long_long_array(m_data,dtype());
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// unsigned integer array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
unsigned_char_array
Node::as_unsigned_char_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                        "as_unsigned_char_array()",
                        unsigned_char_array());
    return unsigned_char_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
unsigned_short_array
Node::as_unsigned_short_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                        "as_unsigned_short_array()",
                        unsigned_short_array());
    return unsigned_short_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
unsigned_int_array
Node::as_unsigned_int_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_INT_ID,
                        "as_unsigned_int_array()",
                        unsigned_int_array());
    return unsigned_int_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
unsigned_long_array
Node::as_unsigned_long_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                        "as_unsigned_long_array()",
                        unsigned_long_array());
    return unsigned_long_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
unsigned_long_long_array
Node::as_unsigned_long_long_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                        "as_unsigned_long_long_array()",
                        unsigned_long_long_array());
    return unsigned_long_long_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// floating point array types via conduit::DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
float_array
Node::as_float_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_FLOAT_ID,
                        "as_float_array()",
                        float_array());
    return float_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
double_array
Node::as_double_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_DOUBLE_ID,
                        "as_double_array()",
                        double_array());
    return double_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
long_double_array
Node::as_long_double_array()
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double_array()",
                        long_double_array());
    return long_double_array(m_data,dtype());
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// signed integer array types via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const char_array
Node::as_char_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_CHAR_ID,
                        "as_char_array() const",
                        char_array());
    return char_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const short_array
Node::as_short_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_SHORT_ID,
                        "as_short_array() const",
                        short_array());
    return short_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const int_array
Node::as_int_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_INT_ID,
                        "as_int_array() const",
                        int_array());
    return int_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const long_array
Node::as_long_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_ID,
                        "as_long_array() const",
                        long_array());
    return long_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
const long_long_array
Node::as_long_long_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_LONG_ID,
                        "as_long_long_array() const",
                        long_long_array());
    return long_long_array(m_data,dtype());
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// unsigned integer array types via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const unsigned_char_array
Node::as_unsigned_char_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                        "as_unsigned_char_array() const",
                        unsigned_char_array());
    return unsigned_char_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const unsigned_short_array
Node::as_unsigned_short_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                        "as_unsigned_short_array() const",
                        unsigned_short_array());
    return unsigned_short_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const unsigned_int_array
Node::as_unsigned_int_array() const
{ 
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_INT_ID,
                        "as_unsigned_int_array() const",
                        unsigned_int_array());
    return unsigned_int_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const unsigned_long_array
Node::as_unsigned_long_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                        "as_unsigned_long_array() const",
                        unsigned_long_array());
    return unsigned_long_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
const unsigned_long_long_array
Node::as_unsigned_long_long_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                        "as_unsigned_long_long_array() const",
                        unsigned_long_long_array());
    return unsigned_long_long_array(m_data,dtype());
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
// floating point array value via conduit::DataArray (const variants)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
const float_array
Node::as_float_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_FLOAT_ID,
                        "as_float_array() const",
                        float_array());
    return float_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
const double_array
Node::as_double_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_DOUBLE_ID,
                        "as_double_array() const",
                        double_array());
    return double_array(m_data,dtype());
}
//---------------------------------------------------------------------------//
#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
const long_double_array
Node::as_long_double_array() const
{
    CONDUIT_CHECK_DTYPE(this,
                        CONDUIT_NATIVE_LONG_DOUBLE_ID,
                        "as_long_double_array() const",
                        long_double_array());
    return long_double_array(m_data,dtype());
}

//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//


//-----------------------------------------------------------------------------
//
// -- end definition of Node value access methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin definition of Interface Warts --
//
//-----------------------------------------------------------------------------

// NOTE: several other warts methods are inlined in Node.h

//---------------------------------------------------------------------------//
void
Node::set_schema_ptr(Schema *schema_ptr)
{
    // if(m_schema->is_root())
    if(m_owns_schema)
    {
        delete m_schema;
        m_owns_schema = false;
    }
    m_schema = schema_ptr;    
}
    
//---------------------------------------------------------------------------//
void
Node::set_data_ptr(void *data)
{
    /// TODO: We need to audit where we actually need release
    //release();
    m_data    = data;
}
    

//-----------------------------------------------------------------------------
//
// -- end definition of Interface Warts --
//
//-----------------------------------------------------------------------------

//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- end conduit::Node public methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================

//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- begin conduit::Node private methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================

//-----------------------------------------------------------------------------
// Node::MMap helper class
//-----------------------------------------------------------------------------
// This private class encapsulates specific logic for both unix and windows
// style memory mapping.
//-----------------------------------------------------------------------------
class Node::MMap
{
  public:
      MMap();
      ~MMap();

      //----------------------------------------------------------------------
      void  open(const std::string &path,
                 index_t data_size);

      //----------------------------------------------------------------------
      void  close();

      //----------------------------------------------------------------------
      void *data_ptr() const
          { return m_data; }

  private:
      void      *m_data;
      int        m_data_size;

#if !defined(CONDUIT_PLATFORM_WINDOWS)
      // memory-map file descriptor
      int       m_mmap_fd;
#else
      // handles for windows mmap
      HANDLE    m_file_hnd;
      HANDLE    m_map_hnd;
#endif

};

//-----------------------------------------------------------------------------
Node::MMap::MMap()
: m_data(NULL),
  m_data_size(0),
#if !defined(CONDUIT_PLATFORM_WINDOWS)
  m_mmap_fd(-1)
#else
  // windows
  m_file_hnd(INVALID_HANDLE_VALUE),
  m_map_hnd(INVALID_HANDLE_VALUE)
#endif
{
    // empty
}

//-----------------------------------------------------------------------------
Node::MMap::~MMap()
{
    close();
}

//-----------------------------------------------------------------------------
void
Node::MMap::open(const std::string &path,
                 index_t data_size)
{
    if(m_data != NULL)
    {
        CONDUIT_ERROR("<Node::mmap> mmap already open");
    }

#if !defined(CONDUIT_PLATFORM_WINDOWS)
    m_mmap_fd   = ::open(path.c_str(),
                         (O_RDWR | O_CREAT),
                         (S_IRUSR | S_IWUSR));

    m_data_size = data_size;

    if (m_mmap_fd == -1) 
        CONDUIT_ERROR("<Node::mmap> failed to open: " << path);

    m_data = ::mmap(0,
                    m_data_size,
                    (PROT_READ | PROT_WRITE),
                    MAP_SHARED,
                    m_mmap_fd, 0);

    if (m_data == MAP_FAILED) 
        CONDUIT_ERROR("<Node::mmap> mmap data = MAP_FAILED" << path);
#else
    m_file_hnd = CreateFile(path.c_str(),
                            (GENERIC_READ | GENERIC_WRITE),
                            0,
                            NULL,
                            OPEN_EXISTING,
                            FILE_FLAG_RANDOM_ACCESS,
                            NULL);

    if (m_file_hnd == INVALID_HANDLE_VALUE)
    {
        CONDUIT_ERROR("<Node::mmap> CreateFile() Failed ");
    }

    m_map_hnd = CreateFileMapping(m_file_hnd,
                                  NULL,
                                  PAGE_READWRITE,
                                  0, 0, 0);

    if (m_map_hnd == NULL)
    {
        CloseHandle(m_file_hnd);
        CONDUIT_ERROR("<Node::mmap> CreateFileMapping() failed with error" << GetLastError());
    }

    m_data = MapViewOfFile(m_map_hnd,
                           FILE_MAP_ALL_ACCESS,
                           0, 0, 0);

    if (m_data == NULL)
    {
        CloseHandle(m_map_hnd);
        CloseHandle(m_file_hnd);
        CONDUIT_ERROR("<Node::mmap> MapViewOfFile() failed with error" << GetLastError());
    }
#endif
}

//-----------------------------------------------------------------------------
void
Node::MMap::close()
{
    // simple return if the mmap isn't active
    if(m_data == NULL)
        return;
    
#if !defined(CONDUIT_PLATFORM_WINDOWS)
    
    if(munmap(m_data, m_data_size) == -1) 
    {
        CONDUIT_ERROR("<Node::mmap> failed to unmap mmap.");
    }
    
    if(::close(m_mmap_fd) == -1)
    {
        CONDUIT_ERROR("<Node::mmap> failed close mmap filed descriptor.");
    }

    m_mmap_fd   = -1;

#else
    UnmapViewOfFile(m_data);
    CloseHandle(m_map_hnd);
    CloseHandle(m_file_hnd);
    m_file_hnd = INVALID_HANDLE_VALUE;
    m_map_hnd  = INVALID_HANDLE_VALUE;
#endif

    // clear data pointer and size member
    m_data      = NULL;
    m_data_size = 0;

}




//-----------------------------------------------------------------------------
//
// -- private methods that help with init, memory allocation, and cleanup --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::init(const DataType& dtype)
{
    if(this->dtype().compatible(dtype))
        return;
    
    if(m_data != NULL)
    {
        release();
    }

    index_t dt_id = dtype.id();
    if(dt_id == DataType::OBJECT_ID ||
       dt_id == DataType::LIST_ID)
    {
        m_children.clear();
    }
    else if(dt_id != DataType::EMPTY_ID)
    {
        allocate(dtype);
    }
    
    m_schema->set(dtype); 
}


//---------------------------------------------------------------------------//
void
Node::allocate(const DataType &dtype)
{
    // TODO: This implies compact storage
    allocate(dtype.number_of_elements()*dtype.element_bytes());
}

//---------------------------------------------------------------------------//
void
Node::allocate(index_t dsize)
{
    m_data      = malloc(dsize);
    m_data_size = dsize;
    m_alloced   = true;
    m_mmaped    = false;
}


//---------------------------------------------------------------------------//
void
Node::mmap(const std::string &stream_path, index_t data_size)
{
    m_mmap = new MMap();
    m_mmap->open(stream_path,data_size);
    m_data = m_mmap->data_ptr();
    m_data_size = data_size;
    m_alloced = false;
    m_mmaped  = true;
}


//---------------------------------------------------------------------------//
void
Node::release()
{
    // delete all children
    for (index_t i = 0; i < (index_t) m_children.size(); i++)
    {
        Node* node = m_children[i];
        delete node;
    }
    m_children.clear();

    // clean up any allocated or mmaped buffers
    if(m_alloced && m_data)
    {
        ///
        /// TODO: why do we need to check for empty here?
        ///
        if(dtype().id() != DataType::EMPTY_ID)
        {   
            // clean up our storage
            free(m_data);
            m_data = NULL;
            m_data_size = 0;
            m_alloced   = false;
        }
    }   
    else if(m_mmaped && m_mmap)
    {
        delete m_mmap;
        m_data = NULL;
        m_data_size = 0;
        m_mmaped    = false;
        m_mmap      = NULL;
    }
}

//---------------------------------------------------------------------------//
void
Node::cleanup()
{
    release();
    // if(m_schema->is_root())
    if(m_owns_schema && m_schema != NULL)
    {
        delete m_schema;
        m_schema = NULL;
        m_owns_schema = false;
    }
    else if(m_schema != NULL)
    {
        m_schema->set(DataType::EMPTY_ID);
    }
}


//---------------------------------------------------------------------------//
void
Node::init_list()
{
    init(DataType::list());
}
 
//---------------------------------------------------------------------------//
void
Node::init_object()
{
    init(DataType::object());
}




//---------------------------------------------------------------------------//
void
Node::init_defaults()
{
    m_data = NULL;
    m_data_size = 0;
    m_alloced = false;

    m_mmaped    = false;
    m_mmap      = NULL;

    m_schema = new Schema(DataType::EMPTY_ID);
    m_owns_schema = true;
    
    m_parent = NULL;
}


//-----------------------------------------------------------------------------
//
// -- private methods that help with hierarchical construction --
//
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void 
Node::walk_schema(Node   *node, 
                  Schema *schema,
                  void   *data)
{
    // we can have an object, list, or leaf
    node->set_data_ptr(data);
    if(schema->dtype().id() == DataType::OBJECT_ID)
    {
        for(index_t i=0;i<(index_t)schema->children().size(); i++)
        {
    
            std::string curr_name = schema->object_order()[i];
            Schema *curr_schema   = schema->fetch_ptr(curr_name);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append_node_ptr(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_ID)
    {
        index_t num_entries = schema->number_of_children();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = schema->child_ptr(i);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append_node_ptr(curr_node);
        }
    }

  
}

//---------------------------------------------------------------------------//
void 
Node::mirror_node(Node   *node,
                  Schema *schema,
                  Node   *src)
{
    // we can have an object, list, or leaf
    node->set_data_ptr(src->m_data);
    
    if(schema->dtype().id() == DataType::OBJECT_ID)
    {
        for(index_t i=0;i< (index_t) schema->children().size(); i++)
        {
    
            std::string curr_name = schema->object_order()[i];
            Schema *curr_schema   = schema->fetch_ptr(curr_name);
            Node *curr_node = new Node();
            Node *curr_src = src->child_ptr(i);
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            mirror_node(curr_node,curr_schema,curr_src);
            node->append_node_ptr(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_ID)
    {
        index_t num_entries = schema->number_of_children();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = schema->child_ptr(i);
            Node *curr_node = new Node();
            Node *curr_src = src->child_ptr(i);
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            mirror_node(curr_node,curr_schema,curr_src);
            node->append_node_ptr(curr_node);
        }
    }

    
}

//-----------------------------------------------------------------------------
//
// -- private methods that help with compaction, serialization, and info  --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::compact_to(uint8 *data, index_t curr_offset) const
{
    CONDUIT_ASSERT( (m_schema != NULL) , "Corrupt schema found in compact_to call");
    
    index_t dtype_id = dtype().id();
    if(dtype_id == DataType::OBJECT_ID ||
       dtype_id == DataType::LIST_ID)
    {
            std::vector<Node*>::const_iterator itr;
            for(itr = m_children.begin(); itr < m_children.end(); ++itr)
            {
                (*itr)->compact_to(data,curr_offset);
                curr_offset +=  (*itr)->total_bytes_compact();
            }
    }
    else
    {
        compact_elements_to(&data[curr_offset]);
    }
}


//---------------------------------------------------------------------------//
void
Node::compact_elements_to(uint8 *data) const
{
    index_t dtype_id = dtype().id();
    if(dtype_id == DataType::OBJECT_ID ||
       dtype_id == DataType::LIST_ID ||
       dtype_id == DataType::EMPTY_ID)
    {
        // TODO: error
    }
    else
    {
        // copy all elements 
        index_t num_ele   = dtype().number_of_elements();
        index_t ele_bytes = DataType::default_bytes(dtype_id);
        uint8 *data_ptr = data;
        for(index_t i=0;i<num_ele;i++)
        {
            memcpy(data_ptr,
                   element_ptr(i),
                   ele_bytes);
            data_ptr+=ele_bytes;
        }
    }
}


//---------------------------------------------------------------------------//
void
Node::serialize(uint8 *data,index_t curr_offset) const
{
    if(dtype().id() == DataType::OBJECT_ID ||
       dtype().id() == DataType::LIST_ID)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin(); itr < m_children.end(); ++itr)
        {
            (*itr)->serialize(&data[0],curr_offset);
            curr_offset+=(*itr)->total_bytes();
        }
    }
    else
    {
        if(is_compact())
        {
            memcpy(&data[curr_offset],
                   m_data,
                   total_bytes());
        }
        else // ser as is. This copies stride * num_ele bytes
        {
            // copy all elements 
            compact_elements_to(&data[curr_offset]);      
        }
       
    }
}

//---------------------------------------------------------------------------//
bool
Node::is_contiguous() const
{
    uint8 *end_addy= NULL;
    return contiguous_with(NULL,end_addy);
}

//---------------------------------------------------------------------------//
bool
Node::contiguous_with(void *address) const
{
    // not handling NULL as input will case an issue b/c NULL 
    // is used to start recursion for the is_contiguous() case.
    if(address == NULL)
    {
        return false;
    }

    uint8 *end_addy= NULL;
    
    return contiguous_with((uint8*)address,end_addy);
}


//---------------------------------------------------------------------------//
bool
Node::contiguous_with(const Node &n) const
{
    uint8* end_addy=NULL;
    
    // use check_contiguous_after to get final address from passed node
    // if the passed node isn't contiguous, this call returns false

    return n.contiguous_with(NULL,end_addy) && 
           this->contiguous_with(end_addy);
}


//---------------------------------------------------------------------------//
bool
Node::contiguous_with(uint8 *start_addy, uint8 *&end_addy) const
{
    bool res = true;
    
    index_t dtype_id = dtype().id();
    
    if(dtype_id == DataType::OBJECT_ID ||
       dtype_id == DataType::LIST_ID)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin();
            itr < m_children.end();
            ++itr)
        {
            res = (*itr)->contiguous_with(start_addy,
                                          end_addy);
            
            if(res)
            {
                // ok, advance
                start_addy = end_addy;
            }
            else
            {
                // no need to check more children
                break;
            }
        }
    }
    else if(dtype_id != DataType::EMPTY_ID)
    {
        uint8 *curr_addy = (uint8*)element_ptr(0);
        if(start_addy != NULL)
        {
            // make sure element ptr is not NULL
            // and that it is our starting address
            if(curr_addy != NULL && curr_addy == start_addy)
            {
                // ok, advance the end pointer
                end_addy = curr_addy + total_bytes();
            }
            else // bad
            {
                res = false;
                end_addy = NULL;
            }
        }
        else if(curr_addy != NULL)
        {
            // this is the first leaf that actually has data
            // 
            // by definition it is contiguous, so we simply 
            // advance the end pointer
            end_addy  = curr_addy + total_bytes();
        }
        else // current address is NULL, nothing is contig with NULL
        {
            res = false;
            end_addy = NULL;
        }
    }
    
    return res;
}


//---------------------------------------------------------------------------//
void
Node::info(Node &res, const std::string &curr_path) const
{
    // extract
    // mem_spaces:
    //  node path, pointer, alloced, mmaped or external, bytes

    if(m_data != NULL)
    {
        std::string ptr_key = utils::to_hex_string(m_data);

        if(!res["mem_spaces"].has_path(ptr_key))
        {
            Node &ptr_ref = res["mem_spaces"][ptr_key];
            ptr_ref["path"] = curr_path;
            if(m_alloced)
            {
                ptr_ref["type"]  = "alloced";
                ptr_ref["bytes"] = m_data_size;
            }
            else if(m_mmaped)
            {
                ptr_ref["type"]  = "mmap";
                ptr_ref["bytes"] = m_data_size;
            }
            else
            {
                ptr_ref["type"]  = "external";
            }
        }
    }
    
    index_t dtype_id = dtype().id();
    if(dtype_id == DataType::OBJECT_ID)
    {
        std::ostringstream oss;
        index_t nchildren = m_children.size();
        for(index_t i=0; i < nchildren;i++)
        {
            oss.str("");
            if(curr_path == "")
            {
                oss << m_schema->object_order()[i];
            }
            else
            {
                oss << curr_path << "/" << m_schema->object_order()[i];
            }
            m_children[i]->info(res,oss.str());
        }
    }
    else if(dtype_id == DataType::LIST_ID)
    {
        std::ostringstream oss;
        index_t nchildren = m_children.size();
        for(index_t i=0; i < nchildren;i++)
        {
            oss.str("");
            oss << curr_path << "[" << i << "]";
            m_children[i]->info(res,oss.str());
        }
    }    
}


//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- end conduit::Node private methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

