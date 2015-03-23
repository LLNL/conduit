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
/// file: Node.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Node.h"
#include "Utils.h"
#include "Generator.h"

//-----------------------------------------------------------------------------
// -- standard includes -- 
//-----------------------------------------------------------------------------
#include <iostream>
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
#endif

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
    m_schema->set(DataType::EMPTY_T);
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
    Generator g(json_schema,data);

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
    Generator g(json_schema,data);
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
Node::generate(const std::string &json_schema)
{
    Generator g(json_schema);
    generate(g);
}

//---------------------------------------------------------------------------//
void
Node::generate(const std::string &json_schema,
               const std::string &protocol)
               
{
    Generator g(json_schema,protocol);
    generate(g);
}   

//---------------------------------------------------------------------------//
void
Node::generate(const std::string &json_schema,
               void *data)
{
    Generator g(json_schema,data);
    generate(g);
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
                        void *data)
{
    Generator g(json_schema,data);
    generate_external(g);
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
Node::load(const Schema &schema,
           const std::string &stream_path)
{
    index_t dsize = schema.total_bytes();

    allocate(dsize);
    std::ifstream ifs;
    ifs.open(stream_path.c_str());
    if(!ifs.is_open())
        THROW_ERROR("<Node::load> failed to open: " << stream_path);
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
Node::load(const std::string &ibase)
{
    Schema s;
    std::string ifschema = ibase + ".conduit_json";
    std::string ifdata   = ibase + ".conduit_bin";
    s.load(ifschema);
    load(s,ifdata);
}

//---------------------------------------------------------------------------//
void
Node::save(const std::string &obase) const
{
    Node res;
    compact_to(res);
    std::string ofschema = obase + ".conduit_json";
    std::string ofdata   = obase + ".conduit_bin";
    res.schema().save(ofschema);
    res.serialize(ofdata);
}

//---------------------------------------------------------------------------//
void
Node::mmap(const std::string &ibase)
{
    Schema s;
    std::string ifschema = ibase + ".conduit_json";
    std::string ifdata   = ibase + ".conduit_bin";
    s.load(ifschema);
    mmap(s,ifdata);
}


//---------------------------------------------------------------------------//
void 
Node::mmap(const Schema &schema,
           const std::string &stream_path)
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
Node::set(const Node &node)
{
    if(node.dtype().id() == DataType::OBJECT_T)
    {
        init(DataType::object());
        std::vector<std::string> paths;
        node.paths(paths);

        for (std::vector<std::string>::const_iterator itr = paths.begin();
             itr < paths.end(); ++itr)
        {
            Schema *curr_schema = this->m_schema->fetch_pointer(*itr);
            index_t idx = this->m_schema->child_index(*itr);
            Node *curr_node = new Node();
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(this);
            curr_node->set(*node.m_children[idx]);
            this->append_node_pointer(curr_node);       
        }        
    }
    else if(node.dtype().id() == DataType::LIST_T)       
    {   
        init(DataType::list());
        for(index_t i=0;i<node.m_children.size();i++)
        {
            this->m_schema->append();
            Schema *curr_schema = this->m_schema->child_pointer(i);
            Node *curr_node = new Node();
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(this);
            curr_node->set(*node.m_children[i]);
            this->append_node_pointer(curr_node);
        }
    }
    else if (node.dtype().id() != DataType::EMPTY_T)
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
Node::set(const DataType &dtype)
{
    init(dtype);
}

//---------------------------------------------------------------------------//
void
Node::set(const Schema &schema)
{
    release();
    m_schema->set(schema);
    // allocate data
    allocate(m_schema->total_bytes());
    // call walk w/ internal data pointer
    walk_schema(this,m_schema,m_data);
}

//---------------------------------------------------------------------------//
void
Node::set(const Schema &schema, void *data)
{
    release();
    m_schema->set(schema);   
    allocate(m_schema->total_bytes());
    memcpy(m_data, data, m_schema->total_bytes());
    walk_schema(this,m_schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set(const DataType &dtype, void *data)
{
    release();
    m_schema->set(dtype);
    allocate(m_schema->total_bytes());
    memcpy(m_data, data, m_schema->total_bytes());
    walk_schema(this,m_schema,data);
}


//-----------------------------------------------------------------------------
// -- set for scalar types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(int8 data)
{
    init(DataType::int8());
    *(int8*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(int16 data)
{
    init(DataType::int16());
    *(int16*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(int32 data)
{
    init(DataType::int32());
    *(int32*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(int64 data)
{
    init(DataType::int64());
    *(int64*)((char*)m_data + schema().element_index(0)) = data;
}

//-----------------------------------------------------------------------------
// unsigned integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(uint8 data)
{
    init(DataType::uint8());
    *(uint8*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(uint16 data)
{
    init(DataType::uint16());
    *(uint16*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(uint32 data)
{
    init(DataType::uint32());
    *(uint32*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(uint64 data)
{
    init(DataType::uint64());
    *(uint64*)((char*)m_data + schema().element_index(0)) = data;
}

//-----------------------------------------------------------------------------
// floating point scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(float32 data)
{
    init(DataType::float32());
    *(float32*)((char*)m_data + schema().element_index(0)) = data;
}


//---------------------------------------------------------------------------//
void 
Node::set(float64 data)
{
    init(DataType::float64());
    *(float64*)((char*)m_data + schema().element_index(0)) = data;
}


//-----------------------------------------------------------------------------
// -- set for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int8>  &data)
{
    DataType vec_t(DataType::INT8_T,
                   (index_t)data.size(),
                   0,
                   sizeof(int8),
                   sizeof(int8),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(int8)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int16>  &data)
{
    DataType vec_t(DataType::INT16_T,
                   (index_t)data.size(),
                   0,
                   sizeof(int16),
                   sizeof(int16),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(int16)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int32>  &data)
{
    DataType vec_t(DataType::INT32_T,
                   (index_t)data.size(),
                   0,
                   sizeof(int32),
                   sizeof(int32),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(int32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<int64>  &data)
{
    DataType vec_t(DataType::INT64_T,
                   (index_t)data.size(),
                   0,
                   sizeof(int64),
                   sizeof(int64),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(int64)*data.size());
}


//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint8>  &data)
{
    DataType vec_t(DataType::UINT8_T,
                   (index_t)data.size(),
                   0,
                   sizeof(uint8),
                   sizeof(uint8),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(uint8)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint16>  &data)
{
    DataType vec_t(DataType::UINT16_T,
                   (index_t)data.size(),
                   0,
                   sizeof(uint16),
                   sizeof(uint16),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(uint16)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint32>  &data)
{
    DataType vec_t(DataType::UINT32_T,
                   (index_t)data.size(),
                   0,
                   sizeof(uint32),
                   sizeof(uint32),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(uint32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<uint64>  &data)
{
    DataType vec_t(DataType::UINT64_T,
                   (index_t)data.size(),
                   0,
                   sizeof(uint64),
                   sizeof(uint64),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(uint64)*data.size());
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<float32>  &data)
{
    DataType vec_t(DataType::FLOAT32_T,
                   (index_t)data.size(),
                   0,
                   sizeof(float32),
                   sizeof(float32),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(float32)*data.size());
}

//---------------------------------------------------------------------------//
void 
Node::set(const std::vector<float64>  &data)
{
    DataType vec_t(DataType::FLOAT64_T,
                   (index_t)data.size(),
                   0,
                   sizeof(float64),
                   sizeof(float64),
                   Endianness::DEFAULT_T);
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(float64)*data.size());
}

//-----------------------------------------------------------------------------
// -- set for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const int8_array  &data)
{
    init(DataType::int8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int16_array  &data)
{
    init(DataType::int16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int32_array  &data)
{
    init(DataType::int32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const int64_array  &data)
{
    init(DataType::int64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}


//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const uint8_array  &data)
{
    init(DataType::uint8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint16_array  &data)
{
    init(DataType::uint16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint32_array  &data)
{
    init(DataType::uint32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const uint64_array  &data)
{
    init(DataType::uint64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const float32_array  &data)
{
    init(DataType::float32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//---------------------------------------------------------------------------//
void 
Node::set(const float64_array  &data)
{
    init(DataType::float64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//-----------------------------------------------------------------------------
// -- set for string types -- 
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// char8_str use cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(const std::string  &data)
{
    release();
    // size including the null term
    index_t str_size_with_term = data.length()+1;
    DataType str_t(DataType::CHAR8_STR_T,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_T);
    init(str_t);
    memcpy(m_data,data.c_str(),sizeof(char)*str_size_with_term);
}

//---------------------------------------------------------------------------//
void 
Node::set_char8_str(const char *data)
{
    release();
    // size including the null term
    index_t str_size_with_term = strlen(data)+1;
    DataType str_t(DataType::CHAR8_STR_T,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_T);
                   init(str_t);
                   memcpy(m_data,data,sizeof(char)*str_size_with_term);
}

//-----------------------------------------------------------------------------
// -- set via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(int8  *data,
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
Node::set(int16 *data, 
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
Node::set(int32 *data,
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
Node::set(int64 *data,
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


//-----------------------------------------------------------------------------
// unsigned integer pointer cases
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
void 
Node::set(uint8  *data,
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
Node::set(uint16 *data,
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
Node::set(uint32 *data, 
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
Node::set(uint64 *data,
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

//-----------------------------------------------------------------------------
// floating point pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set(float32 *data,
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
Node::set(float64 *data, 
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
Node::set_path(const std::string &path,
               const Node& data) 
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const DataType& dtype)
{
    fetch(path).set(dtype);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const Schema &schema)
{
    fetch(path).set(schema);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const Schema &schema,
               void *data)
{
    fetch(path).set(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const DataType &dtype,
               void *data)
{
    fetch(path).set(dtype,data);
}


//-----------------------------------------------------------------------------
// -- set_path for scalar types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,int8 data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,int16 data)
{
    fetch(path).set(data);
}
 
//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,int32 data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,int64 data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// unsigned integer scalar types 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,uint8 data)
{
    fetch(path).set(data);
}
 
//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,uint16 data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,uint32 data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,uint64 data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// floating point scalar types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,float32 data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,float64 data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// -- set_path for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<int8>   &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<int16>  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<int32>  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<int64>  &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<uint8>   &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<uint16>  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<uint32>  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<uint64>  &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<float32> &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const std::vector<float64> &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// -- set_path for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const int8_array  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const int16_array &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const int32_array &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const int64_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const uint8_array  &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const uint16_array &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const uint32_array &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const uint64_array &data)
{
    fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const float32_array &data)
{
fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,const float64_array &data)
{
fetch(path).set(data);
}

//-----------------------------------------------------------------------------
// -- set_path for string types -- 
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path(const std::string &path,
               const std::string &data)
{
    fetch(path).set(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_char8_str(const std::string &path,
                         const char* data)
{
    fetch(path).set_char8_str(data);
}



//-----------------------------------------------------------------------------
// -- set_path via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

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
    fetch(path).set(data,
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
    fetch(path).set(data,
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
    fetch(path).set(data,
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
    fetch(path).set(data,
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
Node::set_path(const std::string &path,
               uint16 *data,
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
Node::set_path(const std::string &path,
               uint32 *data, 
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
Node::set_path(const std::string &path,
               uint64 *data,
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
// floating point integer pointer cases
//-----------------------------------------------------------------------------

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
    fetch(path).set(data,
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
    fetch(path).set(data,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

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
Node::set_external(Node &node)
{
    reset();
    m_schema->set(node.schema());
    mirror_node(this,m_schema,&node);
}

//---------------------------------------------------------------------------//
void
Node::set_external(const Schema &schema, void *data)
{
    reset();
    m_schema->set(schema);
    walk_schema(this,m_schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_external(const DataType &dtype, void *data)
{
    reset();
    m_data    = data;
    m_schema->set(dtype);
}


//-----------------------------------------------------------------------------
// -- set_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(int8 *data,
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
Node::set_external(int16 *data,
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
Node::set_external(int32 *data,
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
Node::set_external(int64 *data,
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



//-----------------------------------------------------------------------------
// unsigned integer pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(uint8 *data,
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
Node::set_external(uint16 *data,
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
Node::set_external(uint32 *data,
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
Node::set_external(uint64 *data,
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

//-----------------------------------------------------------------------------
// floating point pointer cases
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(float32 *data,
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
Node::set_external(float64 *data,
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
    



//-----------------------------------------------------------------------------
// -- set_external for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int8>  &data)
{
    release();
    m_schema->set(DataType::int8((index_t)data.size()));
    m_data  = &data[0];
}
    

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int16>  &data)
{
    release();
    m_schema->set(DataType::int16((index_t)data.size()));
    m_data  = &data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int32>  &data)
{
    release();
    m_schema->set(DataType::int32((index_t)data.size()));
    m_data  = &data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<int64>  &data)
{
    release();
    m_schema->set(DataType::int64((index_t)data.size()));
    m_data  = &data[0];
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint8>  &data)
{
    release();
    m_schema->set(DataType::uint8((index_t)data.size()));
    m_data  = &data[0];
}


//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint16>  &data)
{
    release();
    m_schema->set(DataType::uint16((index_t)data.size()));
    m_data  = &data[0];
}


//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint32>  &data)
{
    release();
    m_schema->set(DataType::uint32((index_t)data.size()));
    m_data  = &data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<uint64>  &data)
{
    release();
    m_schema->set(DataType::uint64((index_t)data.size()));
    m_data  = &data[0];
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<float32>  &data)
{
    release();
    m_schema->set(DataType::float32((index_t)data.size()));
    m_data  = &data[0];
}

//---------------------------------------------------------------------------//
void 
Node::set_external(std::vector<float64>  &data)
{
    release();
    m_schema->set(DataType::float64((index_t)data.size()));
    m_data  = &data[0];
}


    //-----------------------------------------------------------------------------
// -- set_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(const int8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data   = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const int64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const uint64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}


//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void 
Node::set_external(const float32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}

//---------------------------------------------------------------------------//
void 
Node::set_external(const float64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_pointer();
}


//---------------------------------------------------------------------------//
void 
Node::set_external_char8_str(char *data)
{
    release();
    
    // size including the null term
    index_t str_size_with_term = strlen(data)+1;
    DataType str_t(DataType::CHAR8_STR_T,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_T);

    m_schema->set(str_t);
    m_data  = data;
}


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
Node::set_path_external(const std::string &path,
                        const Schema &schema,
                        void *data)
{
    fetch(path).set_external(schema,data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        const DataType &dtype,
                        void *data)
{
    fetch(path).set_external(dtype,data);
}

//-----------------------------------------------------------------------------
// -- set_path_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer pointer cases
//-----------------------------------------------------------------------------

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
    fetch(path).set_external(data,
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
    fetch(path).set_external(data,
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
    fetch(path).set_external(data,
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
    fetch(path).set_external(data,
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
Node::set_path_external(const std::string &path,
                        uint8  *data,
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
    fetch(path).set_external(data,
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
    fetch(path).set_external(data,
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
    fetch(path).set_external(data,
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
Node::set_path_external(const std::string &path,
           float32 *data,
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
    fetch(path).set_external(data,
                             num_elements,
                             offset,
                             stride,
                             element_bytes,
                             endianness);
}


//-----------------------------------------------------------------------------
// -- set_path_external for std::vector types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int8> &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int16> &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int32> &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<int64> &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint8>   &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint16>  &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint32>  &data)
{
    fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<uint64>  &data)
{
    fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
// floating point array types via std::vector
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<float32> &data)
{
    fetch(path).set_external(data);
}
//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                        std::vector<float64> &data)
{
    fetch(path).set_external(data);
}

    //-----------------------------------------------------------------------------
// -- set_path_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// signed integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const int8_array  &data)
{
        fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const int16_array &data)
{
        fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const int32_array &data)
{
        fetch(path).set_external(data);
}
//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const int64_array &data)
{
        fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
// unsigned integer array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const uint8_array  &data)
{
        fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const uint16_array &data)
{
        fetch(path).set_external(data);
}

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const uint32_array &data)
{
        fetch(path).set_external(data);
}
void
Node::set_path_external(const std::string &path,
                       const uint64_array &data)
{
        fetch(path).set_external(data);
}

//-----------------------------------------------------------------------------
// floating point array types via conduit::DataArray
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::set_path_external(const std::string &path,
                       const float32_array &data)
{
        fetch(path).set_external(data);
}
void
Node::set_path_external(const std::string &path,
                       const float64_array &data)
{
        fetch(path).set_external(data);
}

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
// unsigned integer array ttypes via conduit::DataArray
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
        THROW_ERROR("<Node::serialize> failed to open: " << stream_path);
    serialize(ofs);
    ofs.close();
}


//---------------------------------------------------------------------------//
void
Node::serialize(std::ofstream &ofs) const
{
    index_t dtype_id = dtype().id();
    if( dtype_id == DataType::OBJECT_T ||
        dtype_id == DataType::LIST_T)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin(); itr < m_children.end(); ++itr)
        {
            (*itr)->serialize(ofs);
        }
    }
    else if( dtype_id != DataType::EMPTY_T)
    {
        if(is_compact())
        {
            // ser as is. This copies stride * num_ele bytes
            {
                ofs.write((const char*)element_pointer(0),
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
Node::compact()
{
    // TODO: we can do this more efficiently
    Node n;
    compact_to(n);
    set(n);
}


//---------------------------------------------------------------------------//
Node
Node::compact_to()const
{
    // NOTE: very ineff w/o move semantics
    Node res;
    compact_to(res);
    return res;
}


//---------------------------------------------------------------------------//
void
Node::compact_to(Node &n_dest) const
{
    n_dest.reset();
    index_t c_size = total_bytes_compact();
    m_schema->compact_to(*n_dest.schema_pointer());
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
Node::update(Node &n_src)
{
    // walk src and add it contents to this node
    // OBJECT_T is the only special case here?
    /// TODO:
    /// arrays and non empty leafs will simply overwrite the current
    /// node, these semantics seem sensible, but we could revisit this
    index_t dtype_id = n_src.dtype().id();
    if( dtype_id == DataType::OBJECT_T)
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
    else if(dtype_id != DataType::EMPTY_T)
    {
        if(this->dtype().is_compatible(n_src.dtype()))
        {
            memcpy(element_pointer(0),
                   n_src.element_pointer(0), 
                   m_schema->total_bytes());
        }
        else // not compatible
        {
            n_src.compact_to(*this);
        }
        //set(n_src);
    }
}

//-----------------------------------------------------------------------------
// -- leaf coercion methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
int64
Node::to_int64() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_T:  return (int64)as_int8();
        case DataType::INT16_T: return (int64)as_int16();
        case DataType::INT32_T: return (int64)as_int32();
        case DataType::INT64_T: return as_int64();
        /* uints */
        case DataType::UINT8_T:  return (int64)as_uint8();
        case DataType::UINT16_T: return (int64)as_uint16();
        case DataType::UINT32_T: return (int64)as_uint32();
        case DataType::UINT64_T: return (int64)as_uint64();
        /* floats */
        case DataType::FLOAT32_T: return (int64)as_float32();
        case DataType::FLOAT64_T: return (int64)as_float64();
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
        case DataType::INT8_T:  return (uint64)as_int8();
        case DataType::INT16_T: return (uint64)as_int16();
        case DataType::INT32_T: return (uint64)as_int32();
        case DataType::INT64_T: return (uint64)as_int64();
        /* uints */
        case DataType::UINT8_T:  return (uint64)as_uint8();
        case DataType::UINT16_T: return (uint64)as_uint16();
        case DataType::UINT32_T: return (uint64)as_uint32();
        case DataType::UINT64_T: return as_uint64();
        /* floats */
        case DataType::FLOAT32_T: return (uint64)as_float32();
        case DataType::FLOAT64_T: return (uint64)as_float64();
    }
    return 0;
}

//---------------------------------------------------------------------------//
float64
Node::to_float64() const
{
    switch(dtype().id())
    {
        /* ints */
        case DataType::INT8_T:  return (float64)as_int8();
        case DataType::INT16_T: return (float64)as_int16();
        case DataType::INT32_T: return (float64)as_int32();
        case DataType::INT64_T: return (float64)as_int64();
        /* uints */
        case DataType::UINT8_T:  return (float64)as_uint8();
        case DataType::UINT16_T: return (float64)as_uint16();
        case DataType::UINT32_T: return (float64)as_uint32();
        case DataType::UINT64_T: return (float64)as_uint64();
        /* floats */
        case DataType::FLOAT32_T: return (float64)as_float32();
        case DataType::FLOAT64_T: return as_float64();
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
        case DataType::INT8_T:  return (index_t)as_int8();
        case DataType::INT16_T: return (index_t)as_int16();
        case DataType::INT32_T: return (index_t)as_int32();
        case DataType::INT64_T: return (index_t)as_int64();
        /* uints */
        case DataType::UINT8_T:  return (index_t)as_uint8();
        case DataType::UINT16_T: return (index_t)as_uint16();
        case DataType::UINT32_T: return (index_t)as_uint32();
        case DataType::UINT64_T: return (index_t)as_uint64();
        /* floats */
        case DataType::FLOAT32_T: return (index_t)as_float32();
        case DataType::FLOAT64_T: return (index_t)as_float64();
    }
    return 0;
}

//-----------------------------------------------------------------------------
// -- JSON construction methods ---
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
std::string 
Node::to_json(bool detailed,
              index_t indent, 
              index_t depth,
              const std::string &pad,
              const std::string &eoe) const
{
   std::ostringstream oss;
   to_json(oss,detailed,indent,depth,pad,eoe);
   return oss.str();
}

//---------------------------------------------------------------------------//
void
Node::to_json(std::ostringstream &oss,
              bool detailed, 
              index_t indent, 
              index_t depth,
              const std::string &pad,
              const std::string &eoe) const
{
    if(dtype().id() == DataType::OBJECT_T)
    {
        oss << eoe;
        utils::indent(oss,indent,depth,pad);
        oss << "{" << eoe;
    
        index_t nchildren = m_children.size();
        for(index_t i=0; i < nchildren;i++)
        {
            utils::indent(oss,indent,depth+1,pad);
            oss << "\""<< m_schema->object_order()[i] << "\": ";
            m_children[i]->to_json(oss,
                                   detailed,
                                   indent,
                                   depth+1,
                                   pad,
                                   eoe);
            if(i < nchildren-1)
                oss << ",";
            oss << eoe;
        }
        utils::indent(oss,indent,depth,pad);
        oss << "}";
    }
    else if(dtype().id() == DataType::LIST_T)
    {
        oss << eoe;
        utils::indent(oss,indent,depth,pad);
        oss << "[" << eoe;
        
        index_t nchildren = m_children.size();
        for(index_t i=0; i < nchildren;i++)
        {
            utils::indent(oss,indent,depth+1,pad);
            m_children[i]->to_json(oss,
                                   detailed,
                                   indent,
                                   depth+1,
                                   pad,
                                   eoe);
            if(i < nchildren-1)
                oss << ",";
            oss << eoe;
        }
        utils::indent(oss,indent,depth,pad);
        oss << "]";      
    }
    else // assume leaf data type
    {
        std::ostringstream value_oss; 
        switch(dtype().id())
        {
            // ints 
            case DataType::INT8_T:
                as_int8_array().to_json(value_oss);
                break;
            case DataType::INT16_T:
                as_int16_array().to_json(value_oss);
                break;
            case DataType::INT32_T:
                as_int32_array().to_json(value_oss);
                break;
            case DataType::INT64_T:
                as_int64_array().to_json(value_oss);
                break;
            // uints 
            case DataType::UINT8_T:
                as_uint8_array().to_json(value_oss);
                break;
            case DataType::UINT16_T: 
                as_uint16_array().to_json(value_oss);
                break;
            case DataType::UINT32_T:
                as_uint32_array().to_json(value_oss);
                break;
            case DataType::UINT64_T:
                as_uint64_array().to_json(value_oss);
                break;
            // floats 
            case DataType::FLOAT32_T:
                as_float32_array().to_json(value_oss);
                break;
            case DataType::FLOAT64_T:
                as_float64_array().to_json(value_oss);
                break;
            // char8_str
            case DataType::CHAR8_STR_T: 
                value_oss << "\"" << as_char8_str() << "\""; 
                break;
        }

        if(!detailed)
            oss << value_oss.str();
        else
            dtype().to_json(oss,value_oss.str());
    }  
}

//---------------------------------------------------------------------------//
std::string
Node::to_pure_json(index_t indent) const
{
    return to_json(false,indent);
}

void
Node::to_pure_json(std::ostringstream &oss,
                   index_t indent) const
{
    to_json(oss,false,indent);
}

//---------------------------------------------------------------------------//
std::string
Node::to_detailed_json(index_t indent, 
                       index_t depth,
                       const std::string &pad,
                       const std::string &eoe) const
{
    return to_json(true,indent,depth,pad,eoe);
}

//---------------------------------------------------------------------------//
void
Node::to_detailed_json(std::ostringstream &oss,
                       index_t indent, 
                       index_t depth,
                       const std::string &pad,
                       const std::string &eoe) const
{
    to_json(oss,true,indent,depth,pad,eoe);
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
    
    NodeIterator itr = mspaces.iterator();
    
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
    // NOTE: very ineff w/o move semantics
    Node res;
    info(res);
    return res;
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
Node::iterator()
{
    return NodeIterator(this,0);
}

//---------------------------------------------------------------------------//
Node&
Node::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    if(dtype().id() != DataType::OBJECT_T)
    {
        init(DataType::object());
    }
    
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // check for parent
    if(p_curr == "..")
    {
        if(m_parent != NULL) // TODO: check for error (no parent) ?
           return m_parent->fetch(p_next);
    }

    // if this node doesn't exist yet, we need to create it and
    // link it to a schema
        
    index_t idx;
    if(!m_schema->has_path(p_curr))
    {
        Schema *schema_ptr = m_schema->fetch_pointer(p_curr);
        Node *curr_node = new Node();
        curr_node->set_schema_pointer(schema_ptr);
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
Node&
Node::child(index_t idx)
{
    return *m_children[idx];
}


//---------------------------------------------------------------------------//
Node *
Node::fetch_pointer(const std::string &path)
{
    return &fetch(path);
}

//---------------------------------------------------------------------------//
Node *
Node::child_pointer(index_t idx)
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
Node&
Node::operator[](index_t idx)
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
Node &
Node::append()
{
    init_list();
    index_t idx = m_children.size();
    //
    // This makes a proper copy of the schema for us to use
    //
    m_schema->append();
    Schema *schema_ptr = m_schema->child_pointer(idx);

    Node *res_node = new Node();
    res_node->set_schema_pointer(schema_ptr);
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

//-----------------------------------------------------------------------------
//
// -- end definition of Node entry access methods --
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
Node::set_schema_pointer(Schema *schema_ptr)
{
    if(m_schema->is_root())
        delete m_schema;
    m_schema = schema_ptr;    
}
    
//---------------------------------------------------------------------------//
void
Node::set_data_pointer(void *data)
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
//
// -- private methods that help with init, memory allocation, and cleanup --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
Node::init(const DataType& dtype)
{
    if(this->dtype().is_compatible(dtype))
        return;
    
    if(m_data != NULL)
    {
        release();
    }

    index_t dt_id = dtype.id();
    if(dt_id == DataType::OBJECT_T ||
       dt_id == DataType::LIST_T)
    {
        m_children.clear();
    }
    else if(dt_id != DataType::EMPTY_T)
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
Node::mmap(const std::string &stream_path, index_t dsize)
{
#if defined(CONDUIT_PLATFORM_WINDOWS)
    ///
    /// TODO: mmap isn't supported on windows, we need to use a 
    /// a windows specific API.  
    /// See: https://lc.llnl.gov/jira/browse/CON-38
    ///
    /// For now, we simply throw an error
    ///
    THROW_ERROR("<Node::mmap> conduit does not yet support mmap on Windows");
#else    
    m_mmap_fd   = open(stream_path.c_str(),O_RDWR| O_CREAT);
    m_data_size = dsize;

    if (m_mmap_fd == -1) 
        THROW_ERROR("<Node::mmap> failed to open: " << stream_path);

    m_data = ::mmap(0, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, m_mmap_fd, 0);

    if (m_data == MAP_FAILED) 
        THROW_ERROR("<Node::mmap> MAP_FAILED" << stream_path);
    
    m_alloced = false;
    m_mmaped  = true;
#endif    
}


//---------------------------------------------------------------------------//
void
Node::release()
{
    for (index_t i = 0; i < m_children.size(); i++) {
        Node* node = m_children[i];
        delete node;
    }
    m_children.clear();

    if(m_alloced && m_data)
    {
        if(dtype().id() != DataType::EMPTY_T)
        {   
            // clean up our storage
            free(m_data);
            m_data = NULL;
            m_alloced = false;
            m_data_size = 0;
        }
    }   
#if !defined(CONDUIT_PLATFORM_WINDOWS)
    ///
    /// TODO: mmap isn't yet supported on windows
    ///
    else if(m_mmaped && m_data)
    {
        if(munmap(m_data, m_data_size) == -1) 
        {
            // TODO error
        }
        close(m_mmap_fd);
        m_data      = NULL;
        m_mmap_fd   = -1;
        m_data_size = 0;
    }
#endif
}
    

//---------------------------------------------------------------------------//
void
Node::cleanup()
{
    release();
    if(m_schema->is_root())
    {
        if(m_schema != NULL)
        {
            delete m_schema;
            m_schema = NULL;
        }
    }
    else if(m_schema != NULL)
    {
        m_schema->set(DataType::EMPTY_T);
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
    m_mmap_fd   = -1;

    m_schema = new Schema(DataType::EMPTY_T);
    
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
    node->set_schema_pointer(schema);
    node->set_data_pointer(data);
    if(schema->dtype().id() == DataType::OBJECT_T)
    {
        for(index_t i=0;i<schema->children().size();i++)
        {
    
            std::string curr_name = schema->object_order()[i];
            Schema *curr_schema   = schema->fetch_pointer(curr_name);
            Node *curr_node = new Node();
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append_node_pointer(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_T)
    {
        index_t num_entries = schema->number_of_children();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = schema->child_pointer(i);
            Node *curr_node = new Node();
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append_node_pointer(curr_node);
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
    node->set_schema_pointer(schema);
    node->set_data_pointer(src->m_data);
    
    if(schema->dtype().id() == DataType::OBJECT_T)
    {
        for(index_t i=0;i<schema->children().size();i++)
        {
    
            std::string curr_name = schema->object_order()[i];
            Schema *curr_schema   = schema->fetch_pointer(curr_name);
            Node *curr_node = new Node();
            Node *curr_src = src->child_pointer(i);
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(node);
            mirror_node(curr_node,curr_schema,curr_src);
            node->append_node_pointer(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_T)
    {
        index_t num_entries = schema->number_of_children();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = schema->child_pointer(i);
            Node *curr_node = new Node();
            Node *curr_src = src->child_pointer(i);
            curr_node->set_schema_pointer(curr_schema);
            curr_node->set_parent(node);
            mirror_node(curr_node,curr_schema,curr_src);
            node->append_node_pointer(curr_node);
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
    index_t dtype_id = dtype().id();
    if(dtype_id == DataType::OBJECT_T ||
       dtype_id == DataType::LIST_T)
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
    if(dtype_id == DataType::OBJECT_T ||
       dtype_id == DataType::LIST_T ||
       dtype_id == DataType::EMPTY_T)
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
                   element_pointer(i),
                   ele_bytes);
            data_ptr+=ele_bytes;
        }
    }
}


//---------------------------------------------------------------------------//
void
Node::serialize(uint8 *data,index_t curr_offset) const
{
    if(dtype().id() == DataType::OBJECT_T ||
       dtype().id() == DataType::LIST_T)
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
    if(dtype_id == DataType::OBJECT_T)
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
    else if(dtype_id == DataType::LIST_T)
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

