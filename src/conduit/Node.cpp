/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: Node.cpp
///

#include "Node.h"
#include "Utils.h"
#include "Generator.h"
#include "rapidjson/document.h"
#include <iostream>
#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace conduit
{

//============================================
/// Node
//============================================
void
Node::init_defaults()
{
    m_data = NULL;
    m_alloced = false;
    m_alloced_size = 0;

    m_mmaped    = false;
    m_mmap_fd   = -1;
    m_mmap_size = 0;

    m_schema = new Schema(DataType::EMPTY_T);
    
    m_parent = NULL;
}

//============================================
Node::Node()
{
    init_defaults();
}

//============================================
Node::Node(const Node &node)
{
    init_defaults();
    set(node);
}

//============================================
Node::Node(const Schema &schema)

{
    init_defaults();
    set(schema);
}
    
//============================================
Node::Node(Schema *schema_ptr)

{
    init_defaults();
    m_schema = schema_ptr;
}
    
//============================================
Node::Node(const Node &node, Schema *schema_ptr)

{
    init_defaults();
    set(node,schema_ptr);
}

    
//============================================
Node::Node(const Schema &schema, const std::string &stream_path, bool mmap)
{    
    init_defaults();
    if(mmap)
        conduit::Node::mmap(schema,stream_path);
    else
        load(schema,stream_path);
}


//============================================
Node::Node(const Generator &gen)
{
    init_defaults(); 
    gen.walk(*this);
}

//============================================
Node::Node(const std::string &json_schema, void *data)
{
    init_defaults(); 
    Generator g(json_schema,data);
    g.walk(*this);

}


//============================================
Node::Node(const Schema &schema, void *data)
{
    init_defaults();
    std::string json_schema =schema.to_json(); 
    Generator g(json_schema,data);
    g.walk(*this);
}


//============================================
Node::Node(const DataType &dtype, void *data)
{    
    init_defaults();
    set(dtype,data);
}

//============================================
/* int vec types */
//============================================

//============================================
Node::Node(const std::vector<int8>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<int16>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<int32>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<int64>  &data)
{
   init_defaults();
   set(data);
}

//============================================
/* uint vec types */
//============================================

//============================================
Node::Node(const std::vector<uint8>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<uint16>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<uint32>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<uint64>  &data)
{
   init_defaults();
   set(data);
}

//============================================
/* float vec types */
//============================================

//============================================
Node::Node(const std::vector<float32>  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const std::vector<float64>  &data)
{
   init_defaults();
   set(data);
}

//============================================
/* int array types */
//============================================

//============================================
Node::Node(const int8_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const int16_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const int32_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const int64_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
/* uint array types */
//============================================

//============================================
Node::Node(const uint8_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const uint16_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const uint32_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const uint64_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
/* float arr types */
//============================================

//============================================
Node::Node(const float32_array  &data)
{
   init_defaults();
   set(data);
}

//============================================
Node::Node(const float64_array  &data)
{
   init_defaults();
   set(data);
}


//============================================
Node::Node(const DataType &dtype)
{
    init_defaults();
    set(dtype);
}

//============================================
/// int types
//============================================

//============================================
Node::Node(int8  data)
{
    init_defaults();
    set(data);
}

//============================================
Node::Node(int16  data)
{
    init_defaults();
    set(data);
}
    
//============================================
Node::Node(int32  data)
{
    init_defaults();
    set(data);
}

//============================================
Node::Node(int64  data)
{    
    init_defaults();
    set(data);
}


//============================================
/// uint types
//============================================

//============================================
Node::Node(uint8  data)
{
    init_defaults();
    set(data);
}

//============================================
Node::Node(uint16  data)
{
    init_defaults();
    set(data);
}
    
//============================================
Node::Node(uint32  data)
{
    init_defaults();
    set(data);
}

//============================================
Node::Node(uint64  data)
{
    init_defaults();
    set(data);
}

//============================================
/// float types
//============================================

//============================================
Node::Node(float32 data)
{
    init_defaults();
    set(data);
}


//============================================
Node::Node(float64 data)
{
    init_defaults();
    set(data);
}


//============================================
Node::~Node()
{
    cleanup();
}

//============================================
void
Node::reset()
{
    release();
    m_schema->set(DataType::EMPTY_T);
}

//============================================
void 
Node::load(const Schema &schema, const std::string &stream_path)
{
    index_t dsize = schema.total_bytes();

    allocate(dsize);
    std::ifstream ifs;
    ifs.open(stream_path.c_str());
    ifs.read((char *)m_data,dsize);
    ifs.close();       
    
    ///
    /// See Below
    ///
    m_alloced = false;
    
    walk_schema(schema,m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_alloced = true;
}

//============================================
void 
Node::mmap(const Schema &schema, const std::string &stream_path)
{
    reset();
    index_t dsize = schema.total_bytes();
    Node::mmap(stream_path,dsize);

    ///
    /// See Below
    ///
    m_mmaped = false;
    
    walk_schema(schema,m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_mmaped = true;
}

// --- begin set ---

//============================================
void 
Node::set(const Node &node)
{
    set(node, NULL);
}

void
Node::set(const Node& node, Schema* schema)
{
    if (node.dtype().id() != DataType::EMPTY_T)
    {
    
        if(node.dtype().id() == DataType::OBJECT_T || 
           node.dtype().id() == DataType::LIST_T)
        {
            init(node.dtype());

            // If we are making a new head, copy the schema, otherwise, use
            // the pointer we were given
            if (schema != NULL)
            {
                m_schema = schema;
            } 
            else 
            {
                m_schema = new Schema(node.schema());
            }
            
            for(index_t i=0;i<node.m_children.size();i++)
            {
                Node *child = new Node();
                child->m_parent = this;
                child->set(*node.m_children[i],m_schema->children()[i]);
                m_children.push_back(child);
            }
        }
        else
        {
            if(this->dtype().is_compatible(node.dtype()))
            {
                memcpy(element_pointer(0), node.element_pointer(0), m_schema->total_bytes());
            }
            else
            {
                init(node.dtype());
                memcpy(m_data, node.m_data, m_schema->total_bytes());
            }
            
//            // check if compatiable
//            if (node.m_alloced) 
//            {
//                // TODO: compaction?
//                init(node.dtype());
//                memcpy(m_data, node.m_data, m_schema->total_bytes());
//            }
//            else 
//            {
//                // TODO: this needs to be handled by set external ...
//                m_alloced = false;
//                m_data    = node.m_data;
//                m_schema->set(node.schema());
//            }
        }
    }
    else
    {
        // if passed node is empty -- reset this.
        reset();
    }

}

//============================================
void 
Node::set(const DataType &dtype)
{
    init(dtype);
}
    
    
//============================================
void 
Node::set(bool8 data)
{
    init(DataType::Scalars::bool8());
    *(bool8*)((char*)m_data + schema().element_index(0)) = data;
}
    

//============================================
/// int types
//============================================

//============================================
void 
Node::set(int8 data)
{
    init(DataType::Scalars::int8());
    *(int8*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(int16 data)
{
    init(DataType::Scalars::int16());
    *(int16*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(int32 data)
{
    init(DataType::Scalars::int32());
    *(int32*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(int64 data)
{
    init(DataType::Scalars::int64());
    *(int64*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
/// uint types
//============================================

//============================================
void 
Node::set(uint8 data)
{
    init(DataType::Scalars::uint8());
    *(uint8*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(uint16 data)
{
    init(DataType::Scalars::uint16());
    *(uint16*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(uint32 data)
{
    init(DataType::Scalars::uint32());
    *(uint32*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(uint64 data)
{
    init(DataType::Scalars::uint64());
    *(uint64*)((char*)m_data + schema().element_index(0)) = data;
}

//============================================
/// float types
//============================================

//============================================
void 
Node::set(float32 data)
{
    init(DataType::Scalars::float32());
    *(float32*)((char*)m_data + schema().element_index(0)) = data;
}


//============================================
void 
Node::set(float64 data)
{
    init(DataType::Scalars::float64());
    *(float64*)((char*)m_data + schema().element_index(0)) = data;
}

//============================================
/// int vec types
//============================================

//============================================
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

//============================================
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

//============================================
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

//============================================
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


//============================================
/// uint vec types
//============================================

//============================================
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

//============================================
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

//============================================
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

//============================================
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

//============================================
/// float vec types
//============================================

//============================================
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

//============================================
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

//============================================
/// int array types
//============================================

//============================================
void 
Node::set(const int8_array  &data)
{
    init(DataType::Arrays::int8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const int16_array  &data)
{
    init(DataType::Arrays::int16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const int32_array  &data)
{
    init(DataType::Arrays::int32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const int64_array  &data)
{
    init(DataType::Arrays::int64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}


//============================================
/// uint array types
//============================================

//============================================

void 
Node::set(const uint8_array  &data)
{
    init(DataType::Arrays::uint8(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const uint16_array  &data)
{
    init(DataType::Arrays::uint16(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const uint32_array  &data)
{
    init(DataType::Arrays::uint32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const uint64_array  &data)
{
    init(DataType::Arrays::uint64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}
//============================================
/// float array types
//============================================

//============================================
void 
Node::set(const float32_array  &data)
{
    init(DataType::Arrays::uint32(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const float64_array  &data)
{
    init(DataType::Arrays::uint64(data.number_of_elements()));
    data.compact_elements_to((uint8*)m_data);
}

//============================================
void 
Node::set(const std::string  &data)
{
    release();
    // size including the null term
    index_t str_size_with_term = data.length()+1;
    DataType str_t(DataType::BYTESTR_T,
                   str_size_with_term,
                   0,
                   sizeof(char),
                   sizeof(char),
                   Endianness::DEFAULT_T);
    init(str_t);
    memcpy(m_data,data.c_str(),sizeof(char)*str_size_with_term);
}

//============================================
void 
Node::set(const char *data, index_t dtype_id)
{
    if(dtype_id == DataType::BYTESTR_T)
    {
        release();
        // size including the null term
        index_t str_size_with_term = strlen(data)+1;
        DataType str_t(DataType::BYTESTR_T,
                       str_size_with_term,
                       0,
                       sizeof(char),
                       sizeof(char),
                       Endianness::DEFAULT_T);
                       init(str_t);
                       memcpy(m_data,data,sizeof(char)*str_size_with_term);
    }
    else
    {
        //TODO: Error (add support for alaised 8-bit int case?)
    }
}

// --- end set ---

// --- begin set_external ---

    
//============================================
void 
Node::set_external(bool8 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::bool8(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}
    

//============================================
/// int types
//============================================

//============================================
void 
Node::set_external(int8 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::int8(num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(int16 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::int16(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(int32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::int32(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(int64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::int32(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}



//============================================
/// uint types
//============================================

//============================================
void 
Node::set_external(uint8 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::uint8(num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness));
    m_data  = data;
}



//============================================
void 
Node::set_external(uint16 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::uint16(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(uint32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::uint32(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness));
    m_data  = data;
}

//============================================
void 
Node::set_external(uint64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::uint64(num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness));
    m_data  = data;
}

//============================================
/// float types
//============================================

//============================================
void 
Node::set_external(float32 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::float32(num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(float64 *data,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    m_schema->set(DataType::Arrays::float64(num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness));
    m_data  = data;
}


//============================================
void 
Node::set_external(std::vector<bool8>  &data)
{
    release();
    m_schema->set(DataType::Arrays::bool8((index_t)data.size()));
    /*
     TODO:: BUG Why doesn't this compile? Are we ended up with a special 1-bit stl vector?
    m_data  = &data[0];
    // clang++ error message:
    /Users/harrison37/Work/conduit/src/conduit/Node.cpp:1167:13: error: assigning to 'void *' from incompatible
          type '__bit_iterator<std::__1::vector<bool, std::__1::allocator<bool> >, false>'
        m_data  = &data[0];
                ^ ~~~~~~~~
    */
}
    

//============================================
/// int vec types
//============================================

//============================================
void 
Node::set_external(std::vector<int8>  &data)
{
    release();
    m_schema->set(DataType::Arrays::int8((index_t)data.size()));
    m_data  = &data[0];
}
    

//============================================
void 
Node::set_external(std::vector<int16>  &data)
{
    release();
    m_schema->set(DataType::Arrays::int16((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
void 
Node::set_external(std::vector<int32>  &data)
{
    release();
    m_schema->set(DataType::Arrays::int32((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
void 
Node::set_external(std::vector<int64>  &data)
{
    release();
    m_schema->set(DataType::Arrays::int64((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
/// uint vec types
//============================================

//============================================
void 
Node::set_external(std::vector<uint8>  &data)
{
    release();
    m_schema->set(DataType::Arrays::uint8((index_t)data.size()));
    m_data  = &data[0];
}


//============================================
void 
Node::set_external(std::vector<uint16>  &data)
{
    release();
    m_schema->set(DataType::Arrays::uint16((index_t)data.size()));
    m_data  = &data[0];
}


//============================================
void 
Node::set_external(std::vector<uint32>  &data)
{
    release();
    m_schema->set(DataType::Arrays::uint32((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
void 
Node::set_external(std::vector<uint64>  &data)
{
    release();
    m_schema->set(DataType::Arrays::uint64((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
/// float vec types
//============================================

//============================================
void 
Node::set_external(std::vector<float32>  &data)
{
    release();
    m_schema->set(DataType::Arrays::float32((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
void 
Node::set_external(std::vector<float64>  &data)
{
    release();
    m_schema->set(DataType::Arrays::float64((index_t)data.size()));
    m_data  = &data[0];
}

//============================================
/// int array types
//============================================

//============================================
void 
Node::set_external(const int8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data   = data.data_ptr();
}

//============================================
void 
Node::set_external(const int16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const int32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const int64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}


//============================================
/// uint array types
//============================================

//============================================

void 
Node::set_external(const uint8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const uint16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const uint32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const uint64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
//============================================
/// float array types
//============================================

//============================================
void 
Node::set_external(const float32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

//============================================
void 
Node::set_external(const float64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}


//============================================
void 
Node::set_external(char *data,
                   index_t dtype_id,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    release();
    // size including the null term

    if(num_elements == 0 && dtype_id == DataType::BYTESTR_T)
    {
        num_elements= strlen(data)+1;
    }
    else
    {
        // TODO: Error
    }

    DataType vec_t(dtype_id,
                   num_elements,
                   offset,
                   stride,
                   element_bytes,
                   endianness);

    m_schema->set(vec_t);
    m_data  = data;
}

//==== --- end set_external --- 

//============================================
void
Node::set(const Schema &schema)
{
    walk_schema(schema);    
}

//============================================
void
Node::set(const Schema &schema,void* data)
{
    walk_schema(schema,data);    
}

//============================================
void
Node::set(Schema *schema_ptr)
{
    if(m_schema->is_root())
        delete m_schema;
    m_schema = schema_ptr;    
}
    
//============================================
void
Node::set(Schema *schema_ptr,void *data)
{
    set(schema_ptr);
    release();
    m_data    = data;    
}
    
//============================================
void
Node::set(const DataType &dtype, void *data)
{
    release();
    m_alloced = false;
    m_data    = data;
    m_schema->set(dtype);
}

//============================================
Node &
Node::operator=(const Node &node)
{
    if(this != &node)
    {
        set(node);
    }
    return *this;
}

//============================================
Node &
Node::operator=(DataType dtype)
{
    set(dtype);
    return *this;
}

//============================================
/// uint types
//============================================

//============================================
Node &
Node::operator=(uint8 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(uint16 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(uint32 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(uint64 data)
{
    set(data);
    return *this;
}

//============================================
/// int types
//============================================

//============================================
Node &
Node::operator=(int8 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(int16 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(int32 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(int64 data)
{
    set(data);
    return *this;
}

//============================================
/// float types
//============================================

//============================================
Node &
Node::operator=(float32 data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(float64 data)
{
    set(data);
    return *this;
}

//============================================
/// int vec types
//============================================

//============================================
Node &
Node::operator=(const std::vector<int8> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<int16> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<int32> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<int64> &data)
{
    set(data);
    return *this;
}

//============================================
/// uint vec types
//============================================

//============================================
Node &
Node::operator=(const std::vector<uint8> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<uint16> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<uint32> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<uint64> &data)
{
    set(data);
    return *this;
}

//============================================
/// float vec types
//============================================

//============================================
Node &
Node::operator=(const std::vector<float32> &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const std::vector<float64> &data)
{
    set(data);
    return *this;
}

//============================================
/// int array types
//============================================

//============================================
Node &
Node::operator=(const int8_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const int16_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const int32_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const int64_array &data)
{
    set(data);
    return *this;
}

//============================================
/// uint vec types
//============================================

//============================================
Node &
Node::operator=(const uint8_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const uint16_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const uint32_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const uint64_array &data)
{
    set(data);
    return *this;
}

//============================================
/// float vec types
//============================================

//============================================
Node &
Node::operator=(const float32_array &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const float64_array &data)
{
    set(data);
    return *this;
}

//============================================
/// bytestr types
//============================================

//============================================
Node &
Node::operator=(const std::string &data)
{
    set(data);
    return *this;
}

//============================================
Node &
Node::operator=(const char *data)
{
    set(data);
    return *this;
}


//============================================
void
Node::serialize(std::vector<uint8> &data,bool compact) const
{
    data = std::vector<uint8>(total_bytes(),0);
    serialize(&data[0],0,compact);
}

//============================================
void
Node::serialize(const std::string &stream_path,
                bool compact) const
{
    std::ofstream ofs;
    ofs.open(stream_path.c_str());
    serialize(ofs,compact);
    ofs.close();
}

//============================================
void
Node::save(const std::string &obase) const
{
    Node res;
    compact_to(res);
    std::string ofschema = obase + ".conduit_json";
    std::string ofdata   = obase + ".conduit_bin";
    res.schema().save(ofschema);
    res.serialize(ofdata,true);
}


//============================================
void
Node::load(const std::string &ibase)
{
    Schema s;
    std::string ifschema = ibase + ".conduit_json";
    std::string ifdata   = ibase + ".conduit_bin";
    s.load(ifschema);
    load(s,ifdata);
}


//============================================
void
Node::mmap(const std::string &ibase)
{
    Schema s;
    std::string ifschema = ibase + ".conduit_json";
    std::string ifdata   = ibase + ".conduit_bin";
    s.load(ifschema);
    mmap(s,ifdata);
}



//============================================
void
Node::serialize(std::ofstream &ofs,
                bool compact) const
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


//============================================
void
Node::serialize(uint8 *data,index_t curr_offset,bool compact) const
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

//============================================
NodeIterator
Node::iterator()
{
    return NodeIterator(this,0);
}

//============================================
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

//============================================
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
                ptr_ref["bytes"] = m_alloced_size;
            }
            else if(m_mmaped)
            {
                ptr_ref["type"]  = "mmap";
                ptr_ref["bytes"] = m_mmap_size;
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

//============================================
Node
Node::info()const
{
    // NOTE: very ineff w/o move semantics
    Node res;
    info(res);
    return res;
}



//============================================
/// TODO: update option with set_external
void
Node::update(Node &n_src)
{
    // walk src and add it concents to this node
    // OBJECT_T is the only special case here.
    /// TODO:
    /// arrays and non empty leafs will simply overwrite the current
    /// node, these semantics seem sensbile, but we could revisit this
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
        set(n_src);
    }
}


//============================================
void
Node::compact()
{
    // TODO: we can do this more efficently
    Node n;
    compact_to(n);
    set(n);
}


//============================================
Node
Node::compact_to()const
{
    // NOTE: very ineff w/o move semantics
    Node res;
    compact_to(res);
    return res;
}


//============================================
void
Node::compact_to(Node &n_dest) const
{
    n_dest.reset();
    index_t c_size = total_bytes_compact();
    m_schema->compact_to(*n_dest.schema_pointer());
    n_dest.allocate(c_size);
    
    uint8 *n_dest_data = (uint8*)n_dest.m_data;
    compact_to(n_dest_data,0);
    n_dest.m_data = NULL; // TODO evil, brian doesn't like this.

    // need node structure
    walk_schema(&n_dest,n_dest.m_schema,n_dest_data);


}


//============================================
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


//============================================
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





//============================================
// bool             
// Node::compare(const Node &n, Node &cmp_results) const
// {
// /// TODO: cmp_results will describe the diffs between this & n    
// }
// 
// 
// //============================================
// bool             
// Node::operator==(const Node &n) const
// {
// /// TODO value comparison
//     return false;
// }


//============================================
Node&
Node::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    if(dtype().id() != DataType::OBJECT_T)
    {
        init(DataType::Objects::object());
    }
    
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // if this node doesn't exist, we need to 
    // link it to a schema
    
    // check for parent
    if(p_curr == "..")
    {
        if(m_parent != NULL) // TODO: check for erro (no parent)
           return m_parent->fetch(p_next);
    }
    
    index_t idx;
    if(!m_schema->has_path(p_curr))
    {
        Schema *schema_ptr = &m_schema->fetch(p_curr);
        Node *new_node = new Node(schema_ptr);
        new_node->m_parent = this;
        m_children.push_back(new_node);
        idx = m_children.size() - 1;
    } else {
        idx = m_schema->entry_index(p_curr);
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


//============================================
Node&
Node::fetch(index_t idx)
{
    // if(dtype().id() != DataType::LIST_T)
    // {
    // }
    // we could also potentially support index fetch on:
    //   OBJECT_T (imp-order)
    return *m_children[idx];
}


//============================================
Node *
Node::fetch_pointer(const std::string &path)
{
    return &fetch(path);
}

//============================================
Node *
Node::fetch_pointer(index_t idx)
{
    return &fetch(idx);
}

//============================================
Node&
Node::operator[](const std::string &path)
{
    return fetch(path);
}

//============================================
Node&
Node::operator[](index_t idx)
{
    return fetch(idx);
}


//============================================
bool           
Node::has_path(const std::string &path) const
{
    return m_schema->has_path(path);
}


//============================================
void
Node::paths(std::vector<std::string> &paths, bool walk) const
{
    m_schema->paths(paths,walk);
}

//============================================
index_t 
Node::number_of_entries() const 
{
    return m_schema->number_of_entries();
}

//============================================
void    
Node::remove(index_t idx)
{
 
    m_schema->remove(idx);
    // remove the proper list entry
    delete m_children[idx];
    m_children.erase(m_children.begin() + idx);
}

//============================================
void
Node::remove(const std::string &path)
{
    // schema will do a path check
    m_schema->remove(path);

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    index_t idx=m_schema->entry_index(p_curr);

    if(!p_next.empty())
    {
        m_children[idx]->remove(p_next);
    }
    
    delete m_children[idx];
    m_children.erase(m_children.begin() + idx);
}

//============================================
int64
Node::to_int64() const
{
    switch(dtype().id())
    {
        case DataType::BOOL8_T: return (int64)as_bool8();
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

//============================================
uint64
Node::to_uint64() const
{
    switch(dtype().id())
    {
        case DataType::BOOL8_T: return (uint64)as_bool8();
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

//============================================
float64
Node::to_float64() const
{
    switch(dtype().id())
    {
        case DataType::BOOL8_T: return (float64)as_bool8();
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


//============================================
index_t
Node::to_index_t() const
{
    switch(dtype().id())
    {
        case DataType::BOOL8_T: return (index_t)as_bool8();
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

//============================================
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

//============================================
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
            m_children[i]->to_json(oss,detailed,indent,depth+1,pad,eoe);
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
            m_children[i]->to_json(oss,detailed,indent,depth+1,pad,eoe);
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
            /* bool*/
            case DataType::BOOL8_T: as_bool8_array().to_json(value_oss); break;
            /* ints */
            case DataType::INT8_T:  as_int8_array().to_json(value_oss); break;
            case DataType::INT16_T: as_int16_array().to_json(value_oss); break;
            case DataType::INT32_T: as_int32_array().to_json(value_oss); break;
            case DataType::INT64_T: as_int64_array().to_json(value_oss); break;
            /* uints */
            case DataType::UINT8_T:  as_uint8_array().to_json(value_oss); break;
            case DataType::UINT16_T: as_uint16_array().to_json(value_oss); break;
            case DataType::UINT32_T: as_uint32_array().to_json(value_oss); break;
            case DataType::UINT64_T: as_uint64_array().to_json(value_oss); break;
            /* floats */
            case DataType::FLOAT32_T: as_float32_array().to_json(value_oss); break;
            case DataType::FLOAT64_T: as_float64_array().to_json(value_oss); break;
            /* bytestr */
            case DataType::BYTESTR_T: value_oss << "\"" << as_bytestr() << "\""; break;
        }

        if(!detailed)
            oss << value_oss.str();
        else
            dtype().to_json(oss,value_oss.str());
    }  
}
    
//============================================
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


//============================================
void
Node::allocate(const DataType &dtype)
{
    // TODO: This implies compact storage
    allocate(dtype.number_of_elements()*dtype.element_bytes());
}

//============================================
void
Node::allocate(index_t dsize)
{
    m_data    = malloc(dsize);
    m_alloced = true;
    m_alloced_size = dsize;
    m_mmaped  = false;
}


//============================================
void
Node::mmap(const std::string &stream_path, index_t dsize)
{
    m_mmap_fd   = open(stream_path.c_str(),O_RDWR| O_CREAT);
    m_mmap_size = dsize;

    if (m_mmap_fd == -1) 
        THROW_ERROR("<Node::mmap> failed to open: " << stream_path);

    m_data = ::mmap(0, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, m_mmap_fd, 0);

    if (m_data == MAP_FAILED) 
        THROW_ERROR("<Node::mmap> MAP_FAILED" << stream_path);
    
    m_alloced = false;
    m_mmaped  = true;
}


//============================================
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
        }
    }   
    else if(m_mmaped && m_data)
    {
        if(munmap(m_data, m_mmap_size) == -1) 
        {
            // TODO error
        }
        close(m_mmap_fd);
        m_data      = NULL;
        m_mmap_fd   = -1;
        m_mmap_size = 0;
    }
}
    

//============================================
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


//============================================
void
Node::init_list()
{
    init(DataType::Objects::list());
}
 
//============================================
void
Node::init_object()
{
    init(DataType::Objects::object());
}
    
//============================================
void
Node::list_append(const Node &node)
{
    init_list();
    index_t idx = m_children.size();
    m_schema->append(node.schema());
    Schema *schema_ptr = &m_schema->fetch(idx);
    Node *res_node = new Node(node,schema_ptr);
    res_node->m_parent=this;
    m_children.push_back(res_node);
}

//============================================
void 
Node::walk_schema(const Schema &schema)
{
    m_data    = NULL;
    m_alloced = false;
    m_schema->set(schema);
    // allocate data
    allocate(m_schema->total_bytes());
    // call walk w/ data
    walk_schema(this,m_schema,m_data);
}


//============================================
void 
Node::walk_schema(const Schema &schema, void *data)
{
    m_schema->set(schema);
    walk_schema(this,m_schema,data);
}

//============================================
void 
Node::walk_schema(Node   *node, 
                  Schema *schema,
                  void   *data)
{
    // we can have an object, list, or leaf
    
    if(schema->dtype().id() == DataType::OBJECT_T)
    {
        for(index_t i=0;i<schema->children().size();i++)
        {
    
            std::string curr_name = schema->object_order()[i];
            Schema *curr_schema   = schema->fetch_pointer(curr_name);
            Node *curr_node       = new Node(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_T)
    {
        index_t num_entries = schema->number_of_entries();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = schema->fetch_pointer(i);
            Node *curr_node = new Node(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data);
            node->append(curr_node);
        }
    }
    else
    {
        // link the current node to the schema
        node->set(schema,data);
    } 
}


}

