///
/// file: Node.cpp
///

#include "Node.h"
#include "Utils.h"
#include "rapidjson/document.h"
#include <iostream>
#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace conduit
{

///============================================
/// walk_schema helper
///============================================

/* use these funcs to avoid having to include rapidjson headers  in Node.h
 (rapidjson::Values resolve to a complex templated type that we can't forward declare) 
*/
void walk_schema(Node &node,
                 void *data, 
                 const rapidjson::Value &jvalue, 
                 index_t curr_offset);

void walk_schema(Node &node, 
                 const rapidjson::Value &jvalue);


///============================================
/// Node
///============================================
void
Node::init_defaults()
{
    m_data = NULL;
    m_alloced = false;

    m_mmaped    = false;
    m_mmap_fd   = -1;
    m_mmap_size = 0;

    m_schema = new Schema(DataType::EMPTY_T);
    
}

///============================================
Node::Node()
{
    init_defaults();
}

///============================================
Node::Node(const Node &node)
{
    init_defaults();
    set(node);
}

///============================================
Node::Node(Schema &schema)

{
    init_defaults();
    walk_schema(schema.to_json());
}
	
///============================================
Node::Node(Schema *schema_ptr)

{
	init_defaults();
	m_schema = schema_ptr;
}
	
///============================================
Node::Node(const Node &node, Schema *schema_ptr)

{
	init_defaults();
	set(node,schema_ptr);
}

	
///============================================
Node::Node(Schema &schema, const std::string &stream_path, bool mmap)
{    
    init_defaults();
    if(mmap)
        conduit::Node::mmap(schema,stream_path);
    else
        load(schema,stream_path);
}


///============================================
Node::Node(Schema &schema, std::ifstream &ifs)
{
    init_defaults();
    walk_schema(schema.to_json(),ifs);
}


///============================================
Node::Node(Schema &schema, void *data)
{
    init_defaults();
	std::string json_schema =schema.to_json(); 
	std::cout << "json_schema_rc:" << json_schema << std::endl;
	walk_schema(schema,data);
}


///============================================
Node::Node(const DataType &dtype, void *data)
{    
    init_defaults();
    set(dtype,data);
}

///============================================
/* int vec types */
///============================================

///============================================
Node::Node(const std::vector<int8>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<int16>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<int32>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<int64>  &data)
{
   init_defaults();
   set(data);
}

///============================================
/* uint vec types */
///============================================

///============================================
Node::Node(const std::vector<uint8>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<uint16>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<uint32>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<uint64>  &data)
{
   init_defaults();
   set(data);
}

///============================================
/* float vec types */
///============================================

///============================================
Node::Node(const std::vector<float32>  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const std::vector<float64>  &data)
{
   init_defaults();
   set(data);
}

///============================================
/* int array types */
///============================================

///============================================
Node::Node(const int8_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const int16_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const int32_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const int64_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
/* uint array types */
///============================================

///============================================
Node::Node(const uint8_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const uint16_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const uint32_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const uint64_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
/* float arr types */
///============================================

///============================================
Node::Node(const float32_array  &data)
{
   init_defaults();
   set(data);
}

///============================================
Node::Node(const float64_array  &data)
{
   init_defaults();
   set(data);
}


///============================================
Node::Node(const DataType &dtype)
{
    init_defaults();
    set(dtype);
}

///============================================
/// int types
///============================================

///============================================
Node::Node(int8  data)
{
    init_defaults();
    set(data);
}

///============================================
Node::Node(int16  data)
{
    init_defaults();
    set(data);
}
    
///============================================
Node::Node(int32  data)
{
    init_defaults();
    set(data);
}

///============================================
Node::Node(int64  data)
{    
    init_defaults();
    set(data);
}


///============================================
/// uint types
///============================================

///============================================
Node::Node(uint8  data)
{
    init_defaults();
    set(data);
}

///============================================
Node::Node(uint16  data)
{
    init_defaults();
    set(data);
}
    
///============================================
Node::Node(uint32  data)
{
    init_defaults();
    set(data);
}

///============================================
Node::Node(uint64  data)
{
    init_defaults();
    set(data);
}

///============================================
/// float types
///============================================

///============================================
Node::Node(float32 data)
{
    init_defaults();
    set(data);
}


///============================================
Node::Node(float64 data)
{
    init_defaults();
    set(data);
}


///============================================
Node::~Node()
{
	cleanup();
}

///============================================
void
Node::reset()
{
	release();
	m_schema->set(DataType::EMPTY_T);
}

///============================================
void 
Node::load(Schema &schema, const std::string &stream_path)
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
    
    walk_schema(schema.to_json(),m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_alloced = true;
}

///============================================
void 
Node::mmap(Schema &schema, const std::string &stream_path)
{
    reset();
    index_t dsize = schema.total_bytes();
    Node::mmap(stream_path,dsize);

    ///
    /// See Below
    ///
    m_mmaped = false;
    
    walk_schema(schema.to_json(),m_data);

    ///
    /// TODO: Design Issue
    ///
    /// The bookkeeping here is not very intuitive 
    /// The walk process may reset the node, which would free
    /// our data before we can set it up. So for now, we wait  
    /// to indicate ownership until after the node is fully setup
    m_mmaped = true;
}



///============================================
void 
Node::set(const Node &node)
{
    set(node, NULL);
}

void
Node::set(const Node& node, Schema* schema)
{
    if (!node.dtype().id() == DataType::EMPTY_T)
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
                child->set(*node.m_children[i],m_schema->children()[i]);
                m_children.push_back(child);
            }
        }
        else
        { 
            if (node.m_alloced) 
            {
                // TODO: compaction?
                init(node.dtype());
                memcpy(m_data, node.m_data, m_schema->total_bytes());
            }
            else 
            {
                m_alloced = false;
                m_data    = node.m_data;
                m_schema->set(node.schema());
            }
        }
    }
    else
    {
        // if passed node is empty -- reset this.
        reset();
    }

}

///============================================
void 
Node::set(const DataType &dtype)
{
    init(dtype);
}
	
	
///============================================
void 
Node::set(bool8 data)
{
    init(DataType::Scalars::bool8());
    *(bool8*)((char*)m_data + schema().element_index(0)) = data;
}
	

///============================================
/// int types
///============================================

///============================================
void 
Node::set(int8 data)
{
    init(DataType::Scalars::int8());
    *(int8*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(int16 data)
{
    init(DataType::Scalars::int16());
    *(int16*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(int32 data)
{
    init(DataType::Scalars::int32());
    *(int32*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(int64 data)
{
    init(DataType::Scalars::int64());
    *(int64*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
/// uint types
///============================================

///============================================
void 
Node::set(uint8 data)
{
    init(DataType::Scalars::uint8());
    *(uint8*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(uint16 data)
{
    init(DataType::Scalars::uint16());
    *(uint16*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(uint32 data)
{
    init(DataType::Scalars::uint32());
    *(uint32*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(uint64 data)
{
    init(DataType::Scalars::uint64());
    *(uint64*)((char*)m_data + schema().element_index(0)) = data;
}

///============================================
/// float types
///============================================

///============================================
void 
Node::set(float32 data)
{
    init(DataType::Scalars::float32());
    *(float32*)((char*)m_data + schema().element_index(0)) = data;
}


///============================================
void 
Node::set(float64 data)
{
    init(DataType::Scalars::float64());
    *(float64*)((char*)m_data + schema().element_index(0)) = data;
}

///============================================
/// int vec types
///============================================

///============================================
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

///============================================
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

///============================================
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

///============================================
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


///============================================
/// uint vec types
///============================================

///============================================
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

///============================================
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

///============================================
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

///============================================
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

///============================================
/// float vec types
///============================================

///============================================
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

///============================================
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

///============================================
/// int array types
///============================================

///============================================
void 
Node::set(const int8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data   = data.data_ptr();
}

///============================================
void 
Node::set(const int16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const int32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const int64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}


///============================================
/// uint array types
///============================================

///============================================

void 
Node::set(const uint8_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const uint16_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const uint32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const uint64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}
///============================================
/// float array types
///============================================

///============================================
void 
Node::set(const float32_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}

///============================================
void 
Node::set(const float64_array  &data)
{
    release();
    m_schema->set(data.dtype());
    m_data  = data.data_ptr();
}


///============================================
void
Node::set(Schema &schema,void* data)
{
    walk_schema(schema.to_json(),data);    
}

///============================================
void
Node::set(Schema *schema_ptr)
{
    if(m_schema->is_root())
        delete m_schema;
    m_schema = schema_ptr;    
}
	
///============================================
void
Node::set(Schema *schema_ptr,void *data)
{
    set(schema_ptr);
	release();
	m_data    = data;    
}
	
///============================================
void
Node::set(const DataType &dtype, void *data)
{
    release();
    m_alloced = false;
    m_data    = data;
    m_schema->set(dtype);
}

///============================================
Node &
Node::operator=(const Node &node)
{
    if(this != &node)
    {
        set(node);
    }
    return *this;
}

///============================================
Node &
Node::operator=(DataType dtype)
{
    set(dtype);
    return *this;
}

///============================================
/// uint types
///============================================

///============================================
Node &
Node::operator=(uint8 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(uint16 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(uint32 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(uint64 data)
{
    set(data);
    return *this;
}

///============================================
/// int types
///============================================

///============================================
Node &
Node::operator=(int8 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(int16 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(int32 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(int64 data)
{
    set(data);
    return *this;
}

///============================================
/// float types
///============================================

///============================================
Node &
Node::operator=(float32 data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(float64 data)
{
    set(data);
    return *this;
}

///============================================
/// int vec types
///============================================

///============================================
Node &
Node::operator=(const std::vector<int8> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<int16> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<int32> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<int64> &data)
{
    set(data);
    return *this;
}

///============================================
/// uint vec types
///============================================

///============================================
Node &
Node::operator=(const std::vector<uint8> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<uint16> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<uint32> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<uint64> &data)
{
    set(data);
    return *this;
}

///============================================
/// float vec types
///============================================

///============================================
Node &
Node::operator=(const std::vector<float32> &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const std::vector<float64> &data)
{
    set(data);
    return *this;
}

///============================================
/// int array types
///============================================

///============================================
Node &
Node::operator=(const int8_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const int16_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const int32_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const int64_array &data)
{
    set(data);
    return *this;
}

///============================================
/// uint vec types
///============================================

///============================================
Node &
Node::operator=(const uint8_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const uint16_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const uint32_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const uint64_array &data)
{
    set(data);
    return *this;
}

///============================================
/// float vec types
///============================================

///============================================
Node &
Node::operator=(const float32_array &data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(const float64_array &data)
{
    set(data);
    return *this;
}


///============================================
void
Node::serialize(std::vector<uint8> &data,bool compact) const
{
    data = std::vector<uint8>(total_bytes(),0);
    serialize(&data[0],0,compact);
}

///============================================
void
Node::serialize(const std::string &stream_path,
                bool compact) const
{
	std::ofstream ofs;
    ofs.open(stream_path.c_str());
    serialize(ofs,compact);
    ofs.close();
}


///============================================
void
Node::serialize(std::ofstream &ofs,
                bool compact) const
{
    if(dtype().id() == DataType::OBJECT_T ||
       dtype().id() == DataType::LIST_T)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin(); itr != m_children.end(); ++itr)
        {
            (*itr)->serialize(ofs);
        }
    }
    else // assume data value type for now
    {
        // TODO: Compact?
        ofs.write((const char*)element_pointer(0),total_bytes());
    }
}


///============================================
void
Node::serialize(uint8 *data,index_t curr_offset,bool compact) const
{
    if(dtype().id() == DataType::OBJECT_T ||
       dtype().id() == DataType::LIST_T)
    {
        std::vector<Node*>::const_iterator itr;
        for(itr = m_children.begin(); itr != m_children.end(); ++itr)
        {
            (*itr)->serialize(&data[0],curr_offset);
            curr_offset+=(*itr)->total_bytes();
        }
    }
    else // assume data value type for now
    {
        // TODO: Compact?
        memcpy(&data[curr_offset],m_data,total_bytes());
    }
}

///============================================
// bool             
// Node::compare(const Node &n, Node &cmp_results) const
// {
// /// TODO: cmp_results will describe the diffs between this & n    
// }
// 
// 
// ///============================================
// bool             
// Node::operator==(const Node &n) const
// {
// /// TODO value comparison
//     return false;
// }


///============================================
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
    index_t idx;
    if(!m_schema->has_path(p_curr))
    {
        Schema *schema_ptr = &m_schema->fetch(p_curr);
        Node *new_node = new Node(schema_ptr);
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


///============================================
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

///============================================
Node&
Node::operator[](const std::string &path)
{
    return fetch(path);
}

///============================================
Node&
Node::operator[](index_t idx)
{
    return fetch(idx);
}


///============================================
bool           
Node::has_path(const std::string &path) const
{
    return m_schema->has_path(path);
}


///============================================
void
Node::paths(std::vector<std::string> &paths, bool walk) const
{
    m_schema->paths(paths,walk);
}

///============================================
index_t 
Node::number_of_entries() const 
{
    return m_schema->number_of_entries();
}

///============================================
void    
Node::remove(index_t idx)
{
 
    m_schema->remove(idx);
    // remove the proper list entry
    delete m_children[idx];
    m_children.erase(m_children.begin() + idx);
}

///============================================
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

///============================================
int64
Node::to_int() const
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

///============================================
uint64
Node::to_uint() const
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

///============================================
float64
Node::to_float() const
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

///============================================
std::string 
Node::to_json(bool simple, index_t indent) const
{
   std::ostringstream oss;
   to_json(oss,simple,indent);
   return oss.str();
}

///============================================
void
Node::to_json(std::ostringstream &oss,
              bool simple, index_t indent) const
{
    if(dtype().id() == DataType::OBJECT_T)
    {
        oss << "{";
		bool first=true;
        
		index_t nchildren = m_children.size();
		for(index_t i=0; i < nchildren;i++)
		{
		    if(!first)
                oss << ", ";
        	oss << "\""<< m_schema->object_order()[i] << "\": ";
            m_children[i]->to_json(oss,simple,indent);
			
		    first=false;
        }
        oss << "}\n";
    }
    else if(dtype().id() == DataType::LIST_T)
    {
        oss << "[";
        
		index_t nchildren = m_children.size();
		bool first=true;
		for(index_t i=0; i < nchildren;i++)
		{
			if(!first)
                oss << ", ";
			m_children[i]->to_json(oss,simple,indent);
			oss << "]\n";
			first=false;
		}
		
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
            case DataType::BYTESTR_T: oss << as_bytestr(); break;
        }

        if(simple)
            oss << value_oss.str();
        else
            dtype().to_json(oss,value_oss.str());
    }  
}

    
///============================================
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


///============================================
void
Node::allocate(const DataType &dtype)
{
    // TODO: This implies compact storage
    allocate(dtype.number_of_elements()*dtype.element_bytes());
}

///============================================
void
Node::allocate(index_t dsize)
{
    m_data    = malloc(dsize);
    m_alloced = true;
    m_mmaped  = false;
}


///============================================
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


///============================================
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
            // error
        }
        close(m_mmap_fd);
        m_data      = NULL;
        m_mmap_fd   = -1;
        m_mmap_size = 0;
    }
}
	

///============================================
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


///============================================
void
Node::init_list()
{
    init(DataType::Objects::list());
}
 
///============================================
void
Node::init_object()
{
    init(DataType::Objects::object());
}
    
///============================================
void
Node::list_append(const Node &node)
{
    init_list();
    index_t idx = m_children.size();
	m_schema->append(node.schema());
    Schema *schema_ptr = &m_schema->fetch(idx);
    m_children.push_back(new Node(node,schema_ptr));
}

///============================================
void 
Node::walk_schema(const Schema &schema)
{
    m_data    = NULL;
    m_alloced = false;
    m_schema->set(DataType::OBJECT_T);
    
    rapidjson::Document document;
    document.Parse<0>(schema.to_json().c_str());
    
    conduit::walk_schema(*this,document);
}


///============================================
void 
walk_schema(Node &node, 
            const rapidjson::Value &jvalue)
{
    if (jvalue.HasMember("dtype"))
    {
        // if dtype is an object, we have a "list_of" case or tree node
        const rapidjson::Value &dt_value = jvalue["dtype"];
        if(dt_value.IsObject() && jvalue.HasMember("source"))
        {
            std::string path(jvalue["source"].GetString());
            // read source
            
            //node = Node();
            //walk_schema(node,data,jvalue,0);
        }
        else // we will alloc a data buffer that can hold all of the node data
        {
            // walk_schema(node,data,jvalue,0);
        }
    }
}


///============================================
void 
Node::walk_schema(const Schema &schema, void *data)
{
    m_data = data;
    m_alloced = false;
    m_schema->set(schema);

    return walk_schema(*this,m_schema,data);
}

///============================================
void 
Node::walk_schema(Node &node, 
                   Schema *schema,
                   void *data)
{
    // we can have an object, list, or leaf
    
    if(schema->dtype().id() == DataType::OBJECT_T)
    {
		for(index_t i=0;i<schema->children().size();i++)
		{
	
			std::string curr_name = schema->object_order()[i];
            Schema *curr_schema = &schema->fetch(curr_name);
			Node *curr_node = new Node(curr_schema);
            walk_schema(*curr_node,curr_schema,data);
			node.m_children.push_back(curr_node);
        }                   
    }
    else if(schema->dtype().id() == DataType::LIST_T)
    {
        index_t num_entries = schema->number_of_entries();
        for(index_t i=0;i<num_entries;i++)
        {
            Schema *curr_schema = &schema->fetch(i);
			Node *curr_node = new Node(curr_schema);
            walk_schema(*curr_node,curr_schema,data);
			node.m_children.push_back(curr_node);
        }
    }
    else
    {
        // link the current node to the schema
        node.set(schema,data);
    } 
}

///============================================
void 
walk_schema(Node &node, 
            void *data,
            const rapidjson::Value &jvalue,
            index_t curr_offset)
{
    ///
    /// NOTE: We will need some portion of this to parse inline data.
    ///
    
    if(jvalue.IsObject())
    {
        /*
        static const char* kTypeNames[] = { "Null", 
                                            "False", 
                                            "True", 
                                            "Object", 
                                            "Array", 
                                            "String", 
                                            "Number" };
        */
        if (jvalue.HasMember("dtype"))
        {
            // if dtype is an object, we have a "list_of" case
            const rapidjson::Value &dt_value = jvalue["dtype"];
            if(dt_value.IsObject())
            {
                int length =1;
                if(jvalue.HasMember("length"))
                {
                    length = jvalue["length"].GetInt();
                }
                            
                // we will create `length' # of objects of obj des by dt_value
                 
                // TODO: we only need to parse this once, not leng # of times
                // but this is the easiest way to start. 
                for(int i=0;i< length;i++)
                {
                    Node curr_node(DataType::Objects::object());
                    walk_schema(curr_node,data, dt_value, curr_offset);
                    node.append(curr_node);
                    curr_offset += curr_node.total_bytes();
                }
            }
            else
            {
                // handle leaf node with explicit props
                std::string dtype_name(jvalue["dtype"].GetString());
                int length = jvalue["length"].GetInt();
                
                // TODO: Parse optional values:
                //  offset
                //  stride
                //  element_bytes
                //  endianness
                
                //  value
                
                const DataType df_dtype = DataType::default_dtype(dtype_name);
                index_t type_id = df_dtype.id();
                index_t size    = df_dtype.element_bytes();
                DataType dtype(type_id,
                               length,
                               curr_offset,
                               size, 
                               size,
                               Endianness::DEFAULT_T);
                node.set(dtype,data);
            }
        }
        else
        {
            // loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Node curr_node(DataType::Objects::object());
                walk_schema(curr_node,data, itr->value, curr_offset);
                node[entry_name] = curr_node;
                curr_offset += curr_node.total_bytes();
            }
        }
    }
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
			Node curr_node(DataType::Objects::object());
            walk_schema(curr_node,data, jvalue[i], curr_offset);
            curr_offset += curr_node.total_bytes();
            // this will coerce to a list
            node.append(curr_node);
        }
    }
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         DataType df_dtype = DataType::default_dtype(dtype_name);
         index_t size = df_dtype.element_bytes();
         DataType dtype(df_dtype.id(),1,curr_offset,size,size,Endianness::DEFAULT_T);
         node.set(dtype,data);
    }

}



}

