///
/// file: Node.cpp
///

#include "Node.h"
#include "rapidjson/document.h"

namespace conduit
{

///============================================
/// Node
///============================================

///============================================
Node::Node()
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::NODE_T)
{}

///============================================
Node::Node(const Node &node)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::NODE_T)
{
    set(node);
}

///============================================
Node::Node(void *data, const std::string &schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::NODE_T)
{
    walk_schema(data,schema);
}

///============================================
Node::Node(void *data, const Node *schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::NODE_T)
{
    set(data,schema);
}

///============================================
Node::Node(void *data, const DataType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
    set(data,dtype);
}

Node::Node(const std::vector<uint32>  &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
   set(data);
}

Node::Node(const std::vector<float64>  &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
   set(data);
}

///============================================
Node::Node(const DataType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
    set(dtype);
}
    
///============================================
Node::Node(uint32  data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
    set(data);
}

///============================================
Node::Node(float64 data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(DataType::EMPTY_T)
{
    set(data);
}


///============================================
Node::~Node()
{
  cleanup();
}

///============================================
void 
Node::set(const Node &node)
{
    if (node.m_dtype.id() != DataType::EMPTY_T)
    {
        if (node.m_alloced) 
        {
            // TODO: compaction?
            init(node.m_dtype);
            memcpy(m_data, node.m_data, m_dtype.total_bytes());
        }
        else 
        {
            m_alloced = false;
            m_data = node.m_data;
            m_dtype.reset(node.m_dtype);
        }
        // TODO: Replace
        m_entries = node.m_entries;
        m_list_data = node.m_list_data;
    }
}

///============================================
void 
Node::set(const DataType &dtype)
{
    // TODO: Is this right?
    // We need to cleanup and set the dtype w/o storage
    m_dtype.reset(dtype);
}

///============================================
void 
Node::set(uint32 data)
{
    // TODO check for compatible, don't always re-init
    // NOTE: comp check happens in init
    init(DataType::uint32_dtype);
    *(uint32*)((char*)m_data + m_dtype.element_index(0)) = data;
}


///============================================
void 
Node::set(float64 data)
{
    // TODO check for compatible, don't always re-init
    // NOTE: comp check happens in init
    init(DataType::float64_dtype);
    *(float64*)((char*)m_data + m_dtype.element_index(0)) = data;
}


///============================================
void 
Node::set(const std::vector<uint32>  &data)
{
    DataType vec_t(DataType::UINT32_T,
                   (index_t)data.size(),
                   0,
                   sizeof(uint32),
                   sizeof(uint32));
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(uint32)*data.size());
}

///============================================
void 
Node::set(const std::vector<float64>  &data)
{
    DataType vec_t(DataType::FLOAT64_T,
                   (index_t)data.size(),
                   0,
                   sizeof(float64),
                   sizeof(float64));
    init(vec_t);
    memcpy(m_data,&data[0],sizeof(float64)*data.size());
}



void
Node::set(void* data, const Node* schema)
{
}

void
Node::set( void *data, const DataType &dtype)
{
    m_alloced = false;
    m_data    = data;
    m_dtype.reset(dtype);
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
Node &
Node::operator=(uint32 data)
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
Node &
Node::operator=(const std::vector<uint32> &data)
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

index_t
Node::total_bytes() const
{
    index_t size = 0;
    switch (m_dtype.id()) 
    {

        case DataType::UINT32_T:
        case DataType::UINT64_T:
        case DataType::FLOAT64_T:
            size = m_dtype.total_bytes();
            break;
        case DataType::NODE_T:
        {
            const std::map<std::string, Node> &ents = entries();
            for (std::map<std::string, Node>::const_iterator itr = ents.begin();
                 itr != ents.end(); ++itr) 
            {
                size += itr->second.total_bytes();
            }
        }
            break;
        case DataType::LIST_T:
        {
            const std::vector<Node> &lst = list();
            for (std::vector<Node>::const_iterator itr = lst.begin();
                 itr != lst.end(); ++itr)
            {
                size += itr->total_bytes();
            }
        }
            break;
        default:
             // error
             break;
    }
    return size;
}

///============================================
std::string
Node::schema() const
{
    std::ostringstream oss;
    schema(oss);
    return oss.str();
}


///============================================
void
Node::schema(std::ostringstream &oss) const
{
    if(m_dtype.id() == DataType::NODE_T)
    {
        oss << "{";
        std::map<std::string,Node>::const_iterator itr;
        const std::map<std::string, Node> &ents = entries();
        bool first=true;
        for(itr = ents.begin(); itr != ents.end(); ++itr)
        {
            if(!first)
                oss << ",";
            oss << "\""<< itr->first << "\" : ";
            oss << itr->second.schema() << "\n";
            first=false;
        }
        oss << "}\n";
    }
    else if(m_dtype.id() == DataType::LIST_T)
    {
        oss << "[";
        std::vector<Node>::const_iterator itr;
        const std::vector<Node> &lst = list();
        bool first=true;
        for(itr = lst.begin(); itr != lst.end(); ++itr)
        {
            if(!first)
                oss << ",";
            oss << (*itr).schema() << "\n";
            first=false;
        }
        oss << "]\n";
    }
    else // assume data value type for now
    {
        m_dtype.schema(oss);
    }
}

///============================================
void
Node::serialize(std::vector<uint8> &data) const
{
    data = std::vector<uint8>(total_bytes(),0);
    serialize(&data[0],0);
}
///============================================
void
Node::serialize(uint8 *data,index_t curr_offset) const
{
    if(m_dtype.id() == DataType::NODE_T)
    {
        std::map<std::string,Node>::const_iterator itr;
        const std::map<std::string,Node> &ent = entries();
        for(itr = ent.begin(); itr != ent.end(); ++itr)
        {
            itr->second.serialize(&data[0],curr_offset);
            curr_offset+=itr->second.total_bytes();
        }
    }
    else if(m_dtype.id() == DataType::LIST_T)
    {
        std::vector<Node>::const_iterator itr;
        const std::vector<Node> &lst = list();
        for(itr = lst.begin(); itr != lst.end(); ++itr)
        {
            (*itr).serialize(&data[0],curr_offset);
            curr_offset+=(*itr).total_bytes();
        }
    }
    else // assume data value type for now
    {
        // TODO: Compact?
        memcpy(&data[curr_offset],m_data,total_bytes());
    }
}


///============================================
Node&
Node::fetch(const std::string &path)
{
    // TODO: Error checking ...
    // TODO: Nested paths
    return m_entries[path];
}

///============================================
bool           
Node::has_path(const std::string &path) const
{
    // TODO: Imp
}


///============================================
void
Node::paths(std::vector<std::string> &paths,bool expand) const
{
    // TODO: Imp
    // TODO: Nested paths
}


///============================================
index_t
Node::to_integer() const
{
    // TODO: Imp
}

///============================================
float64
Node::to_real() const
{
    // TODO: Imp
}
    
///============================================
void
Node::init(const DataType& dtype)
{
    if (!compatible_storage(dtype) || m_data == NULL) 
    {
        cleanup();
        switch (dtype.id())
        {
            case DataType::UINT32_T:
            case DataType::FLOAT64_T:
                // TODO: This implies compact storage
                m_data = new char[dtype.number_of_elements()*dtype.element_bytes()];
                break;
            case DataType::NODE_T:
                // TODO: alloced map
                break;
            case DataType::LIST_T:
                // TODO: alloced vec
                break;
        }
        m_alloced = true;
        m_dtype.reset(dtype);
    }
}

// TODO: Many more init cases

///============================================
void
Node::cleanup()
{
    if(m_alloced && m_data)
    {
        if(m_dtype.id() == DataType::NODE_T)
        {
            //TODO: Imp    delete alloced map
        }
        else if(m_dtype.id() == DataType::LIST_T)
        {
            //TODO: Imp    delete alloced vec
        }
        else if(m_dtype.id() == DataType::UINT32_T)
        {
            uint32 *ptr=(uint32*)m_data;
            delete ptr; 
        }
        else if(m_dtype.id() == DataType::FLOAT64_T)
        {
            float64 *ptr=(float64*)m_data;
            delete ptr; 
        }
        // TODO: etc
    
    }   
    m_data    = NULL;
    m_alloced = false;
}
    
///============================================
index_t          
Node::element_index(index_t   idx) const
{
    return m_dtype.offset() + m_dtype.stride()*idx;
}

///============================================
std::map<std::string, Node> &  
Node::entries()
{
   return m_entries;
}

///============================================
std::vector<Node> &  
Node::list()
{
   return m_list_data;
}


///============================================
const std::map<std::string, Node> &  
Node::entries() const
{
   return m_entries;
}

///============================================
const std::vector<Node> &  
Node::list() const
{
   return m_list_data;
}



///============================================
void 
Node::walk_schema(void *data, const std::string &schema)
{
    // clean up before this
    m_data    = data;
    m_alloced = false;
    m_dtype.reset(DataType::NODE_T);
    
    rapidjson::Document document;
    document.Parse<0>(schema.c_str());
    index_t current_offset = 0;
    walk_schema(data, document,current_offset);
}

void 
Node::walk_schema(void *data, const rapidjson::Value &jvalue, index_t curr_offset)
{
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
            std::string dtype(jvalue["dtype"].GetString());
            int length = jvalue["length"].GetInt();
            index_t type_id = DataType::type_name_to_id(dtype);
            index_t size    = DataType::size_of_type_id(type_id);
            m_dtype.reset(type_id, length, curr_offset,
                          size, size);
            m_data = data;
        }
        else
        {
            std::map<std::string, Node> &ents = entries();
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Node node;
                node.walk_schema(data, itr->value, curr_offset);
                ents[entry_name] = node;
                curr_offset += node.total_bytes();
            }
        }
    }
    else if (jvalue.IsArray()) 
    {
        m_dtype.reset(DataType::LIST_T);
        std::vector<Node> &lst = list();
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            Node node;
            node.walk_schema(data, jvalue[i], curr_offset);
            curr_offset += node.total_bytes();
            lst.push_back(node);
        }
    }
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         index_t type = DataType::type_name_to_id(dtype_name);
         index_t size = DataType::size_of_type_id(type);
         m_dtype.reset(type,1,curr_offset,size,size);
         m_data = data;
    }
 
    ///
    /// each entry will either be:
    ///  a string that describes a dtype 
    ///  "uint32", "float64"
    ///  an object that describes a dtype
    ///   in this case, the object will have a "dtype" key, at a min
    ///    {"dtype":"<dtype_str>","length":<int>}
    /// or, more complex cases:
    ///  an object that describes a nested object
    ///  a list of objects
}

bool
Node::compatible_storage(const DataType& type)
{
    return (type.id() == m_dtype.id() &&
            type.element_bytes() == m_dtype.element_bytes() &&
            type.total_bytes() == m_dtype.total_bytes());
}


}

