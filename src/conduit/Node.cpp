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
 m_dtype(0)
{}

///============================================
Node::Node(const Node &node)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    set(node);
}

///============================================
Node::Node(void *data, const std::string &schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    
    walk_schema(data,schema);
    // set(data,this);
}

///============================================
Node::Node(void *data, const Node *schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    set(data,schema);
}

///============================================
Node::Node(void *data, const DataType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    set(data,dtype);
}

Node::Node(const std::vector<uint32>  &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
   set(data);
}

Node::Node(const std::vector<float64>  &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
   set(data);
}

///============================================
Node::Node(const DataType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    set(dtype);
}
    
///============================================
Node::Node(uint32  data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
{
    set(data);
}

///============================================
Node::Node(float64 data)
:m_data(NULL),
 m_alloced(false),
 m_dtype(0)
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
    if (node.m_dtype) {
        if (node.m_alloced) {
            init(*node.m_dtype);
            memcpy(m_data, node.m_data, m_dtype->total_bytes());
        } else {
            m_alloced = false;
            m_data = node.m_data;
            m_dtype = new DataType(*node.m_dtype);
        }
        m_entries = node.m_entries;
        m_list_data = node.m_list_data;
    }
    /// TODO
    // init calls cleanup();
    //init(DataType::NODE_T);
    //update(node);
}

///============================================
void 
Node::set(const DataType &dtype)
{
    m_dtype = new DataType(dtype);

    // init calls cleanup
    //init(dtype); // always
}

///============================================
void 
Node::set(uint32 data)
{
    // TODO check for compatible, don't always re-init
    init(DataType::uint32_dtype);
    *((uint32*)m_data) = data;
}


///============================================
void 
Node::set(float64 data)
{
    // TODO check for compatible, don't always re-init
    init(DataType::float64_dtype);
    *((float64*)m_data) = data;
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
    m_data = data;
    m_dtype = new DataType(dtype);
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
    if (m_dtype == 0) {
        return size;
    }

    switch (m_dtype->id()) {

        case DataType::UINT32_T:
        case DataType::UINT64_T:
        case DataType::FLOAT64_T:
            size = m_dtype->total_bytes();
            break;
        case DataType::NODE_T:
            for (std::map<std::string, Node>::const_iterator itr = m_entries.begin();
                 itr != m_entries.end(); ++itr) {
                size += itr->second.total_bytes();
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
     // TODO: Imp
     return "{}\n";
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

//    List             as_list();
    
///============================================
void
Node::init(const DataType& dtype)
{
   if (m_alloced) {
      char* data = static_cast<char*>(data);
      delete[] data;
   }
   m_alloced = true;
   m_data = new char[dtype.number_of_elements()*dtype.element_bytes()];
   m_dtype = new DataType(dtype);
}

// TODO: Many more init cases

///============================================
void
Node::cleanup()
{
    if(m_alloced)
    {
        if(m_dtype->id() == DataType::NODE_T)
        {
            //TODO: Imp    delete entries_ptr();
        }
        else if(m_dtype->id() == DataType::UINT32_T)
        {
            uint32 *ptr=(uint32*)m_data;
            delete ptr; 
            //TODO: delete vs delete[] ? depends on alloc
        }
        // TODO: etc
    
    }   
    m_data = NULL;
    m_alloced = false;
    m_dtype = 0;
}
    
///============================================
index_t          
Node::element_index(index_t   idx) const
{
    return m_dtype->offset() + m_dtype->stride()*idx;
}

///============================================
std::map<std::string, Node> &  
Node::entries()
{
   return m_entries;
}



///============================================
void 
Node::walk_schema(void *data, const std::string &schema)
{
    // clean up before this
    m_data    = data;
    m_alloced = false;
    m_dtype   = new DataType(DataType::NODE_T);
    
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
        static const char* kTypeNames[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };
        for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); itr != jvalue.MemberEnd(); ++itr)
        {
            printf("Type of member %s is %s\n", itr->name.GetString(), kTypeNames[itr->value.GetType()]);
            std::string entry_name(itr->name.GetString());
            if(itr->value.IsString())
            {
                std::string dtype_name(itr->value.GetString());
                printf("%s: %s\n", entry_name.c_str(),dtype_name.c_str());
                // uses this paradigm
                index_t type = DataType::type_name_to_id(dtype_name);
                index_t size = DataType::size_of_type_id(type);
                DataType dtype(type,1,curr_offset,size,size);
                m_entries[entry_name] = Node(data,dtype);
                // calc offset (currenlty wrong b/c we have to pass all params to Type
                // dont want to look up element_size in here, type needs default settings
                // to handle this case)
                curr_offset += dtype.total_bytes();
            } else if (itr->value.IsObject()) {
                DataType dtype(DataType::NODE_T);
                Node node(dtype);
                node.walk_schema(data, itr->value, curr_offset);

                curr_offset += node.total_bytes();
                m_entries[entry_name] = node;
            }
        }
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



}

