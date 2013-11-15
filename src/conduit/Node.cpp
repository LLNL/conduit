///
/// file: Node.cpp
///

#include "Node.h"

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
    // walk_schema(&this);
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
Node::Node(void *data, const BaseType &dtype)
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
Node::Node(const BaseType &dtype)
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
    /// TODO
    // init calls cleanup();
    //init(BaseType::NODE_T);
    //update(node);
}

///============================================
void 
Node::set(BaseType dtype)
{
    // init calls cleanup
    //init(dtype); // always
}

///============================================
void 
Node::set(uint32 data)
{
    // TODO check for compatible, don't always re-init
    init(ValueType::uint32_dtype);
    *((uint32*)m_data) = data;
}


///============================================
void 
Node::set(float64 data)
{
    // TODO check for compatible, don't always re-init
    init(ValueType::float64_dtype);
    *((float64*)m_data) = data;
}


///============================================
void 
Node::set(const std::vector<uint32>  &data)
{
    ValueType vec_t(BaseType::UINT32_T,
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
    ValueType vec_t(BaseType::FLOAT64_T,
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
Node::set( void *data, const BaseType &dtype)
{
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
Node::operator=(BaseType dtype)
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
Node::init(const BaseType &dtype)
{
}

void
Node::init(const ValueType& dtype)
{
   if (m_alloced) {
      char* data = static_cast<char*>(data);
      delete data;
   }
   m_alloced = true;
   m_data = new char[dtype.number_of_elements()*dtype.element_bytes()];
   m_dtype = new ValueType(dtype);
}

// TODO: Many more init cases

///============================================
void
Node::cleanup()
{
    if(m_alloced)
    {
        if(m_dtype->id() == BaseType::NODE_T)
        {
            //TODO: Imp    delete entries_ptr();
        }
        else if(m_dtype->id() == BaseType::UINT32_T)
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
    // TODO: Imp
    return 0;
}

///============================================
std::map<std::string, Node> &  
Node::entries()
{
   return m_entries;
}


}
