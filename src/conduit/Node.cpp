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
 m_dtype()
{}

///============================================
Node::Node(const Node &node)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(node);
}

///============================================
Node::Node(void *data, const std::string &schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    // walk_schema(&this);
    // set(data,this);
}

///============================================
Node::Node(void *data, const Node *schema)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data,schema);
}

///============================================
Node::Node(void *data, const BaseType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data,dtype);
}

///============================================
Node::Node(const BaseType &dtype)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(dtype);
}
    
///============================================
Node::Node(uint32  data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data);
}

///============================================
Node::Node(float64 data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data);
}


///============================================
Node::Node(uint32_array  *data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data);
}

///============================================
Node::Node(float64_array *data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data);
}

///============================================
Node::Node(const uint32_array  &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data); // copy
}

///============================================
Node::Node(const float64_array &data)
:m_data(NULL),
 m_alloced(false),
 m_dtype()
{
    set(data); // copy
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
    //init(uint32_dtype);
    // *((uint32*)m_data) = data;
}


///============================================
void 
Node::set(float64 data)
{
    // TODO check for compatible, don't always re-init
    //init(float64_dtype);
    // *((float64*)m_data) = data;
}


///============================================
void 
Node::set(const uint32_array  *data)
{
    // TODO check for compatible, don't always re-init
    //init(data->ptr(),data->dtype());
}

///============================================
void 
Node::set(const float64_array  *data)
{
    // TODO check for compatible, don't always re-init
    //init(data->ptr(),data->dtype());
}


///============================================
void 
Node::set(const uint32_array  &data)
{
    // TODO check for compatible, don't always re-init
    //init(data); // copy?
}

///============================================
void 
Node::set(const float64_array  &data)
{
    // TODO check for compatible, don't always re-init
    //init(data); // copy?
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
Node::operator=(uint32_array *data)
{
    set(data);
    return *this;
}

///============================================
Node &
Node::operator=(float64_array *data)
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
Node::operator=(const float64_array &data)
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
    return *m_entries[path];
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
{}
// TODO: Many more init cases

///============================================
void
Node::cleanup()
{
    if(m_alloced)
    {
        if(m_dtype.id() == BaseType::NODE_T)
        {
            //TODO: Imp    delete entries_ptr();
        }
        else if(m_dtype.id() == BaseType::UINT32_T)
        {
            uint32 *ptr=(uint32*)m_data;
            delete ptr; 
            //TODO: delete vs delete[] ? depends on alloc
        }
        // TODO: etc
    
    }   
    m_data = NULL;
    m_alloced = false;
    m_dtype = BaseType();
}
    
///============================================
index_t          
Node::element_index(index_t   idx) const
{
    // TODO: Imp
}

///============================================
std::map<std::string, Node*> &  
Node::entries()
{
   return m_entries;
}


}
