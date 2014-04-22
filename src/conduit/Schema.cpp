///
/// file: Schema.cpp
///

#include <stdio.h>
#include "Schema.h"
#include "Error.h"
#include "Utils.h"
#include "rapidjson/document.h"

namespace conduit
{


void
walk_schema(Schema &schema,
            const rapidjson::Value &jvalue,
            index_t curr_offset);

///============================================
/// Schema
///============================================

///============================================
Schema::Schema()
{
    init_defaults();
}

///============================================
Schema::Schema(const Schema &schema)
{
    init_defaults();
    set(schema);
}

///============================================
Schema::Schema(index_t dtype_id)
{
    init_defaults();
    set(dtype_id);
}

///============================================
Schema::Schema(const DataType &dtype)
{
    init_defaults();
    set(dtype);
}


///============================================
Schema::Schema(const std::string &json_schema)
{
    init_defaults();
    set(json_schema);
}


///============================================
Schema::~Schema()
{release();}

///============================================
void
Schema::release()
{
    if(dtype().id() == DataType::OBJECT_T ||
       dtype().id() == DataType::LIST_T)
    {
        std::vector<Schema*> chld = children();
        for(index_t i=0;i< chld.size();i++)
        {
            delete chld[i];
        }
    }
    
    if(dtype().id() == DataType::OBJECT_T)
    { 
        delete object_hierarchy();
    }
    else if(dtype().id() == DataType::LIST_T)
    { 
        delete list_hierarchy();
    }
}

///============================================
void
Schema::reset()
{
    release();
    init_defaults();
}


///============================================
void
Schema::init_defaults()
{
    m_dtype  = DataType::Objects::empty();
    m_hierarchy_data = NULL;
    m_root   = false;
    m_static = false;
}

///============================================
void
Schema::init_object()
{
    if(dtype().id() != DataType::OBJECT_T)
    {
        reset();
        m_dtype  = DataType::Objects::object();
        m_hierarchy_data = new Schema_Object_Hierarchy();
    }
}

///============================================
void
Schema::init_list()
{
    if(dtype().id() != DataType::LIST_T)
    {
        reset();
        m_dtype  = DataType::Objects::list();
        m_hierarchy_data = new Schema_List_Hierarchy();
    }
}

///============================================
void 
Schema::set(const Schema &schema)
{
    bool init_children = false;
	index_t dt_id = schema.m_dtype.id();
    if (dt_id == DataType::OBJECT_T)
    {
       init_object();
       init_children = true;

       object_map() = schema.object_map();
       object_order() = schema.object_order();
    } 
    else if (dt_id == DataType::LIST_T)
    {
       init_list();
       init_children = true;
    }
	else 
	{
		m_dtype = schema.m_dtype;
	}

	
    if (init_children) 
    {
       std::vector<Schema*> &my_ents = children();
       const std::vector<Schema*> &their_ents = schema.children();
       for (index_t i = 0; i < their_ents.size(); i++) 
       {
           Schema *entry_schema = new Schema(*their_ents[i]);
           my_ents.push_back(entry_schema);
       }
    }
}


///============================================
void 
Schema::set(index_t dtype_id)
{
    reset();
    m_dtype.set(dtype_id);
}


///============================================
void 
Schema::set(const DataType &dtype)
{
    reset();
    if (dtype.id() == DataType::OBJECT_T) {
        init_object();
    } else if (dtype.id() == DataType::LIST_T) {
        init_list();
    }
    m_dtype = dtype;
}

///============================================
void 
Schema::set(const std::string &json_schema)
{
    reset();
    walk_schema(json_schema);
}


///============================================
Schema &
Schema::operator=(const Schema &schema)
{
    if(this != &schema)
    {
        set(schema);
    }
    return *this;
}

///============================================
Schema &
Schema::operator=(const std::string &json_schema)
{
    set(json_schema);
    return *this;
}

///============================================
Schema &
Schema::entry(index_t idx)
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::entry[LIST_T]>: Schema is not LIST_T");
    return *children()[idx];
}

///============================================
const Schema &
Schema::entry(index_t idx) const
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::entry[LIST_T]>: Schema is not LIST_T");

    return *children()[idx];
}


///============================================
Schema&
Schema::entry(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::entry[OBJECT_T]>: Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    index_t idx = entry_index(p_curr);
    
    if(p_next.empty())
    {
        return *children()[idx];
    }
    else
    {
        return children()[idx]->entry(p_next);
    }
}


///============================================
const Schema &
Schema::entry(const std::string &path) const
{
    // fetch w/ path forces OBJECT_T
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::entry[OBJECT_T]>: Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    index_t idx = entry_index(p_curr);
    
    if(p_next.empty())
    {
        return *children()[idx];
    }
    else
    {
        return children()[idx]->entry(p_next);
    }
}

index_t
Schema::entry_index(const std::string &path) const
{
    // find p_curr with an iterator
    std::map<std::string, index_t>::const_iterator itr = object_map().find(path);
    // return Empty if the entry does not exist (static/locked case)
    if(itr == object_map().end())
    {
        ///
        /// Full path errors would be nice here. 
        ///
        THROW_ERROR("<Schema::entry_index[OBJECT_T]>"
                    << "Attempt to access invalid entry:" << path);
                      
    }

    return itr->second;
}

///============================================
Schema &
Schema::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    init_object();
        
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    if (!has_path(p_curr)) {
        Schema* my_schema = new Schema();
        children().push_back(my_schema);
        object_map()[p_curr] = children().size() - 1;
        object_order().push_back(p_curr);
    }

    index_t idx = entry_index(p_curr);
    if(p_next.empty())
    {
        return *children()[idx];
    }
    else
    {
        return children()[idx]->fetch(p_next);
    }

}


///============================================
Schema &
Schema::fetch(index_t idx)
{
    return *children()[idx];
}

///============================================
Schema &
Schema::operator[](const std::string &path)
{
    //if(!m_locked)
        return fetch(path);
    //else
    //    return entry(path);
}

///============================================
Schema &
Schema::operator[](index_t idx)
{
    //if(!m_locked)
        return fetch(idx);
    //else
    //    return entry(idx);
}

/// Const variants use const get
///============================================
const Schema &
Schema::operator[](const std::string &path) const
{
    return entry(path);
}

///============================================
const Schema &
Schema::operator[](index_t idx) const
{
    return entry(idx);
}

///============================================
bool           
Schema::has_path(const std::string &path) const
{
    if(m_dtype.id() == DataType::EMPTY_T)
        return false;
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::has_path[OBJECT_T]> Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);
    const std::map<std::string,index_t> &ents = object_map();

    if(ents.find(p_curr) == ents.end())
    {
        return false;
    }

    if(!p_next.empty())
    {
        index_t idx = ents.find(p_curr)->second;
        return children()[idx]->has_path(p_next);
    }
    else
    {
        return true;
    }
}


///============================================
void
Schema::paths(std::vector<std::string> &paths, bool walk) const
{
    paths = object_order();
    // TODO: walk == True, show nested paths?
}

///============================================
index_t 
Schema::number_of_entries() const 
{
    // LIST_T only for now, overload for OBJECT_T
    if(m_dtype.id() != DataType::LIST_T)
        return 0;
    return children().size();
}

///============================================
void    
Schema::remove(index_t idx)
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::remove[LIST_T]> Schema is not LIST_T");
    
    std::vector<Schema*>  &lst = children();
    if(idx > lst.size())
    {
        THROW_ERROR("<Schema::remove[LIST_T]> Invalid list index:" 
                    << idx << ">" << lst.size() <<  "(list_size)");
    }

    Schema* myschema = children()[idx];
    delete myschema;
    lst.erase(lst.begin() + idx);
}

///============================================
void    
Schema::remove(const std::string &path)
{
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::remove[OBJECT_T]> Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);
    index_t idx = entry_index(p_curr);
    Schema* myschema = children()[idx];

    if(!p_next.empty())
    {
        myschema->remove(p_next);
    }

    object_map().erase(p_curr);
    children().erase(children().begin() + idx);
    object_order().erase(object_order().begin() + idx);
    delete myschema;

    for (index_t i = idx; i < object_order().size(); i++) {
        object_map()[object_order()[idx]]--;
    }
}

///============================================
index_t
Schema::total_bytes() const
{
    index_t res = 0;
    index_t dt_id = m_dtype.id();
    if(dt_id == DataType::OBJECT_T || dt_id == DataType::LIST_T)
    {
        const std::vector<Schema*> &lst = children();
        for (std::vector<Schema*>::const_iterator itr = lst.begin();
             itr != lst.end(); ++itr)
        {
            res += (*itr)->total_bytes();
        }
    }
    else if (dt_id != DataType::EMPTY_T)
    {
        res = m_dtype.total_bytes();
    }
    return res;
}


///============================================
void 
Schema::walk_schema(const std::string &json_schema)
{
    reset();
    rapidjson::Document document;
    document.Parse<0>(json_schema.c_str());
    index_t curr_offset = 0;
    conduit::walk_schema(*this,document,curr_offset);
}

///============================================
void 
walk_schema(Schema &schema,
            const rapidjson::Value &jvalue,
            index_t curr_offset)
{
    if(jvalue.IsObject())
    {
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
                    Schema curr_schema(DataType::Objects::list());
                    walk_schema(curr_schema,dt_value, curr_offset);
                    schema.append(curr_schema);
                    curr_offset += curr_schema.total_bytes();
                }
            }
            else
            {
                // handle leaf node with explicit props
                std::string dtype_name(jvalue["dtype"].GetString());
                int length = jvalue["length"].GetInt();
                const DataType df_dtype = DataType::default_dtype(dtype_name);
                index_t type_id = df_dtype.id();
                index_t size    = df_dtype.element_bytes();
                // TODO: Parse endianness
                DataType dtype(type_id,
                               length,
                               curr_offset,
                               size, 
                               size,
                               Endianness::DEFAULT_T);
                schema.set(dtype);
            }
        }
        else
        {
            // loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
				Schema &curr_schema = schema.fetch(entry_name);
				curr_schema.set(DataType::Objects::object());
                walk_schema(curr_schema,itr->value, curr_offset);
                curr_offset += curr_schema.total_bytes();
            }
        }
    }
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            Schema curr_schema(DataType::Objects::list());
            walk_schema(curr_schema,jvalue[i], curr_offset);
            curr_offset += curr_schema.total_bytes();
            // this will coerce to a list
            schema.append(curr_schema);
        }
    }
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         DataType df_dtype = DataType::default_dtype(dtype_name);
         index_t size = df_dtype.element_bytes();
         DataType dtype(df_dtype.id(),1,curr_offset,size,size,Endianness::DEFAULT_T);
         schema.set(dtype);
    }
}

///============================================
std::string
Schema::to_json() const
{
    std::ostringstream oss;
    to_json(oss);
    return oss.str();
}

///============================================
void
Schema::to_json(std::ostringstream &oss) const
{
    if(m_dtype.id() == DataType::OBJECT_T)
    {
        oss << "{";
        bool first=true;
        for (index_t i = 0; i < children().size(); i++) {
            if(!first)
                oss << ",";
            oss << "\""<< object_order()[i] << "\" : ";
            children()[i]->to_json(oss);
            oss << "\n";
            first=false;
        }
		oss << "}";
    }
	else if(m_dtype.id() == DataType::LIST_T)
	{
		oss << "[";
        bool first=true;
        for (index_t i = 0; i < children().size(); i++) {
            if(!first)
                oss << ",";
            children()[i]->to_json(oss);
            oss << "\n";
            first=false;
        }
		oss << "]";
	}
    else // assume leaf data type
    {
        m_dtype.to_json(oss);
    }
}


///============================================
Schema::Schema_Object_Hierarchy *
Schema::object_hierarchy()
{
    return static_cast<Schema_Object_Hierarchy*>(m_hierarchy_data);
}

///============================================
Schema::Schema_List_Hierarchy *
Schema::list_hierarchy()
{
    return static_cast<Schema_List_Hierarchy*>(m_hierarchy_data);
}


///============================================
const Schema::Schema_Object_Hierarchy *
Schema::object_hierarchy() const 
{
    return static_cast<Schema_Object_Hierarchy*>(m_hierarchy_data);
}

///============================================
const Schema::Schema_List_Hierarchy *
Schema::list_hierarchy() const 
{
    return static_cast<Schema_List_Hierarchy*>(m_hierarchy_data);
}


///============================================
std::map<std::string, index_t> &
Schema::object_map()
{
    return object_hierarchy()->object_map;
}

///============================================
std::vector<Schema*> &
Schema::children()
{
    if (m_dtype.id() == DataType::OBJECT_T)
    {
        return object_hierarchy()->children;
    } 
    else 
    {
        return list_hierarchy()->children;
    }
}

///============================================
std::vector<std::string> &
Schema::object_order()
{
    return object_hierarchy()->object_order;
}

///============================================
const std::map<std::string, index_t> &
Schema::object_map() const
{
    return object_hierarchy()->object_map;
}

///============================================
const std::vector<Schema*> &
Schema::children() const
{
    if (m_dtype.id() == DataType::OBJECT_T)
    {
        return object_hierarchy()->children;
    } 
    else 
    {
        return list_hierarchy()->children;
    }
}

///============================================
const std::vector<std::string> &
Schema::object_order() const
{
    return object_hierarchy()->object_order;
}

}
