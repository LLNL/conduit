///
/// file: Schema.cpp
///

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
{}

///============================================
void
Schema::reset()
{
    init_defaults();
}


///============================================
void
Schema::init_defaults()
{
    m_dtype  = DataType::Objects::empty();
    m_obj_insert_order.clear();
    m_obj_entries.clear();
    m_list_entries.clear();
}

///============================================
void
Schema::init_object()
{
    if(dtype().id() != DataType::OBJECT_T)
    {
        reset();
        m_dtype  = DataType::Objects::object();
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
    }
}

///============================================
void 
Schema::set(const Schema &schema)
{
    m_dtype             = schema.m_dtype;
    m_obj_entries       = schema.m_obj_entries;
    m_obj_insert_order  = schema.m_obj_insert_order;
    m_list_entries      = schema.m_list_entries;
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
    return list_entries()[idx];
}

///============================================
const Schema &
Schema::entry(index_t idx) const
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::entry[LIST_T]>: Schema is not LIST_T");

    return list_entries()[idx];
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
    // find p_curr with an iterator
    std::map<std::string, Schema> &ents = obj_entries();
    std::map<std::string, Schema>::iterator itr = ents.find(p_curr);
    // return Empty if the entry does not exist (static/locked case)
    if(itr == ents.end())
    {
        ///
        /// Full path errors would be nice here. 
        ///
        THROW_ERROR("<Schema::entry[OBJECT_T]>"
                    << "Attempt to access invalid entry:" << path);
                      
    }
    
    
    if(p_next.empty())
    {
        return itr->second;
    }
    else
    {
        return itr->second.entry(p_next);
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
    // find p_curr with an iterator
    const std::map<std::string, Schema> &ents = obj_entries();
    std::map<std::string, Schema>::const_iterator itr = ents.find(p_curr);
    // return Empty if the entry does not exist (static/locked case)
    if(itr == ents.end())
    {
        ///
        /// Full path errors would be nice here. 
        ///
        THROW_ERROR("<Schema::entry[OBJECT_T]>"
                    << "Attempt to access invalid entry:" << path);
    }
    
    if(p_next.empty())
        return itr->second;
    else
        return itr->second.entry(p_next);
}


///============================================
Schema &
Schema::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    if(m_dtype.id() != DataType::OBJECT_T)
        set(DataType::Objects::object());
        
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);
    if(p_next.empty())
    {
        // check if this is a new entry, if so record insert order
        const std::map<std::string, Schema> &ents = obj_entries();
        const std::map<std::string, Schema>::const_iterator itr = ents.find(p_curr);
        if(itr == ents.end())
            m_obj_insert_order.push_back(p_curr);
        return obj_entries()[p_curr];
    }
    else
        return obj_entries()[p_curr].fetch(p_next);
}


///============================================
Schema &
Schema::fetch(index_t idx)
{
    return list_entries()[idx];
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
    const std::map<std::string,Schema> &ents = obj_entries();

    if(ents.find(p_curr) == ents.end())
    {
        return false;
    }

    if(!p_next.empty())
    {
        const Schema &s = ents.find(p_curr)->second;
        return s.has_path(p_next);
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
    paths.clear();
    // TODO: Imp
    // TODO: walk == True, show nested paths
}

///============================================
index_t 
Schema::number_of_entries() const 
{
    // LIST_T only for now, overload for OBJECT_T
    if(m_dtype.id() != DataType::LIST_T)
        return 0;
    return list_entries().size();
}

///============================================
void    
Schema::remove(index_t idx)
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::remove[LIST_T]> Schema is not LIST_T");
    
    std::vector<Schema>  &lst = list_entries();
    if(idx > lst.size())
    {
        THROW_ERROR("<Schema::remove[LIST_T]> Invalid list index:" 
                    << idx << ">" << lst.size() <<  "(list_size)");
    }
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
    std::map<std::string,Schema> &ents = obj_entries();

    if(ents.find(p_curr) == ents.end())
    {
        ///
        /// Full path errors would be nice here
        ///
        THROW_ERROR("<Schema::remove[OBJECT_T]>"
                    << "Attempt to remove non-existant entry:" << path); 
    }

    if(!p_next.empty())
    {
        Schema &s = ents.find(p_curr)->second;
        return s.remove(p_next);
        
    }
    else
    {
        ents.erase(p_curr);
    }
}

///============================================
index_t
Schema::total_bytes() const
{
    index_t res = 0;
    index_t dt_id = m_dtype.id();
    if(dt_id == DataType::OBJECT_T)
    {
        const std::map<std::string, Schema> &ents = obj_entries();
        for (std::map<std::string, Schema>::const_iterator itr = ents.begin();
             itr != ents.end(); ++itr) 
        {
            res += itr->second.total_bytes();
        }
    }
    else if(dt_id == DataType::LIST_T)
    {
        const std::vector<Schema> &lst = list_entries();
        for (std::vector<Schema>::const_iterator itr = lst.begin();
             itr != lst.end(); ++itr)
        {
            res += itr->total_bytes();
        }
    }
    else if (dt_id != DataType::EMPTY_T)
    {
        res = m_dtype.total_bytes();
    }
    return res;
}


///
/// TODO This parsing is basically duplicated in Node,
/// need to better define resp of the Schema and of the Node

///============================================
void 
Schema::walk_schema(const std::string &json_schema)
{
    reset();
    m_dtype.set(DataType::OBJECT_T);
    
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
                Schema curr_schema(DataType::Objects::object());
                schema[entry_name] = Schema(DataType::Objects::object());
                walk_schema(curr_schema,itr->value, curr_offset);
                schema[entry_name] = curr_schema;
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

        std::vector<std::string>::const_iterator order_itr;
        std::map<std::string,Schema>::const_iterator ent_itr;

        const std::map<std::string, Schema> &ents = obj_entries();

        bool first=true;
        for(order_itr = m_obj_insert_order.begin(); 
            order_itr != m_obj_insert_order.end(); 
            ++order_itr)
        {
            const std::string &ent_name = *order_itr;
            if(!first)
                oss << ",";
            oss << "\""<< ent_name << "\" : ";
            ent_itr = ents.find(ent_name);
            ent_itr->second.to_json(oss);
            oss << "\n";
            first=false;
        }
        oss << "}\n";
    }
    else if(m_dtype.id() == DataType::LIST_T)
    {
        oss << "[";
        std::vector<Schema>::const_iterator itr;
        const std::vector<Schema> &lst = list_entries();
        bool first=true;
        for(itr = lst.begin(); itr != lst.end(); ++itr)
        {
            if(!first)
                oss << ",";
            (*itr).to_json(oss);
            oss << "\n";
            first=false;
        }
        oss << "]\n";
    }
    else // assume leaf data type
    {
        m_dtype.to_json(oss);
    }
}

///============================================
std::map<std::string, Schema> &
Schema::obj_entries()
{
    return m_obj_entries;
}

///============================================
std::vector<Schema> &
Schema::list_entries()
{
    return m_list_entries;;
}

///============================================
const std::map<std::string, Schema> & 
Schema::obj_entries() const
{
    return m_obj_entries;
}

///============================================
const std::vector<Schema> &
Schema::list_entries() const
{
    return m_list_entries;
}

}

