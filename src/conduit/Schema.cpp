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
/// file: Schema.cpp
///

#include <stdio.h>
#include "Schema.h"
#include "Generator.h"
#include "Error.h"
#include "Utils.h"
#include "rapidjson/document.h"

namespace conduit
{

//============================================
/// Schema
//============================================

//============================================
Schema::Schema()
{
    init_defaults();
}

//============================================
Schema::Schema(const Schema &schema)
{
    init_defaults();
    set(schema);
}

//============================================
Schema::Schema(index_t dtype_id)
{
    init_defaults();
    set(dtype_id);
}

//============================================
Schema::Schema(const DataType &dtype)
{
    init_defaults();
    set(dtype);
}


//============================================
Schema::Schema(const std::string &json_schema)
{
    init_defaults();
    set(json_schema);
}


//============================================
Schema::~Schema()
{release();}

//============================================
void            
Schema::save(const std::string &ofname,
             bool detailed, 
             index_t indent, 
             index_t depth,
             const std::string &pad,
             const std::string &eoe) const
{
    // TODO: this is ineff, get base class rep correct?
    std::ostringstream oss;
    to_json(oss,detailed,indent,depth,pad,eoe);    

    std::ofstream ofile;
    ofile.open(ofname.c_str());
    // TODO: check for open error
    ofile << oss.str();
    ofile.close();
}

//============================================
void            
Schema::load(const std::string &ifname)
{
    std::ostringstream oss;
    std::ifstream ifile;
    ifile.open(ifname.c_str());
    // TODO: check for open error
    std::string res((std::istreambuf_iterator<char>(ifile)), std::istreambuf_iterator<char>());
    set(res);
}            

//============================================
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

//============================================
void
Schema::reset()
{
    release();
    init_defaults();
}


//============================================
void
Schema::init_defaults()
{
    m_dtype  = DataType::Objects::empty();
    m_hierarchy_data = NULL;
    m_parent = NULL;
    m_root   = false;
}

//============================================
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

//============================================
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

//============================================
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
           entry_schema->m_parent = this;
           my_ents.push_back(entry_schema);
       }
    }
}


//============================================
void 
Schema::set(index_t dtype_id)
{
    reset();
    m_dtype.set(dtype_id);
}


//============================================
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

//============================================
void 
Schema::set(const std::string &json_schema)
{
    reset();
    walk_schema(json_schema);
}


//============================================
Schema &
Schema::operator=(const Schema &schema)
{
    if(this != &schema)
    {
        set(schema);
    }
    return *this;
}

//============================================
Schema &
Schema::operator=(const DataType &dtype)
{
    set(dtype);
    return *this;
}


//============================================
Schema &
Schema::operator=(const std::string &json_schema)
{
    set(json_schema);
    return *this;
}

//============================================
Schema &
Schema::entry(index_t idx)
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::entry[LIST_T]>: Schema is not LIST_T");
    return *children()[idx];
}

//============================================
const Schema &
Schema::entry(index_t idx) const
{
    if(m_dtype.id() != DataType::LIST_T)
        THROW_ERROR("<Schema::entry[LIST_T]>: Schema is not LIST_T");

    return *children()[idx];
}


//============================================
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
    
    // check for parent
    if(p_curr == "..")
    {
        if(m_parent != NULL) // TODO: check for erro (no parent)
           return m_parent->entry(p_next);
    }
    
    if(p_next.empty())
    {
        return *children()[idx];
    }
    else
    {
        return children()[idx]->entry(p_next);
    }
}


//============================================
const Schema &
Schema::entry(const std::string &path) const
{
    // fetch w/ path forces OBJECT_T
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::entry[OBJECT_T]>: Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // check for parent
    if(p_curr == "..")
    {
        if(m_parent != NULL) // TODO: check for erro (no parent)
           return m_parent->entry(p_next);
    }

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
    // return Empty if the entry does not exist (static/locked case ?)
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

//============================================
Schema &
Schema::fetch(const std::string &path)
{
    // fetch w/ path forces OBJECT_T
    init_object();
        
    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);

    // handle parent 
    // check for parent
    if(p_curr == "..")
    {
        if(m_parent != NULL) // TODO: check for erro (no parent)
           return m_parent->fetch(p_next);
    }
    
    if (!has_path(p_curr)) 
    {
        Schema* my_schema = new Schema();
        my_schema->m_parent = this;
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


//============================================
Schema &
Schema::fetch(index_t idx)
{
    return *children()[idx];
}

//============================================
Schema *
Schema::fetch_pointer(const std::string &path)
{
    return &fetch(path);
}

//============================================
Schema *
Schema::fetch_pointer(index_t idx)
{
    return &fetch(idx);
}


//============================================
Schema &
Schema::operator[](const std::string &path)
{
    //if(!m_locked)
        return fetch(path);
    //else
    //    return entry(path);
}

//============================================
Schema &
Schema::operator[](index_t idx)
{
    //if(!m_locked)
        return fetch(idx);
    //else
    //    return entry(idx);
}

/// Const variants use const get
//============================================
const Schema &
Schema::operator[](const std::string &path) const
{
    return entry(path);
}

//============================================
const Schema &
Schema::operator[](index_t idx) const
{
    return entry(idx);
}

//============================================
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
    
    // handle parent case (..)
    
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


//============================================
void
Schema::paths(std::vector<std::string> &paths, bool walk) const
{
    paths = object_order();
    // TODO: walk == True, show nested paths?
}

//============================================
index_t 
Schema::number_of_entries() const 
{
    // LIST_T only for now, overload for OBJECT_T
    if(m_dtype.id() != DataType::LIST_T  &&
       m_dtype.id() != DataType::OBJECT_T)
        return 0;
    return children().size();
}

//============================================
void    
Schema::compact_to(Schema &s_dest) const
{
    s_dest.reset();
    compact_to(s_dest,0);
}

//============================================
void    
Schema::compact_to(Schema &s_dest, index_t curr_offset) const
{
    index_t dtype_id = m_dtype.id();
    
    if(dtype_id == DataType::OBJECT_T )
    {
        index_t nchildren = children().size();
        for(index_t i=0; i < nchildren;i++)
        {
            Schema  *cld_src = children()[i];
            Schema &cld_dest = s_dest.fetch(object_order()[i]);
            cld_src->compact_to(cld_dest,curr_offset);
            curr_offset += cld_dest.total_bytes();
        }
    }
    else if(dtype_id == DataType::LIST_T)
    {
        index_t nchildren = children().size();
        for(index_t i=0; i < nchildren ;i++)
        {            
            Schema  *cld_src = children()[i];
            s_dest.append();
            Schema &cld_dest = s_dest.fetch(i);
            cld_src->compact_to(cld_dest,curr_offset);
            curr_offset += cld_dest.total_bytes();
        }
    }
    else if (dtype_id != DataType::EMPTY_T)
    {
        // create a compact data type
        m_dtype.compact_to(s_dest.m_dtype);
        s_dest.m_dtype.set_offset(curr_offset);
    }
}

//============================================
void    
Schema::remove(index_t idx)
{
    index_t dtype_id = m_dtype.id();
    if(! (dtype_id == DataType::LIST_T || dtype_id == DataType::OBJECT_T))
    {
        THROW_ERROR("<Schema::remove> Schema is not LIST_T or OBJECT_T, dtype is" << DataType::id_to_name(dtype_id));
    }
    
    std::vector<Schema*>  &chldrn = children();
    if(idx > chldrn.size())
    {
        THROW_ERROR("<Schema::remove> Invalid index:" 
                    << idx << ">" << chldrn.size() <<  "(list_size)");
    }

    if(dtype_id == DataType::OBJECT_T)
    {
        // any index above the current needs to shift down by one
        for (index_t i = idx; i < object_order().size(); i++)
        {
            object_map()[object_order()[i]]--;
        }
        
        object_map().erase(object_order()[idx]);
        object_order().erase(object_order().begin() + idx);
    }

    Schema* child = chldrn[idx];
    delete child;
    chldrn.erase(chldrn.begin() + idx);
}

//============================================
void    
Schema::remove(const std::string &path)
{
    if(m_dtype.id() != DataType::OBJECT_T)
        THROW_ERROR("<Schema::remove[OBJECT_T]> Schema is not OBJECT_T");

    std::string p_curr;
    std::string p_next;
    utils::split_path(path,p_curr,p_next);
    index_t idx = entry_index(p_curr);
    Schema *child = children()[idx];

    if(!p_next.empty())
    {
        child->remove(p_next);
    }
    else
    {
        // any index above the current needs to shift down by one
        for (index_t i = idx; i < object_order().size(); i++)
        {
            object_map()[object_order()[i]]--;
        }
        object_map().erase(p_curr);
        object_order().erase(object_order().begin() + idx);
        children().erase(children().begin() + idx);
        delete child;
    }    
}

//============================================
index_t
Schema::total_bytes() const
{
    index_t res = 0;
    index_t dt_id = m_dtype.id();
    if(dt_id == DataType::OBJECT_T || dt_id == DataType::LIST_T)
    {
        const std::vector<Schema*> &lst = children();
        for (std::vector<Schema*>::const_iterator itr = lst.begin();
             itr < lst.end(); ++itr)
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

//============================================
index_t
Schema::total_bytes_compact() const
{
    index_t res = 0;
    index_t dt_id = m_dtype.id();
    if(dt_id == DataType::OBJECT_T || dt_id == DataType::LIST_T)
    {
        const std::vector<Schema*> &lst = children();
        for (std::vector<Schema*>::const_iterator itr = lst.begin();
             itr < lst.end(); ++itr)
        {
            res += (*itr)->total_bytes_compact();
        }
    }
    else if (dt_id != DataType::EMPTY_T)
    {
        res = m_dtype.total_bytes_compact();
    }
    return res;
}


//============================================
void 
Schema::walk_schema(const std::string &json_schema)
{
    Generator g(json_schema);
    g.walk(*this);
}


//============================================
std::string
Schema::to_json(bool detailed,
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
Schema::to_json(std::ostringstream &oss,
              bool detailed, 
              index_t indent, 
              index_t depth,
              const std::string &pad,
              const std::string &eoe) const
{
    if(m_dtype.id() == DataType::OBJECT_T)
    {
        oss << eoe;
        utils::indent(oss,indent,depth,pad);
        oss << "{" << eoe;
    
        index_t nchildren = children().size();
        for(index_t i=0; i < nchildren;i++)
        {
            utils::indent(oss,indent,depth+1,pad);
            oss << "\""<< object_order()[i] << "\": ";
            children()[i]->to_json(oss,detailed,indent,depth+1,pad,eoe);
            if(i < nchildren-1)
                oss << ",";
            oss << eoe;
        }
        utils::indent(oss,indent,depth,pad);
        oss << "}";
    }
    else if(m_dtype.id() == DataType::LIST_T)
    {
        oss << eoe;
        utils::indent(oss,indent,depth,pad);
        oss << "[" << eoe;
        
        index_t nchildren = children().size();
        for(index_t i=0; i < nchildren;i++)
        {
            utils::indent(oss,indent,depth+1,pad);
            children()[i]->to_json(oss,detailed,indent,depth+1,pad,eoe);
            if(i < nchildren-1)
                oss << ",";
            oss << eoe;
        }
        utils::indent(oss,indent,depth,pad);
        oss << "]";      
    }
    else // assume leaf data type
    {
        m_dtype.to_json(oss);
    }
}


//============================================
Schema::Schema_Object_Hierarchy *
Schema::object_hierarchy()
{
    return static_cast<Schema_Object_Hierarchy*>(m_hierarchy_data);
}

//============================================
Schema::Schema_List_Hierarchy *
Schema::list_hierarchy()
{
    return static_cast<Schema_List_Hierarchy*>(m_hierarchy_data);
}


//============================================
const Schema::Schema_Object_Hierarchy *
Schema::object_hierarchy() const 
{
    return static_cast<Schema_Object_Hierarchy*>(m_hierarchy_data);
}

//============================================
const Schema::Schema_List_Hierarchy *
Schema::list_hierarchy() const 
{
    return static_cast<Schema_List_Hierarchy*>(m_hierarchy_data);
}


//============================================
std::map<std::string, index_t> &
Schema::object_map()
{
    return object_hierarchy()->object_map;
}

//============================================
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

//============================================
std::vector<std::string> &
Schema::object_order()
{
    return object_hierarchy()->object_order;
}

//============================================
const std::map<std::string, index_t> &
Schema::object_map() const
{
    return object_hierarchy()->object_map;
}

//============================================
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

//============================================
const std::vector<std::string> &
Schema::object_order() const
{
    return object_hierarchy()->object_order;
}

//============================================
void
Schema::object_map_print() const
{
    index_t sz = object_order().size();
    for(index_t i=0;i<sz;i++)
    {
        std::cout << object_order()[i] << " ";
    }
    std::cout << std::endl;
}


//============================================
void
Schema::object_order_print() const
{
    std::map<std::string, index_t>::const_iterator itr; 
        
    for(itr = object_map().begin(); itr != object_map().end();itr++)
    {
       std::cout << itr->first << ":" << itr->second << " ";
    }
    std::cout << std::endl;
}


}
