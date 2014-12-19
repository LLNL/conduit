//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: Schema.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_SCHEMA_H
#define __CONDUIT_SCHEMA_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Core.h"
#include "Endianness.h"
#include "DataType.h"

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <map>
#include <vector>
#include <string>
#include <sstream>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::Schema --
//-----------------------------------------------------------------------------
///
/// class: conduit::Schema
///
/// description:
///  TODO
///
//-----------------------------------------------------------------------------
class CONDUIT_API Schema
{
public:
//-----------------------------------------------------------------------------
// -- friends of Schema --
//-----------------------------------------------------------------------------
    friend class Node;
    friend class NodeIterator;

//-----------------------------------------------------------------------------
//
// -- begin declaration of Schema construction and destruction --
//
//-----------------------------------------------------------------------------
///@name Construction and Destruction
///@{
//-----------------------------------------------------------------------------
/// description:
///  Standard construction and destruction methods.
///
/// notes:
///  TODO:
///
//-----------------------------------------------------------------------------
    /// create an empty schema
    Schema(); 
    /// schema copy constructor
    explicit Schema(const Schema &schema);
    /// create a schema for a leaf type given a data type id
    explicit Schema(index_t dtype_id);
    /// create a schema from a DataType
    explicit Schema(const DataType &dtype);
    /// create a schema from a json description (std::string case)
    explicit Schema(const std::string &json_schema);
    /// create a schema from a json description (c string case)
    explicit Schema(const char *json_schema);

    /// Schema Destructor
    ~Schema();
    
    void  reset();

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Schema construction and destruction --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Schema set methods --
//
//-----------------------------------------------------------------------------
///@name Schema::set(...)
///@{
//-----------------------------------------------------------------------------
/// description:
///   TODO
//-----------------------------------------------------------------------------
    void set(const Schema &schema); 

    void set(index_t dtype_id);
    void set(const DataType &dtype);
    void set(const std::string &json_schema);

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Schema set methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Schema assignment operators --
//
//-----------------------------------------------------------------------------
///@name Schema Assignment Operators
///@{
//-----------------------------------------------------------------------------
/// description:
///  TODO
//-----------------------------------------------------------------------------
    Schema &operator=(const Schema &schema);
    Schema &operator=(index_t dtype_id);
    Schema &operator=(const DataType &dtype);
    Schema &operator=(const std::string &json_schema);

//-----------------------------------------------------------------------------
///@}                      
//-----------------------------------------------------------------------------
//
// -- end declaration of Schema assignment operators --
//
//-----------------------------------------------------------------------------

    /// info
    const DataType &dtype() const 
                    {return m_dtype;}

    index_t         total_bytes() const;
    index_t         total_bytes_compact() const;

    void            print() const
                        {std::cout << to_json() << std::endl;}

    bool            compare(const Schema &n, Node &cmp_results) const;
    bool            operator==(const Schema &n) const;

    /// transformations
    void            compact_to(Schema &s_dest) const;

    std::string     to_json(bool detailed=true, 
                                index_t indent=2, 
                                index_t depth=0,
                                const std::string &pad=" ",
                                const std::string &eoe="\n") const;

    void            to_json(std::ostringstream &oss,
                                bool detailed=true, 
                                index_t indent=2, 
                                index_t depth=0,
                                const std::string &pad=" ",
                                const std::string &eoe="\n") const;

    /// i/o
    void            save(const std::string &ofname,
                         bool detailed=true, 
                         index_t indent=2, 
                         index_t depth=0,
                         const std::string &pad=" ",
                         const std::string &eoe="\n") const;

    void            load(const std::string &ifname);

    /// child access                         

    // the `child' methods don't modify map structure, if a path doesn't exists
    // they will throw an exception
    
    Schema           &child(const std::string &path);
    const Schema     &child(const std::string &path) const;

    Schema           &child(index_t idx);
    const Schema     &child(index_t idx) const;
    Schema           *child_pointer(index_t idx);

    index_t           child_index(const std::string &path) const;

    /// fetch with a path arg methods do modifies map structure 
    /// if a path doesn't exists
    Schema           &fetch(const std::string &path);
    Schema           *fetch_pointer(const std::string &path);

    // this uses the fetch method
    Schema           &operator[](const std::string &path);

    /// these use the "child" methods
    const Schema     &operator[](const std::string &path) const;
    Schema           &operator[](const index_t idx);
    const Schema     &operator[](const index_t idx) const;
  


    index_t           element_index(index_t idx) const 
                        {return m_dtype.element_index(idx);}
     
    

    bool              is_root() const { return m_root;}
    
    ///
    /// Object Interface
    ///
    index_t number_of_children() const; // object and list 
    bool    has_path(const std::string &path) const;
    void    paths(std::vector<std::string> &paths) const;
    void    remove(const std::string &path);

    ///
    /// List Interface
    ///
    void    remove(index_t idx);

    void    append()
        {init_list(); children().push_back(new Schema());}

    void    append(const DataType &dtype)
        {init_list(); children().push_back(new Schema(dtype));}

    void    append(const Schema &schema)
        {init_list(); children().push_back(new Schema(schema));}

    void list_of(const Schema &schema, index_t num_elements);

    /// interface warts
    void              set_root(bool value) {m_root = value;}

private:
    // for obj and list interfaces
    std::vector<Schema*>                   &children();
    std::map<std::string, index_t>         &object_map();
    std::vector<std::string>               &object_order();

    void                                   object_map_print()   const;
    void                                   object_order_print() const;

    const std::vector<Schema*>             &children()  const;    
    const std::map<std::string, index_t>   &object_map()   const;
    const std::vector<std::string>         &object_order() const;


    void        init_defaults();
    void        init_list();
    void        init_object();
    void        release();
    
    void        compact_to(Schema &s_dest, index_t curr_offset) const ;
    void        walk_schema(const std::string &json_schema);
    void        walk_schema(Schema &schema, const std::string &json_schema);

    DataType    m_dtype;
    void       *m_hierarchy_data;
    bool        m_root;
    Schema     *m_parent;


    
    struct Schema_Object_Hierarchy 
    {
        std::vector<Schema*>            children;
        std::vector<std::string>        object_order;
        std::map<std::string, index_t>  object_map;
    };

    struct Schema_List_Hierarchy 
    {
        std::vector<Schema*> children;
    };

    Schema_Object_Hierarchy        *object_hierarchy();
    Schema_List_Hierarchy          *list_hierarchy();

    const Schema_Object_Hierarchy  *object_hierarchy() const;
    const Schema_List_Hierarchy    *list_hierarchy()   const;

};
//-----------------------------------------------------------------------------
// -- end conduit::Schema --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
