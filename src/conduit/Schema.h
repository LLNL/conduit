///
/// file: Schema.h
///

#ifndef __CONDUIT_SCHEMA_H
#define __CONDUIT_SCHEMA_H

#include "Core.h"
#include "Endianness.h"
#include "DataType.h"

#include <map>
#include <vector>
#include <string>
#include <sstream>

namespace conduit
{

class Schema
{
public:    
    /* Constructors */
    Schema(); // empty schema
    explicit Schema(index_t dtype_id);
    Schema(const DataType &dtype);
    Schema(const std::string &json_schema);
    Schema(const Schema &schema);

    /* Destructor */
    virtual  ~Schema();

    void set(const Schema &schema); 
    void set(index_t dtype_id);
    void set(const DataType &dtype);
    void set(const std::string &json_schema);
               
    /* Assignment ops */
    Schema &operator=(const Schema &schema);
    Schema &operator=(index_t dtype_id);
    Schema &operator=(const DataType &dtype);
    Schema &operator=(const std::string &json_schema);
   
    index_t         total_bytes() const;
    
    std::string     to_json() const;
    void            to_json(std::ostringstream &oss) const;
    
    bool            compare(const Schema &n, Node &cmp_results) const;
    bool            operator==(const Schema &n) const;
    
    const DataType &dtype() const {return m_dtype;}
    
    // the `entry' methods don't modify map structure, if a path doesn't exists
    // they will return an Empty Locked Node (we could also throw an exception)
    
    Schema           &entry(const std::string &path);
    Schema           &entry(index_t idx);
    const Schema     &entry(const std::string &path) const;
    const Schema     &entry(index_t idx) const;
    
    // the `fetch' methods do modify map structure if a path doesn't exists
    Schema           &fetch(const std::string &path);
    Schema           &fetch(index_t idx);
    
    index_t    element_index(index_t idx) const {return m_dtype.element_index(idx);}
      
    Schema           &operator[](const std::string &path);
    Schema           &operator[](const index_t idx);
    const Schema     &operator[](const std::string &path) const;
    const Schema     &operator[](const index_t idx) const;
  
    void    reset();
    index_t number_of_entries() const;

    ///
    /// Object Interface
    ///
    bool    has_path(const std::string &path) const;
    void    paths(std::vector<std::string> &paths,bool expand=false) const;
    void    remove(const std::string &path);

    ///
    /// List Interface
    ///
    void    remove(index_t idx);

    void append(const DataType &dtype)
        {init_list(); list_entries().push_back(Schema(dtype));}

    void append(const Schema &schema)
        {init_list(); list_entries().push_back(Schema(schema));}

    void append(Schema &schema)
        {init_list(); list_entries().push_back(schema);}

    void list_of(const Schema &schema, index_t num_elements);

    ///
    /// TODO: locking
    ///

private:
    void        init_defaults();
    void        init_list();
    void        init_object();

    void        walk_schema(const std::string &json_schema);
    void        walk_schema(Schema &schema,const std::string &json_schema);

    DataType                      m_dtype;
    
    // for obj and list interfaces
    std::map<std::string, Schema>         &obj_entries();
    std::vector<Schema>                   &list_entries();

    const std::map<std::string, Schema>   &obj_entries() const;
    const std::vector<Schema>             &list_entries() const;

    std::vector<Schema>          m_list_entries;
    std::vector<std::string>     m_obj_insert_order;
    std::map<std::string,Schema> m_obj_entries;
      
};

}


#endif
