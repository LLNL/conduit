///
/// file: Schema.h
///

#ifndef CONDUIT_SCHEMA_H__
#define CONDUIT_SCHEMA_H__

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
    Schema(const std::string &json_schema);
    Schema(const Schema &schema);

    /* Destructor */
    virtual  ~Schema();

    void set(const Schema &schema);     
    void set(const std::string &json_schema) 
        {init_from_json(json_schema);}

               
    /* Assignment ops */
    Schema &operator=(const Schema &Schema);
    Schema &operator=(const std::string &json_schema);
   

    /*Schema Access */
    std::string to_json() const { return m_json_schema;}

private:
    void init_from_json(const std::string &json_schema);
    std::string m_json_schema;    
      
};

}


#endif
