///
/// file: Schema.cpp
///

#include "Schema.h"

namespace conduit
{

///============================================
/// Schema
///============================================

///============================================
Schema::Schema()
: m_json_schema("")
{}

///============================================
Schema::Schema(const Schema &schema)
: m_json_schema("")
{
    set(schema);
}

///============================================
Schema::Schema(const std::string &json_schema)
: m_json_schema("")
{
    set(json_schema);
}


///============================================
Schema::~Schema()
{

}

///============================================
void 
Schema::set(const Schema &schema)
{
    m_json_schema = schema.m_json_schema;
}

///============================================
void 
Schema::init_from_json(const std::string &json_schema)
{
    m_json_schema = json_schema;
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
    m_json_schema = json_schema;
    return *this;
}

}

