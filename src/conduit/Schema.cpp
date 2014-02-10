///
/// file: Schema.cpp
///

#include "Schema.h"
#include "rapidjson/document.h"

namespace conduit
{


index_t 
walk_schema(const rapidjson::Value &jvalue);

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
    // walk the schema to determine the total # of bytes
    
    m_total_bytes = walk_schema(m_json_schema);
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


///============================================
index_t 
Schema::walk_schema(const std::string &json_schema)
{
    rapidjson::Document document;
    document.Parse<0>(json_schema.c_str());
    return conduit::walk_schema(document);
}

///
/// TODO This parsing is basically duplicated in Node,
/// need to better define resp of the Schema and of the Node

///============================================
index_t 
walk_schema(const rapidjson::Value &jvalue)
{
    index_t  res_bytes = 0;
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
                res_bytes = length * walk_schema(dt_value);
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
                               0,
                               size, 
                               size,
                               Endianness::DEFAULT_T);
                res_bytes = dtype.total_bytes();
            }
        }
        else
        {
            // loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                res_bytes += walk_schema(itr->value);
            }
        }
    }
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            res_bytes += walk_schema(jvalue[i]);
        }
    }
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         DataType df_dtype = DataType::default_dtype(dtype_name);
         index_t size = df_dtype.element_bytes();
         DataType dtype(df_dtype.id(),1,0,size,size,Endianness::DEFAULT_T);
         res_bytes = dtype.total_bytes();
    }
    
    return res_bytes;
}



}

