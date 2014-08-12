///
/// file: Schema.cpp
///

#include <stdio.h>
#include "Generator.h"
#include "Error.h"
#include "Utils.h"
#include "rapidjson/document.h"

namespace conduit
{
    
// we want to isolate the conduit API from the rapidjson headers
// so any methods using rapidjson types are defined in the cpp imp.

void
walk_schema(Schema *schema,
            const   rapidjson::Value &jvalue,
            index_t curr_offset);
    
void
walk_schema(Node   *node,
            Schema *schema,
            void   *data,
            const rapidjson::Value &jvalue,
            index_t curr_offset);    

void 
walk_schema_pure_json(Node   *node,
                      Schema *schema,
                      const rapidjson::Value &jvalue);
       
void 
parse_leaf(const rapidjson::Value &jvalue,
           index_t offset,
           DataType &dtype_res);

void 
parse_inline_value(const rapidjson::Value &jvalue,
                   index_t element_index,
                   Node &node);

///============================================
/// Generator
///============================================


///============================================
Generator::Generator(const std::string &json_schema)
:m_json_schema(json_schema),
 m_protocol("conduit"),
 m_data_ptr(NULL)
{}


///============================================
Generator::Generator(const std::string &json_schema,
                     void *data)
:m_json_schema(json_schema),
 m_protocol("conduit"),
 m_data_ptr(data)
{}

///============================================
Generator::Generator(const std::string &json_schema,
                     const std::string &protocol,
                     void *data)
:m_json_schema(json_schema),
 m_protocol(protocol),
 m_data_ptr(data)
{}



///============================================
void 
Generator::walk(Schema &schema) const
{
    schema.reset();
    rapidjson::Document document;
    std::string res = utils::json_sanitize(m_json_schema);
    document.Parse<0>(res.c_str());
    index_t curr_offset = 0;
    conduit::walk_schema(&schema,document,curr_offset);
}

///============================================
void 
Generator::walk(Node &node) const
{
    node.reset();
    // if data is null, we can parse the schema via the other 'walk' method
    
    if(m_data_ptr == NULL && m_protocol == "conduit")
    {
        Schema s;
        walk(s);
        node.set(s);
    }
    // case for pure json
    else if(m_data_ptr == NULL && m_protocol == "json")
    {
        rapidjson::Document document;
        std::string res = utils::json_sanitize(m_json_schema);
        document.Parse<0>(res.c_str());
        conduit::walk_schema_pure_json(&node,
                                       node.schema_pointer(),
                                       document);
    }
    else
    {
        rapidjson::Document document;
        std::string res = utils::json_sanitize(m_json_schema);
        document.Parse<0>(res.c_str());
        index_t curr_offset = 0;
        conduit::walk_schema(&node,
                             node.schema_pointer(),
                             m_data_ptr,
                             document,
                             curr_offset);
    }
}

///============================================
void
parse_leaf(const rapidjson::Value &jvalue, index_t offset, DataType &dtype_res)
{
    std::string dtype_name(jvalue["dtype"].GetString());
    int length = jvalue["length"].GetInt();
    const DataType df_dtype = DataType::default_dtype(dtype_name);
    index_t type_id = df_dtype.id();
    index_t size    = df_dtype.element_bytes();

    // parse endianness
    index_t endianness = Endianness::DEFAULT_T;
    if(jvalue.HasMember("endianess") && jvalue["endianness"].IsString())
    {
        std::string end_val(jvalue["endianness"].GetString());
        if(end_val == "big")
        {
            endianness = Endianness::BIG_T;
        }
        else
        {
            endianness = Endianness::LITTLE_T;
        }
        
    }
    
    dtype_res.set(type_id,
                  length,
                  offset,
                  size, 
                  size,
                  endianness);
}

///============================================
void
parse_inline_value(const rapidjson::Value &jvalue,
                   index_t element_index,
                   Node &node)
{
    if(jvalue.IsString())
    {
        // TODO check type compat with string
        // throw parsing error if our inline values
        // don't match what we expected
    }
    else if(jvalue.IsBool())
    {
        // TODO check type compat with bool
        // only allow bools to be assigned to the bool type
        // throw parsing error if our inline values
        // don't match what we expected
    }
    else if(jvalue.IsNumber())
    {
        // TODO check type compat with numeric
        // only allow numeric to be assigned to a numeric type
        // throw parsing error if our inline values
        // don't match what we expected
    }
}


///============================================
void 
walk_schema(Schema *schema,
            const   rapidjson::Value &jvalue,
            index_t curr_offset)
{
    // object cases
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
                    // TODO: Handle reference 
                    if(jvalue["length"].IsObject() && jvalue["lenght"].HasMember("reference"))
                    {
                        // we shouldn't get here ....
                    }
                    else
                    {
                        length = jvalue["length"].GetInt();
                    }
                }
                // we will create `length' # of objects of obj des by dt_value
                 
                // TODO: we only need to parse this once, not leng # of times
                // but this is the easiest way to start.                             
                for(int i=0;i< length;i++)
                {
                    Schema curr_schema(DataType::Objects::list());
                    walk_schema(&curr_schema,dt_value, curr_offset);
                    schema->append(curr_schema);
                    curr_offset += curr_schema.total_bytes();
                }
            }
            else
            {
                // handle leaf node with explicit props
                DataType dtype;
                parse_leaf(jvalue,curr_offset,dtype);
                schema->set(dtype);
            }
        }
        else
        {
            // loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Schema &curr_schema = schema->fetch(entry_name);
                curr_schema.set(DataType::Objects::object());
                walk_schema(&curr_schema,itr->value, curr_offset);
                curr_offset += curr_schema.total_bytes();
            }
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            Schema curr_schema(DataType::Objects::list());
            walk_schema(&curr_schema,jvalue[i], curr_offset);
            curr_offset += curr_schema.total_bytes();
            // this will coerce to a list
            schema->append(curr_schema);
        }
    }
    // Simplest case, handles "uint32", "float64", with extended type info
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         DataType df_dtype = DataType::default_dtype(dtype_name);
         index_t size = df_dtype.element_bytes();
         DataType dtype(df_dtype.id(),1,curr_offset,size,size,Endianness::DEFAULT_T);
         schema->set(dtype);
    }
}

///============================================
void 
walk_schema_pure_json(Node  *node,
                      Schema *schema,
                      const rapidjson::Value &jvalue)
{
    // object cases
    if(jvalue.IsObject())
    {
        // loop over all entries
        for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
             itr != jvalue.MemberEnd(); ++itr)
        {
            std::string entry_name(itr->name.GetString());
            Schema *curr_schema = schema->fetch_pointer(entry_name);
            Node *curr_node       = new Node(curr_schema);
            curr_node->set_parent(node);
            walk_schema_pure_json(curr_node,curr_schema,itr->value);
            node->append(curr_node);                
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            schema->append();
            Schema *curr_schema = schema->fetch_pointer(i);
            Node   *curr_node   = new Node(curr_schema);
            curr_node->set_parent(node);
            walk_schema_pure_json(curr_node,curr_schema,jvalue[i]);
            node->append(curr_node);
        }
        
    }
    // Simplest case, handles "uint32", "float64", with extended type info
    else if(jvalue.IsString()) // bytestr case
    {
        std::string sval(jvalue.GetString());
        node->set(sval);
    }
    else if(jvalue.IsNull())
    {
        node->reset();
    }
    else if(jvalue.IsBool())
    {
        if(jvalue.IsTrue())
        {
            node->set(true);
        }
        else
        {
            node->set(false);
        }
    }
    else if(jvalue.IsNumber())
    {
        // use 64bit types by default ... 
	    if(jvalue.IsInt())
        {
            node->set((int64)jvalue.GetInt());
        }
        else if(jvalue.IsInt64())
        {
             node->set((int64)jvalue.GetInt64());
        }
        
        else if(jvalue.IsUint())
        {
             node->set((uint64)jvalue.GetInt());
        }
        else if(jvalue.IsUint64())
        {
             node->set((uint64)jvalue.GetUint64());
        }
        else  // double case
        {
            node->set((float64)jvalue.GetDouble());
        }
    }
}


///============================================
void 
walk_schema(Node   *node,
            Schema *schema,
            void   *data,
            const rapidjson::Value &jvalue,
            index_t curr_offset)
{
    // object cases
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
                    if(jvalue["length"].IsNumber())
                    {
                        length = jvalue["length"].GetInt();
                    }
                    else if(jvalue["length"].IsObject() && 
                            jvalue["length"].HasMember("reference"))
                    {
                        std::string ref_path = jvalue["length"]["reference"].GetString();
                        length = node->fetch(ref_path).to_index_t();
                    }
                    
                }
                // we will create `length' # of objects of obj des by dt_value
                 
                // TODO: we only need to parse this once, not leng # of times
                // but this is the easiest way to start.                             
                for(int i=0;i< length;i++)
                {
                    schema->append();
                    Schema *curr_schema = schema->fetch_pointer(i);
                    Node   *curr_node   = new Node(curr_schema);
                    curr_node->set_parent(node);
                    walk_schema(curr_node,curr_schema,data,dt_value, curr_offset);
                    curr_offset += curr_schema->total_bytes();
                    node->append(curr_node);
                }
                
            }
            else
            {
                // handle leaf node with explicit props
                DataType dtype;
                parse_leaf(jvalue,curr_offset,dtype);
                schema->set(dtype);

                // node needs to link schema ptr 
                node->set(schema,data);

                // check for inline json values
                if(jvalue.HasMember("value"))
                {
                    const rapidjson::Value &jinline = jvalue["value"];
                    // we assume a "value" is a leaf or list of compatiable leafs
                    if(jvalue["value"].IsArray())
                    {
                        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
                        {
                            parse_inline_value(jvalue[i],i,*node);
                        }
                    }
                    else
                    {
                        parse_inline_value(jvalue,0,*node);
                    }
                }
            }
        }
        else
        {
            // standard object case - loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Schema *curr_schema = schema->fetch_pointer(entry_name);
                Node *curr_node     = new Node(curr_schema);
                curr_node->set_parent(node);
                walk_schema(curr_node,curr_schema,data,itr->value, curr_offset);
                curr_offset += curr_schema->total_bytes();
                node->append(curr_node);                
            }
            
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            schema->append();
            Schema *curr_schema = schema->fetch_pointer(i);
            Node   *curr_node   = new Node(curr_schema);
            curr_node->set_parent(node);
            walk_schema(curr_node,curr_schema,data,jvalue[i], curr_offset);
            curr_offset += curr_schema->total_bytes();
            node->append(curr_node);
        }
        
    }
    // Simplest case, handles "uint32", "float64", with extended type info
    else if(jvalue.IsString())
    {
         std::string dtype_name(jvalue.GetString());
         DataType df_dtype = DataType::default_dtype(dtype_name);
         index_t size = df_dtype.element_bytes();
         DataType dtype(df_dtype.id(),1,curr_offset,size,size,Endianness::DEFAULT_T);
         schema->set(dtype);
         
         // node needs to link schema ptr 
         node->set(schema,data);
    }
}

};

