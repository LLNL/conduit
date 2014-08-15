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

index_t 
json_to_numeric_dtype(const rapidjson::Value &jvalue)
{
    index_t res = DataType::EMPTY_T; 
    if(jvalue.IsNumber())
    {
        // TODO: We could have better logic for dealing with int vs uint
        if(jvalue.IsUint64() || 
           jvalue.IsInt64()  || 
           jvalue.IsUint()   ||
           jvalue.IsInt())
        {
            res  = DataType::INT64_T; // for int
        }
        else if(jvalue.IsDouble())
        {
            res  = DataType::FLOAT64_T; // for float
        } 
        // else -- value already inite to EMPTY_T
    }
    
    return res;
}

index_t
check_homogenus_json_array(const rapidjson::Value &jvalue)
{
    // check for homogenous array of ints or floats
    // promote to float64 as the most wide type
    // (this is heruistic decison)

    if(jvalue.Size() == 0)
        return DataType::EMPTY_T;

    index_t val_type = json_to_numeric_dtype(jvalue[(rapidjson::SizeType)0]); 
    bool homogenous  = (val_type != DataType::EMPTY_T);

    for (rapidjson::SizeType i = 1; i < jvalue.Size() && homogenous; i++)
    {
        index_t curr_val_type = json_to_numeric_dtype(jvalue[i]);
        if((val_type == DataType::INT64_T || val_type == DataType::INT64_T) &&
           curr_val_type ==  DataType::FLOAT64_T)
        {
            // promote to a double (may lose prec in some cases)
            val_type = DataType::FLOAT64_T;
        }
        else if(curr_val_type == DataType::EMPTY_T)
        {
            // non hmg inline
            homogenous = false;
            val_type = DataType::EMPTY_T;
        }
    }

    return val_type;
}

    
void
parse_json_int64_array(const rapidjson::Value &jvalue,
                        std::vector<int64> &res)
{
   res.resize(jvalue.Size(),0);
   for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
   {
       res[i] = jvalue[i].GetInt64();
   }
}

void
parse_json_int64_array(const rapidjson::Value &jvalue,
                       Node &node)
{
    // TODO: we can make this more efficent 
    std::vector<int64> vals;
    parse_json_int64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        case DataType::INT8_T:   
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_T: 
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_T:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_T:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_T:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_T:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_T:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_T:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_T:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_T:
            node.as_float64_array().set(vals);
            break;
    }
}


void
parse_json_uint64_array(const rapidjson::Value &jvalue,
                         std::vector<uint64> &res)
{
    res.resize(jvalue.Size(),0);
    for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetUint64();
    }
}

void
parse_json_uint64_array(const rapidjson::Value &jvalue,
                        Node &node)
{
    // TODO: we can make this more efficent 
    std::vector<uint64> vals;
    parse_json_uint64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        case DataType::INT8_T:   
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_T: 
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_T:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_T:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_T:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_T:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_T:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_T:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_T:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_T:
            node.as_float64_array().set(vals);
            break;
    }
}

void
parse_json_float64_array(const rapidjson::Value &jvalue,
                         std::vector<float64> &res)
{
    res.resize(jvalue.Size(),0);
    for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetDouble();
    }
}

void
parse_json_float64_array(const rapidjson::Value &jvalue,
                         Node &node)
{
    // TODO: we can make this more efficent 
    std::vector<float64> vals;
    parse_json_float64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        case DataType::INT8_T:   
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_T: 
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_T:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_T:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_T:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_T:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_T:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_T:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_T:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_T:
            node.as_float64_array().set(vals);
            break;
    }
}

///============================================
void
parse_leaf_dtype(const rapidjson::Value &jvalue, index_t offset, DataType &dtype_res)
{
    std::string dtype_name(jvalue["dtype"].GetString());
    index_t length=0;
    if(jvalue.HasMember("length"))
    {
        length = jvalue["length"].GetUint64();
    }
    const DataType df_dtype = DataType::default_dtype(dtype_name);
    index_t type_id = df_dtype.id();
    index_t size    = df_dtype.element_bytes();

    // TODO: parse offset (override offset if passed)
    
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
    
    if(length == 0)
    {
        if(jvalue.HasMember("value") &&
           jvalue["value"].IsArray())
        {
            length = jvalue["value"].Size();
        }
        else
        {
            length =1 ;
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
parse_inline_leaf(const rapidjson::Value &jvalue,
                  Node &node)
{
    if(jvalue.IsString())
    {
        if(node.dtype().id() == DataType::BYTESTR_T)
        {
            std::string sval(jvalue.GetString());
            node.set(sval);
        }
        else
        {
             /// TODO: ERROR
             // type incompat with bytestr
             // only allow strings to be assigned to a bytestr type
             // throw parsing error if our inline values
             // don't match what we expected
        }
    }
    else if(jvalue.IsBool())
    {
        if(node.dtype().id() == DataType::BOOL8_T)
        {
            node.set(jvalue.GetBool());
        }
        else
        {
             /// TODO: ERROR
             // type incompat with bool
             // only allow bool to be assigned to a bool8 type
             // throw parsing error if our inline values
             // don't match what we expected
        }
    }
    else if(jvalue.IsNumber())
    {
        switch(node.dtype().id())
        {
            // signed ints
            case DataType::INT8_T:   
                node.set((int8)jvalue.GetInt64());
                break;
            case DataType::INT16_T: 
                node.set((int16)jvalue.GetInt64());
                break;
            case DataType::INT32_T:
                node.set((int32)jvalue.GetInt64());
                break;
            case DataType::INT64_T:
                node.set((int64)jvalue.GetInt64());
                break;
            // unsigned ints
            case DataType::UINT8_T:
                node.set((uint8)jvalue.GetUint64());
                break;
            case DataType::UINT16_T:
                node.set((uint16)jvalue.GetUint64());
                break;
            case DataType::UINT32_T:
                node.set((uint32)jvalue.GetUint64());
                break;
            case DataType::UINT64_T:
                node.set((uint64)jvalue.GetUint64());
                break;  
            //floats
            case DataType::FLOAT32_T:
                node.set((float32)jvalue.GetDouble());
                break;
            case DataType::FLOAT64_T:
                node.set((float64)jvalue.GetDouble());
                break;
            // case default:
            //     /// TODO: ERROR
            //     // type incompat with numeric
            //     // only allow numeric to be assigned to a numeric type
            //     // throw parsing error if our inline values
            //     // don't match what we expected
            //     ;
            //     break;
        }
    }
}

///============================================
void
parse_inline_value(const rapidjson::Value &jvalue,
                   Node &node)
{
    if(jvalue.IsArray())
    {
        // we assume a "value" is a leaf or list of compatiable leafs
        index_t hval_type = check_homogenus_json_array(jvalue);
        
        if(node.dtype().number_of_elements() < jvalue.Size())
        {
            std::cout << "ERROR" << std::endl;
            // TODO: error
            // DataType df_dtype = DataType::default_dtype(node.dtype().id());
            // index_t ele_size = df_dtype.element_bytes();
            // DataType dtype(df_dtype.id(),
            //                jvalue.Size(),
            //                0,
            //                ele_size,
            //                ele_size,
            //                Endianness::DEFAULT_T);
            // // this will force an init
            // node.set(dtype);
        }
        
        if(hval_type == DataType::INT64_T)
        {
            if(node.dtype().is_unsigned_integer())
            {
                parse_json_uint64_array(jvalue,node);                
            }
            else
            {
                parse_json_int64_array(jvalue,node);
            }
        }
        else if(hval_type == DataType::FLOAT64_T)
        {
            parse_json_float64_array(jvalue,node);
        }
        else // error
        {
            //TODO: Parsing Error, not hmg
        }
    }
    else
    {
        parse_inline_leaf(jvalue,node);
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
                    if(jvalue["length"].IsObject() && jvalue["length"].HasMember("reference"))
                    {
                        // in some cases we shouldn't get here ...
                        // TODO ref without "data" could be a problem
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
                parse_leaf_dtype(jvalue,curr_offset,dtype);
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
    // Simplest case, handles "uint32", "float64", etc
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
        index_t hval_type = check_homogenus_json_array(jvalue);
        if(hval_type == DataType::INT64_T)
        {
            std::vector<int64> res;
            parse_json_int64_array(jvalue,res);
            node->set(res);
        }
        else if(hval_type == DataType::FLOAT64_T)
        {
            std::vector<float64> res;
            parse_json_float64_array(jvalue,res);
            node->set(res);            
        }
        else // not numeric array
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
	    if(jvalue.IsInt() || jvalue.IsInt64())
        {
            node->set((int64)jvalue.GetInt64());
        }
        else if(jvalue.IsUint() || jvalue.IsUint64())
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
                    // auto offset only makes sense when we have data
                    if(data != NULL)
                        curr_offset += curr_schema->total_bytes();
                    node->append(curr_node);
                }
                
            }
            else
            {
                // handle leaf node with explicit props
                DataType dtype;
                
                parse_leaf_dtype(jvalue,curr_offset,dtype);
   
                if(data != NULL)
                {
                    // node needs to link schema ptr
                    schema->set(dtype);
                    node->set(schema,data);
                }
                else
                {
                    node->set(schema); // properly links back to schema tree
                    // we need to dynamically alloc
                    node->set(dtype);  // causes an init
                }

                // check for inline json values
                if(jvalue.HasMember("value"))
                {
    
                    parse_inline_value(jvalue["value"],*node);
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
                // auto offset only makes sense when we have data
                if(data != NULL)
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
            // auto offset only makes sense when we have data
            if(data != NULL)
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
         
         if(data != NULL)
         {
             // node needs to link schema ptr 
             node->set(schema,data);
         }
         else
         {
             // sets the pointer
             node->set(schema); // properly links back to schema tree
             // we need to dynamically alloc
             node->set(dtype);  // causes an init
         }
    }
}


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
    
    if(m_protocol == "json")
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


};

