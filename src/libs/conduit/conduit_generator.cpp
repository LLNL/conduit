//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_generator.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_generator.hpp"


//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <stdio.h>

//-----------------------------------------------------------------------------
// -- rapidjson includes -- 
//-----------------------------------------------------------------------------
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "conduit_error.hpp"
#include "conduit_utils.hpp"


//-----------------------------------------------------------------------------
//
/// The CONDUIT_JSON_PARSE_ERROR macro use as a single place for handling
/// errors related to rapidjson parsing.
//
//-----------------------------------------------------------------------------
#define CONDUIT_JSON_PARSE_ERROR(json_str, document )                        \
{                                                                            \
    std::ostringstream __json_parse_oss;                                     \
    Generator::Parser::parse_error_details( json_str,                        \
                                            document,                        \
                                            __json_parse_oss);               \
    CONDUIT_ERROR("JSON parse error: \n"                                     \
                  << __json_parse_oss.str()                                  \
                  << "\n");                                                  \
}

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::Generator::Parser --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Generator::Parser handles parsing via rapidjson.
// We want to isolate the conduit API from the rapidjson headers
// so any methods using rapidjson types are defined in here.
// "friend Generator" in Node, allows Generator::Parser to construct complex
// nodes. 
//-----------------------------------------------------------------------------

 class Generator::Parser
{
public:
    static index_t json_to_numeric_dtype(const rapidjson::Value &jvalue);
    
    static index_t check_homogenous_json_array(const rapidjson::Value &jvalue);
    
    static void    parse_json_int64_array(const rapidjson::Value &jvalue,
                                          std::vector<int64> &res);
                                         
    static void    parse_json_int64_array(const rapidjson::Value &jvalue,
                                          Node &node);
                                          
    static void    parse_json_uint64_array(const rapidjson::Value &jvalue,
                                           std::vector<uint64> &res);
                                           
    static void    parse_json_uint64_array(const rapidjson::Value &jvalue,
                                           Node &node);
                                           
    static void    parse_json_float64_array(const rapidjson::Value &jvalue,
                                            std::vector<float64> &res);

    static void    parse_json_float64_array(const rapidjson::Value &jvalue,
                                            Node &node);
    static index_t parse_leaf_dtype_name(const std::string &dtype_name);
    
    static void    parse_leaf_dtype(const rapidjson::Value &jvalue,
                                    index_t offset,
                                    DataType &dtype_res);
                                    
    static void    parse_inline_leaf(const rapidjson::Value &jvalue,
                                     Node &node);
                                     
    static void    parse_inline_value(const rapidjson::Value &jvalue,
                                      Node &node);
                                      
    static void    walk_json_schema(Schema *schema,
                                    const   rapidjson::Value &jvalue,
                                    index_t curr_offset);
                                    
    static void    walk_pure_json_schema(Node  *node,
                                         Schema *schema,
                                         const rapidjson::Value &jvalue);

    static void    walk_json_schema(Node   *node,
                                    Schema *schema,
                                    void   *data,
                                    const rapidjson::Value &jvalue,
                                    index_t curr_offset);
    
    static void    parse_base64(Node *node,
                                const rapidjson::Value &jvalue);

    static void    parse_error_details(const std::string &json,
                                       const rapidjson::Document &document,
                                       std::ostream &os);

};

//---------------------------------------------------------------------------//
index_t 
Generator::Parser::json_to_numeric_dtype(const rapidjson::Value &jvalue)
{
    index_t res = DataType::EMPTY_ID; 
    if(jvalue.IsNumber())
    {
        // TODO: We could have better logic for dealing with int vs uint
        if(jvalue.IsUint64() || 
           jvalue.IsInt64()  || 
           jvalue.IsUint()   ||
           jvalue.IsInt())
        {
            res  = DataType::INT64_ID; // for int
        }
        else if(jvalue.IsDouble())
        {
            res  = DataType::FLOAT64_ID; // for float
        } 
        // else -- value already inited to EMPTY_ID
    }
    
    return res;
}

//---------------------------------------------------------------------------//
index_t
Generator::Parser::check_homogenous_json_array(const rapidjson::Value &jvalue)
{
    // check for homogenous array of ints or floats
    // promote to float64 as the most wide type
    // (this is a heuristic decision)

    if(jvalue.Size() == 0)
        return DataType::EMPTY_ID;

    index_t val_type = json_to_numeric_dtype(jvalue[(rapidjson::SizeType)0]); 
    bool homogenous  = (val_type != DataType::EMPTY_ID);

    for (rapidjson::SizeType i = 1; i < jvalue.Size() && homogenous; i++)
    {
        index_t curr_val_type = json_to_numeric_dtype(jvalue[i]);
        if(val_type == DataType::INT64_ID  &&
           curr_val_type ==  DataType::FLOAT64_ID)
        {
            // promote to a double (may be lossy in some cases)
            val_type = DataType::FLOAT64_ID;
        }
        else if(curr_val_type == DataType::EMPTY_ID)
        {
            // non homogenous inline
            homogenous = false;
            val_type = DataType::EMPTY_ID;
        }
    }

    return val_type;
}

//---------------------------------------------------------------------------// 
void
Generator::Parser::parse_json_int64_array(const rapidjson::Value &jvalue,
                                          std::vector<int64> &res)
{
   res.resize(jvalue.Size(),0);
   for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
   {
       res[i] = jvalue[i].GetInt64();
   }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_json_int64_array(const rapidjson::Value &jvalue,
                                          Node &node)
{
    // TODO: we can make this more efficient 
    std::vector<int64> vals;
    parse_json_int64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        // signed ints
        case DataType::INT8_ID:
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_ID:
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_ID:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_ID:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_ID:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_ID:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_ID:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_ID:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_ID:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_ID:
            node.as_float64_array().set(vals);
            break;
        default:
            CONDUIT_ERROR("JSON Generator error:\n"
                           << "attempting to set non-numeric Node with"
                           << " int64 array");
            break;
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_json_uint64_array(const rapidjson::Value &jvalue,
                                           std::vector<uint64> &res)
{
    res.resize(jvalue.Size(),0);
    for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetUint64();
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_json_uint64_array(const rapidjson::Value &jvalue,
                                           Node &node)
{
    // TODO: we can make this more efficient 
    std::vector<uint64> vals;
    parse_json_uint64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        // signed ints
        case DataType::INT8_ID:
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_ID:
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_ID:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_ID:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_ID:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_ID:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_ID:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_ID:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_ID:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_ID:
            node.as_float64_array().set(vals);
            break;
        default:
            CONDUIT_ERROR("JSON Generator error:\n"
                           << "attempting to set non-numeric Node with"
                           << " uint64 array");
            break;
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_json_float64_array(const rapidjson::Value &jvalue,
                                            std::vector<float64> &res)
{
    res.resize(jvalue.Size(),0);
    for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetDouble();
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_json_float64_array(const rapidjson::Value &jvalue,
                                            Node &node)
{
    // TODO: we can make this more efficient 
    std::vector<float64> vals;
    parse_json_float64_array(jvalue,vals);
    
    switch(node.dtype().id())
    {
        case DataType::INT8_ID:
            node.as_int8_array().set(vals);
            break;
        case DataType::INT16_ID:
            node.as_int16_array().set(vals);
            break;
        case DataType::INT32_ID:
            node.as_int32_array().set(vals);
            break;
        case DataType::INT64_ID:
            node.as_int64_array().set(vals);
            break;
        // unsigned ints
        case DataType::UINT8_ID:
            node.as_uint8_array().set(vals);
            break;
        case DataType::UINT16_ID:
            node.as_uint16_array().set(vals);
            break;
        case DataType::UINT32_ID:
            node.as_uint32_array().set(vals);
            break;
        case DataType::UINT64_ID:
            node.as_uint64_array().set(vals);
            break;  
        //floats
        case DataType::FLOAT32_ID:
            node.as_float32_array().set(vals);
            break;
        case DataType::FLOAT64_ID:
            node.as_float64_array().set(vals);
            break;
        default:
            CONDUIT_ERROR("JSON Generator error:\n"
                           << "attempting to set non-numeric Node with"
                           << " float64 array");
            break;
    }
}


//---------------------------------------------------------------------------//
index_t 
Generator::Parser::parse_leaf_dtype_name(const std::string &dtype_name)
{
    index_t dtype_id = DataType::name_to_id(dtype_name);
    if(dtype_id == DataType::EMPTY_ID)
    {
        // also try native type names
        dtype_id = DataType::c_type_name_to_id(dtype_name);
    }

    // do an explicit check for empty
    if(dtype_id == DataType::EMPTY_ID && dtype_name != "empty")
    {
        CONDUIT_ERROR("JSON Generator error:\n"
                       << "invalid leaf type "
                       << "\""  <<  dtype_name << "\"");
    }
    return dtype_id;
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_leaf_dtype(const rapidjson::Value &jvalue,
                                    index_t offset,
                                    DataType &dtype_res)
{
    
    if(jvalue.IsString())
    {
        std::string dtype_name(jvalue.GetString());
        index_t dtype_id = parse_leaf_dtype_name(dtype_name);
        index_t ele_size = DataType::default_bytes(dtype_id);
        dtype_res.set(dtype_id,
                      1,
                      offset,
                      ele_size,
                      ele_size,
                      Endianness::DEFAULT_ID);
    }
    else if(jvalue.IsObject())
    {
        CONDUIT_ASSERT( ( jvalue.HasMember("dtype") && jvalue["dtype"].IsString() ),
                        "JSON Generator error:\n"
                         << "'dtype' must be a JSON string.");
            
        std::string dtype_name(jvalue["dtype"].GetString());
        
        index_t length=0;
        
        if(jvalue.HasMember("number_of_elements"))
        {
            const rapidjson::Value &json_num_eles = jvalue["number_of_elements"];
            if(json_num_eles.IsNumber())
            {              
                length = json_num_eles.GetUint64();
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                               << "'number_of_elements' must be a number ");
            }
        }
        //
        // DEPRECATE
        //
        // length is the old schema style, we should deprecate this path
        else if(jvalue.HasMember("length"))
        {
            const rapidjson::Value &json_len = jvalue["length"];
            if(json_len.IsNumber())
            {              
                length = json_len.GetUint64();
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                               << "'length' must be a number ");
            }
        }

        index_t dtype_id  = parse_leaf_dtype_name(dtype_name);
        index_t ele_size  = DataType::default_bytes(dtype_id);
        index_t stride    = ele_size;
    
        //  parse offset (override default if passed)
        if(jvalue.HasMember("offset"))
        {
            const rapidjson::Value &json_offset = jvalue["offset"];
            
            if(json_offset.IsNumber())
            {
                offset = json_offset.GetUint64();
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                              << "'offset' must be a number ");
            }
        }

        // parse stride (override default if passed)
        if(jvalue.HasMember("stride") )
        {
            const rapidjson::Value &json_stride = jvalue["stride"];
            
            if(json_stride.IsNumber())
            {
                stride = json_stride.GetUint64();
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                              << "'stride' must be a number ");
            }
        }

        // parse element_bytes (override default if passed)
        if(jvalue.HasMember("element_bytes") )
        {
            const rapidjson::Value &json_ele_bytes = jvalue["element_bytes"];
            
            if(json_ele_bytes.IsNumber())
            {
                ele_size = json_ele_bytes.GetUint64();
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                              << "'element_bytes' must be a number ");
            }
        }
    
    
        // parse endianness (override default if passed)
        index_t endianness = Endianness::DEFAULT_ID;
        if(jvalue.HasMember("endianness"))
        {
            const rapidjson::Value &json_endianness = jvalue["endianness"];
            if(json_endianness.IsString())
            {
                std::string end_val(json_endianness.GetString());
                if(end_val == "big")
                {
                    endianness = Endianness::BIG_ID;
                }
                else if(end_val == "little")
                {
                    endianness = Endianness::LITTLE_ID;
                }
                else
                {
                    CONDUIT_ERROR("JSON Generator error:\n"
                              << "'endianness' must be a string"
                              << " (\"big\" or \"little\")"
                              << " parsed value: " << end_val);
                }
            }
            else
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                          << "'endianness' must be a string"
                          << " (\"big\" or \"little\")");
            }
        }
    
        if(length == 0)
        {
            if(jvalue.HasMember("value") &&
               jvalue["value"].IsArray())
            {
                length = jvalue["value"].Size();
            }
            // support explicit length 0 in a schema
            else if(!jvalue.HasMember("length") && 
                    !jvalue.HasMember("number_of_elements"))
            {
                length = 1;
            }
        }
    
        dtype_res.set(dtype_id,
                      length,
                      offset,
                      stride, 
                      ele_size,
                      endianness);
    }
    else
    {
        CONDUIT_ERROR("JSON Generator error:\n"
                       << "a leaf dtype entry must be a JSON string or"
                       <<  " JSON object.");
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_inline_leaf(const rapidjson::Value &jvalue,
                                     Node &node)
{
    if(jvalue.IsString())
    {
        if(node.dtype().id() == DataType::CHAR8_STR_ID)
        {
            std::string sval(jvalue.GetString());
            node.set(utils::unescape_special_chars(sval));
        }
        else
        {
             // JSON type incompatible with char8_str
             // only allow strings to be assigned to a char8_str type
             // throw parsing error if our inline values
             // don't match what we expected

            CONDUIT_ERROR("JSON Generator error:\n"
                           << "a JSON string can only be used as an inline"
                           << " value for a Conduit CHAR8_STR Node.");
        }
    }
    else if(jvalue.IsBool())
    {
        //
        if(node.dtype().id() == DataType::UINT8_ID)
        {
            node.set((uint8)jvalue.GetBool());
        }
        else
        {
             // JSON type incompatible with uint8
             // only allow JSON bools to be assigned to a uint8 type
             // throw parsing error if our inline values
             // don't match what we expected
            
            CONDUIT_ERROR("JSON Generator error:\n"
                           << "a JSON bool can only be used as an inline"
                           << " value for a Conduit UINT8 Node.");
            
        }
    }
    else if(jvalue.IsNumber())
    {
        switch(node.dtype().id())
        {
            // signed ints
            case DataType::INT8_ID:   
                node.set((int8)jvalue.GetInt64());
                break;
            case DataType::INT16_ID: 
                node.set((int16)jvalue.GetInt64());
                break;
            case DataType::INT32_ID:
                node.set((int32)jvalue.GetInt64());
                break;
            case DataType::INT64_ID:
                node.set((int64)jvalue.GetInt64());
                break;
            // unsigned ints
            case DataType::UINT8_ID:
                node.set((uint8)jvalue.GetUint64());
                break;
            case DataType::UINT16_ID:
                node.set((uint16)jvalue.GetUint64());
                break;
            case DataType::UINT32_ID:
                node.set((uint32)jvalue.GetUint64());
                break;
            case DataType::UINT64_ID:
                node.set((uint64)jvalue.GetUint64());
                break;  
            //floats
            case DataType::FLOAT32_ID:
                node.set((float32)jvalue.GetDouble());
                break;
            case DataType::FLOAT64_ID:
                node.set((float64)jvalue.GetDouble());
                break;
            default:
                // JSON type incompatible with numeric
                // only allow numeric to be assigned to a numeric type
                // throw parsing error if our inline values
                // don't match what we expected
                CONDUIT_ERROR("JSON Generator error:\n"
                              << "a JSON number can only be used as an inline"
                              << " value for a Conduit Numeric Node.");
                break;
        }
    }
    else if(jvalue.IsNull())
    {
        // empty data type
        node.reset();
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::parse_inline_value(const rapidjson::Value &jvalue,
                                      Node &node)
{
    if(jvalue.IsArray())
    {
        // we assume a "value" is a leaf or list of compatible leaves
        index_t hval_type = check_homogenous_json_array(jvalue);
        
        CONDUIT_ASSERT( (node.dtype().number_of_elements() >= jvalue.Size() ),
                       "JSON Generator error:\n" 
                        << "number of elements in JSON array is more"
                        << "than dtype can hold");
        
        if(hval_type == DataType::INT64_ID)
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
        else if(hval_type == DataType::FLOAT64_ID)
        {
            parse_json_float64_array(jvalue,node);
        }
        else
        {
            // Parsing Error, not homogenous
            CONDUIT_ERROR("JSON Generator error:\n"
                        << "a JSON array for value initialization"
                        << " is not homogenous");
        }
    }
    else
    {
        parse_inline_leaf(jvalue,node);
    }
}


//---------------------------------------------------------------------------//
void 
Generator::Parser::walk_json_schema(Schema *schema,
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
                    const rapidjson::Value &len_value = jvalue["length"];
                    if(len_value.IsObject() && 
                       len_value.HasMember("reference"))
                    {
                        CONDUIT_ERROR("JSON Generator error:\n"
                                      << "'reference' option is not supported"
                                      << " when parsing to a Schema because"
                                      << " reference data does not exist.");
                    }
                    else if(len_value.IsNumber())
                    {
                        length = len_value.GetInt();
                    }
                    else
                    {
                        CONDUIT_ERROR("JSON Generator error:\n"
                                      << "'length' must be a JSON Object or"
                                      << " JSON number");
                    }
                }
                // we will create `length' # of objects of obj des by dt_value
                 
                // TODO: we only need to parse this once, not leng # of times
                // but this is the easiest way to start.
                for(int i=0;i< length;i++)
                {
                    Schema &curr_schema =schema->append();
                    curr_schema.set(DataType::list());
                    walk_json_schema(&curr_schema,dt_value, curr_offset);
                    curr_offset += curr_schema.total_strided_bytes();
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
            // if we make it here and have an empty json object
            // we still want the conduit schema to take on the
            // object role
            schema->set(DataType::object());
            
            // loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr =
                 jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Schema &curr_schema = schema->fetch(entry_name);
                curr_schema.set(DataType::object());
                walk_json_schema(&curr_schema,itr->value, curr_offset);
                curr_offset += curr_schema.total_strided_bytes();
            }
        }
    }
    // List case 
    else if(jvalue.IsArray()) 
    { 
        // if we make it here and have an empty json list
        // we still want the conduit schema to take on the
        // list role
        schema->set(DataType::list());

        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {

            Schema &curr_schema = schema->append();
            curr_schema.set(DataType::list());
            walk_json_schema(&curr_schema,jvalue[i], curr_offset);
            curr_offset += curr_schema.total_strided_bytes();
        }
    }
    // Simplest case, handles "uint32", "float64", etc
    else if(jvalue.IsString())
    {
        DataType dtype;
        parse_leaf_dtype(jvalue,curr_offset,dtype);
        schema->set(dtype);
    }
    else
    {
        CONDUIT_ERROR("JSON Generator error:\n"
                      << "Invalid JSON type for parsing Schema."
                      << "Expected: JSON Object, Array, or String");
    }
}

//---------------------------------------------------------------------------//
void 
Generator::Parser::walk_pure_json_schema(Node *node,
                                         Schema *schema,
                                         const rapidjson::Value &jvalue)
{
    // object cases
    if(jvalue.IsObject())
    {
        // if we make it here and have an empty json object
        // we still want the conduit node to take on the
        // object role
        schema->set(DataType::object());
        // loop over all entries
        for (rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
             itr != jvalue.MemberEnd(); ++itr)
        {
            std::string entry_name(itr->name.GetString());
            
            // json files may have duplicate object names
            // we could provide some clear semantics, such as:
            //   always use first instance, or always use last instance
            // however duplicate object names are most likely a
            // typo, so it's best to throw an error

            if(schema->has_child(entry_name))
            {
                CONDUIT_ERROR("JSON Generator error:\n"
                              << "Duplicate JSON object name: " 
                              << "\"" << entry_name << "\"");
            }

            Schema *curr_schema = schema->fetch_ptr(entry_name);

            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            
            walk_pure_json_schema(curr_node,
                                  curr_schema,
                                  itr->value);
        
            node->append_node_ptr(curr_node);
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        index_t hval_type = check_homogenous_json_array(jvalue);
        if(hval_type == DataType::INT64_ID)
        {
            std::vector<int64> res;
            parse_json_int64_array(jvalue,res);
            node->set(res);
        }
        else if(hval_type == DataType::FLOAT64_ID)
        {
            std::vector<float64> res;
            parse_json_float64_array(jvalue,res);
            node->set(res);
        }
        else // not numeric array
        {
            // if we make it here and have an empty json list
            // we still want the conduit node to take on the
            // list role
            schema->set(DataType::list());
            
            for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
            {
                schema->append();
                Schema *curr_schema = schema->child_ptr(i);
                Node * curr_node = new Node();
                curr_node->set_schema_ptr(curr_schema);
                curr_node->set_parent(node);
                walk_pure_json_schema(curr_node,curr_schema,jvalue[i]);
                node->append_node_ptr(curr_node);
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
        // we store bools as uint8s
        if(jvalue.IsTrue())
        {
            node->set((uint8)1);
        }
        else
        {
            node->set((uint8)0);
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
    else
    {
        // not sure if can an even land here, but catch error just in case.
        CONDUIT_ERROR("JSON Generator error:\n"
                      << "Invalid JSON type for parsing Node from pure JSON."
                      << " Expected: JSON Object, Array, String, Null,"
                      << " Boolean, or Number");
    }
}


//---------------------------------------------------------------------------//
void 
Generator::Parser::walk_json_schema(Node   *node,
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
                index_t length =1;
                if(jvalue.HasMember("length"))
                {
                    if(jvalue["length"].IsNumber())
                    {
                        length = jvalue["length"].GetInt();
                    }
                    else if(jvalue["length"].IsObject() && 
                            jvalue["length"].HasMember("reference"))
                    {
                        std::string ref_path = 
                          jvalue["length"]["reference"].GetString();
                        length = node->fetch(ref_path).to_index_t();
                    }
                    else
                    {
                        CONDUIT_ERROR("JSON Parsing error:\n"
                                      << "'length' must be a number "
                                      << "or reference.");
                    }
                    
                }
                // we will create `length' # of objects of obj des by dt_value
                 
                // TODO: we only need to parse this once, not leng # of times
                // but this is the easiest way to start.
                for(index_t i=0;i< length;i++)
                {
                    schema->append();
                    Schema *curr_schema = schema->child_ptr(i);
                    Node *curr_node = new Node();
                    curr_node->set_schema_ptr(curr_schema);
                    curr_node->set_parent(node);
                    walk_json_schema(curr_node,
                                     curr_schema,
                                     data,
                                     dt_value,
                                     curr_offset);
                    // auto offset only makes sense when we have data
                    if(data != NULL)
                        curr_offset += curr_schema->total_strided_bytes();
                    node->append_node_ptr(curr_node);
                }
                
            }
            else
            {
                // handle leaf node with explicit props
                DataType dtype;
                
                parse_leaf_dtype(jvalue,curr_offset,dtype);
   
                if(data != NULL)
                {
                    // node is already linked to the schema pointer
                    schema->set(dtype);
                    node->set_data_ptr(data);
                }
                else
                {
                    // node is already linked to the schema pointer
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
        else // object case
        {
            schema->set(DataType::object());
            // standard object case - loop over all entries
            for (rapidjson::Value::ConstMemberIterator itr = 
                 jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                
                // json files may have duplicate object names
                // we could provide some clear semantics, such as:
                //   always use first instance, or always use last instance.
                // however duplicate object names are most likely a
                // typo, so it's best to throw an error
                //
                // also its highly unlikely that the auto offset case
                // could safely deal with offsets for the
                // duplicate key case

                if(schema->has_child(entry_name))
                {
                    CONDUIT_ERROR("JSON Generator error:\n"
                                  << "Duplicate JSON object name: " 
                                  << "\"" << entry_name << "\"");
                }

                Schema *curr_schema = schema->fetch_ptr(entry_name);
                
                Node *curr_node = new Node();
                curr_node->set_schema_ptr(curr_schema);
                curr_node->set_parent(node);
                
                walk_json_schema(curr_node,
                                 curr_schema,
                                 data,
                                 itr->value,
                                 curr_offset);
                
                // auto offset only makes sense when we have data
                if(data != NULL)
                    curr_offset += curr_schema->total_strided_bytes();
                
                node->append_node_ptr(curr_node);
            }
            
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        schema->set(DataType::list());

        for (rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            schema->append();
            Schema *curr_schema = schema->child_ptr(i);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            walk_json_schema(curr_node,
                             curr_schema,
                             data,
                             jvalue[i],
                             curr_offset);
            // auto offset only makes sense when we have data
            if(data != NULL)
                curr_offset += curr_schema->total_strided_bytes();
            node->append_node_ptr(curr_node);
        }
        
    }
    // Simplest case, handles "uint32", "float64", with extended type info
    else if(jvalue.IsString())
    {
        DataType dtype;
        parse_leaf_dtype(jvalue,curr_offset,dtype);
        schema->set(dtype);
        
        if(data != NULL)
        {
             // node is already linked to the schema pointer
             node->set_data_ptr(data);
             
        }
        else
        {
             // node is already linked to the schema pointer
             // we need to dynamically alloc
             node->set(dtype);  // causes an init
        }
    }
    else
    {
        CONDUIT_ERROR("JSON Generator error:\n"
                      << "Invalid JSON type for parsing Node."
                      << " Expected: JSON Object, Array, or String");
    }
}

//---------------------------------------------------------------------------//
void 
Generator::Parser::parse_base64(Node *node,
                                const rapidjson::Value &jvalue)
{
    // object case

    std::string base64_str = "";
    
    if(jvalue.IsObject())
    {
        Schema s;
        if (jvalue.HasMember("data") && jvalue["data"].HasMember("base64"))
        {
            base64_str = jvalue["data"]["base64"].GetString();
        }
        else
        {
            CONDUIT_ERROR("conduit_base64_json protocol error: missing data/base64");
        }
        
        if (jvalue.HasMember("schema"))
        {
            // parse schema
            index_t curr_offset = 0;
            Parser::walk_json_schema(&s,jvalue["schema"],curr_offset);
        }
        else
        {
            CONDUIT_ERROR("conduit_base64_json protocol error: missing schema");
        }
        
        const char *src_ptr = base64_str.c_str();
        index_t encoded_len = (index_t) base64_str.length();
        index_t dec_buff_size = utils::base64_decode_buffer_size(encoded_len);

        // decode buffer
        Node bb64_decode;
        bb64_decode.set(DataType::char8_str(dec_buff_size));
        char *decode_ptr = (char*)bb64_decode.data_ptr();
        memset(decode_ptr,0,dec_buff_size);

        utils::base64_decode(src_ptr,
                             encoded_len,
                             decode_ptr);

        node->set(s,decode_ptr);

    }
    else
    {
        CONDUIT_ERROR("conduit_base64_json protocol error: missing schema and data/base64");
    }
}

//---------------------------------------------------------------------------//
void 
Generator::Parser::parse_error_details(const std::string &json,
                                       const rapidjson::Document &document,
                                       std::ostream &os)
{
    // provide message with line + char from rapidjson parse error offset 
    index_t doc_offset = (index_t)document.GetErrorOffset();
    std::string json_curr = json.substr(0,doc_offset);

    std::string curr = "";
    std::string next = " ";
    
    index_t doc_line   = 0;
    index_t doc_char   = 0;

    while(!next.empty())
    {
        utils::split_string(json_curr, "\n", curr, next);
        doc_char = curr.size();
        json_curr = next;
        if(!next.empty())
        {
            doc_line++;
        }
    }

    os << " parse error message:\n"
       << GetParseError_En(document.GetParseError()) << "\n"
       << " offset: "    << doc_offset << "\n"
       << " line: "      << doc_line << "\n"
       << " character: " << doc_char << "\n"
       << " json:\n"     << json << "\n"; 
}

//-----------------------------------------------------------------------------
// -- end conduit::Generator::Parser --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit::Generator --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Generator Construction and Destruction
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Generator::Generator()
:m_json_schema(""),
 m_protocol("conduit_json"),
 m_data(NULL)
{}


//---------------------------------------------------------------------------//
Generator::Generator(const std::string &json_schema,
                     const std::string &protocol,
                     void *data)
:m_json_schema(json_schema),
 m_protocol(protocol),
 m_data(data)
{}

//---------------------------------------------------------------------------//
void
Generator::set_json_schema(const std::string &json_schema)
{
    m_json_schema = json_schema;
}

//---------------------------------------------------------------------------//
void
Generator::set_protocol(const std::string &protocol)
{
    m_protocol = protocol;
}

//---------------------------------------------------------------------------//
void
Generator::set_data_ptr(void *data_ptr)
{
    m_data = data_ptr;
}

//---------------------------------------------------------------------------//
const std::string &
Generator::json_schema() const
{
    return m_json_schema;
}

//---------------------------------------------------------------------------//
const std::string &
Generator::protocol() const
{
    return m_protocol;
}

//---------------------------------------------------------------------------//
void *
Generator::data_ptr() const
{
    return m_data;
}


//-----------------------------------------------------------------------------
// JSON Parsing interface
//-----------------------------------------------------------------------------s

const rapidjson::ParseFlag RAPIDJSON_PARSE_OPTS = rapidjson::kParseNoFlags;

//---------------------------------------------------------------------------//
void 
Generator::walk(Schema &schema) const
{
    schema.reset();
    rapidjson::Document document;
    std::string res = utils::json_sanitize(m_json_schema);

    if(document.Parse<RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
    {
        CONDUIT_JSON_PARSE_ERROR(res, document);
    }
    index_t curr_offset = 0;
    Parser::walk_json_schema(&schema,document,curr_offset);
}

//---------------------------------------------------------------------------//
void 
Generator::walk(Node &node) const
{
    /// TODO: This is an inefficient code path, need better solution?
    Node n;
    walk_external(n);
    n.compact_to(node);
}

//---------------------------------------------------------------------------//
void 
Generator::walk_external(Node &node) const
{
    node.reset();
    // if data is null, we can parse the schema via the other 'walk' method
    if(m_protocol == "json")
    {
        rapidjson::Document document;
        std::string res = utils::json_sanitize(m_json_schema);
                
        if(document.Parse<RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }

        Parser::walk_pure_json_schema(&node,
                                      node.schema_ptr(),
                                      document);
    }
    else if( m_protocol == "conduit_base64_json")
    {
        rapidjson::Document document;
        std::string res = utils::json_sanitize(m_json_schema);
        
        if(document.Parse<RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }
        Parser::parse_base64(&node,
                             document);
    }
    else if( m_protocol == "conduit_json")
    {
        rapidjson::Document document;
        std::string res = utils::json_sanitize(m_json_schema);
        
        if(document.Parse<RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }
        index_t curr_offset = 0;

        Parser::walk_json_schema(&node,
                                 node.schema_ptr(),
                                 m_data,
                                 document,
                                 curr_offset);
    }
    else
    {
        CONDUIT_ERROR("Generator unknown parsing protocol: " << m_protocol);
    }
}


//-----------------------------------------------------------------------------
// -- end conduit::Generator --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

