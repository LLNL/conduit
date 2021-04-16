// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
#include <cstdlib>

//-----------------------------------------------------------------------------
// -- rapidjson includes -- 
//-----------------------------------------------------------------------------
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

//-----------------------------------------------------------------------------
// -- libyaml includes -- 
//-----------------------------------------------------------------------------
#include "yaml.h"


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
    Generator::Parser::JSON::parse_error_details( json_str,                  \
                                                 document,                   \
                                                __json_parse_oss);           \
    CONDUIT_ERROR("JSON parse error: \n"                                     \
                  << __json_parse_oss.str()                                  \
                  << "\n");                                                  \
}


//-----------------------------------------------------------------------------
//
/// The CONDUIT_YAML_PARSE_ERROR macro use as a single place for handling
/// errors related to libyaml parsing.
//
//-----------------------------------------------------------------------------
#define CONDUIT_YAML_PARSE_ERROR( yaml_doc, yaml_parser )                    \
{                                                                            \
    std::ostringstream __yaml_parse_oss;                                     \
    Generator::Parser::YAML::parse_error_details( yaml_parser,               \
                                                __yaml_parse_oss);           \
    CONDUIT_ERROR("YAML parse error: \n"                                     \
                  << __yaml_parse_oss.str()                                  \
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
// Generator::Parser -- concrete parsing implementations
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class Generator::Parser
{
public:
//-----------------------------------------------------------------------------
// Generator::Parser::JSON handles parsing via rapidjson.
// We want to isolate the conduit API from the rapidjson headers
// so any methods using rapidjson types are defined in here.
// "friend Generator" in Node, allows Generator::Parser to construct complex
// nodes. 
//-----------------------------------------------------------------------------
  class JSON
  {
  public:
      
    static const conduit_rapidjson::ParseFlag RAPIDJSON_PARSE_OPTS = conduit_rapidjson::kParseNoFlags;
    
    static index_t json_to_numeric_dtype(const conduit_rapidjson::Value &jvalue);
    
    static index_t check_homogenous_json_array(const conduit_rapidjson::Value &jvalue);
    
    static void    parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
                                          std::vector<int64> &res);

    // for efficiency - assumes res is already alloced to proper size
    static void    parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
                                          int64_array &res);

    static void    parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
                                          Node &node);
                                          
    static void    parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
                                           std::vector<uint64> &res);

    // for efficiency - assumes res is already alloced to proper size
    static void    parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
                                          uint64_array &res);

                                           
    static void    parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
                                           Node &node);
                                           
    static void    parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
                                            std::vector<float64> &res);

    // for efficiency - assumes res is already alloced to proper size
    static void    parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
                                            float64_array &res);

    static void    parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
                                            Node &node);
    static index_t parse_leaf_dtype_name(const std::string &dtype_name);
    
    static void    parse_leaf_dtype(const conduit_rapidjson::Value &jvalue,
                                    index_t offset,
                                    DataType &dtype_res);
                                    
    static void    parse_inline_leaf(const conduit_rapidjson::Value &jvalue,
                                     Node &node);
                                     
    static void    parse_inline_value(const conduit_rapidjson::Value &jvalue,
                                      Node &node);
                                      
    static void    walk_json_schema(Schema *schema,
                                    const   conduit_rapidjson::Value &jvalue,
                                    index_t curr_offset);
                                    
    static void    walk_pure_json_schema(Node  *node,
                                         Schema *schema,
                                         const conduit_rapidjson::Value &jvalue);

    static void    walk_json_schema(Node   *node,
                                    Schema *schema,
                                    void   *data,
                                    const conduit_rapidjson::Value &jvalue,
                                    index_t curr_offset);
    
    static void    parse_base64(Node *node,
                                const conduit_rapidjson::Value &jvalue);

    static void    parse_error_details(const std::string &json,
                                       const conduit_rapidjson::Document &document,
                                       std::ostream &os);
  };
//-----------------------------------------------------------------------------
// Generator::Parser::YAML handles parsing via libyaml.
// We want to isolate the conduit API from the libyaml headers
// so any methods using libyaml types are defined in here.
// "friend Generator" in Node, allows Generator::Parser to construct complex
// nodes. 
//-----------------------------------------------------------------------------
  class YAML
  {
  public:


    //-----------------------------------------------------------------------------
    // Wrappers around libyaml c API that help us parse
    //-----------------------------------------------------------------------------
     
    // YAMLParserWrapper class helps with libyaml cleanup when 
    // exceptions are thrown during parsing
    class YAMLParserWrapper
    {
    public:
        YAMLParserWrapper();
       ~YAMLParserWrapper();

       // parses and creates doc tree, throws exception
       // when things go wrong
       void         parse(const char *yaml_txt);

       yaml_document_t *yaml_doc_ptr();
       yaml_node_t     *yaml_doc_root_ptr();

    private:
        yaml_document_t m_yaml_doc;
        yaml_parser_t   m_yaml_parser;

        bool m_yaml_parser_is_valid;
        bool m_yaml_doc_is_valid;

    };

    // 
    // yaml scalar (aka leaf) values are always strings, however that is
    // not a very useful way to parse into Conduit tree. We apply json
    // rules to the yaml leaves to get more useful types in Conduit
    //
    // excluded from the JSON-like rules are:
    //  boolean literals (true, false)
    //  the null literal (null)
    // 
    // This is b/c we can't distinguish between string values like
    //    "true"
    // vs non-quoted literals like 
    //    true
    // with the yaml parser
    //

    // TODO: try to inline these helpers? Not sure it matters since
    // they call other routines

    // checks if c-string is a null pointer or empty
    static bool string_is_empty(const char *txt_value);

    // checks if input c-string is an integer or a double
    static bool string_is_number(const char *txt_value);

    // checks if input string holds something that converts
    // to a double (integer c-string will pass this check )
    static bool string_is_double(const char *txt_value);

    // checks if input c-string holds something that converts
    // to an integer
    static bool string_is_integer(const char *txt_value);

    // converts c-string to double
    static double string_to_double(const char *txt_value);
    // converts c-string to long
    static long int string_to_long(const char *txt_value);

    // assumes res is already inited to DataType::int64 w/ proper size
    static void parse_yaml_int64_array(yaml_document_t *yaml_doc,
                                       yaml_node_t *yaml_node,
                                       Node &res);

    // assumes res is already inited to DataType::float64 w/ proper size
    static void parse_yaml_float64_array(yaml_document_t *yaml_doc,
                                         yaml_node_t *yaml_node,
                                         Node &res);
    // parses generic leaf and places value in res
    static void parse_yaml_inline_leaf(const char *yaml_txt,
                                       Node &res);

    // finds if leaf string is int64, float64, or neither (DataType::EMPTY_T)
    static index_t yaml_leaf_to_numeric_dtype(const char *txt_value);

    // checks if the input yaml node is a homogenous numeric sequence
    // 
    // if not: returns DataType::EMPTY_T and seq_size = -1
    //
    // if so:
    //  seq_size contains the sequence length and:
    //  if homogenous integer sequence returns DataType::INT64_T 
    //  if homogenous floating point sequence returns DataType::FLOAT64_T 
    static index_t check_homogenous_yaml_numeric_sequence(const Node &node,
                                                          yaml_document_t *yaml_doc,
                                                          yaml_node_t *yaml_node,
                                                          index_t &seq_size);

    // main entry point for parsing pure yaml
    static void    walk_pure_yaml_schema(Node  *node,
                                         Schema *schema,
                                         const char *yaml_txt);

    // workhorse for parsing a pure yaml tree
    static void    walk_pure_yaml_schema(Node  *node,
                                         Schema *schema,
                                         yaml_document_t *yaml_doc,
                                         yaml_node_t *yaml_node);
    
    // extract human readable parser errors
    static void    parse_error_details(yaml_parser_t *yaml_parser,
                                       std::ostream &os);

  };

};

//-----------------------------------------------------------------------------
// -- begin conduit::Generator::Parser::JSON --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t 
Generator::Parser::JSON::json_to_numeric_dtype(const conduit_rapidjson::Value &jvalue)
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
Generator::Parser::JSON::check_homogenous_json_array(const conduit_rapidjson::Value &jvalue)
{
    // check for homogenous array of ints or floats
    // promote to float64 as the most wide type
    // (this is a heuristic decision)

    if(jvalue.Size() == 0)
        return DataType::EMPTY_ID;

    index_t val_type = json_to_numeric_dtype(jvalue[(conduit_rapidjson::SizeType)0]); 
    bool homogenous  = (val_type != DataType::EMPTY_ID);

    for (conduit_rapidjson::SizeType i = 1; i < jvalue.Size() && homogenous; i++)
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
Generator::Parser::JSON::parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
                                                std::vector<int64> &res)
{
   res.resize(jvalue.Size(),0);
   for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
   {
       res[i] = jvalue[i].GetInt64();
   }
}

//---------------------------------------------------------------------------// 
void
Generator::Parser::JSON::parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
                                                int64_array &res)
{
    // for efficiency - assumes res is already alloced to proper size
    for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
       res[i] = jvalue[i].GetInt64();
    }
}


//---------------------------------------------------------------------------//
void
Generator::Parser::JSON::parse_json_int64_array(const conduit_rapidjson::Value &jvalue,
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
Generator::Parser::JSON::parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
                                                 std::vector<uint64> &res)
{
    res.resize(jvalue.Size(),0);
    for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetUint64();
    }
}

//---------------------------------------------------------------------------// 
void
Generator::Parser::JSON::parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
                                                 uint64_array &res)
{
    // for efficiency - assumes res is already alloced to proper size
    for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
       res[i] = jvalue[i].GetUint64();
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::JSON::parse_json_uint64_array(const conduit_rapidjson::Value &jvalue,
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
Generator::Parser::JSON::parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
                                                  std::vector<float64> &res)
{
    res.resize(jvalue.Size(),0);
    for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
        res[i] = jvalue[i].GetDouble();
    }
}

//---------------------------------------------------------------------------// 
void
Generator::Parser::JSON::parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
                                                  float64_array &res)
{
    // for efficiency - assumes res is already alloced to proper size
    for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
    {
       res[i] = jvalue[i].GetDouble();
    }
}


//---------------------------------------------------------------------------//
void
Generator::Parser::JSON::parse_json_float64_array(const conduit_rapidjson::Value &jvalue,
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
Generator::Parser::JSON::parse_leaf_dtype_name(const std::string &dtype_name)
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
Generator::Parser::JSON::parse_leaf_dtype(const conduit_rapidjson::Value &jvalue,
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
            const conduit_rapidjson::Value &json_num_eles = jvalue["number_of_elements"];
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
            const conduit_rapidjson::Value &json_len = jvalue["length"];
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
            const conduit_rapidjson::Value &json_offset = jvalue["offset"];
            
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
            const conduit_rapidjson::Value &json_stride = jvalue["stride"];
            
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
            const conduit_rapidjson::Value &json_ele_bytes = jvalue["element_bytes"];
            
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
            const conduit_rapidjson::Value &json_endianness = jvalue["endianness"];
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
Generator::Parser::JSON::parse_inline_leaf(const conduit_rapidjson::Value &jvalue,
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
Generator::Parser::JSON::parse_inline_value(const conduit_rapidjson::Value &jvalue,
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
Generator::Parser::JSON::walk_json_schema(Schema *schema,
                                          const   conduit_rapidjson::Value &jvalue,
                                          index_t curr_offset)
{
    // object cases
    if(jvalue.IsObject())
    {
        if (jvalue.HasMember("dtype"))
        {
            // if dtype is an object, we have a "list_of" case
            const conduit_rapidjson::Value &dt_value = jvalue["dtype"];
            if(dt_value.IsObject())
            {
                int length =1;
                if(jvalue.HasMember("length"))
                {
                    const conduit_rapidjson::Value &len_value = jvalue["length"];
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
            for (conduit_rapidjson::Value::ConstMemberIterator itr =
                 jvalue.MemberBegin(); 
                 itr != jvalue.MemberEnd(); ++itr)
            {
                std::string entry_name(itr->name.GetString());
                Schema &curr_schema = schema->add_child(entry_name);
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

        for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
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
Generator::Parser::JSON::walk_pure_json_schema(Node *node,
                                               Schema *schema,
                                               const conduit_rapidjson::Value &jvalue)
{
    // object cases
    if(jvalue.IsObject())
    {
        // if we make it here and have an empty json object
        // we still want the conduit node to take on the
        // object role
        schema->set(DataType::object());
        // loop over all entries
        for (conduit_rapidjson::Value::ConstMemberIterator itr = jvalue.MemberBegin(); 
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
                              << utils::join_path(node->path(),entry_name));
            }

            Schema *curr_schema = &schema->add_child(entry_name);

            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            node->append_node_ptr(curr_node);

            walk_pure_json_schema(curr_node,
                                  curr_schema,
                                  itr->value);
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        index_t hval_type = check_homogenous_json_array(jvalue);
        if(hval_type == DataType::INT64_ID)
        {
            node->set(DataType::int64(jvalue.Size()));
            int64_array vals = node->value();
            parse_json_int64_array(jvalue,vals);
        }
        else if(hval_type == DataType::FLOAT64_ID)
        {
            node->set(DataType::float64(jvalue.Size()));
            float64_array vals = node->value();
            parse_json_float64_array(jvalue,vals);
        }
        else // not numeric array
        {
            // if we make it here and have an empty json list
            // we still want the conduit node to take on the
            // list role
            schema->set(DataType::list());
            
            for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
            {
                schema->append();
                Schema *curr_schema = schema->child_ptr(i);
                Node * curr_node = new Node();
                curr_node->set_schema_ptr(curr_schema);
                curr_node->set_parent(node);
                node->append_node_ptr(curr_node);
                walk_pure_json_schema(curr_node,curr_schema,jvalue[i]);
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
Generator::Parser::JSON::walk_json_schema(Node   *node,
                                          Schema *schema,
                                          void   *data,
                                          const conduit_rapidjson::Value &jvalue,
                                          index_t curr_offset)
{
    // object cases
    if(jvalue.IsObject())
    {

        if (jvalue.HasMember("dtype"))
        {
            // if dtype is an object, we have a "list_of" case
            const conduit_rapidjson::Value &dt_value = jvalue["dtype"];
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
                    node->append_node_ptr(curr_node);
                    walk_json_schema(curr_node,
                                     curr_schema,
                                     data,
                                     dt_value,
                                     curr_offset);
                    // auto offset only makes sense when we have data
                    if(data != NULL)
                        curr_offset += curr_schema->total_strided_bytes();
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
            for (conduit_rapidjson::Value::ConstMemberIterator itr = 
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
                                  << utils::join_path(node->path(),entry_name));
                }

                Schema *curr_schema = &schema->add_child(entry_name);
                
                Node *curr_node = new Node();
                curr_node->set_schema_ptr(curr_schema);
                curr_node->set_parent(node);
                node->append_node_ptr(curr_node);
                walk_json_schema(curr_node,
                                 curr_schema,
                                 data,
                                 itr->value,
                                 curr_offset);
                
                // auto offset only makes sense when we have data
                if(data != NULL)
                    curr_offset += curr_schema->total_strided_bytes();
            }
            
        }
    }
    // List case 
    else if (jvalue.IsArray()) 
    {
        schema->set(DataType::list());

        for (conduit_rapidjson::SizeType i = 0; i < jvalue.Size(); i++)
        {
            schema->append();
            Schema *curr_schema = schema->child_ptr(i);
            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            node->append_node_ptr(curr_node);
            walk_json_schema(curr_node,
                             curr_schema,
                             data,
                             jvalue[i],
                             curr_offset);
            // auto offset only makes sense when we have data
            if(data != NULL)
                curr_offset += curr_schema->total_strided_bytes();
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
Generator::Parser::JSON::parse_base64(Node *node,
                                      const conduit_rapidjson::Value &jvalue)
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
            walk_json_schema(&s,jvalue["schema"],curr_offset);
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
Generator::Parser::JSON::parse_error_details(const std::string &json,
                                             const conduit_rapidjson::Document &document,
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
// -- end conduit::Generator::Parser::JSON --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::Generator::YAML --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit::Generator::YAML::YAMLParserWrapper --
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
Generator::Parser::YAML::YAMLParserWrapper::YAMLParserWrapper()
: m_yaml_parser_is_valid(false),
  m_yaml_doc_is_valid(false)
{

}

//---------------------------------------------------------------------------//
Generator::Parser::YAML::YAMLParserWrapper::~YAMLParserWrapper()
{
    // cleanup!
    if(m_yaml_parser_is_valid)
    {
        yaml_parser_delete(&m_yaml_parser);
    }

    if(m_yaml_doc_is_valid)
    {
        yaml_document_delete(&m_yaml_doc);
    }
}

//---------------------------------------------------------------------------//
void
Generator::Parser::YAML::YAMLParserWrapper::parse(const char *yaml_txt)
{
    // Initialize parser
    if(yaml_parser_initialize(&m_yaml_parser) == 0)
    {
        // error!
        CONDUIT_ERROR("yaml_parser_initialize failed");
    }
    else
    {
        m_yaml_parser_is_valid = true;
    }

    // set input
    yaml_parser_set_input_string(&m_yaml_parser,
                                 (const unsigned char*)yaml_txt,
                                 strlen(yaml_txt));

    // use parser to construct document
    if( yaml_parser_load(&m_yaml_parser, &m_yaml_doc) == 0 )
    {
        CONDUIT_YAML_PARSE_ERROR(&m_yaml_doc,
                                 &m_yaml_parser);
    }
    else
    {
        m_yaml_doc_is_valid = true;
    }
}

//---------------------------------------------------------------------------//
yaml_document_t *
Generator::Parser::YAML::YAMLParserWrapper::yaml_doc_ptr()
{
    yaml_document_t *res = NULL;

    if(m_yaml_doc_is_valid)
    {
        res = &m_yaml_doc;
    }

    return res;
}

//---------------------------------------------------------------------------//
yaml_node_t *
Generator::Parser::YAML::YAMLParserWrapper::yaml_doc_root_ptr()
{
    yaml_node_t *res = NULL;

    if(m_yaml_doc_is_valid)
    {
        res = yaml_document_get_root_node(&m_yaml_doc);
    }

    return res;
}

//-----------------------------------------------------------------------------
// -- end conduit::Generator::YAML::YAMLParserWrapper --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
// checks if input string is a null pointer or empty
bool
Generator::Parser::YAML::string_is_empty(const char *txt_value)
{
    if(txt_value == NULL)
        return true;
    return strlen(txt_value) == 0;
}

//---------------------------------------------------------------------------//
// checks if input string holds something that converts
// to a double (integer strings will pass this check )
bool
Generator::Parser::YAML::string_is_number(const char *txt_value)
{
    return string_is_integer(txt_value) || string_is_double(txt_value);
}

//---------------------------------------------------------------------------//
// checks if input string holds something that converts
// to a double (integer strings will pass this check )
bool
Generator::Parser::YAML::string_is_double(const char *txt_value)
{
    // TODO: inline check for empty ?
    if(string_is_empty(txt_value))
        return false;
    char *val_end = NULL;
    strtod(txt_value,&val_end);
    return *val_end == 0;
}

//---------------------------------------------------------------------------//
// checks if input string holds something that converts
// to an integer
bool
Generator::Parser::YAML::string_is_integer(const char *txt_value)
{
    // TODO: inline check for empty ?
    if(string_is_empty(txt_value))
        return false;
    char *val_end = NULL;
    strtol(txt_value,&val_end,10);
    return *val_end == 0;
}

//---------------------------------------------------------------------------//
double 
Generator::Parser::YAML::string_to_double(const char *txt_value)
{
    char *val_end = NULL;
    return strtod(txt_value,&val_end);
}

//---------------------------------------------------------------------------//
long int
Generator::Parser::YAML::string_to_long(const char *txt_value)
{
    char *val_end = NULL;
    return strtol(txt_value,&val_end,10);
}

//---------------------------------------------------------------------------//
index_t 
Generator::Parser::YAML::yaml_leaf_to_numeric_dtype(const char *txt_value)
{
    index_t res = DataType::EMPTY_ID;
    if(string_is_integer(txt_value))
    {
        res = DataType::INT64_ID;
    }
    else if(string_is_double(txt_value))
    {
        res = DataType::FLOAT64_ID;
    }
    //else, already inited to DataType::EMPTY_ID

    return res;
}

//---------------------------------------------------------------------------//
// NOTE: Assumes Node res is already DataType::int64, w/ proper len
void
Generator::Parser::YAML::parse_yaml_int64_array(yaml_document_t *yaml_doc,
                                                yaml_node_t *yaml_node,
                                                Node &res)
{
    int64_array res_vals = res.value();
    int cld_idx = 0;
    while( yaml_node->data.sequence.items.start + cld_idx < yaml_node->data.sequence.items.top )
    {
        yaml_node_t *yaml_child = yaml_document_get_node(yaml_doc,
                                                 yaml_node->data.sequence.items.start[cld_idx]);
                                     
        if(yaml_child == NULL || yaml_child->type != YAML_SCALAR_NODE )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid int64 array value at path: "
                          << res.path() << "[" << cld_idx << "]");
        }


        // check type of string contents
        const char *yaml_value_str = (const char*)yaml_child->data.scalar.value;

        if(yaml_value_str == NULL )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid int64 array value at path: "
                          << res.path() << "[" << cld_idx << "]");
        }

        res_vals[cld_idx] = (int64) string_to_long(yaml_value_str);

        cld_idx++;
    }
}

//---------------------------------------------------------------------------//
// NOTE: Assumes Node res is already DataType::float64, w/ proper len
void
Generator::Parser::YAML::parse_yaml_float64_array(yaml_document_t *yaml_doc,
                                                  yaml_node_t *yaml_node,
                                                  Node &res)
{
    float64_array res_vals = res.value();
    int cld_idx = 0;
    while( yaml_node->data.sequence.items.start + cld_idx < yaml_node->data.sequence.items.top )
    {
        yaml_node_t *yaml_child = yaml_document_get_node(yaml_doc,
                                                 yaml_node->data.sequence.items.start[cld_idx]);
                                     
        if(yaml_child == NULL || yaml_child->type != YAML_SCALAR_NODE )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid float64 array value at path: "
                          << res.path() << "[" << cld_idx << "]");
        }


        // check type of string contents
        const char *yaml_value_str = (const char*)yaml_child->data.scalar.value;

        if(yaml_value_str == NULL )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid float64 array value at path: "
                          << res.path() << "[" << cld_idx << "]");
        }

        res_vals[cld_idx] = (float64) string_to_double(yaml_value_str);

        cld_idx++;
    }
}

//---------------------------------------------------------------------------//
index_t
Generator::Parser::YAML::check_homogenous_yaml_numeric_sequence(const Node &node,
                                                                yaml_document_t *yaml_doc,
                                                                yaml_node_t *yaml_node,
                                                                index_t &seq_size)
{
     index_t res = DataType::EMPTY_ID;
     seq_size = -1;
     bool ok = true;
     int cld_idx = 0;
     while( ok && yaml_node->data.sequence.items.start + cld_idx < yaml_node->data.sequence.items.top )
     {
         yaml_node_t *yaml_child = yaml_document_get_node(yaml_doc,
                                                 yaml_node->data.sequence.items.start[cld_idx]);
                                                 
        if(yaml_child == NULL )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid sequence child at path: "
                          << node.path() << "[" << cld_idx << "]");
        }

        // first make sure we only have yaml scalars
        if(yaml_child->type == YAML_SCALAR_NODE)
        {

            // check type of string contents
            const char *yaml_value_str = (const char*)yaml_child->data.scalar.value;

            if(yaml_value_str == NULL )
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "Invalid value for sequence child at path: "
                              << node.path() << "[" << cld_idx << "]");
            }

            // check for integers, then widen to floats

            index_t child_dtype_id = yaml_leaf_to_numeric_dtype(yaml_value_str);

            if(child_dtype_id == DataType::EMPTY_ID)
            {
                ok = false;
            }
            else if(res == DataType::EMPTY_ID)
            {
                // good so far, promote to child's dtype
                res = child_dtype_id;
            }
            else if( res == DataType::INT64_ID && child_dtype_id == DataType::FLOAT64_ID)
            {
                // promote to float64
                res = DataType::FLOAT64_ID;
            }
        }
        else
        {
            ok = false;
        }

        cld_idx++;
     }

     // if we are ok, seq_size is the final cld_idx
     if(ok)
     {
         seq_size = cld_idx;
     }
     else
     {
        res = DataType::EMPTY_ID;
     }

     return res;
}

//---------------------------------------------------------------------------//
void
Generator::Parser::YAML::parse_yaml_inline_leaf(const char *yaml_txt,
                                                Node &node)
{
    if(string_is_integer(yaml_txt))
    {
        node.set((int64)string_to_long(yaml_txt));
    }
    else if(string_is_double(yaml_txt))
    {
        node.set((float64)string_to_double(yaml_txt));
    }
    else if(string_is_empty(yaml_txt))
    {
        node.reset();
    }
    else // general string case
    {
        node.set_char8_str(yaml_txt);
    }
}


//---------------------------------------------------------------------------//
void 
Generator::Parser::YAML::walk_pure_yaml_schema(Node *node,
                                               Schema *schema,
                                               const char *yaml_txt)
{
    YAMLParserWrapper parser;
    parser.parse(yaml_txt);

    yaml_document_t *yaml_doc  = parser.yaml_doc_ptr();
    yaml_node_t     *yaml_node = parser.yaml_doc_root_ptr();


    if(yaml_doc == NULL || yaml_node == NULL)
    {
        CONDUIT_ERROR("failed to fetch yaml document root");
    }

    walk_pure_yaml_schema(node,
                          schema,
                          yaml_doc,
                          yaml_node);

    // YAMLParserWrapper cleans up for us
}


//---------------------------------------------------------------------------//
void 
Generator::Parser::YAML::walk_pure_yaml_schema(Node *node,
                                               Schema *schema,
                                               yaml_document_t *yaml_doc,
                                               yaml_node_t *yaml_node)
{
    
    // object cases
    if( yaml_node->type == YAML_MAPPING_NODE )
    {
        // if we make it here and have an empty json object
        // we still want the conduit node to take on the
        // object role
        schema->set(DataType::object());
        // loop over all entries
        
        int cld_idx = 0;
        // while (has_next) --> grab next, then process
        while( (yaml_node->data.mapping.pairs.start + cld_idx) < yaml_node->data.mapping.pairs.top)
        {
            yaml_node_pair_t *yaml_pair = yaml_node->data.mapping.pairs.start + cld_idx;

            if(yaml_pair == NULL)
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "failed to fetch mapping pair at path: "
                              << node->path() << "[" << cld_idx << "]");
            }

            yaml_node_t *yaml_key = yaml_document_get_node(yaml_doc, yaml_pair->key);
            
            if(yaml_key == NULL)
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "failed to fetch mapping key at path: "
                              << node->path() << "[" << cld_idx << "]");
            }

            if(yaml_key->type != YAML_SCALAR_NODE )
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "Invalid mapping key type at path: "
                              << node->path() << "[" << cld_idx << "]");
            }

            const char *yaml_key_str = (const char *) yaml_key->data.scalar.value;

            if(yaml_key_str == NULL )
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "Invalid mapping key value at path: "
                              << node->path() << "[" << cld_idx << "]");
            }
            
            std::string entry_name(yaml_key_str);

            yaml_node_t *yaml_child = yaml_document_get_node(yaml_doc, yaml_pair->value);

            if(yaml_child == NULL )
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "Invalid mapping child at path: "
                              << utils::join_path(node->path(),entry_name));
            }

            // yaml files may have duplicate object names
            // we could provide some clear semantics, such as:
            //   always use first instance, or always use last instance
            // however duplicate object names are most likely a
            // typo, so it's best to throw an error

            if(schema->has_child(entry_name))
            {
                CONDUIT_ERROR("YAML Generator error:\n"
                              << "Duplicate YAML object name: "
                              << utils::join_path(node->path(),entry_name));
            }

            Schema *curr_schema = &schema->add_child(entry_name);

            Node *curr_node = new Node();
            curr_node->set_schema_ptr(curr_schema);
            curr_node->set_parent(node);
            node->append_node_ptr(curr_node);
        
            walk_pure_yaml_schema(curr_node,
                                  curr_schema,
                                  yaml_doc,
                                  yaml_child);
            cld_idx++;
        }
    }
    // List case
    else if( yaml_node->type == YAML_SEQUENCE_NODE )
    {
        index_t seq_size  = -1;
        index_t hval_type = check_homogenous_yaml_numeric_sequence(*node,
                                                                   yaml_doc,
                                                                   yaml_node,
                                                                   seq_size);

        if(hval_type == DataType::INT64_ID)
        {

            node->set(DataType::int64(seq_size));
            parse_yaml_int64_array(yaml_doc, yaml_node, *node);

        }
        else if(hval_type == DataType::FLOAT64_ID)
        {
            node->set(DataType::float64(seq_size));
            parse_yaml_float64_array(yaml_doc, yaml_node, *node);
        }
        else
        {
            // general case (not a numeric array)
            index_t cld_idx = 0;
            while( yaml_node->data.sequence.items.start + cld_idx < yaml_node->data.sequence.items.top )
            {
                yaml_node_t *yaml_child = yaml_document_get_node(yaml_doc,
                                                        yaml_node->data.sequence.items.start[cld_idx]);
            
                if(yaml_child == NULL )
                {
                    CONDUIT_ERROR("YAML Generator error:\n"
                                  << "Invalid sequence child at path: "
                                  << node->path() << "[" << cld_idx << "]");
                }

                schema->append();
                Schema *curr_schema = schema->child_ptr(cld_idx);
                Node * curr_node = new Node();
                curr_node->set_schema_ptr(curr_schema);
                curr_node->set_parent(node);
                node->append_node_ptr(curr_node);
                walk_pure_yaml_schema(curr_node,
                                      curr_schema,
                                      yaml_doc,
                                      yaml_child);
                cld_idx++;
            }
        }
    }
    else if(yaml_node->type == YAML_SCALAR_NODE)// bytestr case
    {
        const char *yaml_value_str = (const char*)yaml_node->data.scalar.value;

        if( yaml_value_str == NULL )
        {
            CONDUIT_ERROR("YAML Generator error:\n"
                          << "Invalid yaml scalar value at path: "
                          << node->path());
        }

        parse_yaml_inline_leaf(yaml_value_str,*node);
    }
    else // this will include unknown enum vals and YAML_NO_NODE
    {
        // not sure if can an even land here, but catch error just in case.
        CONDUIT_ERROR("YAML Generator error:\n"
                      << "Invalid YAML type for parsing Node from pure YAML."
                      << " Expected: YAML Map, Sequence, String, Null,"
                      << " Boolean, or Number");
    }
}


//-----------------------------------------------------------------------------
void
Generator::Parser::YAML::parse_error_details(yaml_parser_t *yaml_parser,
                                             std::ostream &os)
{
    os << "YAML Parsing Error (";
    switch (yaml_parser->error)
    {
        case YAML_NO_ERROR:
            os << "YAML_NO_ERROR";
            break;
        case YAML_MEMORY_ERROR:
            os << "YAML_MEMORY_ERROR";
            break;
        case YAML_READER_ERROR:
            os << "YAML_MEMORY_ERROR";
            break;
        case YAML_SCANNER_ERROR:
            os << "YAML_SCANNER_ERROR";
            break;
        case YAML_PARSER_ERROR:
            os << "YAML_PARSER_ERROR";
            break;
        case YAML_COMPOSER_ERROR:
            os << "YAML_COMPOSER_ERROR";
            break;
        case YAML_WRITER_ERROR:
            os << "YAML_WRITER_ERROR";
            break;
        case YAML_EMITTER_ERROR:
            os << "YAML_EMITTER_ERROR";
            break;
        default:
            os << "[Unknown Error!]";
            break;
    }
    
    // Q: Is yaml_parser->problem_mark.index useful here?
    //    that might be the only case where we need the yaml_doc
    //    otherwise using yaml_parser is sufficient

    if(yaml_parser->problem != NULL)
    {
        os << ")\n Problem:\n" << yaml_parser->problem << "\n"
           << "  Problem Line: "   << yaml_parser->problem_mark.line << "\n"
           << "  Problem Column: " << yaml_parser->problem_mark.column << "\n";
    }
    else
    {
        os << "unexpected: yaml_parser->problem is NULL (missing)\n";
    }
    if(yaml_parser->context != NULL)
    {
       os << " Context\n"         << yaml_parser->context << "\n"
          << "  Context Line: "   << yaml_parser->context_mark.line << "\n"
          << "  Context Column: " << yaml_parser->context_mark.column<< "\n";
    }
    os << std::endl;
}

//-----------------------------------------------------------------------------
// -- end conduit::Generator::Parser::YAML --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Generator Construction and Destruction
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
Generator::Generator()
:m_schema(""),
 m_protocol("conduit_json"),
 m_data(NULL)
{}


//---------------------------------------------------------------------------//
Generator::Generator(const std::string &schema,
                     const std::string &protocol,
                     void *data)
:m_schema(schema),
 m_protocol(protocol),
 m_data(data)
{}

//---------------------------------------------------------------------------//
void
Generator::set_schema(const std::string &schema)
{
    m_schema = schema;
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
Generator::schema() const
{
    return m_schema;
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


//---------------------------------------------------------------------------//
void 
Generator::walk(Schema &schema) const
{
    schema.reset();
    conduit_rapidjson::Document document;
    std::string res = utils::json_sanitize(m_schema);

    if(document.Parse<Parser::JSON::RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
    {
        CONDUIT_JSON_PARSE_ERROR(res, document);
    }
    index_t curr_offset = 0;
    Parser::JSON::walk_json_schema(&schema,document,curr_offset);
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
        conduit_rapidjson::Document document;
        std::string res = utils::json_sanitize(m_schema);
                
        if(document.Parse<Parser::JSON::RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }

        Parser::JSON::walk_pure_json_schema(&node,
                                            node.schema_ptr(),
                                            document);
    }
    else if(m_protocol == "yaml")
    {
        // errors will flow up from this call 
        Parser::YAML::walk_pure_yaml_schema(&node,
                                            node.schema_ptr(),
                                            m_schema.c_str());
    }
    else if( m_protocol == "conduit_base64_json")
    {
        conduit_rapidjson::Document document;
        std::string res = utils::json_sanitize(m_schema);
        
        if(document.Parse<Parser::JSON::RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }

        Parser::JSON::parse_base64(&node,
                                   document);
    }
    else if( m_protocol == "conduit_json")
    {
        conduit_rapidjson::Document document;
        std::string res = utils::json_sanitize(m_schema);
        
        if(document.Parse<Parser::JSON::RAPIDJSON_PARSE_OPTS>(res.c_str()).HasParseError())
        {
            CONDUIT_JSON_PARSE_ERROR(res, document);
        }
        index_t curr_offset = 0;

        Parser::JSON::walk_json_schema(&node,
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

