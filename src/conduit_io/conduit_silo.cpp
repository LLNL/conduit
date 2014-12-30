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
/// file: conduit_silo.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_silo.h"
#include <iostream>
#include <silo.h>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

    //-----------------------------------------------------------------------------
// -- begin conduit::io --
//-----------------------------------------------------------------------------
namespace io
{


//---------------------------------------------------------------------------//
void 
silo_save(const  Node &node,
          const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 silo_obj_base);

    /// If silo_obj_base is empty, we have a problem ... 
    if(silo_obj_base.size() == 0)
    {
        THROW_ERROR("Invalid path for save: " << path);
    }

    silo_save(node,file_path,silo_obj_base);
}

//---------------------------------------------------------------------------//
void
silo_load(const std::string &path,
          Node &node)
{
    // check for ":" split    
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 silo_obj_base);

    /// If silo_obj_base is empty, we have a problem ... 
    if(silo_obj_base.size() == 0)
    {
        THROW_ERROR("Invalid path for load: " << path);
    }

    silo_load(file_path,silo_obj_base,node);
}


//---------------------------------------------------------------------------//
void silo_save(const  Node &node,
               const std::string &file_path,
               const std::string &silo_obj_path)
{
    DBfile *dbfile = DBCreate(file_path.c_str(),
                              DB_CLOBBER,
                              DB_LOCAL,
                              NULL,
                              DB_HDF5);

    if(dbfile)
    {
        silo_save(node,dbfile,silo_obj_path);
    }
    else 
    {
        THROW_ERROR("Error opening Silo file for writting: " << file_path );
        return;
    }
    
    if(DBClose(dbfile) != 0)
    {
        THROW_ERROR("Error closing Silo file: " << file_path);
    }
}

//---------------------------------------------------------------------------//
void silo_load(const std::string &file_path,
               const std::string &silo_obj_path,
               Node &n)
{
    DBfile *dbfile = DBOpen(file_path.c_str(), DB_HDF5, DB_READ);

    if(dbfile)
    {
        silo_load(dbfile,silo_obj_path,n);
    }
    else 
    {
        THROW_ERROR("Error opening Silo file for reading: " << file_path );
    }
    
    if(DBClose(dbfile) != 0)
    {
        THROW_ERROR("Error closing Silo file: " << file_path );
    }
}


//---------------------------------------------------------------------------//
void silo_save(const  Node &node,
               DBfile *dbfile,
               const std::string &silo_obj_path)
{
    Schema schema_c;
    node.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();
    int schema_len = schema.length() + 1;

    std::vector<uint8> data;
    node.serialize(data);
    int data_len = data.size();

    // use path to construct dest silo obj paths
    
    std::string dest_json = silo_obj_path +  "_conduit_json";
    std::string dest_data = silo_obj_path +  "_conduit_bin";

    int silo_error = 0;
    silo_error += DBWrite(dbfile,
                          dest_json.c_str(),
                          schema.c_str(),
                          &schema_len,
                          1,
                          DB_CHAR);
    silo_error += DBWrite(dbfile,
                          dest_data.c_str(),
                          &data[0],
                          &data_len,
                          1,
                          DB_CHAR);

    if(silo_error != 0)
    {
        THROW_ERROR("Error writing conduit Node to Silo file");
    }
}


//---------------------------------------------------------------------------//
void CONDUIT_IO_API silo_load(DBfile *dbfile,
                              const std::string &silo_obj_path,
                              Node &node)
{
    std::string src_json = silo_obj_path +  "_conduit_json";
    std::string src_data = silo_obj_path +  "_conduit_bin";

    int schema_len = DBGetVarLength(dbfile, src_json.c_str());
    int data_len   = DBGetVarLength(dbfile, src_data.c_str());

    char *schema = new char[schema_len];
    char *data   = new char[data_len];



    DBReadVar(dbfile, src_json.c_str(), schema);
    DBReadVar(dbfile, src_data.c_str(), data);

    if (schema == NULL || data == NULL) 
    {
        THROW_ERROR("Error extracting data conduit Node from Silo file");
    }

    Generator node_gen(schema, data);
    /// gen copy 
    node_gen.walk(node,false);
    
    delete [] schema;
    delete [] data;
}


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
