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
/// file: conduit_io.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_io.h"
#include <iostream>

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
identify_io_type(const std::string &path,
                 std::string &io_type)
{
    io_type = "conduit_bin";

    std::string file_path;
    std::string obj_base;

    // check for ":" split
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 obj_base);

    if(obj_base.size() != 0)
    {
        std::string file_name_base;
        std::string file_name_ext;
        
        // find file extension to auto match 
        conduit::utils::rsplit_string(file_path,
                                      std::string("."),
                                      file_name_base,
                                      file_name_ext);

        if(file_name_ext == "silo")
        {
            io_type = "conduit_silo";
        }
    }
}

//---------------------------------------------------------------------------//
void 
save(const  Node &node,
     const std::string &path)
{
    std::string io_type;
    identify_io_type(path,io_type);

    if(io_type == "conduit_bin")
    {
        node.save(path);
    }
    else if( io_type == "conduit_silo")
    {
#ifdef CONDUIT_IO_ENABLE_SILO
        silo_save(node,path);
#else
        THROW_ERROR("conduit_io lacks Silo support: " << 
                    "Failed to save conduit node to path " << path);
#endif
    }

}

//---------------------------------------------------------------------------//
void
load(const std::string &path,
     Node &node)
{
    std::string io_type;
    identify_io_type(path,io_type);

    if(io_type == "conduit_bin")
    {
        node.load(path);
    }
    else if( io_type == "conduit_silo")
    {
#ifdef CONDUIT_IO_ENABLE_SILO
        silo_load(path,node);
#else
        THROW_ERROR("conduit_io lacks Silo support: " << 
                    "Failed to load conduit node from path " << path);
#endif
    }


}

//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    io::about(n);
    return n.to_json(true,2);
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();
    Node &nn = n["io_endpoints"];

    // standard binary io
    nn.append().set("conduit_bin");
    
    // silo
#ifdef CONDUIT_IO_ENABLE_SILO
    nn.append().set("conduit_silo");
#endif


}


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


