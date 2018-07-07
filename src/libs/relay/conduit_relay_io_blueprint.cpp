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
/// file: conduit_relay_io_blueprint.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_io_blueprint.hpp"
#include "conduit_relay_io.hpp"

#include "conduit_blueprint.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::io_blueprint --
//-----------------------------------------------------------------------------
namespace io_blueprint
{

// TODO(JRC): This functionality was copied directly from "conduit::relay::io"
// to prevent exposing it to users. It should potentially be abstracted and made
// available in some form of "conduit::relay" utility header ultimately.
//---------------------------------------------------------------------------//
void
identify_protocol(const std::string &path,
                  std::string &io_type)
{
    io_type = "conduit_bin";

    std::string file_path;
    std::string obj_base;

    // check for ":" split
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    obj_base);

    std::string file_name_base;
    std::string file_name_ext;

    // find file extension to auto match
    conduit::utils::rsplit_string(file_path,
                                  std::string("."),
                                  file_name_ext,
                                  file_name_base);


    if(file_name_ext == "hdf5" ||
       file_name_ext == "h5")
    {
        io_type = "hdf5";
    }
    else if(file_name_ext == "silo")
    {
        io_type = "conduit_silo";
    }
    else if(file_name_ext == "json")
    {
        io_type = "json";
    }
    else if(file_name_ext == "conduit_json")
    {
        io_type = "conduit_json";
    }
    else if(file_name_ext == "conduit_base64_json")
    {
        io_type = "conduit_base64_json";
    }

    // default to conduit_bin
}

//---------------------------------------------------------------------------//
void
save(const Node &mesh,
     const std::string &path)
{
    std::string protocol;
    identify_protocol(path,protocol);
    save(mesh,path,protocol);
}

//---------------------------------------------------------------------------//
void
save(const Node &mesh,
     const std::string &path,
     const std::string &protocol)
{
    Node info;
    if(protocol != "json" && protocol != "hdf5")
    {
        CONDUIT_ERROR("Blueprint I/O doesn't support '" << protocol << "' outputs: " <<
                      "Failed to save mesh to path " << path);
    }
    if(!blueprint::mesh::verify(mesh, info))
    {
        CONDUIT_ERROR("Given node isn't a valid Blueprint mesh: " <<
                      "Failed to save mesh to path " << path);
    }

    // NOTE(JRC): The code below is used in lieu of `blueprint::mesh::to_multi_domain`
    // because the official Blueprint function produces results that are incompatible
    // with HDF5 outputs (because they include Conduit lists instead of dictionaries).
    Node mmesh;
    if(blueprint::mesh::is_multi_domain(mesh))
    {
        mmesh.set_external(mesh);
    }
    else
    {
        mmesh["mesh"].set_external(mesh);
    }

    Node mindex;
    Node &bpindex = mindex["blueprint_index"];
    {
        NodeConstIterator domain_iter = mmesh.children();
        while(domain_iter.has_next())
        {
            const Node &domain = domain_iter.next();
            const std::string domain_name = domain_iter.name();

            // NOTE: Skip all domains containing one or more mixed-shape topologies
            // because this type of mesh isn't fully supported yet.
            bool is_domain_index_valid = true;
            NodeConstIterator topo_iter = domain["topologies"].children();
            while(topo_iter.has_next())
            {
                const Node &topo = topo_iter.next();
                is_domain_index_valid &= (
                    !blueprint::mesh::topology::unstructured::verify(topo, info) ||
                    !topo["elements"].has_child("element_types"));
            }

            if(is_domain_index_valid)
            {
                blueprint::mesh::generate_index(
                    domain,domain_name,1,bpindex[domain_name]);
            }
        }
    }

    if(bpindex.number_of_children() == 0)
    {
        CONDUIT_INFO("No valid domains in given Blueprint mesh: " <<
                     "Skipping save of mesh to path " << path);
    }
    else
    {
        std::string path_base, path_ext;
        conduit::utils::rsplit_string(path,std::string("."),path_ext,path_base);
        std::string index_path = path_base + std::string(".blueprint_root");
        std::string data_path = path_base + std::string(".") + path_ext;

        mindex["protocol/name"].set(protocol);
        mindex["protocol/version"].set(CONDUIT_VERSION);

        mindex["number_of_files"].set(1);
        mindex["number_of_trees"].set(1);
        mindex["file_pattern"].set(data_path);
        mindex["tree_pattern"].set((protocol == "hdf5") ? "/" : "");

        relay::io::save(mindex,index_path,protocol);
        relay::io::save(mmesh,data_path);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io_blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
