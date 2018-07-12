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
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io_blueprint
{

//---------------------------------------------------------------------------//
std::string
identify_protocol(const std::string &path)
{
    std::string file_path, obj_base;
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    obj_base);

    std::string file_name_base, file_name_ext;
    conduit::utils::rsplit_string(file_path,
                                  std::string("."),
                                  file_name_ext,
                                  file_name_base);

    std::string io_type = "bin";
    if(file_name_ext.find("blueprint_root") == 0)
    {
        std::string file_name_true_ext = file_name_ext.substr(
            std::string("blueprint_root").length(), file_name_ext.length());
        if(file_name_true_ext == "")
        {
            io_type = "json";
        }
        else if(file_name_true_ext == "_hdf5" || file_name_true_ext == "_h5")
        {
            io_type = "hdf5";
        }
        else if(file_name_true_ext == "_silo")
        {
            io_type = "silo";
        }
    }

    return io_type;
}

//---------------------------------------------------------------------------//
void
save(const Node &mesh,
     const std::string &path)
{
    save(mesh,path,identify_protocol(path));
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
        CONDUIT_ERROR("Blueprint I/O doesn't support '" << protocol << "' outputs; "
                      "output type must be 'blueprint_root' (JSON) or 'blueprint_root_hdf5' (HDF5): " <<
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
    Node index;
    if(blueprint::mesh::is_multi_domain(mesh))
    {
        index["data"].set_external(mesh);
    }
    else
    {
        index["data/mesh"].set_external(mesh);
    }

    Node &bpindex = index["blueprint_index"];
    {
        NodeConstIterator domain_iter = index["data"].children();
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
        index["protocol/name"].set(protocol);
        index["protocol/version"].set(CONDUIT_VERSION);

        index["number_of_files"].set(1);
        index["number_of_trees"].set(1);
        index["file_pattern"].set(path);
        index["tree_pattern"].set((protocol == "hdf5") ? "data/" : "data");

        relay::io::save(index,path,protocol);
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
