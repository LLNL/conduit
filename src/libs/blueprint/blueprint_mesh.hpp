//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: blueprint_mesh.hpp
///
//-----------------------------------------------------------------------------

#ifndef BLUEPRINT_MESH_HPP
#define BLUEPRINT_MESH_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "blueprint_exports.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

namespace mesh 
{

//-----------------------------------------------------------------------------
/// blueprint protocol verify interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Interface to call verify on nested mesh protocols by name.
///   supports: coordset
///             topology
///             field
///             index
///             coordset/index,
///             topology/index,
///             field/index

//-----------------------------------------------------------------------------
bool BLUEPRINT_API verify(const std::string &protocol,
                          const conduit::Node &n,
                          conduit::Node &info);

//-----------------------------------------------------------------------------
bool BLUEPRINT_API verify(const conduit::Node &n,
                          conduit::Node &info);

//-------------------------------------------------------------------------
void BLUEPRINT_API generate_index(const conduit::Node &mesh,
                                  const std::string &ref_path,
                                  index_t num_domains,
                                  Node &index_out);

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset protocol interface
//-----------------------------------------------------------------------------
namespace coordset
{
    //-------------------------------------------------------------------------
    bool BLUEPRINT_API verify(const conduit::Node &n,
                              conduit::Node &info);
                              
                              
    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::uniform protocol interface
    //-------------------------------------------------------------------------
    namespace uniform
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::rectilinear protocol interface
    //-------------------------------------------------------------------------
    namespace rectilinear
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::explicit protocol interface
    //-------------------------------------------------------------------------
    namespace _explicit
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }


    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::coord_system protocol interface
    //-------------------------------------------------------------------------
    namespace coord_system
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// blueprint::mesh::topology protocol interface
//-----------------------------------------------------------------------------
namespace topology
{
    //-------------------------------------------------------------------------
    bool BLUEPRINT_API verify(const conduit::Node &n,
                              conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::field protocol interface
//-----------------------------------------------------------------------------
namespace field
{
    //-------------------------------------------------------------------------
    bool BLUEPRINT_API verify(const conduit::Node &n,
                              conduit::Node &info);
                              
    //-------------------------------------------------------------------------
    // blueprint::mesh::field::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
                              
    //-------------------------------------------------------------------------
    // blueprint::mesh::field::association  protocol interface
    //-------------------------------------------------------------------------
    namespace association
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
    //-------------------------------------------------------------------------
    // blueprint::mesh::field::basis protocol interface
    //-------------------------------------------------------------------------
    namespace basis
    {
        //---------------------------------------------------------------------
        bool BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);
    }
    
    
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::field --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{
    //-------------------------------------------------------------------------
    bool BLUEPRINT_API verify(const conduit::Node &n,
                              conduit::Node &info);

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::index --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif 



