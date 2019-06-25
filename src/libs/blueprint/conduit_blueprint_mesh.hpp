//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_blueprint_mesh.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_HPP
#define CONDUIT_BLUEPRINT_MESH_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"

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
bool CONDUIT_BLUEPRINT_API verify(const std::string &protocol,
                                  const conduit::Node &n,
                                  conduit::Node &info);

//-----------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                  conduit::Node &info);


//-----------------------------------------------------------------------------
/// blueprint mesh property and transform methods
/// 
/// These methods can be called on any verified blueprint mesh.
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API is_multi_domain(const conduit::Node &n);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API to_multi_domain(const conduit::Node &n,
                                           conduit::Node &dest);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_index(const conduit::Node &mesh,
                                          const std::string &ref_path,
                                          index_t num_domains,
                                          Node &index_out);

//-----------------------------------------------------------------------------
// blueprint::mesh::logical_dims protocol interface
//-----------------------------------------------------------------------------
namespace logical_dims
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::association protocol interface
//-----------------------------------------------------------------------------
namespace association
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset protocol interface
//-----------------------------------------------------------------------------
namespace coordset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API dims(const conduit::Node &n);

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::uniform protocol interface
    //-------------------------------------------------------------------------
    namespace uniform
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_rectilinear(const conduit::Node &n,
                                                  conduit::Node &dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_explicit(const conduit::Node &n,
                                               conduit::Node &dest);

        //---------------------------------------------------------------------
        // blueprint::mesh::coordset::uniform::origin protocol interface
        //---------------------------------------------------------------------
        namespace origin
        {
            //-----------------------------------------------------------------
            bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                              conduit::Node &info);
        }

        //---------------------------------------------------------------------
        // blueprint::mesh::coordset::uniform::spacing protocol interface
        //---------------------------------------------------------------------
        namespace spacing
        {
            //-----------------------------------------------------------------
            bool CONDUIT_BLUEPRINT_API  verify(const conduit::Node &n,
                                               conduit::Node &info);
        }

    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::rectilinear protocol interface
    //-------------------------------------------------------------------------
    namespace rectilinear
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_explicit(const conduit::Node &n,
                                               conduit::Node &dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::explicit protocol interface
    //-------------------------------------------------------------------------
    namespace _explicit
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::coord_system protocol interface
    //-------------------------------------------------------------------------
    namespace coord_system
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::points protocol interface
    //-------------------------------------------------------------------------
    namespace points
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::uniform protocol interface
    //-------------------------------------------------------------------------
    namespace uniform
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_rectilinear(const conduit::Node &n,
                                                  conduit::Node &dest,
                                                  conduit::Node &cdest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_structured(const conduit::Node &n,
                                                 conduit::Node &dest,
                                                 conduit::Node &cdest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &n,
                                                   conduit::Node &dest,
                                                   conduit::Node &cdest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::rectilinear protocol interface
    //-------------------------------------------------------------------------
    namespace rectilinear
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_structured(const conduit::Node &n,
                                                 conduit::Node &dest,
                                                 conduit::Node &cdest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &n,
                                                   conduit::Node &dest,
                                                   conduit::Node &cdest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::structured protocol interface
    //-------------------------------------------------------------------------
    namespace structured
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &n,
                                                   conduit::Node &dest,
                                                   conduit::Node &cdest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::unstructured protocol interface
    //-------------------------------------------------------------------------
    namespace unstructured
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_polygonal(const conduit::Node &n,
                                                conduit::Node &dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_points(const conduit::Node &n,
                                                   conduit::Node &dest,
                                                   conduit::Node &s2dmap,
                                                   conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_lines(const conduit::Node &n,
                                                  conduit::Node &dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_faces(const conduit::Node &n,
                                                  conduit::Node &dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_centroids(const conduit::Node &n,
                                                      conduit::Node &dest,
                                                      conduit::Node &cdest,
                                                      conduit::Node &s2dmap,
                                                      conduit::Node &d2smap);

        //---------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_sides(const conduit::Node &n,
                                                  conduit::Node &dest,
                                                  conduit::Node &cdest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //---------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_corners(const conduit::Node &n,
                                                    conduit::Node &dest,
                                                    conduit::Node &cdest,
                                                    conduit::Node &s2dmap,
                                                    conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_offsets(const conduit::Node &n,
                                                    conduit::Node &dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::shape protocol interface
    //-------------------------------------------------------------------------
    namespace shape
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::matset protocol interface
//-----------------------------------------------------------------------------
namespace matset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::matset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::matset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::field protocol interface
//-----------------------------------------------------------------------------
namespace field
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::field::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::field::basis protocol interface
    //-------------------------------------------------------------------------
    namespace basis
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::field --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::specset protocol interface
//-----------------------------------------------------------------------------
namespace specset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::specset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::specset--
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::adjset protocol interface
//-----------------------------------------------------------------------------
namespace adjset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::adjset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::adjset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::nestset protocol interface
//-----------------------------------------------------------------------------
namespace nestset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::nestset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::nestset::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
                                          conduit::Node &info);
    }
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::nestset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// blueprint::mesh::index protocol interface
//-----------------------------------------------------------------------------
namespace index
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &n,
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



