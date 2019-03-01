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
/// file: conduit_blueprint_mesh_examples.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_exports.h"

//-----------------------------------------------------------------------------
// -- begin conduit::--
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
/// Methods that generate example meshes.
//-----------------------------------------------------------------------------
namespace examples
{
    /// Generates a uniform grid with a scalar field that assigns a unique,
    /// monotonically increasing value to each element.
    void CONDUIT_BLUEPRINT_API basic(const std::string &mesh_type,
                                     conduit::index_t nx,
                                     conduit::index_t ny,
                                     conduit::index_t nz,
                                     conduit::Node &res);

    /// Generates a braid-like example mesh that covers elements defined in a
    /// rectilinear grid. The element type (e.g. triangles, quads, their 3D
    /// counterparts, or a mixture) and the coordinate set/topology
    /// types can be configured by specifying different "mesh_type" values
    /// (see the Conduit documentation for details).
    void CONDUIT_BLUEPRINT_API braid(const std::string &mesh_type,
                                     conduit::index_t nx,
                                     conduit::index_t ny,
                                     conduit::index_t nz,
                                     conduit::Node &res);

    /// Generates a rectilinear grid with a scalar field that
    /// visualizes the julia set (https://en.wikipedia.org/wiki/Julia_set)
    void CONDUIT_BLUEPRINT_API julia(conduit::index_t nx,
                                     conduit::index_t ny,
                                     conduit::float64 x_min,
                                     conduit::float64 x_max,
                                     conduit::float64 y_min,
                                     conduit::float64 y_max,
                                     conduit::float64 c_re,
                                     conduit::float64 c_im,
                                     conduit::Node &res);

    /// Generates a multi-domain fibonacci estimation of a golden spiral.
    void CONDUIT_BLUEPRINT_API spiral(conduit::index_t ndomains,
                                      conduit::Node &res);

    /// Generates a tessellated heterogeneous polygonal mesh consisting of
    /// packed octogons and rectangles.
    void CONDUIT_BLUEPRINT_API polytess(conduit::index_t nlevels,
                                        conduit::Node &res);

    /// Generates an assortment of extra meshes that demonstrate the use of
    /// less common concepts (e.g. adjacency sets, amr blocks, etc.).
    void CONDUIT_BLUEPRINT_API misc(const std::string &mesh_type,
                                    conduit::index_t nx,
                                    conduit::index_t ny,
                                    conduit::index_t nz,
                                    conduit::Node &res);
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
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
// -- end conduit --
//-----------------------------------------------------------------------------


#endif 



