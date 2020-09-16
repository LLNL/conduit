// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

    /// Generates a mesh that uses uniform adjsets
    void CONDUIT_BLUEPRINT_API adjset_uniform(conduit::Node &res);
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



