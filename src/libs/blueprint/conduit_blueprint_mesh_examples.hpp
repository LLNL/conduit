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

    /// Generates a structured grid with a scalar field that assigns a unique,
    /// monotonically increasing value to each element.  Calling code can
    /// specify the dimension of the storage array of the elements and
    /// vertices, which can differ.  Pass the extra specifications with
    /// a conduit::Node:
    ///
    /// vertex_data:
    ///   shape: [vx, vy, vz]
    ///   origin: [wx, wy, wz]
    /// element_data:
    ///   shape: [ex, ey, ez]
    ///   origin: [fx, fy, fz]
    ///
    /// It is an error if the vertex or element data array shapes are too
    /// small to contain the requested mesh.
    void CONDUIT_BLUEPRINT_API strided_structured(conduit::Node &desc,
                                                  conduit::index_t nx,
                                                  conduit::index_t ny,
                                                  conduit::index_t nz,
                                                  conduit::Node &res);

    /// Generates a multidomain uniform grid of 'basic' examples for each
    /// domain/grid element.
    void CONDUIT_BLUEPRINT_API grid(const std::string &mesh_type,
                                     conduit::index_t nx,
                                     conduit::index_t ny,
                                     conduit::index_t nz,
                                     conduit::index_t dx,
                                     conduit::index_t dy,
                                     conduit::index_t dz,
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
    /// packed octogons and rectangles. The parameter nz can be any nonzero
    /// natural number. An nz value of 1 will produce a polytess in 2D,
    /// while an nz value of 2 will produce a polytess in 3D, which can be
    /// explained as follows: first, polytess is placed into 3D space, and
    /// then a copy of it is placed into a plane parallel to the original.
    /// Then "walls" are added, and finally polyhedra are specified that use
    /// faces from the original polytess, the reflected copy, and the walls.
    /// An nz value of 3 or more will simply add layers to this setup,
    /// essentially stacking "sheets" of polytess on top of one another.
    void CONDUIT_BLUEPRINT_API polytess(conduit::index_t nlevels,
                                        conduit::index_t nz,
                                        conduit::Node &res);

    /// Generates a chain of cubes and triangular prisms
    void CONDUIT_BLUEPRINT_API polychain(const conduit::index_t length,
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



