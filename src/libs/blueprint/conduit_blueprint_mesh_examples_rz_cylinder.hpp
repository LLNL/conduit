// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_rz_cylinder.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_RZ_CYLINDER_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_RZ_CYLINDER_HPP

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
    /// Generates an example 2D cylindrical (RZ) mesh.
    ///
    /// This follows the Blueprints suggested coordinate order for
    /// 2D cylindrical meshes:
    ///   Z is the first axis
    ///   R is the second axis
    ///
    /// Arguments:
    ///    nz --> number of elements in z
    ///    nr --> number of elements in r
    ///    mesh_type --> type of topology and coordset to generate 
    //       mesh_type options:
    ///       - uniform
    ///       - rectilinear
    ///       - structured
    ///       - unstructured
    ///
    void CONDUIT_BLUEPRINT_API rz_cylinder(const std::string &mesh_type,
                                           index_t nz,
                                           index_t nr,
                                           Node &res);

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



