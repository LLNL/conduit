// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_julia.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_JULIA_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_JULIA_HPP

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

    /// Generates a simple 8x8 two level julia nestset with
    /// two domains
    void CONDUIT_BLUEPRINT_API julia_nestsets_simple(conduit::float64 x_min,
                                                     conduit::float64 x_max,
                                                     conduit::float64 y_min,
                                                     conduit::float64 y_max,
                                                     conduit::float64 c_re,
                                                     conduit::float64 c_im,
                                                     conduit::Node &res);


    void CONDUIT_BLUEPRINT_API julia_nestsets_complex(conduit::index_t nx,
                                                      conduit::index_t ny,
                                                      conduit::float64 x_min,
                                                      conduit::float64 x_max,
                                                      conduit::float64 y_min,
                                                      conduit::float64 y_max,
                                                      conduit::float64 c_re,
                                                      conduit::float64 c_im,
                                                      conduit::index_t levels,
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



