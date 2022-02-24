// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_related_boundary.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_RELATED_BOUNDARY_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_RELATED_BOUNDARY_HPP

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
    /*
        conduit::blueprint::mesh::examples::related_boundary
    
        Creates an example mesh that uses the following pattern:

        |------------|------------|
        |            |            |
        |  domain 1  |            |
        |            |            |
        |------------|  domain 2  |
        |            |            |
        |  domain 0  |            |
        |            |            |
        |------------|------------|

        The base grid i and j are used to size domains 0 and 1, domain 2
         uses grid i and j*2 as dims. The logical dimensions 
        `base_ele_dims_i` and `base_ele_dims_j` are in terms of elements.

        There are two topologies: `main` and `boundary`.

        The `main` topology is a structured mesh of quads.
        The `boundary` topology is an unstructured mesh of lines,
        the line elements correspond to the boundary faces of `main`.

        The `main` and `boundary` topologies are defined using the same
        explicit coordset. Because of this we can relate the elements
        between them.

        The field `ele_local_id` provides local ids (unique within a domain)
        for elements of the `main` topology.

        The field `ele_global_id` provides globally unique ids for elements
        of the `main` topology.

        The field `domain_id` provides the domain number for elements
        of the `main` topology.

        The field `bndry_val` is defined on the `boundary` topology
        as 1 for elements on the external mesh boundary, and 0 for
        elements on an internal mesh boundary.

        The field `bndry_local_id` provides local ids (unique within a domain)
        for elements of the `boundary` topology.

        The field `bndry_global_id` provides globally unique ids for elements
        of the `boundary` topology.

        The field `bndry_to_main_local` provides a map from each `boundary`
        topology element to the local element id of the `main` topology it
        is connected to.

        The field `bndry_to_main_global` provides a map from each `boundary`
        topology element to the global element id of the `main` topology it
        is connected to.

        It also provides an adjacency set `main_adjset` that describes
        how the domains of the `main` topology abut.

    */
    void CONDUIT_BLUEPRINT_API related_boundary(conduit::index_t base_ele_dims_i,
                                                conduit::index_t base_ele_dims_j,
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



