// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &mesh,
                                  conduit::Node &info);


//-----------------------------------------------------------------------------
/// blueprint mesh property and transform methods
/// 
/// These methods can be called on any verified blueprint mesh.
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
bool CONDUIT_BLUEPRINT_API is_multi_domain(const conduit::Node &mesh);

//-------------------------------------------------------------------------
index_t CONDUIT_BLUEPRINT_API number_of_domains(const conduit::Node &mesh);


//-----------------------------------------------------------------------------
std::vector<conduit::Node *>       CONDUIT_BLUEPRINT_API domains(Node &mesh);
std::vector<const conduit::Node *> CONDUIT_BLUEPRINT_API domains(const Node &mesh);


//-----------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API domains(const Node &mesh,
                                   std::vector<const conduit::Node *> &res);

//-----------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API domains(Node &mesh,
                                   std::vector<conduit::Node *> &res);

/// Note: to_multi_domain uses Node::set_external to avoid copying data.
/// If you need a copy of the data unlinked from the input, set into
/// another node.
//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API to_multi_domain(const conduit::Node &mesh,
                                           conduit::Node &dest);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_index(const conduit::Node &mesh,
                                          const std::string &ref_path,
                                          index_t number_of_domains,
                                          Node &index_out);

//-------------------------------------------------------------------------
void CONDUIT_BLUEPRINT_API generate_index_for_single_domain(const conduit::Node &mesh,
                                                            const std::string &ref_path,
                                                            Node &index_out);


//-------------------------------------------------------------------------
/**
 @brief Partition an input mesh or set of mesh domains into a different decomposition,
        according to options. This is the serial implementation.
 @param n_mesh  A Conduit node containing a Blueprint mesh or set of mesh domains.
 @param options A Conduit node containing options that govern the partitioning.
 @param[out]    A Conduit node to accept the repartitioned mesh(es).
 */
void CONDUIT_BLUEPRINT_API partition(const conduit::Node &mesh,
                                     const conduit::Node &options,
                                     conduit::Node &output);

//-------------------------------------------------------------------------
/**
 @brief Map fields from a repartitioned mesh back onto the original mesh.
 @param repart_mesh A conduit node containing a repartitioned mesh with fields
                    to map back.
                    This mesh should have a field "original_element_ids"
                    which is generated by enabling the "mapping" option in
                    partition() (enabled by default).
 @param options     A Conduit node containing options that govern the partitioning.
                    Some available options:
                     "fields": a list of variable names to map back
 @param orig_mesh   A Conduit node containing the original mesh onto which
                    fields will be mapped onto.
 @param fields      A list of field names to map back.
 */
void CONDUIT_BLUEPRINT_API partition_map_back(const conduit::Node& repart_mesh,
                                              const conduit::Node& options,
                                              conduit::Node& orig_mesh);

/**
 @brief Convert the given blueprint mesh into a blueprint table.
 @param mesh    A Conduit node containing a blueprint mesh or set of mesh domains.
 @param options A Conduit node containing options for the flatten operation.
 @param[out] output A Conduit node that will contain the blueprint table output.
    Output will contain the two tables "vertex_data" and "element_data". If one
    of these tables is empty it will be removed from the output.
    (example - If no vertex_data then output will contain one child "element_data" table).

 Supported options:
    "topology": The name of the topology to use for reference. (Defaults to first topology).
    "field_names": A list of field names to include in the output table.
        (Defaults to all fields associated with the active "topology")
    "fill_value": The default value of every element in the table (Default 0)
    "add_domain_info": Determines whether "domain_id", "vertex_id", and "element_id"
        are included in the output table. Should be passed as int. (Default 1 (true))
    "add_element_centers": Determines whether the centers of every element in the mesh
        should be calculated and added to the output table. Should be passed as int.
        (Default 1 (true))
    "add_vertex_locations": Determines whether the coordinate locations of every vertex
        in the mesh should be included in the output table. Should be passed as int.
        (Default 1 (true))
*/
void CONDUIT_BLUEPRINT_API flatten(const conduit::Node &mesh,
                                   const conduit::Node &options,
                                   conduit::Node &output);

//-----------------------------------------------------------------------------
// blueprint::mesh::logical_dims protocol interface
//-----------------------------------------------------------------------------
namespace logical_dims
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &dims,
                                      conduit::Node &info);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::association protocol interface
//-----------------------------------------------------------------------------
namespace association
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &assoc,
                                      conduit::Node &info);
}

//-----------------------------------------------------------------------------
// blueprint::mesh::coordset protocol interface
//-----------------------------------------------------------------------------
namespace coordset
{
    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coordset,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API dims(const conduit::Node &coordset);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API length(const conduit::Node &coordset);

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::uniform protocol interface
    //-------------------------------------------------------------------------
    namespace uniform
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coordset,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_rectilinear(const conduit::Node &coordset,
                                                  conduit::Node &dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_explicit(const conduit::Node &coordset,
                                               conduit::Node &dest);

        //---------------------------------------------------------------------
        // blueprint::mesh::coordset::uniform::origin protocol interface
        //---------------------------------------------------------------------
        namespace origin
        {
            //-----------------------------------------------------------------
            bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &origin,
                                              conduit::Node &info);
        }

        //---------------------------------------------------------------------
        // blueprint::mesh::coordset::uniform::spacing protocol interface
        //---------------------------------------------------------------------
        namespace spacing
        {
            //-----------------------------------------------------------------
            bool CONDUIT_BLUEPRINT_API  verify(const conduit::Node &spacing,
                                               conduit::Node &info);
        }

    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::rectilinear protocol interface
    //-------------------------------------------------------------------------
    namespace rectilinear
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coordset,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_explicit(const conduit::Node &coordset,
                                               conduit::Node &dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::explicit protocol interface
    //-------------------------------------------------------------------------
    namespace _explicit
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coordset,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coordset_idx,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &type,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::coordset::coord_system protocol interface
    //-------------------------------------------------------------------------
    namespace coord_system
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &coord_sys,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API dims(const conduit::Node &topo);

    //-------------------------------------------------------------------------
    index_t CONDUIT_BLUEPRINT_API length(const conduit::Node &topo);

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::points protocol interface
    //-------------------------------------------------------------------------
    namespace points
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::uniform protocol interface
    //-------------------------------------------------------------------------
    namespace uniform
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_rectilinear(const conduit::Node &topo,
                                                  conduit::Node &topo_dest,
                                                  conduit::Node &coords_dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_structured(const conduit::Node &topo,
                                                 conduit::Node &topo_dest,
                                                 conduit::Node &coords_dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &topo,
                                                   conduit::Node &topo_dest,
                                                   conduit::Node &coords_dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::rectilinear protocol interface
    //-------------------------------------------------------------------------
    namespace rectilinear
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_structured(const conduit::Node &topo,
                                                 conduit::Node &topo_dest,
                                                 conduit::Node &coords_dest);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &topo,
                                                   conduit::Node &topo_dest,
                                                   conduit::Node &coords_dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::structured protocol interface
    //-------------------------------------------------------------------------
    namespace structured
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_unstructured(const conduit::Node &topo,
                                                   conduit::Node &topo_dest,
                                                   conduit::Node &coords_dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::unstructured protocol interface
    //-------------------------------------------------------------------------
    namespace unstructured
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo,
                                          conduit::Node &info);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_polygonal(const conduit::Node &topo,
                                                conduit::Node &dest);

        // Note: 
        // this is an alias to `to_polygonal`
        // to_polytopal is a better name for our existing to_polygonal
        // since it supports both polygons and polyhedra
        // to_polygonal may be deprecated in the future
        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API to_polytopal(const conduit::Node &topo,
                                                conduit::Node &dest);


        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_points(const conduit::Node &topo,
                                                   conduit::Node &dest,
                                                   conduit::Node &s2dmap,
                                                   conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_lines(const conduit::Node &topo,
                                                  conduit::Node &dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_faces(const conduit::Node &topo,
                                                  conduit::Node &dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_centroids(const conduit::Node &topo,
                                                      conduit::Node &topo_dest,
                                                      conduit::Node &coords_dest,
                                                      conduit::Node &s2dmap,
                                                      conduit::Node &d2smap);

        //---------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_sides(const conduit::Node &topo,
                                                  conduit::Node &topo_dest,
                                                  conduit::Node &coords_dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);

        //---------------------------------------------------------------------
        // this variant of the function call will also map the fields specified in
        // the options node. The options node can have a child "field_prefix", 
        // which should be a string that allows the user to specify a prefix
        // to insert into the names of the fields stored in fields_dest. The options
        // node can also have a child "field_names", which should be a string or list
        // of strings that allow the user to specify which fields they want to be 
        // mapped from the original set of fields.
        void CONDUIT_BLUEPRINT_API generate_sides(const conduit::Node &topo,
                                                  conduit::Node &topo_dest,
                                                  conduit::Node &coords_dest,
                                                  conduit::Node &fields_dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap,
                                                  const conduit::Node &options);

        //---------------------------------------------------------------------
        // this variant of the function same as generate sides and map fields
        // with empty options
        void CONDUIT_BLUEPRINT_API generate_sides(const conduit::Node &topo,
                                                  conduit::Node &topo_dest,
                                                  conduit::Node &coords_dest,
                                                  conduit::Node &fields_dest,
                                                  conduit::Node &s2dmap,
                                                  conduit::Node &d2smap);


        //---------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_corners(const conduit::Node &topo,
                                                    conduit::Node &topo_dest,
                                                    conduit::Node &coords_dest,
                                                    conduit::Node &s2dmap,
                                                    conduit::Node &d2smap);

        //-------------------------------------------------------------------------
        void CONDUIT_BLUEPRINT_API generate_offsets(const conduit::Node &topo,
                                                    conduit::Node &dest);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &topo_idx,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &type,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::topology::shape protocol interface
    //-------------------------------------------------------------------------
    namespace shape
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &shape,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &matset,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_multi_buffer(const conduit::Node &matset);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_uni_buffer(const conduit::Node &matset);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_element_dominant(const conduit::Node &matset);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_material_dominant(const conduit::Node &matset);

    //-------------------------------------------------------------------------
    // Converts a blueprint matset to the silo style sparse mixed slot 
    // representation.
    //
    // For details about the silo format, see documentation for 
    // 'DBPutMaterial' at:
    // https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/LLNL-SM-654357.pdf
    void CONDUIT_BLUEPRINT_API to_silo(const conduit::Node &matset,
                                       conduit::Node &dest,
                                       const float64 epsilon = CONDUIT_EPSILON);

    //-------------------------------------------------------------------------
    // blueprint::mesh::matset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &matset_idx,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &field,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // Given a blueprint field and matset, converts the matset and the field
    // values + matset_values to the silo style sparse mixed slot
    // representation.
    //
    // For details about the silo format, see documentation for 
    // 'DBPutZZZVar' methods `mixvar` / `mixlen` params at:
    // https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/LLNL-SM-654357.pdf
    void CONDUIT_BLUEPRINT_API to_silo(const conduit::Node &field,
                                       const conduit::Node &matset,
                                       conduit::Node &dest,
                                       const float64 epsilon = CONDUIT_EPSILON);

    //-------------------------------------------------------------------------
    // blueprint::mesh::field::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &field_idx,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::field::basis protocol interface
    //-------------------------------------------------------------------------
    namespace basis
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &basis,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &specset,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::specset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &specset,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &adjset,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_pairwise(const conduit::Node &adjset);

    //-------------------------------------------------------------------------
    bool CONDUIT_BLUEPRINT_API is_maxshare(const conduit::Node &adjset);

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API to_pairwise(const conduit::Node &adjset,
                                           conduit::Node &dest);

    //-------------------------------------------------------------------------
    void CONDUIT_BLUEPRINT_API to_maxshare(const conduit::Node &adjset,
                                           conduit::Node &dest);

    //-------------------------------------------------------------------------
    // blueprint::mesh::adjset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &adjset_idx,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &nestset,
                                      conduit::Node &info);

    //-------------------------------------------------------------------------
    // blueprint::mesh::nestset::index protocol interface
    //-------------------------------------------------------------------------
    namespace index
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &nestset_idx,
                                          conduit::Node &info);
    }

    //-------------------------------------------------------------------------
    // blueprint::mesh::nestset::type protocol interface
    //-------------------------------------------------------------------------
    namespace type
    {
        //---------------------------------------------------------------------
        bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &type,
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
    bool CONDUIT_BLUEPRINT_API verify(const conduit::Node &idx,
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



