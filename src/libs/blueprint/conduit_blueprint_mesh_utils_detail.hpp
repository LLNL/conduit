// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_util_detail.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_UTILS_DETAIL_HPP
#define CONDUIT_BLUEPRINT_MESH_UTILS_DETAIL_HPP

//-----------------------------------------------------------------------------
// std includes
//-----------------------------------------------------------------------------
#include <map>
#include <set>
#include <string>
#include <vector>

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
// -- begin conduit::blueprint::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::detail --
//-----------------------------------------------------------------------------
namespace detail
{

/**
 This class accepts a node containing a topology and it puts that topology
 through a shape cascade. This means turning elements->faces->edges->points
 and producing associations (maps) that allow the caller to query various
 relations such as the elements that own a face, or the lines in an element.
 All of the various entities in the different topology levels have been made
 unique.

 This class differs from the previous implementation in the following ways:

 * It uses recursion and template functions to make sure we can access
   connectivity data using either native pointers or and index_t_accessor
   so no Conduit casts are needed.

 * Associations are mostly created only when asked for (or indirectly needed)
   and this is done largely after the cascade has been performed.

 * Associations are stored contiguously in memory or are implicit.

 * Access to associations is made using a "vector_view" which returns a window
   into the bulk association data so the caller can use the data without
   having to copy it out. Local association data is returned as a "range_vector"
   so the local ranges can be represented implicitly while still looking
   like a vector.
*/
class CONDUIT_BLUEPRINT_API NewTopologyMetadata
{
    struct association
    {
        // The association owns this storage.
        std::vector<index_t> data;
        std::vector<index_t> sizes;
        std::vector<index_t> offsets;
        // The association is using data from a topology.
        index_t    *data_ptr{nullptr};
        index_t    *sizes_ptr{nullptr};
        index_t    *offsets_ptr{nullptr};
        index_t     data_ptr_size{0};
        index_t     sizes_ptr_size{0};
        index_t     offsets_ptr_size{0};
        // Other fields
        int                  single_size{1};
        bool                 requested{false};

        // Helper methods.
        bool get_data(const index_t *&array, index_t &arraylen) const;
        bool get_sizes(const index_t *&array, index_t &arraylen) const;
        bool get_offsets(const index_t *&array, index_t &arraylen) const;

        inline std::pair<index_t *, index_t> get_data(index_t entity_id) const;
        inline index_t get_size(index_t entity_id) const;
        inline index_t get_offset(index_t entity_id) const;
    };

    // Data members
    const conduit::Node *topo, *coords;
    const ShapeCascade topo_cascade;
    const ShapeType topo_shape;
    size_t lowest_cascade_dim;
    index_t coords_length;
    conduit::Node dim_topos[MAX_ENTITY_DIMS];
    index_t dim_topo_lengths[MAX_ENTITY_DIMS];
    association G[MAX_ENTITY_DIMS][MAX_ENTITY_DIMS];  

public:
    constexpr static size_t MAX_ENTITY_DIMS = 4;

    enum IndexType { GLOBAL = 0, LOCAL = 1 };

    //-----------------------------------------------------------------------
    /**
     @brief Legacy constructor, which builds all of the topology levels in the shape
            cascade as well as all associations.

     @param topology     The input topology node.
     @param coordset     The input coordset associated with the topology.
     */
    NewTopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset);

    //-----------------------------------------------------------------------
    /**
     @brief Constructor for the NewTopologyMetadata class. This constructor
            lets the caller be more selective about which topology levels and 
            associations are created, possibly saving time.

     @param topology     The input topology node.
     @param coordset     The input coordset associated with the topology.
     @param lowest_dim   The lowest level of shape cascade that we're interested in.
                         If we only want faces, for example, then we don't need to
                         refine to edges or points.
     @param desired      A vector of (entity_dim,assoc_dim) pairs that indicate
                         the associations that will be requested by the client.
     */
    NewTopologyMetadata(const conduit::Node &topology,
                        const conduit::Node &coordset,
                        size_t lowest_dim,
                        const std::vector<std::pair<size_t,size_t> > &desired_maps);

    //-----------------------------------------------------------------------
    /**
     @brief Get the highest shape dimension.
     @return The highest shape dimension.
     */
    int dimension() const { return topo_shape.dim; }

    //-----------------------------------------------------------------------
    /**
     @brief Get the topologies array.
     @return The topologies array.
     */
    const conduit::Node *get_topologies() const { return dim_topos; }

    //-----------------------------------------------------------------------
    /**
     @brief Get the legnths arrays for the topologies. Any topologies that
            were not produced will have length 0.
     @return The topology lengths array.
     */
    const index_t *get_topology_lengths() const { return dim_topo_lengths; }

    //-----------------------------------------------------------------------
    /**
     @brief Get a global association for a particular entity_id in the entity_dim
            dimension. This lets us ask a question like: which face ids are in
            element 5? Or, which faces contain edge 12?

     @param entity_id The entity id whose information we want to obtain. This
                      value must be valid for the entity dimension in question.
     @param entity_dim The dimension of the entity whose information we want.
     @param assoc_dim  The dimension of the association we want.

     Example: Which faces are in element 5?
              get_global_association(5,  // element 5
                                     3,  // elements are 3 dimensional
                                     2); // faces are 2 dimensional

     Example 2: Which faces contain edge 2?
              get_global_association(2,  // edge 2
                                     1,  // edges are 1 dimensional
                                     2); // faces are 2 dimensional

     @note The data are returned as a vector_view, which is an object that acts
           like a vector but does not have to copy bulk data. The data it points
           to for a given return are in the associations.
     */
    conduit::vector_view<index_t> get_global_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const;

    conduit::range_vector<index_t> get_local_association(index_t entity_id, index_t entity_dim, index_t assoc_dim) const;

    //-----------------------------------------------------------------------
    /**
     @brief Get whether the association e,a was requested when the object
            was initialized. If so, the association will exist. Otherwise,
            the association does not exist.

     @param entity_dim The starting dimension.
     @param assoc_dim  The destination dimension.

     @return True if the association exists; False otherwise.
     */
    bool association_requested(index_t entity_dim, index_t assoc_dim) const;

    //-----------------------------------------------------------------------
    /**
      @brief Return all of the data for an association by COPYING it into the 
             provided Conduit node.

      @param type Whether we want GLOBAL or LOCAL data.
      @param src_dim The source dimension of the desired association.
      @param dst_dim The destination dimension of the desired association.
      @param[out] map_node The Conduit node that will contain the copied association data.

      @note This method guarantees that all bulk arrays for values, sizes, and
            offsets will be index_t.

      @return True if the map exists(was requested); False otherwise.
     */
    bool get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const;
private:

    //-----------------------------------------------------------------------
    /**
     @brief This method causes the topologies and maps to be created,
            initializing the object.

     @param desired A vector of (e,a) pairs that indicate which associations
                    we want to compute.
     */
    void initialize(const std::vector<std::pair<size_t, size_t> > &desired);

    //-----------------------------------------------------------------------
    /**
     @brief Examine the vector of desired maps and mark the associations
            as requested.

     @param desired A vector of (e,a) pairs that indicate which associations
                    we want to compute.
     */
    void request_associations(const std::vector<std::pair<size_t, size_t> > &desired);

    //-----------------------------------------------------------------------
    /**
     @brief Build any associations that were not produced during the cascade.
     */
    void build_associations();

    //-----------------------------------------------------------------------
    /**
     @brief Build the association from dimension e to dimension 0.

     @param e The source dimension.
     */
    void build_association_e_0(int e);

    //-----------------------------------------------------------------------
    /**
     @brief Given a 3D non-PH shape, return the set of unique line segments
            in traversal order.

     @param shape The input shape, which should be a 3D non-PH element.
     @return A vector of edge endpoint pairs that represent the unique edges
             for the element.
     */
    std::vector<index_t> embedding_3_1_edges(const ShapeType &shape) const;

    //-----------------------------------------------------------------------
    /**
     @brief Build the G(3,1) and G(3,0) associations together because they
            need to do most of the same work.
     */
    void build_association_3_1_and_3_0();
    void build_association_3_1_and_3_0_ph();
    void build_association_3_1_and_3_0_nonph();

    //-----------------------------------------------------------------------
    /**
     @brief Print association G[e][a].
     */
    void print_association(int e, int a) const;

    //-----------------------------------------------------------------------
    /**
     @brief This method builds a child to parent association by reversing
            an existing parent to child association.

     @param e The entity dimension (the thing we have: element=3,face=2,...)
     @param a The association dimension (the thing we want to know about with
              respect to the entity dimension). e < a.
     */
    void build_child_to_parent_association(int e, int a);

    /**
      @brief Return all of the data for a GLOBAL association by COPYING it into
             the provided Conduit node.

      @param src_dim The source dimension of the desired association.
      @param dst_dim The destination dimension of the desired association.
      @param[out] map_node The Conduit node that will contain the copied association data.

      @note This method guarantees that all bulk arrays for values, sizes, and
            offsets will be index_t.
     */
    bool get_global_dim_map(index_t src_dim, index_t dst_dim, Node &map_node) const;
    bool get_local_dim_map(index_t src_dim, index_t dst_dim, Node &map_node) const;
    index_t get_local_association_entity_range(int e, int a) const;

    //-----------------------------------------------------------------------
    /**
     @brief Takes the input topology and reuses it has the highest topology.
            The method makes sure that the topology will have offsets too.
     */
    void make_highest_topology();

    //-----------------------------------------------------------------------
    /**
     @brief Identify sequences of unique ids and give them the same unique id
            that is an increasing number rather than a random id.

     @param faceid_to_ef A vector of pairs that contain a faceid to element face.
                         Or, rather, a hashid to element entity.
     @param ef_to_unique: A reverse map of element face to unique id.

     @return The number of unique ids.
    */
    index_t
    make_unique(const std::vector<std::pair<uint64, uint64>> &faceid_to_ef, 
                std::vector<std::pair<uint64, uint64>> &ef_to_unique) const;

    //-----------------------------------------------------------------------
    /**
     @brief This method casts the sizes node into some usable types and then
            invokes dispatch_shape_ph built for those types.

     @param subel The polyhedral subelement node.
     @param sizes The sizes node under the subelement node.
     */
    void dispatch_connectivity_ph(const conduit::Node &subel,
                                  const conduit::Node &sizes);

    //-----------------------------------------------------------------------
    /**
     @brief This method is called with a shape and connectivity and it casts
            the connectivity to usable types and calls dispatch_shape methods
            that operate on those types.

     @param shape The shape that the connectivity represents.
     @param conn  A node that contains the bulk connectivity data.
     */
    void  dispatch_connectivity(const ShapeType &shape,
                                const conduit::Node &conn);

    //-----------------------------------------------------------------------
    /**
     @brief This method accepts a node containing polyhedral subelement data,
            which contains the actual face definitions that make up the PH
            mesh. We invoke the method to make the face connectivity and then
            dispatch again to make the lines.

     @param subel The polyhedral subelement node.
     @param sizes The sizes in the subelement node cast to a more usable type
                  such as a pointer or an accessor.
     @param sizeslen The number of elements in sizes.
     */
    template <typename ConnType>
    void
    dispatch_shape_ph(const conduit::Node &subel,
                      const ConnType &sizes,
                      size_t sizeslen)
    {
        const int embed_dim = 2;

        // Make faces from the PH faces.
        make_embedded_connectivity_ph(subel, sizes, sizeslen);

        // Make lines. Note that we get the embed shape from the 2D topo
        // in case we set it to quads or tris. The topo_cascade would
        // return polygons but we may do a little better with quads/tris
        // if we can assume those.
        const conduit::Node &econn = dim_topos[embed_dim].fetch_existing("elements/connectivity");
        ShapeType embed_shape(dim_topos[embed_dim]);
        dispatch_connectivity(embed_shape, econn);
    }

    //-----------------------------------------------------------------------
    /**
     @brief This method is called with a shape and connectivity data that
            is of a concrete type (or an accessor). The method makes calls
            to create the connectivity for the embedded shape (e.g. faces 
            from elements) and then calls the next level down when done.

     @param shape The shape that the connectivity represents.
     @param conn  An array or accessor that contains the connectivity data.
     @param connlen The number of connectivity array elements.
     */
    template <typename ConnType>
    void
    dispatch_shape(const ShapeType &shape,
                   const ConnType &conn,
                   size_t connlen)
    {
        if(!shape.is_polygonal())
        {
            // Make faces for any of the 3D implicit cell types.
            if(shape.dim == 3)
            {
                make_embedded_connectivity(shape, conn, connlen);

                // Make lines.
                int embed_dim = shape.dim - 1;
                const conduit::Node &econn = dim_topos[embed_dim].fetch_existing("elements/connectivity");
                ShapeType embed_shape = topo_cascade.get_shape(embed_dim);
                dispatch_connectivity(embed_shape, econn);
            }
            else if(shape.dim == 2)
            {
                // All input shapes are the same (e.g. quads or tris).
                // Make lines.
                make_embedded_connectivity(shape, conn, connlen);
            }
        }
        else
        {
            // The shape is said to contain polygons so we have to be a bit more
            // general in how we traverse the connectivity. We want sizes/offsets.

            // TODO: write make_embedded_connectivity_mixed
        }
    }

    //-----------------------------------------------------------------------
    /**
     @brief This method accepts a node containing polyhedral subelement data,
            which contains the actual face definitions that make up the PH
            mesh. We use these faces as the 2D topology as it is already
            made of polygons. We scan it too to see whether it contains all
            triangles or quads, in which case, we change the shape type to
            make downstream line production easier.

     @param subel The polyhedral subelement node.
     @param sizes The sizes in the subelement node cast to a more usable type
                  such as a pointer or an accessor.
     @param sizeslen The number of elements in sizes.
     */
    template <typename ConnType>
    void
    make_embedded_connectivity_ph(const conduit::Node &subel,
                                  const ConnType &sizes,
                                  size_t sizeslen)
    {
        const int embed_dim = 2;

        // Use the subelement information from the PH mesh as the embedded
        // connectivity for the 2D faces.
        conduit::Node &node = dim_topos[embed_dim];
        node["type"] = "unstructured";
        node["coordset"] = coords->name();
        node["elements/shape"] = subel["shape"].as_string();
        node["elements/connectivity"].set_external(subel["connectivity"]);
        // PH geometries should have sizes and offsets too.
        if(subel.has_child("sizes"))
            node["elements/sizes"].set_external(subel["sizes"]);
        if(subel.has_child("offsets"))
            node["elements/offsets"].set_external(subel["offsets"]);
#if 1
        // Check whether the sizes are the same. If they are then we can convert
        // from "polygon" types to tri, or quad.
        bool istri = sizes[0] == 3;
        bool isquad = sizes[0] == 4;
        if(istri || isquad)
        {
            bool same = true;
            for(size_t i = 1; i < sizeslen && same; i++)
                same &= sizes[0] == sizes[i];

            if(same && istri)
            {
                node["elements/shape"] = "tri";
            }
            else if(same && isquad)
            {
                node["elements/shape"] = "quad";
            }
        }
#endif
    }

    //-----------------------------------------------------------------------
    /**
      @brief Make the embedded connectivity from the input connectivity and
             store the results in dim_topos[shape.dim-1]. If we're passed a
             3D cell, we make 2D faces. If we're passed 2D faces, we make 1D
             lines.

      @param shape The input shape.
      @param conn  The connectivity data. This is usually a pointer but can
                   be an accessor too.
      @param connlen The length of the connectivity data.

      - This method assumes that all elements in the connectivity are the 
        same type.   
      - This is a template method so we can pass in basic types (like 
        const int*) for the connectivity.
    */
    template <typename ConnType>
    void
    make_embedded_connectivity(const ShapeType &shape, const ConnType &conn, index_t connlen)
    {
        int embed_dim = shape.dim - 1;
        ShapeType embed_shape = topo_cascade.get_shape(embed_dim);

// TODO: rename some things so the terms are more generic.

        // Get sizes from the shape and embed_shape.
        index_t points_per_elem = shape.indices;
        index_t faces_per_elem = shape.embed_count;
        index_t points_per_face = embed_shape.indices;
        index_t nfacepts = faces_per_elem * points_per_face;

        index_t nelem = connlen / points_per_elem;
        auto nelem_faces = nelem * faces_per_elem;
#ifdef DEBUG_PRINT
        cout << "=======================================================" << endl;
        cout << "make_embedded_connectivity: shape_dim=" << shape.dim << endl;
        cout << "=======================================================" << endl;
        cout << "shape=" << shape.type << endl;
        cout << "embed_shape=" << embed_shape.type << endl;
        cout << "points_per_elem="<<points_per_elem<<endl;
        cout << "faces_per_elem="<<faces_per_elem<<endl;
        cout << "points_per_face="<<points_per_face<<endl;
        cout << "nfacepts="<<nfacepts<<endl;
        cout << "nelem="<<nelem<<endl;
        cout << "nelem_faces="<<nelem_faces<<endl;
        cout << "conn={";
        for(index_t i = 0; i < connlen; i++)
            cout << conn[i] << ", ";
        cout << "}" << endl << endl;
#endif

        // Iterate over each hex cell and compute a faceid for it. Store
        // these in faceid_to_ef. The "ef" stands for element face, which
        // is the element id * faces_per_elem + face.
        std::vector<std::pair<uint64, uint64>> faceid_to_ef(nelem_faces);

#pragma omp parallel for
        for(size_t elem = 0; elem < nelem; elem++)
        {

// TODO: it might be good to keep these the same as the connectivity element
//       type rather than index_t (in case sizeof(index_t) > sizeof(elem_t).
//       That would hash fewer bytes and possibly eliminate casts.

            // Get the element faces, storing them all in face_pts.
            index_t elemstart = elem * points_per_elem;
            index_t face_pts[nfacepts];
            for(size_t i = 0; i < nfacepts; i++)
                face_pts[i] = conn[elemstart + shape.embedding[i]];

            // Make a unique id for each face.
            index_t facestart = elem * faces_per_elem;
            for(size_t face = 0; face < faces_per_elem; face++)
            {
                // Sort the face's points
                index_t *face_pts_start = &face_pts[face * points_per_face];
                index_t *face_pts_end = face_pts_start + points_per_face;

                // encode element and face into element_face.
                uint64 element_face = facestart + face;

#if 0
                cout << "elem=" << elem << ", face=" << face
                     << ", element_face=" << element_face;
                cout << ", pts={";
                for(int q = 0; q < points_per_face; q++)
                    cout << std::setw(2) << face_pts_start[q] << ", ";
                cout << "}, sort={";
#endif
                std::sort(face_pts_start, face_pts_end); // Better 4 item sort
                uint64 faceid = hash_ids(face_pts_start, points_per_face);
#if 0
                for(int q = 0; q < points_per_face; q++)
                    cout << std::setw(2) << face_pts_start[q] << ", ";
                cout << "}, faceid=" << faceid << endl;
#endif

                // Store the faceid and ef values.
                faceid_to_ef[element_face] = std::make_pair(faceid, element_face);
            }
        }

#ifdef DEBUG_PRINT
        cout << "faceid_to_ef = " << faceid_to_ef << endl;
#endif

        // Sort faceid_to_ef so any like faces will be sorted, first by their
        // general faceid, then by their elemface "ef", which should keep the
        // elements in order.
        std::sort(OPTIONAL_PARALLEL_EXECUTION_POLICY faceid_to_ef.begin(), faceid_to_ef.end());
#ifdef DEBUG_PRINT
        cout << "faceid_to_ef.sorted = " << faceid_to_ef << endl;
#endif

        // Faces are sorted. We probably do not want to necessarily create faces
        // in this order though since it would create faces in random order.
        // The spans with like ids should only be 1 or 2 faces long, assuming the
        // hashing did its job correctly.
        std::vector<std::pair<uint64, uint64>> ef_to_unique(nelem_faces);
        index_t unique = make_unique(faceid_to_ef, ef_to_unique);
#ifdef DEBUG_PRINT
        cout << "unique = " << unique << endl;
        cout << "ef_to_unique = " << ef_to_unique << endl;
#endif

        // Sort on ef to get back to a ef->unique mapping.
        std::sort(OPTIONAL_PARALLEL_EXECUTION_POLICY
                  ef_to_unique.begin(), ef_to_unique.end(),
            [&](const std::pair<uint64, uint64> &lhs, const std::pair<uint64, uint64> &rhs)
        {
            // Only sort using the ef value.
            return lhs.first < rhs.first;
        });
#ifdef DEBUG_PRINT
        cout << "ef_to_unique.sorted = " << ef_to_unique << endl;
#endif

        // Store the new embed connectivity data in Conduit nodes.
        conduit::Node &node = dim_topos[embed_shape.dim];
        node["type"] = "unstructured";
        node["coordset"] = coords->name();
        node["elements/shape"] = embed_shape.type;
        node["elements/connectivity"].set(conduit::DataType(index_t_id(), unique * points_per_face));
        index_t *embed_conn = as_index_t_ptr(node["elements/connectivity"]);
        index_t embed_conn_idx = 0;

        // Now ef_to_unique contains a list of unique face ids but they are not
        // expressed in element-creation order.
        //
        // elem 0 faces            elem 1 faces
        // {99, 1, 4, 2, 11, 16}, {4, 22, 44, 67, 55, 12}, ...
        //
        // Now we want to iterate over the ef_to_unique in element face order 
        // and renumber the faces in another pass.
        //
        // If the (3,2) or (2,1) association was requested then we do a little
        // more work to save the map as we build the connectivity. Otherwise,
        // the 2 code blocks do the same thing. We separate to remove branches.
        std::vector<unsigned char> avail(unique, 1);
        if(G[shape.dim][embed_shape.dim].requested)
        {
            // Association data. This is the set of indices that point to the
            // embedded shapes from each input shape. Think of it like the set
            // of polyhedral face ids for each element but it works for other
            // dimensions too.
            std::vector<index_t> face_reorder(unique);
            auto &embed_refs = G[shape.dim][embed_shape.dim].data;
            embed_refs.resize(nelem_faces, 0);
#ifdef DEBUG_PRINT
            cout << "Building G(" << shape.dim << ", " << embed_shape.dim << ")" << endl;
            cout << "points_per_face=" << points_per_face << endl;
            cout << "unique=" << unique << endl;
            cout << "faces_per_elem=" << faces_per_elem << endl;
            cout << "nelem_faces=" << nelem_faces << endl;
#endif

            // Save how many embedded shapes an element would have.
            G[shape.dim][embed_shape.dim].single_size = shape.embed_count;

            // Make the embedded connectivity (and map data)
            index_t embed_refs_idx = 0, final_faceid = 0;
            for(size_t ef = 0; ef < nelem_faces; ef++)
            {
                uint64 unique_face_id = ef_to_unique[ef].second;

                if(avail[unique_face_id])
                {
                    // Store the map data.
                    face_reorder[unique_face_id] = final_faceid++;
                    embed_refs[embed_refs_idx++] = face_reorder[unique_face_id];

                    avail[unique_face_id] = 0;

                    // Emit the face definition (as defined by the first element
                    // that referenced it.
                    int faceelem = ef / faces_per_elem;
                    int facecase = ef % faces_per_elem;
                    index_t elemstart = faceelem * points_per_elem;
                    index_t *embed = &shape.embedding[facecase * points_per_face];
                    for(size_t i = 0; i < points_per_face; i++)
                        embed_conn[embed_conn_idx++] = conn[elemstart + embed[i]];
                }
                else
                {
                    // Store the map data. The entity has already been output,
                    // add its reordered number to the refs.
                    embed_refs[embed_refs_idx++] = face_reorder[unique_face_id];
                }
            }
#ifdef DEBUG_PRINT
            cout << "final embed_refs_idx=" << embed_refs_idx << endl << endl;
            cout << "embed_refs=" << embed_refs << endl;
#endif
        }
        else
        {
            // Make the embedded connectivity
            for(size_t ef = 0; ef < nelem_faces; ef++)
            {
                uint64 unique_face_id = ef_to_unique[ef].second;
                if(avail[unique_face_id])
                {
                    avail[unique_face_id] = 0;

                    // Emit the face definition (as defined by the first element
                    // that referenced it.
                    int faceelem = ef / faces_per_elem;
                    int facecase = ef % faces_per_elem;
                    index_t elemstart = faceelem * points_per_elem;
                    index_t *embed = &shape.embedding[facecase * points_per_face];
                    for(size_t i = 0; i < points_per_face; i++)
                        embed_conn[embed_conn_idx++] = conn[elemstart + embed[i]];
                }
            }
        }
#if 1
// It has to be done like this for the connectivity nodes to match the old TopologyMetadata.
// We have to have offsets for the new topology to match what TopologyMetadata makes.
// Otherwise we could turn it off and let the maps handle the offsets implicitly.

        // Generate offsets in the output connectivity. Some downstream algorithms want it.
        node["elements/offsets"].set(conduit::DataType(index_t_id(), unique));
        index_t *offsets = as_index_t_ptr(node["elements/offsets"]);
        for(size_t ei = 0; ei < unique; ei++)
            offsets[ei] = points_per_face * ei;

#else
// I changed it to this to try and work around a valgrind bug that I saw somewhere.

        // Generate offsets in the output connectivity. Some downstream algorithms want it.
        node["elements/offsets"].set(conduit::DataType(index_t_id(), nelem_faces));
        index_t *offsets = as_index_t_ptr(node["elements/offsets"]);
        for(size_t ei = 0; ei < nelem_faces; ei++)
            offsets[ei] = points_per_face * ei;
#endif
    }
};

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::detail::reference --
//-----------------------------------------------------------------------------
namespace reference
{

// Keep this older class as a reference for now. Remove it when the new
// implementation is up to par.
class CONDUIT_BLUEPRINT_API TopologyMetadata
{
public:
    enum IndexType { GLOBAL = 0, LOCAL = 1 };

    // The 'IndexType' indicates the index space to be used when referring to
    // entities within this topological cascade. The types have the following
    // meanings:
    //
    // - GLOBAL: The unique index for the entity relative to the entire topology.
    //   Though a point may be shared by many lines/faces/cells, it will only
    //   have one global index. This is most commonly used for entity identification.
    // - LOCAL: The index of the entity relative to a cascade context. A point
    //   will have one local index for each line/face/cell that it participates
    //   in along the cascade. This is most commonly used to determine entity orientation.
    //
    // To clarify, consider the following example, with the following local and
    // global identifiers:
    //
    // - GLOBAL Scheme: Each entity has 1 unique identifier depending on FIFO
    //   cascade iteration:
    //
    //   p3               p4              p5
    //   +----------------+----------------+
    //   |       l2       |       l6       |
    //   |                |                |
    //   |                |                |
    //   |l3     f0       |l1     f1     l5|
    //   |                |                |
    //   |                |                |
    //   |       l0       |       l4       |
    //   +----------------+----------------+
    //   p0               p1              p2
    //
    // - LOCAL Scheme: Each entity has an identifier for each occurence within
    //   the cascade:
    //
    //    p5            p4 p13          p12
    //   +----------------+----------------+
    //   |p6     l2     p3|p14    l6    p11|
    //   |                |                |
    //   |                |                |
    //   |l3     f0     l1|l7     f1     l5|
    //   |                |                |
    //   |                |                |
    //   |p7     l0     p2|p15    l4    p10|
    //   +----------------+----------------+
    //    p0            p1 p8            p9
    //

    TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset);

    void add_entity_assoc(IndexType type, index_t e0_id, index_t e0_dim, index_t e1_id, index_t e1_dim);

    std::vector<index_t> &get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim);
    const std::vector<index_t> &get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim) const;
    void get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const;
    void get_point_data(IndexType type, index_t point_id, Node &data) const;

    index_t get_length(index_t dim = -1) const;
    index_t get_embed_length(index_t entity_dim, index_t embed_dim) const;

    std::string to_json() const;

    void expand_assoc_capacity(IndexType type, index_t idx, index_t dim);
    inline index_t next_global_id(index_t dim) const
    {
        index_t tdim1 = topo_shape.dim + 1;
        return dim_geassocs_maps[dim].size() / tdim1;
    }
    inline index_t next_local_id(index_t dim) const
    {
        index_t tdim1 = topo_shape.dim + 1;
        return dim_leassocs_maps[dim].size() / tdim1;
    }

    const conduit::Node *topo, *cset;
    const conduit::DataType int_dtype, float_dtype;
    const ShapeCascade topo_cascade;
    const ShapeType topo_shape;

    /*
      dim_geassocs_maps[dim][element * (topodims+1) + dim] -> associates vector
          [dim 0]
          [dim 1]--->elements
          [dim 2]      [ei=0, dim=0]
          [dim 3]      [ei=0, dim=1]
                       [ei=0, dim=2]--->associates {1,3,5}
                       [ei=0, dim=3]
                       [ei=1, dim=0]
                       [ei=1, dim=1]
                       ...
                       [ei=nelem-1, dim=3]

     */

    // per-dimension topology nodes (mapped onto 'cset' coordinate set)
    std::vector< conduit::Node > dim_topos;
    // per-dimension maps from global entity ids to per-dimension global associate ids
    std::vector< std::vector< std::vector<index_t> > > dim_geassocs_maps;
    // per-dimension maps from local entity ids to per-dimension local associate ids
    std::vector< std::vector< std::vector<index_t> > > dim_leassocs_maps;
    // per-dimension mapping from local entity ids to global entity ids (delegates)
    std::vector< std::vector<index_t> > dim_le2ge_maps;
};

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::detail::reference --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::detail --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
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
