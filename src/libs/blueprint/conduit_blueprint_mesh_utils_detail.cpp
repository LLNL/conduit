// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_utils_detail.cpp
///
//-----------------------------------------------------------------------------

#define EA_INDEX(E,A) ((E)*(MAX_ENTITY_DIMS)+(A))
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

//-----------------------------------------------------------------------
/**
 @brief Hash a series of bytes using a Jenkins hash forwards and backwards
        and combine the results into a uint64 hash.

 @param data A series of bytes to hash.
 @param n The number of bytes.

 @return A hash value that represents the bytes.
 */
uint64
hash_uint8(const uint8 *data, index_t n)
{
    uint32 hashF = 0;

    // Build the length into the hash so {1} and {0,1} hash to different values.
    const auto ldata = reinterpret_cast<const uint8 *>(&n);
    for(size_t e = 0; e < sizeof(n); e++)
    {
      hashF += ldata[e];
      hashF += hashF << 10;
      hashF ^= hashF >> 6;
    }
    // hash the data forward and backwards.
    uint32 hashB = hashF;
    for(index_t i = 0; i < n; i++)
    {
        hashF += data[i];
        hashF += hashF << 10;
        hashF ^= hashF >> 6;

        hashB += data[n - 1 - i];
        hashB += hashB << 10;
        hashB ^= hashB >> 6;
    }
    hashF += hashF << 3;
    hashF ^= hashF >> 11;
    hashF += hashF << 15;

    hashB += hashB << 3;
    hashB ^= hashB >> 11;
    hashB += hashB << 15;

    // Combine the forward, backward into a uint64.
    return (static_cast<uint64>(hashF) << 32) | static_cast<uint64>(hashB);
}

//-----------------------------------------------------------------------
/**
 @brief Make a hash value from a series of index_t values.

 @param data A sorted list of ids.
 @param n The number of ids.

 @return A hash value that represents the ids.
 */
inline uint64
hash_ids(const index_t *data, index_t n)
{
    return hash_uint8(reinterpret_cast<const uint8 *>(data), n * sizeof(index_t));
}

//-----------------------------------------------------------------------
// TODO: move to conduit::DataType.
inline conduit::DataType::TypeID
index_t_id()
{
#ifdef CONDUIT_INDEX_32
    return conduit::DataType::INT32_ID;
#else
    return conduit::DataType::INT64_ID;
#endif
}

//-----------------------------------------------------------------------
// TODO: Add in conduit::Node.
inline index_t *
as_index_t_ptr(conduit::Node &n)
{
#ifdef CONDUIT_INDEX_32
    return n.as_int32_ptr();
#else
    if(n.dtype().id() != conduit::DataType::INT64_ID)
       cout << "as_index_t_ptr: node " << n.name() << " is not index_t. It is " << n.dtype().name() << endl;
    return n.as_int64_ptr();
#endif
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
bool
NewTopologyMetadata::association::get_data(const index_t *&array,
    index_t &arraylen) const
{
    bool retval = false;
    if(data_ptr)
    {
        array = data_ptr;
        arraylen = data_ptr_size;
        retval = true;
    }
    else if(!data.empty())
    {
        array = &data[0];
        arraylen = data.size();
        retval = true;
    }
    else
    {
        array = nullptr;
        arraylen = 0;
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Get the data for one entity.

 @param entity_id The entity id whose data we want to return.
 @return A pair containing the data pointer and array size.
 */
std::pair<index_t *, index_t>
NewTopologyMetadata::association::get_data(index_t entity_id) const
{
    index_t size = get_size(entity_id);
    index_t offset = get_offset(entity_id);

    std::pair<index_t *, index_t> retval;
    if(data_ptr)
    {
        retval = std::make_pair(const_cast<index_t *>(data_ptr + offset), size);
    }
    else if(!data.empty())
    {
        retval = std::make_pair(const_cast<index_t *>(&data[offset]), size);
    }
    else
    {
        retval = std::make_pair(nullptr, 0);
    }
    return retval;
}

//---------------------------------------------------------------------------
index_t
NewTopologyMetadata::association::get_size(index_t entity_id) const
{
    index_t retval = 0;
    if(sizes_ptr)
        retval = sizes_ptr[entity_id];
    else if(!sizes.empty())
        retval = sizes[entity_id];
    else
        retval = single_size;
    return retval;
}

//---------------------------------------------------------------------------
index_t
NewTopologyMetadata::association::get_offset(index_t entity_id) const
{
    index_t retval = 0;
    if(offsets_ptr)
        retval = offsets_ptr[entity_id];
    else if(!offsets.empty())
        retval = offsets[entity_id];
    else
        retval = entity_id * single_size;
    return retval;
}

//---------------------------------------------------------------------------
bool
NewTopologyMetadata::association::get_sizes(const index_t *&array,
    index_t &arraylen) const
{
    bool retval = false;
    if(sizes_ptr)
    {
        array = sizes_ptr;
        arraylen = sizes_ptr_size;
        retval = true;
    }
    else if(!sizes.empty())
    {
        array = &sizes[0];
        arraylen = sizes.size();
        retval = true;
    }
    else
    {
        array = nullptr;
        arraylen = 0;
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
NewTopologyMetadata::association::get_offsets(const index_t *&array,
    index_t &arraylen) const
{
    bool retval = false;
    if(offsets_ptr)
    {
        array = offsets_ptr;
        arraylen = offsets_ptr_size;
        retval = true;
    }
    else if(!offsets.empty())
    {
        array = &offsets[0];
        arraylen = offsets.size();
        retval = true;
    }
    else
    {
        array = nullptr;
        arraylen = 0;
    }
    return retval;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

NewTopologyMetadata::NewTopologyMetadata(const conduit::Node &topology,
    const conduit::Node &coordset) : topo(&topology), coords(&coordset),
    topo_cascade(topology), coords_length(0), topo_shape(topology),
    lowest_cascade_dim(0)
{
    // Select all maps that could be valid for this shape.
    std::vector<std::pair<size_t, size_t> > desired;
    for(size_t e = 0; e <= topo_shape.dim; e++)
    for(size_t a = 0; a <= topo_shape.dim; a++)
        desired.push_back(std::make_pair(e,a));

    initialize(desired);
}

//------------------------------------------------------------------------------
NewTopologyMetadata::NewTopologyMetadata(const conduit::Node &topology,
    const conduit::Node &coordset,
    size_t lowest_dim,
    const std::vector<std::pair<size_t, size_t> > &desired) :
    topo(&topology), coords(&coordset), topo_cascade(topology), topo_shape(topology),
    lowest_cascade_dim(lowest_dim), coords_length(0)
{
    initialize(desired);
}

//------------------------------------------------------------------------------
void
NewTopologyMetadata::initialize(const std::vector<std::pair<size_t, size_t> > &desired)
{
    // The lowest cascade dim is less than or equal to the topo_shape.dim.
    if(lowest_cascade_dim > topo_shape.dim)
    {
        CONDUIT_ERROR("lowest_cascade_dim is greater than the topo_shape.dim!");
    }

    // Initialize nodes/lengths.
    for(int dim = 0; dim < MAX_ENTITY_DIMS; dim++)
    {
        dim_topos[dim].reset();
        dim_topo_lengths[dim] = 0;
    }

    // Request the associations that we need to build.
    request_associations(desired);

    // Make the highest topology
    if(topo_shape.dim > 0)
        make_highest_topology();

    // Make point topology
    coords_length = coordset::length(*coords);
    if(lowest_cascade_dim == 0)
        make_point_topology();

    // If we have lines or faces to make, make them.
    if(lowest_cascade_dim < topo_shape.dim && topo_shape.dim > 1)
    {
        if(topo_shape.is_polyhedral())
        {
            const conduit::Node &subel = topo->fetch_existing("subelements");
            const conduit::Node &sizes = subel.fetch_existing("sizes");
            dispatch_connectivity_ph(subel, sizes);
        }
        else
        {
            const conduit::Node &conn = topo->fetch_existing("elements/connectivity");
            dispatch_connectivity(topo_shape, conn);
        }
    }

    build_associations();
}

//------------------------------------------------------------------------------
void
NewTopologyMetadata::request_associations(const std::vector<std::pair<size_t, size_t> > &desired)
{
    for(size_t i = 0; i < desired.size(); i++)
    {
        auto e = desired[i].first;
        auto a = desired[i].second;
        if(e > topo_shape.dim || a > topo_shape.dim)
        {
            CONDUIT_ERROR("An invalid (e,a) association index was selected: (" <<
               e << ", " << a << ")");
        }
        G[e][a].requested = true;
        if(e < a)
        {
            // This is a child to parent association.
            // Example: G[1][3] requested - all of the elements that contain
            //          edge i. G[1][3] relies on G[3][1] existing.
            G[a][e].requested = true;
        }
    }

#if 0
// No longer true - need to do this for PH cells.
    // A couple of our association cases are built using other associations.
    if(G[3][1].requested)
    {
        G[3][2].requested = true;
        G[2][1].requested = true;
    }
    if(G[3][0].requested)
    {
        G[3][2].requested = true;
        G[2][1].requested = true;
        G[1][0].requested = true;
    }
#endif
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::dispatch_connectivity_ph(const conduit::Node &subel,
    const conduit::Node &sizes)
{
    // Dispatch to different versions based on connectivity type.
    index_t sizeslen = sizes.dtype().number_of_elements();
    if(sizes.dtype().is_int32())
        dispatch_shape_ph(subel, sizes.as_int32_ptr(), sizeslen);
    else if(sizes.dtype().is_uint32())
        dispatch_shape_ph(subel, sizes.as_uint32_ptr(), sizeslen);
    else if(sizes.dtype().is_int64())
        dispatch_shape_ph(subel, sizes.as_int64_ptr(), sizeslen);
    else if(sizes.dtype().is_uint64())
        dispatch_shape_ph(subel, sizes.as_uint64_ptr(), sizeslen);
    else
    {
        // Backup case. Use index_t accessor.
        dispatch_shape_ph(subel, sizes.as_index_t_accessor(), sizeslen);
    }
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::dispatch_connectivity(const ShapeType &shape,
    const conduit::Node &conn)
{
    // Dispatch to different versions based on connectivity type.
    if(conn.dtype().is_int32())
        dispatch_shape(shape, conn.as_int32_ptr(), conn.dtype().number_of_elements());
    else if(conn.dtype().is_uint32())
        dispatch_shape(shape, conn.as_uint32_ptr(), conn.dtype().number_of_elements());
    else if(conn.dtype().is_int64())
        dispatch_shape(shape, conn.as_int64_ptr(), conn.dtype().number_of_elements());
    else if(conn.dtype().is_uint64())
        dispatch_shape(shape, conn.as_uint64_ptr(), conn.dtype().number_of_elements());
    else
    {
        // Backup case. Use index_t accessor.
        dispatch_shape(shape, conn.as_index_t_accessor(), conn.dtype().number_of_elements());
    }
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::make_highest_topology()
{
    // Reuse the input topology as the highest dimension's topology.
    conduit::Node &node = dim_topos[topo_shape.dim];
    node["type"] = "unstructured";
    node["coordset"] = coords->name();
    node["elements/shape"] = topo_shape.type;

    // Copy data as index_t.
    std::vector<std::string> copy_keys{"elements/connectivity",
                                       "elements/sizes",
                                       "elements/offsets"
                                      };
    if(topo_shape.is_polyhedral())
    {
        copy_keys.push_back("subelements/connectivity");
        copy_keys.push_back("subelements/sizes");
        copy_keys.push_back("subelements/offsets");
    }
cout << "copy_keys = " << copy_keys << endl;
    for(const auto &key : copy_keys)
    {
        if(topo->has_path(key))
        {
cout << "Copying " << key << " as index_t." << endl;
            const conduit::Node &src = (*topo)[key];
            conduit::Node &dest = node[key];
            dest.set(DataType(index_t_id(), src.dtype().number_of_elements()));
            src.to_data_type(index_t_id(), dest);
        }
    }

    // Make sure we have offsets. The Conduit helper routines make them
    // in various precisions. We want index_t.
    conduit::Node src_offsets;
    if(topo_shape.is_polyhedral())
    {
        conduit::Node &topo_suboffsets = node["subelements/offsets"];
        topology::unstructured::generate_offsets(*topo,
                                                 src_offsets,
                                                 topo_suboffsets);
    }
    else
    {
        topology::unstructured::generate_offsets(*topo, src_offsets);
    }

    // Make offsets if they do not exist or if they were not index_t.
    conduit::Node &offsets = node["elements/offsets"];
    if(src_offsets.dtype().id() != index_t_id())
    {
cout << "Copying src_offsets to offsets as index_t. It was " << src_offsets.dtype().id() << endl;
        index_t nvalues = src_offsets.dtype().number_of_elements();
        offsets.set(DataType(index_t_id(), nvalues));
        src_offsets.to_data_type(index_t_id(), offsets);
    }
    else
    {
cout << "offsets.set(src_offsets) it was already index_t." << endl;
        offsets.set(src_offsets);
    }
node.print_detailed();
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::make_point_topology()
{
    conduit::Node &node = dim_topos[0];
    node["type"] = "unstructured";
    node["coordset"] = coords->name();
    node["elements/shape"] = "point";
    node["elements/connectivity"].set(conduit::DataType(index_t_id(), coords_length));
    // Also use the connectivity as offsets (works for points).
    node["elements/offsets"].set_external(node["elements/connectivity"]);

    index_t *conn = as_index_t_ptr(node["elements/connectivity"]);
    for(index_t i = 0; i < coords_length; i++)
        conn[i] = i;
}

//---------------------------------------------------------------------------
index_t
NewTopologyMetadata::make_unique(
    const std::vector<std::pair<uint64, uint64>> &faceid_to_ef,
    std::vector<std::pair<uint64, uint64>> &ef_to_unique) const
{
    size_t nids = faceid_to_ef.size();

    index_t unique = 0;
    size_t start = 0;
    for(size_t i = 1; i < nids; i++)
    {
        if(faceid_to_ef[start].first != faceid_to_ef[i].first)
        {
            for(size_t j = start; j < i; j++)
                ef_to_unique[j] = std::make_pair(faceid_to_ef[j].second, unique);
            unique++;
            start = i;
        }
    }
    for(size_t i = start; i < nids; i++)
        ef_to_unique[i] = std::make_pair(faceid_to_ef[i].second, unique);
    unique += (start < nids) ? 1 : 0;

    return unique;
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_associations()
{
    // Some maps will need lengths of the topologies that were produced.
#ifdef DEBUG_PRINT
    cout << "build_associations: topo_shape.dim=" << topo_shape.dim << endl;
#endif
    for(int dim = topo_shape.dim; dim >= 0; dim--)
    {
#ifdef DEBUG_PRINT
        cout << "topo " << dim << endl;
        cout << "=======" << endl;
        yaml_print(cout, dim_topos[dim]);
        cout << endl;
#endif
        conduit::Node info;
        if(conduit::blueprint::mesh::topology::verify(dim_topos[dim], info))
            dim_topo_lengths[dim] = conduit::blueprint::mesh::utils::topology::length(dim_topos[dim]);
        else
            dim_topo_lengths[dim] = 0;
    }

    // e,a association build order. See G[e][a] table.
    const int eaorder[][2] = {{3,3},{2,2},{1,1},{0,0},
                              {3,2},{2,1},{1,0},
                              {3,1},{2,0},
                              {3,0},
                              {2,3},{1,2},{0,1},
                              {1,3},{0,2},
                              {0,3}};

    // Set the single_size values. This is the number of items in the data
    // that are grouped together.
    if(topo_shape.dim >= 3)
    {
// TODO: special case PH?
        G[3][3].single_size = 1;
        G[3][2].single_size = topo_cascade.get_shape(3).embed_count;
        G[3][1].single_size = embedding_3_1_edges(topo_shape).size() / 2; // #unique edges
        G[3][0].single_size = topo_cascade.get_shape(3).indices;
    }
    if(topo_shape.dim >= 2)
    {
        G[2][3].single_size = 1;
        G[2][2].single_size = 1;
        G[2][1].single_size = topo_cascade.get_shape(2).embed_count; // This is probably wrong for PH cells when we switch to quad/tri.
        G[2][0].single_size = topo_cascade.get_shape(2).indices;
    }
    if(topo_shape.dim >= 1)
    {
        G[1][3].single_size = 1;
        G[1][2].single_size = 1;
        G[1][1].single_size = 1;
        G[1][0].single_size = topo_cascade.get_shape(1).indices;
    }
    G[0][3].single_size = 1;
    G[0][2].single_size = 1;
    G[0][1].single_size = 1;
    G[0][0].single_size = 1;

#ifdef DEBUG_PRINT
    for(int e = 3; e >= 0; e--)
    {
        for(int a = 3; a >= 0; a--)
            cout << G[e][a].single_size << ", ";
        cout << endl;
    }
#endif

    bool associations_31_30_need_built = true;
    for(int bi = 0; bi < 16; bi++)
    {
        int e = eaorder[bi][0];
        int a = eaorder[bi][1];
        if(G[e][a].requested)
        {
#ifdef DEBUG_PRINT
            cout << "Building association " << e << ", " << a << endl;
#endif
            index_t mapcase = EA_INDEX(e,a);
            switch(mapcase)
            {
            // Self cases.
            case EA_INDEX(0,0):
                // Falls through
            case EA_INDEX(1,1):
                // Falls through
            case EA_INDEX(2,2):
                // Falls through
            case EA_INDEX(3,3):
                G[e][a].data.resize(dim_topo_lengths[e]);
                std::iota(G[e][a].data.begin(), G[e][a].data.end(), 0);
                // We don't need to fill out sizes, offsets.
                break;

            // Connectivity cases
            case EA_INDEX(1,0):
                build_association_e_0(1);
                break;
            case EA_INDEX(2,0):
                build_association_e_0(2);
                break;

            // Cases that depend on face matching.
            case EA_INDEX(3,1):
                // Falls through
            case EA_INDEX(3,0):
                if(associations_31_30_need_built)
                {
                    build_association_3_1_and_3_0();
                    associations_31_30_need_built = false;
                }
                break;

            // Child to parent cases.
            case EA_INDEX(0,1):
                // Falls through
            case EA_INDEX(0,2):
                // Falls through
            case EA_INDEX(0,3):
                // Falls through
            case EA_INDEX(1,2):
                // Falls through
            case EA_INDEX(1,3):
                // Falls through
            case EA_INDEX(2,3):
                build_child_to_parent_association(e, a);
                break;

            // These are handled during construction. (no-op)
            case EA_INDEX(2,1):
                break;
            case EA_INDEX(3,2):
                break;
            }
        }
    }
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_association_e_0(int e)
{
    conduit::Node &node = dim_topos[e];
    static const std::string keys[]{"elements/connectivity",
                                    "elements/sizes",
                                    "elements/offsets"};
cout << "build_association_e_0(" << e << ")" << endl;
node.print_detailed();
    // Save some pointers from the connectivity in the association.
    if(node.has_path(keys[0]))
    {
        conduit::Node &n = node[keys[0]];
        G[e][0].data_ptr = as_index_t_ptr(n);
        G[e][0].data_ptr_size = n.dtype().number_of_elements();
    }
    if(node.has_path(keys[1]))
    {
        conduit::Node &n = node[keys[1]];
        G[e][0].sizes_ptr = as_index_t_ptr(n);
        G[e][0].sizes_ptr_size = n.dtype().number_of_elements();
    }
    if(node.has_path(keys[2]))
    {
        conduit::Node &n = node[keys[2]];
        G[e][0].offsets_ptr = as_index_t_ptr(n);
        G[e][0].offsets_ptr_size = n.dtype().number_of_elements();
    }
}

//---------------------------------------------------------------------------
std::vector<index_t>
NewTopologyMetadata::embedding_3_1_edges(const ShapeType &shape) const
{
    std::vector<index_t> retval;
    std::set<std::pair<index_t, index_t>> edges;
    auto npts_per_face = TOPO_SHAPE_INDEX_COUNTS[shape.embed_id];
    for(index_t fi = 0; fi < shape.embed_count; fi++)
    {
        const index_t *facepts = shape.embedding + fi * npts_per_face;
        for(index_t pi = 0; pi < npts_per_face; pi++)
        {
            index_t pi_next = (pi + 1) % npts_per_face;
            index_t i0 = facepts[pi];
            index_t i1 = facepts[pi_next];
            auto key = std::make_pair(std::min(i0, i1), std::max(i0, i1));
            if(edges.find(key) == edges.end())
            {
                edges.insert(key);
                retval.push_back(i0);
                retval.push_back(i1);
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_association_3_1_and_3_0()
{
    if(topo_shape.is_polyhedral())
        build_association_3_1_and_3_0_ph();
    else
        build_association_3_1_and_3_0_nonph();
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_association_3_1_and_3_0_ph()
{
    // G(3,2) contains the PH faces.
    const association &map32 = G[3][2];
    index_t nelem = dim_topo_lengths[3];

    // G(2,1) contains the face to edges map
    const association &map21 = G[2][1];
    index_t nedges = dim_topo_lengths[1];
    std::vector<int> edge_count(nedges, 0);

    // Assume hex ph, which have 12 edges per element. For ph elements that
    // do not fit the assumtion, we'll just let the vector resize.
    association &map31 = G[3][1];
    map31.data.reserve(nelem * 12);
    map31.sizes.resize(nelem, 0);
    map31.offsets.resize(nelem, 0);

    // Prepare G(3,0)
    association &map30 = G[3][0];
    const association &map10 = G[1][0];
    std::vector<int> point_count;
    if(map30.requested)
    {
        map30.data.reserve(nelem * 12);
        map30.sizes.resize(nelem, 0);
        map30.offsets.resize(nelem, 0);

        index_t npts = dim_topo_lengths[0];
        point_count.resize(npts, 0);
    }

    // Iterate over the elements and then the faces for each element so we
    // build up a set of edges used in this element.
    for(index_t ei = 0; ei < nelem; ei++)
    {
        // Store where this element's values start.
        map31.offsets[ei] = map31.data.size();
        if(map30.requested)
            map30.offsets[ei] = map30.data.size();
            
        // Iterate over this element's faces.
        auto elemfaces = map32.get_data(ei);
        for(index_t fi = 0; fi < elemfaces.second; fi++)
        {
            index_t faceid = elemfaces.first[fi];
            auto face_edges = map21.get_data(faceid);
            // Iterate over this face's edges
            for(index_t edge_index = 0; edge_index < face_edges.second; edge_index++)
            {
                // NOTE: We could decide to iterate the edges backwards if it's
                //       the second time we see this face overall. This assumes
                //       that the first element that defined the face would have
                //       defined it suitable for itself. Polyhedra otherwise
                //       do not maintain any starting point/edge in Conduit so
                //       we should not need to worry about faces rotated relative
                //       to different elements.

                index_t edgeid = face_edges.first[edge_index];
                // If it is the first time this edge has been seen in this
                // element, add it to the map.
                if(edge_count[edgeid] == 0)
                {
                    map31.data.push_back(edgeid);
                    map31.sizes[ei]++;

                    if(map30.requested)
                    {
                        // Iterate over this edge's points and add them if
                        // they have not been seen in this element.
                        auto edge_pts = map10.get_data(edgeid);
                        for(index_t pi = 0; pi < 2; pi++)
                        {
                            if(point_count[edge_pts.first[pi]] == 0)
                            {
                                map30.data.push_back(edge_pts.first[pi]);
                                map30.sizes[ei]++;
                            }
                            point_count[edge_pts.first[pi]]++;
                        }
                    }
                }
                edge_count[edgeid]++;
            }
        }
        
        // We're done with this element. Zero out the counts for the edges
        // and points that we saw in this element.
        for(index_t edge_index = 0; edge_index < map31.sizes[ei]; edge_index++)
        {
            index_t edgeid = map31.data[map31.offsets[ei] + edge_index];
            edge_count[edgeid] = 0;
        }
        if(map30.requested)
        {
            for(index_t pi = 0; pi < map30.sizes[ei]; pi++)
            {
                index_t ptid = map30.data[map30.offsets[ei] + pi];
                point_count[ptid] = 0;
            }
        }
    }
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_association_3_1_and_3_0_nonph()
{
    // This is the non PH method.

// Q: can we assume that we have index_t connectivity at this stage?

    // This method kind of refines the 3D shape from the start to make edge
    // id pairs that we look up in the edge connectivity. This is done because
    // it turned out to be problematic using the maps alone due to shared faces
    // being rotated among elements and then their edges could have mismatched
    // orientations.
    conduit::index_t_accessor conn3D = dim_topos[3]["elements/connectivity"].value();
    index_t connlen = conn3D.number_of_elements();
    index_t points_per_elem = topo_shape.indices;
    index_t nelem = connlen / points_per_elem;

    // Look through the edge connectivity and make a map that we can use to turn
    // an edge expressed as global point ids into a unique edge id.
    conduit::index_t_accessor conn1D = dim_topos[1]["elements/connectivity"].value();
    index_t nedges = conn1D.number_of_elements() / 2;
    std::vector<std::pair<uint64, index_t>> edge_key_to_id(nedges);
#ifdef DEBUG_PRINT
    cout << "edges_key_to_id = {" << endl;
#endif
#pragma omp parallel for
    for(index_t edge_index = 0; edge_index < nedges; edge_index++)
    {
        // Make a key for this edge.
        index_t edge[2];
        edge[0] = conn1D[edge_index * 2];
        edge[1] = conn1D[edge_index * 2 + 1];
        if(edge[1] > edge[0])
            std::swap(edge[0], edge[1]);
        uint64 key = hash_ids(edge, 2);
        // Store the edge in the map.
        edge_key_to_id[edge_index] = std::make_pair(key, edge_index);
#ifdef DEBUG_PRINT
        cout << std::setw(4) << edge_index << ": key=" <<  std::setw(20) << key
             << ", pts=" << std::setw(8) << edge[0] << ", "
             << std::setw(8) << edge[1] << endl;
#endif
    }
#ifdef DEBUG_PRINT
    cout << "}" << endl;
#endif
    // Sort the edges by the ids.
    std::sort(edge_key_to_id.begin(), edge_key_to_id.end(),
        [&](const std::pair<uint64, index_t> &lhs,
            const std::pair<uint64, index_t> &rhs) 
        {
            return lhs.first < rhs.first;
        });

    // Get the unique edges template for the element type.
    std::vector<index_t> elem_edges = embedding_3_1_edges(topo_shape);
#ifdef DEBUG_PRINT
    cout << "elem_edges=" << elem_edges << endl;
#endif
    // This function looks up a key in edge_key_to_id to return the edge id.
    auto lookup_id = [&](uint64 key) -> index_t
    {
        index_t index = -1;
        index_t left = 0;
        index_t right = edge_key_to_id.size() - 1;
        while(left <= right)
        {
            index_t m = (left + right) / 2;
            if(edge_key_to_id[m].first < key)
                left = m + 1;
            else if(edge_key_to_id[m].first > key)
                right = m - 1;
            else
            {
                index = m;
                break;
            }
        }
        return edge_key_to_id[index].second;
    };

    // Prepare the G(3,1) association.
    index_t edges_per_elem = elem_edges.size() / 2;
    association &map31 = G[3][1];
    map31.data.resize(nelem * edges_per_elem, 0);
    map31.sizes.resize(nelem, 0);
    map31.offsets.resize(nelem, 0);

    // Prepare the G(3,0) association.
    association &map30 = G[3][0];
    if(map30.requested)
    {
        map30.data.resize(nelem * points_per_elem, 0);
        map30.sizes.resize(nelem, 0);
        map30.offsets.resize(nelem, 0);
    }

    // Iterate over the elements, applying the edge template to make unique
    // edges for the element. We look up the edge in edge_key_to_id to get
    // its id.
#pragma omp parallel for
    for(index_t ei = 0; ei < nelem; ei++)
    {
        index_t elem_offset = ei * points_per_elem;

        // Each bit in the ptadded variable indicates whether the point has
        // been added into the 3,0 map. If map30 is not requested add a value
        // with 1's in all the bits so points can't be added.
        size_t ptadded = map30.requested ? 0 : std::numeric_limits<size_t>::max();
        index_t elem_point = elem_offset;

        for(size_t edge_index = 0; edge_index < edges_per_elem; edge_index++)
        {
            // These point ids are in [0,points_per_elem) and are the local
            // point indices for the edges in the cell.
            index_t edge_pt0 = elem_edges[(edge_index << 1)];
            index_t edge_pt1 = elem_edges[(edge_index << 1) + 1];

            // Make the edge.
            index_t edge[2];
            edge[0] = conn3D[elem_offset + edge_pt0];
            edge[1] = conn3D[elem_offset + edge_pt1];

            // Add the points in the edge to the 3,0 mapping if it was requested
            // and if we have not seen them before in this element. We do this
            // before we make the edge key so the point order is preserved.
            size_t pt0mask = 1 << edge_pt0;
            size_t pt1mask = 1 << edge_pt1;
            if((ptadded & pt0mask) == 0)
            {
                ptadded |= pt0mask;
                map30.data[elem_point++] = edge[0];
            }
            if((ptadded & pt1mask) == 0)
            {
                ptadded |= pt1mask;
                map30.data[elem_point++] = edge[1];
            }

            // Make a key from the edge.
            if(edge[1] > edge[0])
                std::swap(edge[0], edge[1]);
            uint64 key = hash_ids(edge, 2);

            // Look up the edge id in our list of real edges.
            index_t edgeid = lookup_id(key);

            // Store the edge id in the map.
            index_t elem_edge = ei * edges_per_elem + edge_index;
            map31.data[elem_edge] = edgeid;
        }

        map31.sizes[ei] = edges_per_elem;
        map31.offsets[ei] = ei * edges_per_elem;
    }

    // If the G(3,0) map was requested then build its sizes and offsets.
    if(map30.requested)
    {
        for(index_t ei = 0; ei < nelem; ei++)
        {
            map30.sizes[ei] = points_per_elem;
            map30.offsets[ei] = ei * points_per_elem;
        }
    }
}

#if 0
// OBSOLETE -- I might want to find my old one for PH elements though.
//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_association_3_1()
{
    // Get the number of faces and lines produced during the cascade.
    size_t numelems = dim_topo_lengths[3]; // Or G[3][2].sizes.size()
    size_t numfaces = dim_topo_lengths[2];
    size_t numlines = dim_topo_lengths[1];

    const association &map32 = G[3][2];
    const association &map21 = G[2][1];
    association &map31 = G[3][1];

#ifdef DEBUG_PRINT
    // sanity check
    cout << "map32:" << endl;
    print_association(3, 2);
    cout << "map21:" << endl;
    print_association(2, 1);
    cout << "first_face_points=" << first_face_points << endl;
#endif

    // Reserve storage
    map31.data.reserve(numlines);
    map31.sizes.resize(numelems, 0);
    map31.offsets.resize(numelems, 0);

    // Keep track of whether a line has been seen before.
    std::vector<int> linecount(numlines, 0);
    // Keep track of whether a face has been seen before.
    std::vector<int> facecount(numfaces, 0);
    // Get the line connectivity since we need it.
    const index_t *lineconn = as_index_t_ptr(dim_topos[1]["elements/connectivity"]);
    const index_t *faceconn = as_index_t_ptr(dim_topos[2]["elements/connectivity"]);
    //const index_t *facesize = as_index_t_ptr(dim_topos[2]["elements/sizes"]);
    const index_t *faceoffset = as_index_t_ptr(dim_topos[2]["elements/offsets"]);

    // For each element, get its faces and add its lines to the map31.data
    // if we have not seen that line before in the element.
    for(size_t ei = 0; ei < numelems; ei++)
    {
        map31.offsets[ei] = map31.data.size();
cout << "Element " << ei << endl;
        auto elemfaces = map32.get_data(ei);
        for(size_t fi = 0; fi < elemfaces.second; fi++)
        {
            index_t faceid = elemfaces.first[fi];
cout << "\tfi=" << fi << ", faceid=" << faceid << endl;
cout << "\t\tlines: ";
            auto facelines = map21.get_data(faceid);
            index_t nfacelines = facelines.second;

            // We have the lines for this face.
            if(facecount[faceid] == 0)
            {
                // This is the first time the face was referenced. The
                // face was defined relative to the current element. We
                // can therefore do the edges in order.
                for(size_t li = 0; li < nfacelines; li++)
                {
                    index_t lineid = facelines.first[li];
cout << lineid;
                    if(linecount[lineid] == 0)
                    {
cout << "+";
                        map31.data.push_back(lineid);
                        map31.sizes[ei]++;
                    }
cout << ", ";
                    linecount[lineid]++;
                }
cout << endl;
            }
            else
            {
                // The points in this face.
                const index_t *face_pts = faceconn + faceoffset[faceid];
                index_t element_face = map32.get_offset(ei) + fi;
                // Look for the starting point in the face connectivity.
                index_t current = 0;
                while(face_pts[current] != first_face_points[element_face])
                    current++;
cout << "[backward][element_face=" << element_face << ", first_face_point="
     << first_face_points[element_face]
     << ", current=" << current << "]";
                for(size_t li = 0; li < nfacelines; li++)
                {
                    index_t next = (current == 0) ? (nfacelines - 1) : (current - 1);
                    index_t edge[2];
                    edge[0] = face_pts[current];
                    edge[1] = face_pts[next];

                    // Look in the edge connectivity for the line id having these ids.
                    // We just need to search the lines in this face.
                    for(size_t li = 0; li < nfacelines; li++)
                    {
                        index_t lineid = facelines.first[li];
                        const index_t *this_edge = &lineconn[lineid * 2];
                        if((edge[0] == this_edge[0] && edge[1] == this_edge[1]) ||
                           (edge[0] == this_edge[1] && edge[1] == this_edge[0]))
                        {
cout << lineid << " (" << this_edge[1] << "," << this_edge[0] << ")";
                            if(linecount[lineid] == 0)
                            {
cout << "+";
                                map31.data.push_back(lineid);
                                map31.sizes[ei]++;
                            }
cout << ", ";
                            linecount[lineid]++;
                            break;
                        }
                    }
                    current = next;
                }
cout << endl;
            }
            // increase the face count.
            facecount[faceid]++;
        }
cout << endl;
        // Prepare for next element by zeroing out the line counts for
        // the lines we used in this element.
        for(index_t i = 0; i < map31.sizes[ei]; i++)
        {
            index_t lineid = map31.data[map31.offsets[ei] + i];
            linecount[lineid] = 0;
        }
    }
#ifdef DEBUG_PRINT
    cout << "map31:" << endl;
    print_association(3, 1);
#endif
}

//---------------------------------------------------------------------------
// OBSOLETE
void
NewTopologyMetadata::build_association_3_0()
{
    // Get the number of faces and lines produced during the cascade.
    size_t numelems = dim_topo_lengths[3];
    size_t numfaces = dim_topo_lengths[2];
    size_t numlines = dim_topo_lengths[1];
    size_t numpoints = dim_topo_lengths[0];

    const association &map32 = G[3][2];
    const association &map21 = G[2][1];
    const association &map10 = G[1][0];
    association &map30 = G[3][0];

#ifdef DEBUG_PRINT
    // sanity check
    cout << "map32:" << endl;
    print_association(3, 2);
    cout << "map21:" << endl;
    print_association(2, 1);
    cout << "map10:" << endl;
    print_association(1, 0);
#endif

    // Reserve storage
    map30.data.reserve(numpoints);
    map30.sizes.resize(numelems, 0);
    map30.offsets.resize(numelems, 0);

    // Keep track of whether a line has been seen before.
    std::vector<int> linecount(numlines, 0);
    // Keep track of whether a face has been seen before.
    std::vector<int> facecount(numfaces, 0);
    // Keep track of whether a point has been seen before.
    std::vector<int> pointcount(numpoints, 0);

    // Sometimes we don't make sizes of offsets in some associations (like when 
    // they contain all the same shape).
    const index_t *map32_data = nullptr, *map32_sizes = nullptr, *map32_offsets = nullptr;
    index_t map32_data_size = 0, map32_sizes_size = 0, map32_offsets_size = 0;
    map32.get_data(map32_data, map32_data_size);
    if(!map32.get_sizes(map32_sizes, map32_sizes_size))
        map32_sizes_size = numelems;
    map32.get_offsets(map32_offsets, map32_offsets_size);

    const index_t *map21_data = nullptr, *map21_sizes = nullptr, *map21_offsets = nullptr;
    index_t map21_data_size = 0, map21_sizes_size = 0, map21_offsets_size = 0;
    map21.get_data(map21_data, map21_data_size);
    if(!map21.get_sizes(map21_sizes, map21_sizes_size))
        map21_sizes_size = numfaces;
    map21.get_offsets(map21_offsets, map21_offsets_size);

    const index_t *map10_data = nullptr, *map10_sizes = nullptr, *map10_offsets = nullptr;
    index_t map10_data_size = 0, map10_sizes_size = 0, map10_offsets_size = 0;
    map10.get_data(map10_data, map10_data_size);
    if(!map10.get_sizes(map10_sizes, map10_sizes_size))
        map10_sizes_size = numlines;
    map10.get_offsets(map10_offsets, map10_offsets_size);

    std::vector<index_t> lineids;
    lineids.reserve(100);

    // For each element, get its faces and add its lines to the map30.data
    // if we have not seen that line before in the element.
    for(size_t ei = 0; ei < map32_sizes_size; ei++)
    {
        map30.offsets[ei] = map30.data.size();

        index_t elemoffset = map32_offsets ? map32_offsets[ei] : (ei * map32.single_size);
        const index_t *elemfaces = &map32_data[elemoffset];
        size_t nelemfaces = map32_sizes ? map32_sizes[ei] : map32.single_size;

        for(size_t fi = 0; fi < nelemfaces; fi++)
        {
            index_t faceid = elemfaces[fi];
            index_t faceoffset = map21_offsets ? map21_offsets[faceid] : (faceid * map21.single_size);
            const index_t *facelines = &map21_data[faceoffset];
            size_t nfacelines = map21_sizes ? map21_sizes[faceid] : map21.single_size;

            for(size_t li = 0; li < nfacelines; li++)
            {
                // If we've seen the face before, do the lines in reverse
                // order since the face should have a reverse orientation
                // in this element.
                index_t lineid = facecount[faceid] ? facelines[nfacelines-1-li] : facelines[li];
                index_t lineoffset = map10_offsets ? map10_offsets[lineid] : (lineid * map10.single_size);
                const index_t *linepoints = &map10_data[lineoffset];

                if(linecount[lineid] == 0)
                {
                    // Get the point id.
                    index_t ptid = std::min(linepoints[0],linepoints[1]);
                    if(pointcount[ptid] == 0)
                    {
                        map30.data.push_back(ptid);
                        map30.sizes[ei]++;
                    }
                    pointcount[ptid]++;

                    index_t ptid2 = std::max(linepoints[0],linepoints[1]);
                    if(pointcount[ptid2] == 0)
                    {
                        map30.data.push_back(ptid2);
                        map30.sizes[ei]++;
                    }
                    pointcount[ptid2]++;

                    lineids.push_back(lineid);
                }
                linecount[lineid]++;
            }
            // increase the face count.
            facecount[faceid]++;
        }

        // Prepare for next element by zeroing out the point and line counts for
        // the ones we used in this element.
        for(index_t i = 0; i < map30.sizes[ei]; i++)
        {
            index_t ptid = map30.data[map30.offsets[ei] + i];
            pointcount[ptid] = 0;
        }
        for(index_t i = 0; i < lineids.size(); i++)
        {
            linecount[lineids[i]] = 0;
        }
        lineids.clear();
    }
#ifdef DEBUG_PRINT
    cout << "map30:" << endl;
    print_association(3, 0);
#endif
}
#endif

//---------------------------------------------------------------------------
void
NewTopologyMetadata::print_association(int e, int a) const
{
    const association &assoc = G[e][a];
    cout << "\tdata=" << assoc.data << endl;
    cout << "\tsizes=" << assoc.sizes << endl;
    cout << "\toffsets=" << assoc.offsets << endl;
    cout << "\tdata_ptr=" << (void*)assoc.data_ptr << "{";
    if(assoc.data_ptr)
    {
        for(int i = 0; i < assoc.data_ptr_size; i++)
            cout << assoc.data_ptr[i] << ", ";
    }
    cout << "}" << endl;
    cout << "\tsizes_ptr=" << (void*)assoc.sizes_ptr << "{";
    if(assoc.sizes_ptr)
    {
        for(int i = 0; i < assoc.sizes_ptr_size; i++)
            cout << assoc.sizes_ptr[i] << ", ";
    }
    cout << "}" << endl;
    cout << "\toffsets_ptr=" << (void*)assoc.offsets_ptr << "{";
    if(assoc.offsets_ptr)
    {
        for(int i = 0; i < assoc.offsets_ptr_size; i++)
            cout << assoc.offsets_ptr[i] << ", ";
    }
    cout << "}" << endl;
    cout << "\tdata_ptr_size=" << assoc.data_ptr_size << endl;
    cout << "\tsizes_ptr_size=" << assoc.sizes_ptr_size << endl;
    cout << "\toffsets_ptr_size=" << assoc.offsets_ptr_size << endl;
    cout << "\tsingle_size=" << assoc.single_size << endl;
    cout << "\trequested=" << assoc.requested << endl;
}

//---------------------------------------------------------------------------
void
NewTopologyMetadata::build_child_to_parent_association(int e, int a)
{
    const association &mapPC = G[a][e]; // parent to child (already exists)
    association &mapCP = G[e][a];       // child to parent (what we're making).

#ifdef DEBUG_PRINT
    cout << "----------------------------------------" << endl;
    cout << "build_child_to_parent_association(" << e << ", " << a <<")" << endl;
    cout << "----------------------------------------" << endl;
    cout << "mapPC:" << endl;
    print_association(a,e);
#endif

    // NOTE: It is possible for the mapPC association to have its data
    //       stored in the connectivity. Thus, we have to use the association's
    //       data_ptr if we really want data.
    const index_t *mapPC_data = mapPC.data_ptr ? mapPC.data_ptr : &mapPC.data[0];
    size_t mapPC_data_size = mapPC.data_ptr ? mapPC.data_ptr_size : mapPC.data.size();
    mapCP.sizes.resize(dim_topo_lengths[e], 0);
    mapCP.offsets.resize(dim_topo_lengths[e], 0);

    // Make sizes by counting how many times an id occurs.
    for(size_t i = 0; i < mapPC_data_size; i++)
        mapCP.sizes[mapPC_data[i]]++;

    // Make offsets from sizes
    int off = 0;
    for(size_t i = 0; i < mapCP.sizes.size(); i++)
    {
        mapCP.offsets[i] = off;
        off += mapCP.sizes[i];
    }

#ifdef DEBUG_PRINT
    cout << "mapPC_data_size=" << mapPC_data_size << endl;
    for(int i =0 ; i < 4; i++)
        cout << i << ", topolen=" << dim_topo_lengths[i] << endl;
    cout << "mapCP.sizes=" << mapCP.sizes << endl;
    cout << "mapCP.offsets=" << mapCP.offsets << endl;
#endif
    // Make a series of ids using the parent-child sizes vector. This will
    // make a pattern like: 0,1,2,3...  or 0,0,0,0,1,1,1,1,2,2,2,2,...
    // according to the size values. We use this to make pairs of
    // parent/child ids.
    std::vector<std::pair<index_t, index_t>> p2c(mapPC_data_size);
    size_t idx = 0;
    if(mapPC.sizes.empty())
    {
        for(; idx < mapPC_data_size; idx++)
        {
            p2c[idx] = std::make_pair(idx / mapPC.single_size, // parent id
                                      mapPC_data[idx]);        // child id
        }
    }
    else
    {
        for(index_t id = 0; id < mapPC.sizes.size(); id++)
        {
            for(index_t i = 0; i < mapPC.sizes[id]; i++)
            {
                p2c[idx].first = id;               // parent id
                p2c[idx].second = mapPC_data[idx]; // child id
                idx++;
            }
        }
    }
#ifdef DEBUG_PRINT
    cout << "p2c=" << p2c << endl;
#endif
    // Sort p2c by child.
    std::sort(OPTIONAL_PARALLEL_EXECUTION_POLICY
              p2c.begin(), p2c.end(),
        [&](const std::pair<index_t, index_t> &lhs,
            const std::pair<index_t, index_t> &rhs)
    {
        return lhs.second < rhs.second;
    });
#ifdef DEBUG_PRINT
    cout << "p2c.sorted=" << p2c << endl;
#endif

    // Extract the permuted list of parents.
    mapCP.data.resize(p2c.size(), 0);
    for(size_t i = 0; i < p2c.size(); i++)
        mapCP.data[i] = p2c[i].first;
#ifdef DEBUG_PRINT
    cout << "mapCP.data=" << mapCP.data << endl;
#endif

    // Sort ids in each bin of parent ids.
    for(size_t i = 0; i < mapCP.sizes.size(); i++)
    {
        if(mapCP.sizes[i] > 1)
        {
            index_t *start = &mapCP.data[mapCP.offsets[i]];
            index_t *end = start + mapCP.sizes[i];
            std::sort(start, end);
        }
    }
#ifdef DEBUG_PRINT
    cout << "mapCP.data.final=" << mapCP.data << endl;
    cout << "mapCP" << endl;
    print_association(e, a);
#endif
}

//---------------------------------------------------------------------------
bool
NewTopologyMetadata::association_requested(index_t entity_dim, index_t assoc_dim) const
{
#ifndef _NDEBUG
    if(entity_dim > topo_shape.dim || assoc_dim > topo_shape.dim)
    {
        CONDUIT_ERROR("A global association map G(" << entity_dim << ", " << assoc_dim
            << ") does not exist because one or more indices is invalid.");
    }
#endif
    return G[entity_dim][assoc_dim].requested;
}

//---------------------------------------------------------------------------
vector_view<index_t>
NewTopologyMetadata::get_global_association(index_t entity_id,
    index_t entity_dim, index_t assoc_dim) const
{
#ifndef _NDEBUG
    if(entity_dim > topo_shape.dim || assoc_dim > topo_shape.dim)
    {
        CONDUIT_ERROR("A global association map G(" << entity_dim << ", " << assoc_dim
            << ") does not exist because one or more indices is invalid.");
    }
    if(!G[entity_dim][assoc_dim].requested)
    {
        CONDUIT_ERROR("A global association map G(" << entity_dim << ", " << assoc_dim
            << ") does not exist because it was not built during metadata initialization.");
    }
#endif

    const association &assoc = G[entity_dim][assoc_dim];
    auto data = assoc.get_data(entity_id);
    return vector_view<index_t>(data.first, data.second);
}

//---------------------------------------------------------------------------
index_t
NewTopologyMetadata::get_local_association_entity_range(int src_dim, int dst_dim) const
{
    index_t ne;
    index_t mapcase = EA_INDEX(src_dim, dst_dim);
    switch(mapcase)
    {
    case EA_INDEX(0,0):
        // Falls through
    case EA_INDEX(0,1):
        // Falls through
    case EA_INDEX(0,2):
        // Falls through
    case EA_INDEX(0,3):
        ne = dim_topo_lengths[3] * G[3][2].single_size * G[2][1].single_size * G[1][0].single_size + coords_length;
        break;

    case EA_INDEX(1,0):
        // Falls through
    case EA_INDEX(1,1):
        // Falls through
    case EA_INDEX(1,2):
        // Falls through
    case EA_INDEX(1,3):
        // This needs to be 96 for hex.  4cells * 6faces/cell * 4lines/face
        ne = dim_topo_lengths[3] * G[3][2].single_size * G[2][1].single_size;
        break;

    case EA_INDEX(2,0):
        // Falls through
    case EA_INDEX(2,1):
        // Falls through
    case EA_INDEX(2,2):
        // Falls through
    case EA_INDEX(2,3):
        //  4                    * 6
        ne = dim_topo_lengths[3] * G[3][2].single_size;
        // This could also be sum(G[3][2].sizes)
        break;

    case EA_INDEX(3,0):
        // Falls through
    case EA_INDEX(3,1):
        // Falls through
    case EA_INDEX(3,2):
        // Falls through
    case EA_INDEX(3,3):
        ne = dim_topo_lengths[src_dim];
        break;
    } 
    return ne;
}

//---------------------------------------------------------------------------
conduit::range_vector<index_t>
NewTopologyMetadata::get_local_association(index_t entity_id,
    index_t entity_dim, index_t assoc_dim) const
{
#ifndef _NDEBUG
    if(entity_dim > topo_shape.dim || assoc_dim > topo_shape.dim)
    {
        CONDUIT_ERROR("A local association map G(" << entity_dim << ", " << assoc_dim
            << ") does not exist because one or more indices is invalid.");
    }
    if(!G[entity_dim][assoc_dim].requested)
    {
        CONDUIT_ERROR("A local association map G(" << entity_dim << ", " << assoc_dim
            << ") does not exist because it was not built during metadata initialization.");
    }
#endif
    // Sometimes we need size/offset information from the global association.
    const association &assoc = G[entity_dim][assoc_dim];
    conduit::range_vector<index_t> vec(entity_id, 1, 1);

    // NOTE: A pattern that occurs when e, a are more than 1 apart is that we
    //       need to use a product of the sizes. Take e,a = 2,0. These are 2 apart
    //       so we multiply G[2][1].get_size(entity_id) * G[1][0].get_size(entity_id)
    //                        |                             |
    //                        -------------------------------

    // NOTE: some of these entity_id * N look like they could be .get_offset(entity_id) * 2.

    index_t mapcase = EA_INDEX(entity_dim, assoc_dim);
    switch(mapcase)
    {
    // Self cases.
    case EA_INDEX(0,0): break; // no-op
    case EA_INDEX(1,1): break; // no-op
    case EA_INDEX(2,2): break; // no-op
    case EA_INDEX(3,3): break; // no-op

    // Connectivity cases. These guys' start is offset by the coords_length.
    case EA_INDEX(1,0):
        {
            auto start = coords_length;
            auto N = 2; // 2 = G[1][0].get_size(entity_id);
            vec = conduit::range_vector<index_t>(start + entity_id * N, 1, N);
        }
        break;
    case EA_INDEX(2,0):
        {
            auto start = coords_length;
            auto N = G[2][1].get_size(entity_id) * 2; // 2 = G[1][0].get_size(entity_id);
            vec = conduit::range_vector<index_t>(start + entity_id * N, 1, N);
        }
        break;
    case EA_INDEX(3,0):
        {
            auto start = coords_length;
            auto N = G[3][2].get_size(entity_id) * G[2][1].get_size(entity_id) * 2; // 2 = G[1][0].get_size(entity_id);
            vec = conduit::range_vector<index_t>(start + entity_id * N, 1, N);
        }
        break;

    // Child to parent cases.
    case EA_INDEX(0,1):
        {
            if(entity_id < coords_length)
                vec = conduit::range_vector<index_t>(0,0,0); // empty
            else
            {
                auto start = (entity_id - coords_length) / 2; // 2 = G[1][0].get_size(entity_id);
                vec = conduit::range_vector<index_t>(start, 1, 1);
            }
        }
        break;
    case EA_INDEX(0,2):
        {
            if(entity_id < coords_length)
                vec = conduit::range_vector<index_t>(0,0,0); // empty
            else
            {
               auto start = (entity_id - coords_length) / (G[2][1].get_size(entity_id) * 2); // 2= G[1][0].get_size(entity_id);
               vec = conduit::range_vector<index_t>(start, 1, 1);
            }
        }
        break;
    case EA_INDEX(0,3):
        {
           if(entity_id < coords_length)
               vec = conduit::range_vector<index_t>(0,0,0); // empty
           else
           {
               auto start = (entity_id - coords_length) / (G[3][2].get_size(entity_id) * G[2][1].get_size(entity_id) * 2); // 2 = G[1][0].get_size(entity_id);
               vec = conduit::range_vector<index_t>(start, 1, 1);
           }
        }
        break;
    case EA_INDEX(1,2):
        {
            index_t start = entity_id / G[2][1].get_size(entity_id);
            vec = conduit::range_vector<index_t>(start, 0, 1);
        }
        break;
    case EA_INDEX(1,3):
        {
            index_t start = entity_id / (G[3][2].get_size(entity_id) * G[2][1].get_size(entity_id));
            vec = conduit::range_vector<index_t>(start, 0, 1);
        }
        break;
    case EA_INDEX(2,3):
        {
            index_t start = entity_id / G[3][2].get_size(entity_id);
            vec = conduit::range_vector<index_t>(start, 0, 1);
        }
        break;

    case EA_INDEX(3,1):
        { // new scope
            index_t N = G[3][1].get_size(entity_id) * 2; // 2 = G[1][0].get_size(entity_id);
            vec = conduit::range_vector<index_t>(entity_id * N, 1, N);
        } // end scope
        break;

    // e,e-1 cases
    case EA_INDEX(2,1):
        // Falls through
    case EA_INDEX(3,2):
        { // new scope
            const association &assoc = G[entity_dim][assoc_dim];
            index_t gsize = assoc.get_size(entity_id);
            index_t goffset = assoc.get_offset(entity_id);
            vec = conduit::range_vector<index_t>(goffset, 1, gsize);
        } // end scope
        break;
    }

    return vec;
}

//---------------------------------------------------------------------------
bool
NewTopologyMetadata::get_dim_map(NewTopologyMetadata::IndexType type,
    index_t src_dim, index_t dst_dim, conduit::Node &map_node) const
{
    bool retval = false;
    if(type == GLOBAL)
        retval = get_global_dim_map(src_dim, dst_dim, map_node);
    else if(type == LOCAL)
        retval = get_local_dim_map(src_dim, dst_dim, map_node);
    return retval;
}

//---------------------------------------------------------------------------
bool
NewTopologyMetadata::get_global_dim_map(index_t src_dim, index_t dst_dim,
    conduit::Node &map_node) const
{
    const association &assoc = G[src_dim][dst_dim];
    if(assoc.requested)
    {
        // Copy the vectors out.

        conduit::Node &values = map_node["values"];
        if(assoc.data_ptr)
        {
            // Copy data from the connectivity
            values.set(dim_topos[src_dim]["elements/connectivity"]);
        }
        else
        {
            values.set(assoc.data);
        }

        conduit::Node &sizes = map_node["sizes"];
        if(assoc.sizes_ptr)
        {
            // Copy sizes from the connectivity
            sizes.set(dim_topos[src_dim]["elements/sizes"]);
        }
        else if(assoc.sizes.empty())
        {
            // Shapes were all the same size. Make new values.
            size_t nembed = dim_topo_lengths[src_dim];
            sizes.set(conduit::DataType(index_t_id(), nembed));
            index_t *ptr = as_index_t_ptr(sizes);
            index_t sz = assoc.single_size;
            for(size_t i = 0; i < nembed; i++)
                ptr[i] = sz;
        }
        else
        {
            sizes.set(G[src_dim][dst_dim].sizes);
        }

        conduit::Node &offsets = map_node["offsets"];
        if(assoc.offsets_ptr)
        {
            // Copy offsets from the connectivity
            offsets.set(dim_topos[src_dim]["elements/offsets"]);
        }
        else if(assoc.offsets.empty())
        {
            // Shapes were all the same size. Make new values.
            size_t nembed = dim_topo_lengths[src_dim];
            offsets.set(conduit::DataType(index_t_id(), nembed));
            index_t *ptr = as_index_t_ptr(offsets);
            index_t sz = assoc.single_size;
            for(size_t i = 0; i < nembed; i++)
                ptr[i] = i * sz;
        }
        else
        {
            offsets.set(G[src_dim][dst_dim].offsets);
        }
        // We could do that trick to wrap a node and then bulk convert to the desired type...
    }
    return assoc.requested;
}

//------------------------------------------------------------------------------
bool
NewTopologyMetadata::get_local_dim_map(index_t src_dim, index_t dst_dim,
    conduit::Node &map_node) const
{
    bool requested = G[src_dim][dst_dim].requested;
    if(requested)
    {
        // Do a sizing pass.
        auto N = get_local_association_entity_range(src_dim, dst_dim);
        index_t total_size = 0;
        for(index_t eid = 0; eid < N; eid++)
        {
            auto lm = get_local_association(eid, src_dim, dst_dim);
            total_size += lm.size();
        }

        // Make the nodes.
        conduit::Node &values = map_node["values"];
        conduit::Node &sizes = map_node["sizes"];
        conduit::Node &offsets = map_node["offsets"];

        values.set(conduit::DataType(index_t_id(), total_size));
        sizes.set(conduit::DataType(index_t_id(),  N));
        offsets.set(conduit::DataType(index_t_id(), N));

        index_t *values_ptr = as_index_t_ptr(values);
        index_t *sizes_ptr = as_index_t_ptr(sizes);
        index_t *offsets_ptr = as_index_t_ptr(offsets);

        // Do another pass to store the data in the nodes.
        index_t off = 0;
        for(index_t eid = 0; eid < N; eid++)
        {
            auto lm = get_local_association(eid, src_dim, dst_dim);
            // Copy lm values into values array.
            for(auto lmval : lm)
                *values_ptr++ = lmval;

            sizes_ptr[eid] = lm.size();
            offsets_ptr[eid] = off;
            off += lm.size();
        }
    }

    return requested;
}

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::detail::reference --
//-----------------------------------------------------------------------------
namespace reference
{

//---------------------------------------------------------------------------//
TopologyMetadata::TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset) :
    topo(&topology), cset(&coordset),
    int_dtype(find_widest_dtype(link_nodes(topology, coordset), DEFAULT_INT_DTYPES)),
    float_dtype(find_widest_dtype(link_nodes(topology, coordset), DEFAULT_FLOAT_DTYPE)),
    topo_cascade(topology), topo_shape(topology)
{
    // NOTE(JRC): This type current only works at forming associations within
    // an unstructured topology's hierarchy.
    Node topo_offsets, topo_suboffsets;
    bool is_polyhedral = topo->has_child("subelements");

    if(is_polyhedral)
    {
        topology::unstructured::generate_offsets(*topo,
                                                 topo_offsets,
                                                 topo_suboffsets);
    }
    else
    {
        topology::unstructured::generate_offsets(*topo, topo_offsets);
    }

    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();
    const index_t topo_num_coords = coordset::length(coordset);

    // Allocate Data Templates for Outputs //

    // per-dimension maps from an entity's point id set to its global entity id
    std::vector< std::map< std::vector<index_t>, index_t > > dim_geid_maps;
    // per-dimension vector to build up connectivity.
    std::vector< std::vector<int64> > dim_buffers(topo_shape.dim + 1);

    dim_topos.resize(topo_shape.dim + 1);
    dim_geid_maps.resize(topo_shape.dim + 1);
    dim_geassocs_maps.resize(topo_shape.dim + 1);
    dim_leassocs_maps.resize(topo_shape.dim + 1);
    dim_le2ge_maps.resize(topo_shape.dim + 1);

    // Start out reserving space for the association spines. These multiples
    // were calibrated using a 2D dataset. Other topological dims may need
    // different values. These were determined using the final sizes of the
    // dim_leassocs_maps, dim_geassocs_maps and comparing to topo_num_elems.
    // 
    size_t le_est_size_multiples[] = {9, 4, 1, 1};
    size_t ge_est_size_multiples[] = {1, 2, 1, 1};
    for(index_t dim = 0; dim <= topo_shape.dim; dim++)
    {
        dim_leassocs_maps[dim].reserve(le_est_size_multiples[dim] *
                                       (topo_shape.dim + 1) *
                                       topo_num_elems);
        dim_geassocs_maps[dim].reserve(ge_est_size_multiples[dim] *
                                       (topo_shape.dim + 1) *
                                       topo_num_elems);
    }

    // Set up the output nodes.
    for(index_t di = 0; di < topo_shape.dim; di++)
    {
        Node &dim_topo = dim_topos[di];
        dim_topo.reset();
        dim_topo["type"].set("unstructured");
        dim_topo["coordset"].set(topology["coordset"].as_string());
        dim_topo["elements/shape"].set(topo_cascade.get_shape(di).type);
    }
    // NOTE: This is done so that the index values for the top-level entities
    // can be extracted by the 'get_entity_data' function before DFS
    // processing below.
    dim_topos[topo_shape.dim].set_external(topology);
    dim_topos[topo_shape.dim]["elements/offsets"].set(topo_offsets);
    if(is_polyhedral)
    {
        dim_topos[topo_shape.dim]["subelements/offsets"].set(topo_suboffsets);
    }

    // Prepare Initial Values for Processing //

    // Temporary nodes to manage pointers to important information (temp) and
    // associated conversations (data).
    Node temp, data;

    // NOTE(JRC): A 'deque' is used so that queue behavior (FIFO)
    // is responsible for ordering the identifiers in the cascade of entities,
    // which more closely follows the average intuition.
    const index_t bag_num_elems = topo_num_coords + topo_num_elems;
    std::deque< std::vector<int64> > entity_index_bag(bag_num_elems);
    std::deque< index_t > entity_dim_bag(bag_num_elems, -1);
    std::deque< index_t > entity_origid_bag(bag_num_elems, -1);
    std::deque< std::vector< std::pair<int64, int64> > > entity_parent_bag(bag_num_elems);

    // NOTE(JRC): We start with processing the points of the topology followed
    // by the top-level elements in order to ensure that order is preserved
    // relative to the original topology for these entities.
    for(index_t pi = 0; pi < topo_num_coords; pi++)
    {
        index_t bi = pi;
        entity_index_bag[bi].push_back(bi);
        entity_dim_bag[bi] = 0;
        // Keep track of original id.
        entity_origid_bag[bi] = pi;
    }
    // Prepopulate point connectivity
    dim_buffers[0].reserve(topo_num_coords);
    for(index_t pi = 0; pi < topo_num_coords; pi++)
        dim_buffers[0].push_back(pi);

    // Add entities for the top-level elements (these will be refined later).
    const Node &topo_elem_conn = dim_topos[topo_shape.dim]["elements/connectivity"];
    const Node &topo_elem_offsets = dim_topos[topo_shape.dim]["elements/offsets"];
    const auto elem_conn_access = topo_elem_conn.as_index_t_accessor();
    const auto elem_offsets_access = topo_elem_offsets.as_index_t_accessor();
    index_t topo_conn_len = 0;
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        index_t bi = topo_num_coords + ei;
        // Use the offsets to compute a size.
        index_t entity_start_index = elem_offsets_access[ei];
        index_t entity_end_index = (ei < topo_num_elems - 1) ? elem_offsets_access[ei + 1] : 
                                       topo_elem_conn.dtype().number_of_elements();
        index_t entity_size = entity_end_index - entity_start_index;
        topo_conn_len += entity_size;

        // Get the vector we'll populate.
        std::vector<int64> &elem_indices = entity_index_bag[bi];
        elem_indices.resize(entity_size);

        // Store the connectivity into the vector.
        for(index_t i = 0; i < entity_size; i++)
            elem_indices[i] = elem_conn_access[entity_start_index + i];

        // Set the entity dimension
        entity_dim_bag[bi] = topo_shape.dim;

        // Keep track of original id.
        entity_origid_bag[bi] = ei;
    }
    // Prepopulate element connectivity
    auto &dim_buffers_topo = dim_buffers[topo_shape.dim];
    dim_buffers_topo.reserve(topo_num_coords);
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        index_t bi = topo_num_coords + ei;
        std::vector<int64> &elem_indices = entity_index_bag[bi];
        dim_buffers_topo.insert(dim_buffers_topo.end(), elem_indices.begin(), elem_indices.end());
    }

    constexpr index_t ENTITY_REQUIRES_ID = -1;

    while(!entity_index_bag.empty())
    {
        // Pop some work off of the deques
        std::vector<int64> entity_indices(std::move(entity_index_bag.front()));
        entity_index_bag.pop_front();
        index_t entity_dim = entity_dim_bag.front();
        entity_dim_bag.pop_front();
        std::vector< std::pair<int64, int64> > entity_parents(std::move(entity_parent_bag.front()));
        entity_parent_bag.pop_front();
        index_t entity_origid = entity_origid_bag.front();
        entity_origid_bag.pop_front();

        // Make some references based on entity_dim.
        std::vector<int64> &dim_buffer = dim_buffers[entity_dim];
        std::map< std::vector<index_t>, index_t > &dim_geid_map = dim_geid_maps[entity_dim];
        auto &dim_geassocs = dim_geassocs_maps[entity_dim];
        auto &dim_leassocs = dim_leassocs_maps[entity_dim];
        std::vector<index_t> &dim_le2ge_map = dim_le2ge_maps[entity_dim];
        const ShapeType &dim_shape = topo_cascade.get_shape(entity_dim);

        // Add Element to Topology //

        index_t global_id = entity_origid;
        const index_t local_id = next_local_id(entity_dim);
        if(global_id == ENTITY_REQUIRES_ID)
        {
            // Make a unique map key from the entity_indices.

            // NOTE: This code assumes that all entities can be uniquely
            // identified by the list of coordinate indices of which they
            // are comprised. This is certainly true of all implicit topologies
            // and of 2D polygonal topologies, but it may not be always the
            // case for 3D polygonal topologies.
            std::vector<int64> vert_ids;
            if(!dim_shape.is_polyhedral())
            {
                vert_ids = entity_indices;
            }
            else // if(dim_shape.is_polyhedral())
            {
                const index_t elem_outer_count = entity_indices.size();
                index_t_accessor elem_inner_sizes   = topo->fetch_existing("subelements/sizes").value();
                index_t_accessor elem_inner_offsets = topo_suboffsets.value();
                index_t_accessor elem_inner_conn    = topo->fetch_existing("subelements/connectivity").value();
            
                for(index_t oi = 0; oi < elem_outer_count; oi++)
                {
                    const index_t elem_inner_size   = elem_inner_sizes[entity_indices[oi]];
                    const index_t elem_inner_offset = elem_inner_offsets[entity_indices[oi]];

                    for(index_t ii = 0; ii < elem_inner_size; ii++)
                    {
                        const index_t vi = elem_inner_conn[elem_inner_offset + ii];
                        if(std::find(vert_ids.begin(), vert_ids.end(), vi) == vert_ids.end())
                            vert_ids.push_back(vi);
                    }
                }
            }
            std::sort(vert_ids.begin(), vert_ids.end());

            // Look up the entity in the map, make global_id.
            const auto dim_geid_it = dim_geid_map.find(vert_ids);
            if(dim_geid_it == dim_geid_map.end())
            {
                // Generate new id.
                global_id = next_global_id(entity_dim);

                // Append the entity indices to the connectivity.
                dim_buffer.insert(dim_buffer.end(), entity_indices.begin(), entity_indices.end());

                // Add vert_ids to the map so it is known.
                std::pair<std::vector<index_t>, index_t> obj(std::move(vert_ids), global_id);
                dim_geid_map.insert(std::move(obj));
            }
            else
            {
                // We've seen the entity before, reuse the id.
                global_id = dim_geid_it->second;
            }
        }

        { // create_entity(global_id, local_id, entity_dim)
            expand_assoc_capacity(IndexType::GLOBAL, global_id, entity_dim);
            expand_assoc_capacity(IndexType::LOCAL, local_id, entity_dim);

            if((index_t)dim_le2ge_map.size() <= local_id)
            {
                dim_le2ge_map.resize(local_id + 1);
            }
            dim_le2ge_map[local_id] = global_id;
        }

        // Add Element to Associations //

        add_entity_assoc(IndexType::GLOBAL, global_id, entity_dim, global_id, entity_dim);
        add_entity_assoc(IndexType::LOCAL, local_id, entity_dim, local_id, entity_dim);
        for(index_t pi = 0; pi < (index_t)entity_parents.size(); pi++)
        {
            index_t plevel = entity_parents.size() - pi - 1;
            const index_t parent_global_id = entity_parents[plevel].first;
            const index_t parent_local_id = entity_parents[plevel].second;
            const index_t parent_dim = entity_dim + pi + 1;
            add_entity_assoc(IndexType::GLOBAL, global_id, entity_dim, parent_global_id, parent_dim);
            add_entity_assoc(IndexType::LOCAL, local_id, entity_dim, parent_local_id, parent_dim);
        }

        // Add Embedded Elements for Further Processing //

        if(entity_dim > 0)
        {
            std::vector< std::pair<int64, int64> > embed_parents(std::move(entity_parents));
            embed_parents.push_back(std::make_pair(global_id, local_id));
            ShapeType embed_shape = topo_cascade.get_shape(entity_dim - 1);

            index_t elem_outer_count = dim_shape.is_poly() ?
                entity_indices.size() : dim_shape.embed_count;

            // NOTE(JRC): This is horribly complicated for the poly case and needs
            // to be refactored so that it's legible. There's a lot of overlap in
            // used variables where it feels unnecessary (e.g. 'poly' being
            // shoehorned into using 'implicit' variables), for example.
            
            
            //
            // NOTE(CYRUSH): Refactored to use accessors, however we still have
            // inner loop accessor fetches, which is less than ideal
            //
            for(index_t oi = 0, ooff = 0; oi < elem_outer_count; oi++)
            {
                index_t elem_inner_count = embed_shape.indices;

                if (dim_shape.is_polyhedral())
                {
                    index_t_accessor subelem_sizes   = topo->fetch_existing("subelements/sizes").value();
                    index_t_accessor subelem_offsets = topo_suboffsets.value();
                    elem_inner_count = subelem_sizes[entity_indices[oi]];
                    ooff = subelem_offsets[entity_indices[oi]];
                }

                std::vector<index_t> embed_indices;
                for(index_t ii = 0; ii < elem_inner_count; ii++)
                {
                    index_t ioff = ooff + (dim_shape.is_poly() ?
                        ii : dim_shape.embedding[oi * elem_inner_count + ii]);

                    if (dim_shape.is_polyhedral())
                    {
                        index_t_accessor subele_conn = topo->fetch_existing("subelements/connectivity").value();
                        embed_indices.push_back(subele_conn[ioff]);
                    }
                    else
                    {
                        embed_indices.push_back(entity_indices[ioff % entity_indices.size()]);
                    }
                }

                ooff += dim_shape.is_polygonal() ? 1 : 0;

                entity_index_bag.push_back(embed_indices);
                entity_dim_bag.push_back(embed_shape.dim);
                entity_parent_bag.push_back(embed_parents);

                // For any entity other than points, we'll want to make a new id.
                entity_origid_bag.push_back((embed_shape.dim == 0) ? embed_indices[0] : ENTITY_REQUIRES_ID);
            }
        }
    }

    // Move Topological Data into Per-Dim Nodes //

    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        Node &dim_conn = dim_topos[di]["elements/connectivity"];
        dim_conn.set(DataType(int_dtype.id(), dim_buffers[di].size()));

        data.reset();
        data.set_external(DataType::int64(dim_buffers[di].size()),
            &(dim_buffers[di][0]));
        data.to_data_type(int_dtype.id(), dim_conn);

        // Initialize element sizes for 2D polygonal mesh generating
        // from 3D polyhedral mesh
        if(di == 2 && topo_shape.is_polyhedral())
        {
            Node &poly_sizes = dim_topos[di]["elements/sizes"];
            poly_sizes.set(DataType(int_dtype.id(), dim_geid_maps[di].size()));

            temp.reset();
            data.reset();

            for(const auto &poly_pair : dim_geid_maps[di])
            {
                const std::vector<index_t> &poly_verts = poly_pair.first;
                const index_t &poly_geid = poly_pair.second;

                temp.set_external(DataType(int_dtype.id(), 1),
                    poly_sizes.element_ptr(poly_geid));
                data.set((index_t)poly_verts.size());
                data.to_data_type(int_dtype.id(), temp);
            }
        }

        topology::unstructured::generate_offsets_inline(dim_topos[di]);
    }
}

//---------------------------------------------------------------------------//
void
TopologyMetadata::expand_assoc_capacity(IndexType type, index_t idx, index_t dim)
{
    auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps[dim] : dim_geassocs_maps[dim];
    index_t tdim1 = topo_shape.dim + 1;
    index_t idxT = tdim1 * idx;
    if(idxT >= (index_t)dim_assocs.size())
    {
        index_t start = dim_assocs.size();
        index_t end = idxT + tdim1;
        dim_assocs.resize(idxT + tdim1);
        // Give each association vector a little memory now to reduce the
        // number of reallocations later.
        for(index_t i = start; i < end; i++)
            dim_assocs[i].reserve(4);
    }
}

//---------------------------------------------------------------------------//
void
TopologyMetadata::add_entity_assoc(IndexType type, index_t e0_id, index_t e0_dim, index_t e1_id, index_t e1_dim)
{
    auto &e0_cross_assocs = get_entity_assocs(type, e0_id, e0_dim, e1_dim);
    if(std::find(e0_cross_assocs.begin(), e0_cross_assocs.end(), e1_id) == e0_cross_assocs.end())
    {
        e0_cross_assocs.push_back(e1_id);
    }

    auto &e1_cross_assocs = get_entity_assocs(type, e1_id, e1_dim, e0_dim);
    if(std::find(e1_cross_assocs.begin(), e1_cross_assocs.end(), e0_id) == e1_cross_assocs.end())
    {
        e1_cross_assocs.push_back(e0_id);
    }
}

//---------------------------------------------------------------------------//
const std::vector<index_t>&
TopologyMetadata::get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim) const
{
    auto &dim_assoc = (type == IndexType::LOCAL) ? dim_leassocs_maps[entity_dim] : dim_geassocs_maps[entity_dim];
    index_t tdim1 = topo_shape.dim + 1;
    index_t sidx = entity_id * tdim1 + assoc_dim;
    return dim_assoc[sidx];
}

//---------------------------------------------------------------------------//
std::vector<index_t>&
TopologyMetadata::get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim)
{
    auto &dim_assoc = (type == IndexType::LOCAL) ? dim_leassocs_maps[entity_dim] : dim_geassocs_maps[entity_dim];
    index_t tdim1 = topo_shape.dim + 1;
    index_t sidx = entity_id * tdim1 + assoc_dim;
    return dim_assoc[sidx];
}


//---------------------------------------------------------------------------//
void
TopologyMetadata::get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const
{
    auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;

    index_t tdim1 = topo_shape.dim + 1;
    index_t dimlen = static_cast<index_t>(dim_assocs[src_dim].size()) / tdim1;
    std::vector<index_t> values, sizes, offsets;
    for(index_t sdi = 0, so = 0; sdi < dimlen; sdi++, so += sizes.back())
    {
        const std::vector<index_t> &src_assocs = get_entity_assocs(type, sdi, src_dim, dst_dim);
        values.insert(values.end(), src_assocs.begin(), src_assocs.end());
        sizes.push_back((index_t)src_assocs.size());
        offsets.push_back(so);
    }
// NOTE: can we store the data directly into the Conduit node?
    std::vector<index_t>* path_data[] = { &values, &sizes, &offsets };
    std::string path_names[] = { "values", "sizes", "offsets" };
    const index_t path_count = sizeof(path_data) / sizeof(path_data[0]);
    for(index_t pi = 0; pi < path_count; pi++)
    {
        Node data;
        data.set(*path_data[pi]);
        data.to_data_type(int_dtype.id(), map_node[path_names[pi]]);
    }
}

//---------------------------------------------------------------------------//
void
TopologyMetadata::get_point_data(IndexType type, index_t point_id, Node &data) const
{
    const index_t point_gid = (type == IndexType::LOCAL) ?
        dim_le2ge_maps[0][point_id] : point_id;

    if(data.dtype().is_empty())
    {
        data.set(DataType::float64(3));
    }
    const DataType data_dtype(data.dtype().id(), 1);

    Node temp1, temp2;
    const std::vector<std::string> csys_axes = coordset::axes(*cset);
    for(index_t di = 0; di < topo_shape.dim; di++)
    {
        temp1.set_external(float_dtype,
            (void*)(*cset)["values"][csys_axes[di]].element_ptr(point_gid));
        temp2.set_external(data_dtype, data.element_ptr(di));
        temp1.to_data_type(data_dtype.id(), temp2);
    }
}


//---------------------------------------------------------------------------//
index_t
TopologyMetadata::get_length(index_t dim) const
{
    // NOTE: The default version of 'get_length' gets the total length of all
    // unique entities in the topology. The parameterized version fetches the
    // length for just that parameter's dimension.

    index_t start_dim = (dim >= 0) ? dim : 0;
    index_t end_dim = (dim >= 0) ? dim : topo_shape.dim;

    index_t topo_length = 0;
    for(index_t di = start_dim; di <= end_dim; di++)
    {
        topo_length += topology::length(dim_topos[di]);
    }

    return topo_length;
}


//---------------------------------------------------------------------------//
index_t
TopologyMetadata::get_embed_length(index_t entity_dim, index_t embed_dim) const
{
    // NOTE: The default version of 'get_embed_length' gets the total number of
    // embeddings for each entity at the top level to the embedding level. The
    // parameterized version just fetches the number of embeddings for one
    // specific entity at the top level.

    std::vector<index_t> entity_index_bag;
    std::vector<index_t> entity_dim_bag;
    for(index_t ei = 0; ei < this->get_length(entity_dim); ei++)
    {
        entity_index_bag.push_back(ei);
        entity_dim_bag.push_back(entity_dim);
    }

    std::set<index_t> embed_set;
    index_t embed_length = 0;
    while(!entity_index_bag.empty())
    {
        index_t entity_index = entity_index_bag.back();
        entity_index_bag.pop_back();
        index_t entity_dim_back = entity_dim_bag.back();
        entity_dim_bag.pop_back();

        if(entity_dim_back == embed_dim)
        {
            if(embed_set.find(entity_index) == embed_set.end())
            {
                embed_length++;
            }
            embed_set.insert(entity_index);
        }
        else
        {
            const std::vector<index_t> &embed_ids = get_entity_assocs(
                TopologyMetadata::LOCAL, entity_index, entity_dim_back, entity_dim_back - 1);
            for(index_t ei = 0; ei < (index_t)embed_ids.size(); ei++)
            {
                entity_index_bag.push_back(embed_ids[ei]);
                entity_dim_bag.push_back(entity_dim_back - 1);
            }
        }
    }

    return embed_length;
}


//---------------------------------------------------------------------------//
std::string
TopologyMetadata::to_json() const
{
    Node mesh;

    Node &mesh_coords = mesh["coordsets"][(*topo)["coordset"].as_string()];
    mesh_coords.set_external(*cset);

    Node &mesh_topos = mesh["topologies"];
    for(index_t di = 0; di <= topo_shape.dim; di++)
    {
        std::ostringstream oss;
        oss << "d" << di;
        mesh_topos[oss.str()].set_external(dim_topos[di]);
    }

    return mesh.to_json();
}

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
