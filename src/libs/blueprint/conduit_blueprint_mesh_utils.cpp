// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_utils.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_blueprint_mesh_utils.hpp"

// access one-to-many index types
namespace O2MIndex = conduit::blueprint::o2mrelation;

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
/// blueprint mesh utility structures
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
ShapeType::ShapeType()
{
    init(-1);
}

//---------------------------------------------------------------------------//
ShapeType::ShapeType(const index_t type_id)
{
    init(type_id);
}

//---------------------------------------------------------------------------//
ShapeType::ShapeType(const std::string &type_name)
{
    init(type_name);
}

//---------------------------------------------------------------------------//
ShapeType::ShapeType(const conduit::Node &topology)
{
    init(-1);

    if(topology["type"].as_string() == "unstructured" &&
        topology["elements"].has_child("shape"))
    {
        init(topology["elements/shape"].as_string());
    }
}


//---------------------------------------------------------------------------//
void
ShapeType::init(const std::string &type_name)
{
    init(-1);

    for(index_t i = 0; i < (index_t)TOPO_SHAPES.size(); i++)
    {
        if(type_name == TOPO_SHAPES[i])
        {
            init(i);
        }
    }
}


//---------------------------------------------------------------------------//
void
ShapeType::init(const index_t type_id)
{
    if(type_id < 0 || type_id >= (index_t)TOPO_SHAPES.size())
    {
        type = "";
        id = dim = indices = embed_id = embed_count = -1;
        embedding = NULL;
    }
    else
    {
        type = TOPO_SHAPES[type_id];
        id = type_id;
        dim = TOPO_SHAPE_DIMS[type_id];
        indices = TOPO_SHAPE_INDEX_COUNTS[type_id];

        embed_id = TOPO_SHAPE_EMBED_TYPES[type_id];
        embed_count = TOPO_SHAPE_EMBED_COUNTS[type_id];
        embedding = const_cast<index_t*>(TOPO_SHAPE_EMBEDDINGS[type_id]);
    }
}


//---------------------------------------------------------------------------//
bool
ShapeType::is_poly() const
{
    return embedding == NULL;
}


//---------------------------------------------------------------------------//
bool
ShapeType::is_polygonal() const
{
    return embedding == NULL && dim == 2;
}


//---------------------------------------------------------------------------//
bool
ShapeType::is_polyhedral() const
{
    return embedding == NULL && dim == 3;
}


//---------------------------------------------------------------------------//
bool
ShapeType::is_valid() const
{
    return id >= 0;
}


//---------------------------------------------------------------------------//
ShapeCascade::ShapeCascade(const conduit::Node &topology)
{
    ShapeType base_type(topology);
    init(base_type);
}

//---------------------------------------------------------------------------//
ShapeCascade::ShapeCascade(const ShapeType &base_type)
{
    init(base_type);
}

//---------------------------------------------------------------------------//
index_t
ShapeCascade::get_num_embedded(const index_t level) const
{
    index_t num_embedded = -1;

    if(!get_shape().is_poly())
    {
        num_embedded = 1;
        for(index_t di = level + 1; di <= dim; di++)
        {
            num_embedded *= dim_types[di].embed_count;
        }
    }

    return num_embedded;
}


//---------------------------------------------------------------------------//
const ShapeType&
ShapeCascade::get_shape(const index_t level) const
{
    return dim_types[level < 0 ? dim : level];
}

//---------------------------------------------------------------------------//
void
ShapeCascade::init(const ShapeType &base_type)
{
    dim = base_type.dim;

    dim_types[base_type.dim] = base_type;
    for(index_t di = base_type.dim - 1; di >= 0; di--)
    {
        dim_types[di] = ShapeType(dim_types[di + 1].embed_id);
    }
}

//---------------------------------------------------------------------------//
TopologyMetadata::TopologyMetadata(const conduit::Node &topology, const conduit::Node &coordset) :
    topo(&topology), cset(&coordset),
    int_dtype(find_widest_dtype(link_nodes(topology, coordset), DEFAULT_INT_DTYPES)),
    float_dtype(find_widest_dtype(link_nodes(topology, coordset), DEFAULT_FLOAT_DTYPE)),
    topo_cascade(topology), topo_shape(topology)
{
    // NOTE(JRC): This type current only works at forming associations within
    // an unstructured topology's hierarchy.
    Node topo_offsets;
    topology::unstructured::generate_offsets(*topo, topo_offsets);
    const index_t topo_num_elems = topo_offsets.dtype().number_of_elements();
    const index_t topo_num_coords = coordset::length(coordset);

    // Allocate Data Templates for Outputs //

    dim_topos.resize(topo_shape.dim + 1);
    dim_geid_maps.resize(topo_shape.dim + 1);
    dim_geassocs_maps.resize(topo_shape.dim + 1);
    dim_leassocs_maps.resize(topo_shape.dim + 1);
    dim_le2ge_maps.resize(topo_shape.dim + 1);

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
    std::vector< std::vector<int64> > dim_buffers(topo_shape.dim + 1);

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
    std::deque< std::vector< std::pair<int64, int64> > > entity_parent_bag(bag_num_elems);

    // NOTE(JRC): We start with processing the points of the topology followed
    // by the top-level elements in order to ensure that order is preserved
    // relative to the original topology for these entities.
    for(index_t pi = 0; pi < topo_num_coords; pi++)
    {
        index_t bi = pi;
        entity_index_bag[bi].push_back(bi);
        entity_dim_bag[bi] = 0;
    }
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        index_t bi = topo_num_coords + ei;

        temp.reset();
        get_entity_data(TopologyMetadata::GLOBAL, ei, topo_shape.dim, temp);

        std::vector<int64> &elem_indices = entity_index_bag[bi];
        elem_indices.resize(temp.dtype().number_of_elements());
        data.set_external(DataType::int64(elem_indices.size()), &elem_indices[0]);
        temp.to_int64_array(data);

        entity_dim_bag[bi] = topo_shape.dim;
    }

    while(!entity_index_bag.empty())
    {
        std::vector<int64> entity_indices = entity_index_bag.front();
        entity_index_bag.pop_front();
        index_t entity_dim = entity_dim_bag.front();
        entity_dim_bag.pop_front();
        std::vector< std::pair<int64, int64> > entity_parents = entity_parent_bag.front();
        entity_parent_bag.pop_front();

        std::vector<int64> &dim_buffer = dim_buffers[entity_dim];
        std::map< std::set<index_t>, index_t > &dim_geid_map = dim_geid_maps[entity_dim];
        auto &dim_geassocs = dim_geassocs_maps[entity_dim];
        auto &dim_leassocs = dim_leassocs_maps[entity_dim];
        std::vector<index_t> &dim_le2ge_map = dim_le2ge_maps[entity_dim];
        ShapeType dim_shape = topo_cascade.get_shape(entity_dim);

        // Add Element to Topology //

        // NOTE: This code assumes that all entities can be uniquely
        // identified by the list of coordinate indices of which they
        // are comprised. This is certainly true of all implicit topologies
        // and of 2D polygonal topologies, but it may not be always the
        // case for 3D polygonal topologies.
        std::set<int64> vert_ids;
        if(!dim_shape.is_polyhedral())
        {
            vert_ids = std::set<int64>(entity_indices.begin(), entity_indices.end());
        }
        else // if(dim_shape.is_polyhedral())
        {
            const index_t elem_outer_count = entity_indices.size();
            for(index_t oi = 0; oi < elem_outer_count; oi++)
            {
                temp.set_external((*topo)["subelements/offsets"]);
                data.set_external(int_dtype, temp.element_ptr(entity_indices[oi]));
                const index_t elem_inner_offset = data.to_index_t();

                temp.set_external((*topo)["subelements/sizes"]);
                data.set_external(int_dtype, temp.element_ptr(entity_indices[oi]));
                const index_t elem_inner_count = data.to_index_t();

                for(index_t ii = 0; ii < elem_inner_count; ii++)
                {
                    temp.set_external((*topo)["subelements/connectivity"]);
                    data.set_external(int_dtype, temp.element_ptr(elem_inner_offset + ii));
                    const index_t vi = data.to_int64();
                    vert_ids.insert(vi);
                }
            }
        }

        const index_t local_id = dim_leassocs.size();
        if(dim_geid_map.find(vert_ids) == dim_geid_map.end())
        {
            const index_t global_id = dim_geassocs.size();
            dim_buffer.insert(dim_buffer.end(), entity_indices.begin(), entity_indices.end());
            dim_geid_map[vert_ids] = global_id;
        }
        const index_t global_id = dim_geid_map.find(vert_ids)->second;

        { // create_entity(global_id, local_id, entity_dim)
            if((index_t)dim_geassocs.size() <= global_id)
            {
                dim_geassocs.resize(global_id + 1);
            }
            if((index_t)dim_leassocs.size() <= local_id)
            {
                dim_leassocs.resize(local_id + 1);
            }
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
            std::vector< std::pair<int64, int64> > embed_parents = entity_parents;
            embed_parents.push_back(std::make_pair(global_id, local_id));
            ShapeType embed_shape = topo_cascade.get_shape(entity_dim - 1);

            index_t elem_outer_count = dim_shape.is_poly() ?
                entity_indices.size() : dim_shape.embed_count;

            // NOTE(JRC): This is horribly complicated for the poly case and needs
            // to be refactored so that it's legible. There's a lot of overlap in
            // used variables where it feels unnecessary (e.g. 'poly' being
            // shoehorned into using 'implicit' variables), for example.
            for(index_t oi = 0, ooff = 0; oi < elem_outer_count; oi++)
            {
                index_t elem_inner_count = embed_shape.indices;

                if (dim_shape.is_polyhedral())
                {
                    const Node &subelem_off_const = (*topo)["subelements/offsets"];
                    const Node &subelem_size_const = (*topo)["subelements/sizes"];

                    Node subelem_off; subelem_off.set_external(subelem_off_const);
                    Node subelem_size; subelem_size.set_external(subelem_size_const);

                    temp.set_external(int_dtype,
                        subelem_off.element_ptr(entity_indices[oi]));
                    ooff = temp.to_int64();
                    temp.set_external(int_dtype,
                        subelem_size.element_ptr(entity_indices[oi]));
                    elem_inner_count = temp.to_int64();
                }

                std::vector<int64> embed_indices;
                for(index_t ii = 0; ii < elem_inner_count; ii++)
                {
                    index_t ioff = ooff + (dim_shape.is_poly() ?
                        ii : dim_shape.embedding[oi * elem_inner_count + ii]);

                    if (dim_shape.is_polyhedral())
                    {
                        const Node &subelem_conn_const = (*topo)["subelements/connectivity"];
                        Node subelem_conn; subelem_conn.set_external(subelem_conn_const);

                        temp.set_external(int_dtype,
                            subelem_conn.element_ptr(ioff));
                        embed_indices.push_back(temp.to_int64());
                    }
                    else
                    {
                        embed_indices.push_back(
                            entity_indices[ioff % entity_indices.size()]);
                    }
                }

                ooff += dim_shape.is_polygonal() ? 1 : 0;

                entity_index_bag.push_back(embed_indices);
                entity_dim_bag.push_back(embed_shape.dim);
                entity_parent_bag.push_back(embed_parents);
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
                const std::set<index_t> &poly_verts = poly_pair.first;
                const index_t &poly_geid = poly_pair.second;

                temp.set_external(DataType(int_dtype.id(), 1),
                    poly_sizes.element_ptr(poly_geid));
                data.set((index_t)poly_verts.size());
                data.to_data_type(int_dtype.id(), temp);
            }
        }

        Node dim_topo_offsets;
        topology::unstructured::generate_offsets(dim_topos[di], dim_topo_offsets);
    }
}

//---------------------------------------------------------------------------//
void
TopologyMetadata::add_entity_assoc(IndexType type, index_t e0_id, index_t e0_dim, index_t e1_id, index_t e1_dim)
{
    auto &assoc_maps = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;
    std::vector< std::pair< std::vector<index_t>, std::set<index_t> > > *entity_assocs[2] = {
        &assoc_maps[e0_dim][e0_id],
        &assoc_maps[e1_dim][e1_id]
    };

    for(index_t ai = 0; ai < 2; ai++)
    {
        auto &curr_assocs = *entity_assocs[ai];
        curr_assocs.resize(topo_shape.dim + 1);

        const index_t cross_id = (ai == 0) ? e1_id : e0_id;
        const index_t cross_dim = (ai == 0) ? e1_dim : e0_dim;
        auto &cross_assocs = curr_assocs[cross_dim];
        if(cross_assocs.second.find(cross_id) == cross_assocs.second.end())
        {
            cross_assocs.first.push_back(cross_id);
            cross_assocs.second.insert(cross_id);
        }
    }
}


//---------------------------------------------------------------------------//
const std::vector<index_t>&
TopologyMetadata::get_entity_assocs(IndexType type, index_t entity_id, index_t entity_dim, index_t assoc_dim) const
{
    auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;
    return dim_assocs[entity_dim][entity_id][assoc_dim].first;
}


//---------------------------------------------------------------------------//
void
TopologyMetadata::get_dim_map(IndexType type, index_t src_dim, index_t dst_dim, Node &map_node) const
{
    auto &dim_assocs = (type == IndexType::LOCAL) ? dim_leassocs_maps : dim_geassocs_maps;

    std::vector<index_t> values, sizes, offsets;
    for(index_t sdi = 0, so = 0; sdi < (index_t)dim_assocs[src_dim].size(); sdi++, so += sizes.back())
    {
        const std::vector<index_t> &src_assocs = get_entity_assocs(type, sdi, src_dim, dst_dim);
        values.insert(values.end(), src_assocs.begin(), src_assocs.end());
        sizes.push_back((index_t)src_assocs.size());
        offsets.push_back(so);
    }

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
TopologyMetadata::get_entity_data(IndexType type, index_t entity_id, index_t entity_dim, Node &data) const
{
    Node temp;

    // NOTE(JRC): This is done in order to get around 'const' casting for
    // data pointers that won't be changed by the function anyway.
    Node dim_conn; dim_conn.set_external(dim_topos[entity_dim]["elements/connectivity"]);
    Node dim_off; dim_off.set_external(dim_topos[entity_dim]["elements/offsets"]);

    const DataType conn_dtype(dim_conn.dtype().id(), 1);
    const DataType off_dtype(dim_off.dtype().id(), 1);
    const DataType data_dtype = data.dtype().is_number() ? data.dtype() : DataType::int64(1);

    // FIXME(JRC): This code assumes that the per-element index data is packed
    // in memory, which isn't guaranteed to be the case (could be stride between
    // values, etc.).

    const index_t entity_gid = (type == IndexType::LOCAL) ?
        dim_le2ge_maps[entity_dim][entity_id] : entity_id;
    temp.set_external(off_dtype, dim_off.element_ptr(entity_gid));
    index_t entity_start_index = temp.to_int64();
    temp.set_external(off_dtype, dim_off.element_ptr(entity_gid + 1));
    index_t entity_end_index = (entity_gid < get_length(entity_dim) - 1) ?
        temp.to_int64() : dim_conn.dtype().number_of_elements();

    index_t entity_size = entity_end_index - entity_start_index;
    temp.set_external(DataType(conn_dtype.id(), entity_size),
        dim_conn.element_ptr(entity_start_index));
    temp.to_data_type(data_dtype.id(), data);
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

//-----------------------------------------------------------------------------
/// blueprint mesh utility query functions
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Node link_nodes(const Node &lhs, const Node &rhs)
{
    Node linker;
    linker.append().set_external(lhs);
    linker.append().set_external(rhs);
    return linker;
}


//-----------------------------------------------------------------------------
DataType
find_widest_dtype(const Node &node, const std::vector<DataType> &default_dtypes)
{
    DataType widest_dtype(default_dtypes[0].id(), 0, 0, 0, 0, default_dtypes[0].endianness());

    std::vector<const Node*> node_bag(1, &node);
    while(!node_bag.empty())
    {
        const Node *curr_node = node_bag.back(); node_bag.pop_back();
        const DataType curr_dtype = curr_node->dtype();
        if( curr_dtype.is_list() || curr_dtype.is_object() )
        {
            NodeConstIterator curr_node_it = curr_node->children();
            while(curr_node_it.has_next())
            {
                node_bag.push_back(&curr_node_it.next());
            }
        }
        else
        {
            for(index_t ti = 0; ti < (index_t)default_dtypes.size(); ti++)
            {
                const DataType &valid_dtype = default_dtypes[ti];
                bool is_valid_dtype =
                    (curr_dtype.is_floating_point() && valid_dtype.is_floating_point()) ||
                    (curr_dtype.is_signed_integer() && valid_dtype.is_signed_integer()) ||
                    (curr_dtype.is_unsigned_integer() && valid_dtype.is_unsigned_integer()) ||
                    (curr_dtype.is_string() && valid_dtype.is_string());
                if(is_valid_dtype && (widest_dtype.element_bytes() < curr_dtype.element_bytes()))
                {
                    widest_dtype.set(DataType(curr_dtype.id(), 1));
                }
            }
        }
    }

    bool no_type_found = widest_dtype.element_bytes() == 0;
    return no_type_found ? default_dtypes[0] : widest_dtype;
}


//-----------------------------------------------------------------------------
DataType
find_widest_dtype(const Node &node, const DataType &default_dtype)
{
    return find_widest_dtype(node, std::vector<DataType>(1, default_dtype));
}


//-----------------------------------------------------------------------------
const Node *
find_reference_node(const Node &node, const std::string &ref_key)
{
    const Node *res = nullptr;

    // NOTE: This segment of code is necessary to transform "topology" into
    // "topologies" while keeping all other dependency names (e.g. "coordset")
    // simply plural by just appending an "s" character.
    const std::string ref_section = (ref_key[ref_key.length()-1] != 'y') ?
        ref_key + "s" : ref_key.substr(0, ref_key.length()-1) + "ies";

    if(node.has_child(ref_key))
    {
        const std::string &ref_value = node.fetch(ref_key).as_string();

        const Node *traverse_node = node.parent();
        while(traverse_node != NULL)
        {
            if(traverse_node->has_child(ref_section))
            {
                const Node &ref_parent = traverse_node->fetch(ref_section);
                if(ref_parent.has_child(ref_value))
                {
                    res = &ref_parent[ref_value];
                }
                break;
            }
            traverse_node = traverse_node->parent();
        }
    }

    return res;
}


//-----------------------------------------------------------------------------
// NOTE: 'node' can be any subtree of a Blueprint-compliant mesh
index_t
find_domain_id(const Node &node)
{
    index_t domain_id = -1;

    Node info;
    const Node *curr_node = &node;
    while(curr_node != NULL && domain_id == -1)
    {
        if(blueprint::mesh::verify(*curr_node, info))
        {
            const std::vector<const Node *> domains = blueprint::mesh::domains(*curr_node);
            const Node &domain = *domains.front();
            if(domain.has_path("state/domain_id"))
            {
                domain_id = domain["state/domain_id"].to_index_t();
            }
        }

        curr_node = curr_node->parent();
    }

    return domain_id;
}

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::coordset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::pair<std::string, std::vector<std::string>>
get_coordset_info(const Node &n)
{
    std::pair<std::string, std::vector<std::string>> info;
    std::string &cset_coordsys = info.first;
    std::vector<std::string> &cset_axes = info.second;

    std::string coords_path = "";
    if(n["type"].as_string() == "uniform")
    {
        if(n.has_child("origin"))
        {
            coords_path = "origin";
        }
        else if(n.has_child("spacing"))
        {
            coords_path = "spacing";
        }
        else
        {
            coords_path = "dims";
        }
    }
    else // if(n.has_child("values"))
    {
        coords_path = "values";
    }

    Node axis_names;
    const Node &cset_coords = n[coords_path];
    NodeConstIterator itr = cset_coords.children();
    while(itr.has_next())
    {
        itr.next();
        const std::string axis_name = itr.name();

        if(axis_name[0] == 'd' && axis_name.size() > 1)
        {
            axis_names[axis_name.substr(1, axis_name.length())];
        }
        else
        {
            axis_names[axis_name];
        }
    }

    std::vector<std::string> cset_base_axes;
    cset_coordsys = "unknown";
    if(axis_names.has_child("theta") || axis_names.has_child("phi"))
    {
        cset_coordsys = "spherical";
        cset_base_axes = SPHERICAL_AXES;
    }
    else if(axis_names.has_child("r")) // rz, or r w/o theta, phi
    {
        cset_coordsys = "cylindrical";
        cset_base_axes = CYLINDRICAL_AXES;
    }
    else if(axis_names.has_child("x") || axis_names.has_child("y") || axis_names.has_child("z"))
    {
        cset_coordsys = "cartesian";
        cset_base_axes = CARTESIAN_AXES;
    }
    else if(axis_names.has_child("i") || axis_names.has_child("j") || axis_names.has_child("k"))
    {
        cset_coordsys = "logical";
        cset_base_axes = LOGICAL_AXES;
    }

    // intersect 'cset_base_axes' and 'axis_names.child_names()'
    for(const std::string &base_axis : cset_base_axes)
    {
        for(const std::string &cset_axis : axis_names.child_names())
        {
            if(base_axis == cset_axis)
            {
                cset_axes.push_back(cset_axis);
                break;
            }
        }
    }

    // TODO(JRC): The following is an alterate (though potentially more error-prone) solution:
    // std::vector<std::string>(cset_axes.begin(), cset_axes.begin() + cset_coords.number_of_children());

    return info;
}


//-----------------------------------------------------------------------------
index_t
coordset::dims(const Node &n)
{
    const std::vector<std::string> csys_axes = coordset::axes(n);
    return (index_t)csys_axes.size();
}


//-----------------------------------------------------------------------------
index_t
coordset::length(const Node &n)
{
    index_t coordset_length = 1;

    const std::string csys_type = n["type"].as_string();
    const std::vector<std::string> csys_axes = coordset::axes(n);
    for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
    {
        if(csys_type == "uniform")
        {
            coordset_length *= n["dims"][LOGICAL_AXES[i]].to_index_t();
        }
        else if(csys_type == "rectilinear")
        {
            coordset_length *= n["values"][csys_axes[i]].dtype().number_of_elements();
        }
        else // if(csys_type == "explicit")
        {
            coordset_length = n["values"][csys_axes[i]].dtype().number_of_elements();
        }
    }

    return coordset_length;
}


//-----------------------------------------------------------------------------
std::vector<std::string>
coordset::axes(const Node &n)
{
    return get_coordset_info(n).second;
}


//-----------------------------------------------------------------------------
std::string
coordset::coordsys(const Node &n)
{
    return get_coordset_info(n).first;
}


//-----------------------------------------------------------------------------
std::vector<float64>
coordset::_explicit::coords(const Node &n, const index_t i)
{
    std::vector<float64> cvals;

    Node temp;
    for(const std::string &axis : coordset::axes(n))
    {
        const Node &axis_node = n["values"][axis];
        temp.set_external(DataType(axis_node.dtype().id(), 1),
            (void*)axis_node.element_ptr(i));
        cvals.push_back(temp.to_float64());
    }

    return std::vector<float64>(std::move(cvals));
}

//-----------------------------------------------------------------------------
std::vector<index_t>
coordset::dim_lengths(const conduit::Node &n)
{
    auto info = get_coordset_info(n);
    const std::string cset_type = n["type"].as_string();
    const std::vector<std::string> &cset_axes = info.second;
    const index_t dim = (index_t)cset_axes.size();

    std::vector<index_t> retval(dim, 0);
    for(index_t i = 0; i < dim; i++)
    {
        if(cset_type == "uniform")
        {
            retval[i] = n["dims"][LOGICAL_AXES[i]].to_index_t();
        }
        else if(cset_type == "rectilinear")
        {
            retval[i] = n["values"][cset_axes[i]].dtype().number_of_elements();
        }
        else // if(cset_type == "explicit")
        {
            retval[i] = n["values"][cset_axes[i]].dtype().number_of_elements();
        }
    }

    return retval;
}

//-----------------------------------------------------------------------------
template<typename data_array>
static void
typed_minmax(const data_array &da, float64 &out_min, float64 &out_max)
{
    // Figure out what primitive type we are dealing with
    using T = decltype(std::declval<data_array>().min());
    const index_t nelem = da.number_of_elements();
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();
    for(index_t i = 0; i < nelem; i++)
    {
        min = std::min(min, da[i]);
        max = std::max(max, da[i]);
    }
    out_min = (float64)min;
    out_max = (float64)max;
}

//-----------------------------------------------------------------------------
std::vector<float64>
coordset::extents(const Node &n)
{
    std::vector<float64> cset_extents;
    const std::string csys_type = n["type"].as_string();
    const std::vector<std::string> csys_axes = coordset::axes(n);
    const index_t naxes = (index_t)csys_axes.size();
    cset_extents.reserve(naxes*2);
    for(index_t i = 0; i < naxes; i++)
    {
        float64 min, max;
        if(csys_type == "uniform")
        {
            index_t origin = 0;
            float64 spacing = 1.0;
            index_t dim = n["dims"][LOGICAL_AXES[i]].to_index_t();
            if(n.has_child("origin")
                && n["origin"].has_child(csys_axes[i]))
            {
                origin = n["origin"][csys_axes[i]].to_index_t();
            }
            if(n.has_child("spacing")
                && n["spacing"].has_child("d"+csys_axes[i]))
            {
                spacing = n["spacing"]["d" + csys_axes[i]].to_float64();
            }
            min = (float64)origin;
            max = (float64)origin + (spacing * ((float64)dim - 1.));
            if(spacing < 0.)
            {
                std::swap(min, max);
            }
        }
        else // csys_type == "rectilinear" || csys_type == "explicit"
        {
            const auto &axis = n["values"][csys_axes[i]];
            const auto id = axis.dtype().id();
            switch(id)
            {
            case conduit::DataType::INT8_ID:
                typed_minmax(axis.as_int8_array(), min, max);
                break;
            case conduit::DataType::INT16_ID:
                typed_minmax(axis.as_int16_array(), min, max);
                break;
            case conduit::DataType::INT32_ID:
                typed_minmax(axis.as_int32_array(), min, max);
                break;
            case conduit::DataType::INT64_ID:
                typed_minmax(axis.as_int64_array(), min, max);
                break;
            case conduit::DataType::UINT8_ID:
                typed_minmax(axis.as_uint8_array(), min, max);
                break;
            case conduit::DataType::UINT16_ID:
                typed_minmax(axis.as_uint16_array(), min, max);
                break;
            case conduit::DataType::UINT32_ID:
                typed_minmax(axis.as_uint32_array(), min, max);
                break;
            case conduit::DataType::UINT64_ID:
                typed_minmax(axis.as_uint64_array(), min, max);
                break;
            case conduit::DataType::FLOAT32_ID:
                typed_minmax(axis.as_float32_array(), min, max);
                break;
            case conduit::DataType::FLOAT64_ID:
                typed_minmax(axis.as_float64_array(), min, max);
                break;
            }
        }
        cset_extents.push_back(min);
        cset_extents.push_back(max);
    }
    return cset_extents;
}

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::coordset::uniform --
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
std::vector<double>
coordset::uniform::spacing(const Node &n)
{
    auto info = get_coordset_info(n);
    const auto &cset_axes = info.second;
    std::vector<double> retval(cset_axes.size(), 1);
    if(n.has_child("spacing"))
    {
        const Node &n_spacing = n["spacing"];
        for(index_t i = 0; i < (index_t)cset_axes.size(); i++)
        {
            const std::string child_name = "d"+cset_axes[i];
            if(n_spacing.has_child(child_name))
            {
                retval[i] = n_spacing[child_name].to_double();
            }
        }
    }
    return retval;
}

//-----------------------------------------------------------------------------
std::vector<index_t>
coordset::uniform::origin(const Node &n)
{
    auto info = get_coordset_info(n);
    const auto &cset_axes = info.second;
    std::vector<index_t> retval(cset_axes.size(), 0);
    if(n.has_child("origin"))
    {
        const Node &n_spacing = n["origin"];
        for(index_t i = 0; i < (index_t)cset_axes.size(); i++)
        {
            const std::string child_name = cset_axes[i];
            if(n_spacing.has_child(child_name))
            {
                retval[i] = n_spacing[child_name].to_index_t();
            }
        }
    }
    return retval;
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::coordset::uniform --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::coordset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
index_t
topology::dims(const Node &n)
{
    index_t topology_dims = 1;

    const std::string type = n["type"].as_string();
    if(type != "unstructured")
    {
        const Node *coordset = find_reference_node(n, "coordset");
        topology_dims = coordset::dims(*coordset);
    }
    else // if(type == "unstructured")
    {
        ShapeType shape(n);
        topology_dims = (index_t)shape.dim;
    }

    return topology_dims;
}


//-----------------------------------------------------------------------------
void
topology::logical_dims(const Node &n, index_t *d, index_t maxdims)
{
    for(index_t i = 0; i < maxdims; i++)
        d[i] = 1;

    const std::string type = n["type"].as_string();
    if(type == "uniform" || type == "rectilinear")
    {
        const Node *coordset = find_reference_node(n, "coordset");
        const std::vector<std::string> csys_axes = coordset::axes(*coordset);
        for(index_t i = 0; i < (index_t)csys_axes.size(); i++)
        {
            d[i] = ((type == "uniform") ?
                coordset["dims"][LOGICAL_AXES[i]].to_index_t() :
                coordset["values"][csys_axes[i]].dtype().number_of_elements()) - 1;
        }
    }
    else if(type == "structured")
    {
        const Node &dims = n["elements/dims"];

        for(index_t i = 0; i < (index_t)dims.number_of_children(); i++)
        {
            d[i] = dims[LOGICAL_AXES[i]].to_index_t();
        }
    }
    else // if(type == "unstructured")
    {
        // TODO(JRC): This is rather inefficient because the offsets array
        // is discarded after this calculation is complete.
        Node topo_offsets;
        topology::unstructured::generate_offsets(n, topo_offsets);
        d[0] = topo_offsets.dtype().number_of_elements();
    }
}

//-----------------------------------------------------------------------------
index_t
topology::length(const Node &n)
{
    index_t d[3]={1,1,1};
    logical_dims(n, d, 3);
    return d[0] * d[1] * d[2];
}

//-----------------------------------------------------------------------------
void
topology::unstructured::generate_offsets(Node &n,
                                         Node &dest)
{
    dest.reset();

    // FIXME(JRC): There are weird cases wherein a polyhedral topology can have only
    // the 'elements/offsets' defined and not 'subelements/offsets', which isn't currently
    // properly handled by this function.

    if(n["elements"].has_child("offsets") && !n["elements/offsets"].dtype().is_empty())
    {
        if(&dest != &n["elements/offsets"])
        {
            dest.set_external(n["elements/offsets"]);
        }
    }
    else
    {
        const Node &n_const = n;
        Node &offsets = n["elements/offsets"];
        blueprint::mesh::utils::topology::unstructured::generate_offsets(n_const, offsets);
        if(&dest != &offsets)
        {
            dest.set_external(offsets);
        }
    }
}


//-----------------------------------------------------------------------------
void
topology::unstructured::generate_offsets(const Node &n,
                                         Node &dest)
{
    const ShapeType topo_shape(n);
    const DataType int_dtype = find_widest_dtype(n, DEFAULT_INT_DTYPES);
    std::string key("elements/connectivity"), stream_key("elements/stream");
    if(!n.has_path(key))
        key = stream_key;
    const Node &topo_conn = n[key];

    const DataType topo_dtype(topo_conn.dtype().id(), 1, 0, 0,
        topo_conn.dtype().element_bytes(), topo_conn.dtype().endianness());

    if(n["elements"].has_child("offsets") && !n["elements/offsets"].dtype().is_empty())
    {
        if(&dest != &n["elements/offsets"])
        {
            dest.set_external(n["elements/offsets"]);
        }
    }
    else if(n.has_path(stream_key))
    {
        dest.reset();
        // Mixed element types
        std::map<int,int> stream_id_npts;
        const conduit::Node &n_element_types = n["elements/element_types"];
        for(index_t i = 0; i < n_element_types.number_of_children(); i++)
        {
            const Node &n_et = n_element_types[i];
            auto stream_id = n_et["stream_id"].to_int();
            std::string shape(n_et["shape"].as_string());
            for(size_t j = 0; j < TOPO_SHAPES.size(); j++)
            {
                if(shape == TOPO_SHAPES[j])
                {
                    stream_id_npts[stream_id] = TOPO_SHAPE_EMBED_COUNTS[j];
                    break;
                }
            }
        }

        const Node &n_stream_ids = n["elements/element_index/stream_ids"];
        std::vector<index_t> offsets;
        if(n.has_path("elements/element_index/element_counts"))
        {
            const Node &n_element_counts = n["elements/element_index/element_counts"];

            index_t offset = 0, elemid = 0;
            for(index_t j = 0; j < n_stream_ids.dtype().number_of_elements(); j++)
            {
                // Get the j'th elements from n_stream_ids, n_element_counts
                const Node n_elem_ct_j(int_dtype,
                            const_cast<void*>(n_element_counts.element_ptr(j)), true);
                const Node n_stream_ids_j(int_dtype,
                            const_cast<void*>(n_stream_ids.element_ptr(j)), true);
                auto ec = static_cast<index_t>(n_elem_ct_j.to_int64());
                auto sid = static_cast<index_t>(n_stream_ids_j.to_int64());
                auto npts = stream_id_npts[sid];
                for(index_t i = 0; i < ec; i++)
                {
                    offsets.push_back(offset);
                    offset += npts;
                    elemid++;
                }
            }
        }
        else if(n.has_path("elements/element_index/offsets"))
        {
            const Node &n_stream = n["elements/stream"];
            const Node &n_element_offsets = n["elements/element_index/offsets"];
            index_t offset = 0, elemid = 0;
            for(index_t j = 0; j < n_stream_ids.dtype().number_of_elements(); j++)
            {
                // Get the j'th elements from n_stream_ids, n_element_offsets
                const Node n_stream_ids_j(int_dtype,
                            const_cast<void*>(n_stream_ids.element_ptr(j)), true);
                const Node n_element_offsets_j(int_dtype,
                            const_cast<void*>(n_element_offsets.element_ptr(j)), true);
                offset = n_element_offsets.to_index_t();
                index_t next_offset = offset;
                if(j == n_stream_ids.dtype().number_of_elements() - 1)
                {
                    next_offset = n_stream.dtype().number_of_elements();
                }
                else
                {
                    const Node n_element_offsets_j1(int_dtype,
                                const_cast<void*>(n_element_offsets.element_ptr(j)), true);
                    next_offset = n_element_offsets_j1.to_index_t();
                }
                const auto sid = static_cast<index_t>(n_stream_ids_j.to_int64());
                const auto npts = stream_id_npts[sid];
                while(offset < next_offset) {
                    offsets.push_back(offset);
                    offset += npts;
                    elemid++;
                }
            }
        }
        else
        {
            CONDUIT_ERROR("Stream based mixed topology has no element_counts or offsets.")
        }

        Node off_node;
        off_node.set_external(offsets);
        off_node.to_data_type(int_dtype.id(), dest);
    }
    else if(!topo_shape.is_poly())
    {
        dest.reset();       
        // Single element type
        const index_t num_topo_shapes =
            topo_conn.dtype().number_of_elements() / topo_shape.indices;

        Node shape_node(DataType::int64(num_topo_shapes));
        int64_array shape_array = shape_node.as_int64_array();
        for(index_t s = 0; s < num_topo_shapes; s++)
        {
            shape_array[s] = s * topo_shape.indices;
        }
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polygonal")
    {
        dest.reset();

        const Node &topo_size = n["elements/sizes"];
        std::vector<int64> shape_array;
        index_t i = 0;
        index_t s = 0;
        while(i < topo_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_size.element_ptr(i)), true);
            shape_array.push_back(s);
            s += index_node.to_int64();
            i++;
        }

        Node shape_node;
        shape_node.set_external(shape_array);
        shape_node.to_data_type(int_dtype.id(), dest);
    }
    else if(topo_shape.type == "polyhedral")
    {
        Node &dest_elem_off = const_cast<Node &>(n)["elements/offsets"];
        Node &dest_subelem_off = const_cast<Node &>(n)["subelements/offsets"];

        const Node& topo_elem_size = n["elements/sizes"];
        const Node& topo_subelem_size = n["subelements/sizes"];

        Node elem_node;
        Node subelem_node;

        std::vector<index_t> shape_array;
        index_t ei = 0;
        index_t es = 0;
        while(ei < topo_elem_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_elem_size.element_ptr(ei)), true);
            shape_array.push_back(es);
            es += index_node.to_index_t();
            ei++;
        }

        elem_node.set_external(shape_array);
        elem_node.to_data_type(int_dtype.id(), dest_elem_off);
        elem_node.to_data_type(int_dtype.id(), dest);

        shape_array.clear();
        ei = 0;
        es = 0;
        while(ei < topo_subelem_size.dtype().number_of_elements())
        {
            const Node index_node(int_dtype,
                const_cast<void*>(topo_subelem_size.element_ptr(ei)), true);
            shape_array.push_back(es);
            es += index_node.to_index_t();
            ei++;
        }

        subelem_node.set_external(shape_array);
        subelem_node.to_data_type(int_dtype.id(), dest_subelem_off);
    }
}


//-----------------------------------------------------------------------------
std::vector<index_t>
topology::unstructured::points(const Node &n,
                               const index_t ei)
{
    // NOTE(JRC): This is a workaround to ensure offsets are generated up-front
    // if they don't exist and aren't regenerated for each subcall that needs them.
    Node ntemp;
    ntemp.set_external(n);
    generate_offsets(ntemp, ntemp["elements/offsets"]);

    Node temp;
    const ShapeType topo_shape(ntemp);

    std::set<index_t> pidxs;
    if(!topo_shape.is_poly())
    {
        const Node &poffs_node = ntemp["elements/offsets"];
        temp.set_external(DataType(poffs_node.dtype().id(), 1),
            (void*)poffs_node.element_ptr(ei));
        const index_t eoff = temp.to_index_t();

        const Node &pidxs_node = ntemp["elements/connectivity"];
        for(index_t pi = 0; pi < topo_shape.indices; pi++)
        {
            temp.set_external(DataType(pidxs_node.dtype().id(), 1),
                (void*)pidxs_node.element_ptr(eoff + pi));
            pidxs.insert(temp.to_index_t());
        }
    }
    else // if(topo_shape.is_poly())
    {
        Node enode;
        std::set<index_t> eidxs;
        if(topo_shape.is_polygonal())
        {
            enode.set_external(ntemp["elements"]);

            eidxs.insert(ei);
        }
        else // if(topo_shape.is_polyhedral())
        {
            enode.set_external(ntemp["subelements"]);

            const Node &eidxs_node = ntemp["elements/connectivity"];
            o2mrelation::O2MIterator eiter(ntemp["elements"]);
            eiter.to(ei, O2MIndex::ONE);
            eiter.to_front(O2MIndex::MANY);
            while(eiter.has_next(O2MIndex::MANY))
            {
                eiter.next(O2MIndex::MANY);
                const index_t ii = eiter.index(O2MIndex::DATA);
                temp.set_external(DataType(eidxs_node.dtype().id(), 1),
                    (void*)eidxs_node.element_ptr(ii));
                eidxs.insert(temp.to_index_t());
            }
        }

        for(const index_t eidx : eidxs)
        {
            const Node &pidxs_node = enode["connectivity"];
            o2mrelation::O2MIterator piter(enode);
            piter.to(eidx, O2MIndex::ONE);
            piter.to_front(O2MIndex::MANY);
            while(piter.has_next(O2MIndex::MANY))
            {
                piter.next(O2MIndex::MANY);
                const index_t pi = piter.index(O2MIndex::DATA);
                temp.set_external(DataType(pidxs_node.dtype().id(), 1),
                    (void*)pidxs_node.element_ptr(pi));
                pidxs.insert(temp.to_index_t());
            }
        }
    }

    return std::vector<index_t>(pidxs.begin(), pidxs.end());
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::adjset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
adjset::canonicalize(Node &adjset)
{
    const index_t domain_id = find_domain_id(adjset);

    const std::vector<std::string> &adjset_group_names = adjset["groups"].child_names();
    for(const std::string &old_group_name : adjset_group_names)
    {
        const Node &group_node = adjset["groups"][old_group_name];
        const Node &neighbors_node = group_node["neighbors"];

        std::string new_group_name;
        {
            std::ostringstream oss;
            oss << "group";

            Node temp;
            DataType temp_dtype(neighbors_node.dtype().id(), 1);

            // NOTE(JRC): Need to use a vector instead of direct 'Node::to_index_t'
            // because the local node ID isn't included in the neighbor list and
            // 'DataArray' uses a static array size.
            std::vector<index_t> group_neighbors(1, domain_id);
            for(index_t ni = 0; ni < neighbors_node.dtype().number_of_elements(); ni++)
            {
                temp.set_external(temp_dtype, (void*)neighbors_node.element_ptr(ni));
                group_neighbors.push_back(temp.to_index_t());
            }
            std::sort(group_neighbors.begin(), group_neighbors.end());

            for(const index_t &neighbor_id : group_neighbors)
            {
                oss << "_" << neighbor_id;
            }

            new_group_name = oss.str();
        }

        adjset["groups"].rename_child(old_group_name, new_group_name);
    }
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::adjset --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils --
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

