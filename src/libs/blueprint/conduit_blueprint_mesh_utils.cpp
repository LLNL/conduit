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
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_execution.hpp"
#include "conduit_annotations.hpp"

// access one-to-many index types
namespace o2mrelation = conduit::blueprint::o2mrelation;

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
// -- begin conduit::blueprint::mesh::utils::connectivity --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
connectivity::make_element_2d(std::vector<index_t>& elem,
                              index_t element,
                              index_t iwidth)
{
    index_t ilo = element % iwidth;
    index_t jlo = element / iwidth;
    index_t ihi = ilo + 1;
    index_t jhi = jlo + 1;

    index_t ilo_jlo = (iwidth+1)*jlo + ilo;
    index_t ihi_jlo = (iwidth+1)*jlo + ihi;
    index_t ihi_jhi = (iwidth+1)*jhi + ihi;
    index_t ilo_jhi = (iwidth+1)*jhi + ilo;

    elem.push_back(ilo_jlo);
    elem.push_back(ihi_jlo);
    elem.push_back(ihi_jhi);
    elem.push_back(ilo_jhi);
}


void
connectivity::make_element_3d(ElemType& connect,
                              index_t element,
                              index_t iwidth,
                              index_t jwidth,
                              index_t kwidth,
                              SubelemMap& faces)
{
    index_t ilo = element % iwidth;
    index_t jlo = (element / iwidth) % jwidth;
    index_t klo = element / (iwidth*jwidth);
    index_t ihi = ilo + 1;
    index_t jhi = jlo + 1;
    index_t khi = klo + 1;

    index_t jlo_offset = (iwidth+1)*jlo;
    index_t jhi_offset = (iwidth+1)*jhi;
    index_t klo_offset = (iwidth+1)*(jwidth+1)*klo;
    index_t khi_offset = (iwidth+1)*(jwidth+1)*khi;


    index_t iface_start = 0; 
    index_t jface_start = (iwidth+1)*jwidth*kwidth;
    index_t kface_start = jface_start + iwidth*(jwidth+1)*kwidth;

    //ifaces
    {
        index_t j_offset = jlo_offset; 
        index_t k_offset = (iwidth+1)*jwidth*klo;

        index_t lo_face = iface_start + ilo + j_offset + k_offset;
        index_t hi_face = iface_start + ihi + j_offset + k_offset;


        //ilo face
        if (faces.find(lo_face) == faces.end())
        {
            auto& ilo_face = faces[lo_face];
            ilo_face.push_back(ilo + jlo_offset + klo_offset);
            ilo_face.push_back(ilo + jhi_offset + klo_offset);
            ilo_face.push_back(ilo + jhi_offset + khi_offset);
            ilo_face.push_back(ilo + jlo_offset + khi_offset);
        }
        //ihi face
        if (faces.find(hi_face) == faces.end())
        {
            auto& ihi_face = faces[hi_face];
            ihi_face.push_back(ihi + jlo_offset + klo_offset);
            ihi_face.push_back(ihi + jhi_offset + klo_offset);
            ihi_face.push_back(ihi + jhi_offset + khi_offset);
            ihi_face.push_back(ihi + jlo_offset + khi_offset);
        }
        connect.push_back(lo_face);
        connect.push_back(hi_face);
    }
    //jfaces
    {
        index_t i_offset = ilo;  
        index_t jlo_face_offset = iwidth*jlo; 
        index_t jhi_face_offset = iwidth*jhi; 
        index_t k_offset = iwidth*(jwidth+1)*klo;

        index_t lo_face = jface_start + i_offset + jlo_face_offset + k_offset;
        index_t hi_face = jface_start + i_offset + jhi_face_offset + k_offset;

        //jlo face
        if (faces.find(lo_face) == faces.end())
        {
            auto& jlo_face = faces[lo_face];
            jlo_face.push_back(ilo + jlo_offset + klo_offset);
            jlo_face.push_back(ihi + jlo_offset + klo_offset);
            jlo_face.push_back(ihi + jlo_offset + khi_offset);
            jlo_face.push_back(ilo + jlo_offset + khi_offset);
        }
        //jhi face
        if (faces.find(hi_face) == faces.end())
        {
            auto& jhi_face = faces[hi_face];
            jhi_face.push_back(ilo + jhi_offset + klo_offset);
            jhi_face.push_back(ihi + jhi_offset + klo_offset);
            jhi_face.push_back(ihi + jhi_offset + khi_offset);
            jhi_face.push_back(ilo + jhi_offset + khi_offset);
        }
        connect.push_back(lo_face);
        connect.push_back(hi_face);
    }
    //kfaces
    {
        index_t i_offset = ilo;  
        index_t j_offset = iwidth*jlo; 
        index_t klo_face_offset = iwidth*jwidth*klo;
        index_t khi_face_offset = iwidth*jwidth*khi;

        index_t lo_face = kface_start + i_offset + j_offset + klo_face_offset;
        index_t hi_face = kface_start + i_offset + j_offset + khi_face_offset;

        //klo face
        if (faces.find(lo_face) == faces.end())
        {
            auto& klo_face = faces[lo_face];
            klo_face.push_back(ilo + jlo_offset + klo_offset);
            klo_face.push_back(ihi + jlo_offset + klo_offset);
            klo_face.push_back(ihi + jhi_offset + klo_offset);
            klo_face.push_back(ilo + jhi_offset + klo_offset);
        }
        //khi face
        if (faces.find(hi_face) == faces.end())
        {
            auto& khi_face = faces[hi_face];
            khi_face.push_back(ilo + jlo_offset + khi_offset);
            khi_face.push_back(ihi + jlo_offset + khi_offset);
            khi_face.push_back(ihi + jhi_offset + khi_offset);
            khi_face.push_back(ilo + jhi_offset + khi_offset);
        }
        connect.push_back(/*kface_start +*/lo_face);
        connect.push_back(/*kface_start +*/hi_face);
    }
}




//-----------------------------------------------------------------------------
void
connectivity::create_elements_2d(const Node& ref_win,
                                 index_t i_lo,
                                 index_t j_lo,
                                 index_t iwidth,
                                 std::map<index_t, std::vector<index_t> >& elems)
{
    index_t origin_iref = ref_win["origin/i"].to_index_t();
    index_t origin_jref = ref_win["origin/j"].to_index_t();

    index_t ref_size_i = ref_win["dims/i"].to_index_t();
    index_t ref_size_j = ref_win["dims/j"].to_index_t();

    if (ref_size_i == 1)
    {
        index_t jstart = origin_jref - j_lo;
        index_t jend = origin_jref - j_lo + ref_size_j - 1;
        if (origin_iref == i_lo)
        {
            for (index_t jidx = jstart; jidx < jend; ++jidx)
            {
                index_t offset = jidx * iwidth;
                auto& elem_conn = elems[offset];
                if (elem_conn.empty())
                {
                    connectivity::make_element_2d(elem_conn,
                                                  offset,
                                                  iwidth);
                }
            }
        }
        else
        {
            for (index_t jidx = jstart; jidx < jend; ++jidx)
            {
                index_t offset = jidx * iwidth + (origin_iref - i_lo - 1);
                auto& elem_conn = elems[offset];
                if (elem_conn.empty())
                {
                    connectivity::make_element_2d(elem_conn,
                                                  offset,
                                                  iwidth);
                }
            }
        }
    }
    else if (ref_size_j == 1)
    {
        index_t istart = origin_iref - i_lo;
        index_t iend = origin_iref - i_lo + ref_size_i - 1;
        if (origin_jref == j_lo)
        {
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                auto& elem_conn = elems[iidx];
                if (elem_conn.empty())
                {
                    connectivity::make_element_2d(elem_conn,
                                                  iidx,
                                                  iwidth);
                }
            }
        }
        else
        {
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t offset = iidx + ((origin_jref - j_lo - 1) * iwidth);
                auto& elem_conn = elems[offset];
                if (elem_conn.empty())
                {
                    connectivity::make_element_2d(elem_conn,
                                                  offset,
                                                  iwidth);
                }
            }
        }
    }

    index_t istart = origin_iref - i_lo;
    index_t jstart = origin_jref - j_lo;
    index_t iend = istart + ref_size_i - 1;
    index_t jend = jstart + ref_size_j - 1;

    if (ref_size_i == 1)
    {
        if (origin_iref != i_lo)
        {
            --istart;
        }
        iend = istart + 1;
    }
    if (ref_size_j == 1)
    {
        if (origin_jref != j_lo)
        {
            --jstart;
        }
        jend = jstart + 1;
    }

    for (index_t jidx = jstart; jidx < jend; ++jidx)
    {
        index_t joffset = jidx * iwidth;
        for (index_t iidx = istart; iidx < iend; ++iidx)
        {
            index_t offset = joffset + iidx;
            auto& elem_conn = elems[offset];
            if (elem_conn.empty())
            {
                 connectivity::make_element_2d(elem_conn,
                                               offset,
                                               iwidth);
            }
        }
    }
}


//-----------------------------------------------------------------------------
void
connectivity::create_elements_3d(const Node& ref_win,
                                       index_t i_lo,
                                       index_t j_lo,
                                       index_t k_lo,
                                       index_t iwidth,
                                       index_t jwidth,
                                       index_t kwidth,
                                       std::map<index_t, ElemType>& elems,
                                       SubelemMap& faces)
{
    index_t origin_iref = ref_win["origin/i"].to_index_t();
    index_t origin_jref = ref_win["origin/j"].to_index_t();
    index_t origin_kref = ref_win["origin/k"].to_index_t();

    index_t ref_size_i = ref_win["dims/i"].to_index_t();
    index_t ref_size_j = ref_win["dims/j"].to_index_t();
    index_t ref_size_k = ref_win["dims/k"].to_index_t();

    index_t istart = origin_iref - i_lo;
    index_t jstart = origin_jref - j_lo;
    index_t kstart = origin_kref - k_lo;
    index_t iend = istart + ref_size_i - 1;
    index_t jend = jstart + ref_size_j - 1;
    index_t kend = kstart + ref_size_k - 1;

    if (ref_size_i == 1)
    {
        iend = istart + 1;
    }
    if (ref_size_j == 1)
    {
        jend = jstart + 1;
    }
    if (ref_size_k == 1)
    {
        kend = kstart + 1;
    }

    for (index_t kidx = kstart; kidx < kend; ++kidx)
    {
        index_t koffset = kidx * iwidth * jwidth;
        for (index_t jidx = jstart; jidx < jend; ++jidx)
        {
            index_t joffset = jidx * iwidth; 
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t offset = koffset + joffset + iidx;
                auto& elem_conn = elems[offset];
                if (elem_conn.empty())
                {
                     connectivity::make_element_3d(elem_conn,
                                                   offset,
                                                   iwidth,
                                                   jwidth,
                                                   kwidth,
                                                   faces);
                }
            }
        }
    }
}

void
connectivity::connect_elements_3d(const Node& ref_win,
                                        index_t i_lo,
                                        index_t j_lo,
                                        index_t k_lo,
                                        index_t iwidth,
                                        index_t jwidth,
                                        index_t& new_vertex,
                                        std::map<index_t, ElemType>& elems)
{
    index_t origin_iref = ref_win["origin/i"].to_index_t();
    index_t origin_jref = ref_win["origin/j"].to_index_t();
    index_t origin_kref = ref_win["origin/k"].to_index_t();

    index_t ref_size_i = ref_win["dims/i"].to_index_t();
    index_t ref_size_j = ref_win["dims/j"].to_index_t();
    index_t ref_size_k = ref_win["dims/k"].to_index_t();

    index_t kstart = origin_kref - k_lo;
    index_t kend = origin_kref - k_lo + ref_size_k - 1;
    if (kstart == kend) kend = kstart + 1;
    index_t jstart = origin_jref - j_lo;
    index_t jend = origin_jref - j_lo + ref_size_j - 1;
    if (jstart == jend) jend = jstart + 1;
    index_t istart = origin_iref - i_lo;
    index_t iend = origin_iref - i_lo + ref_size_i - 1;
    if (istart == iend) iend = istart + 1;

    for (index_t kidx = kstart; kidx < kend; ++kidx)
    {
        for (index_t jidx = jstart; jidx < jend; ++jidx)
        {
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t offset = kidx*iwidth*jwidth + jidx*iwidth + iidx;
                auto& elem_conn = elems[offset];
                elem_conn.push_back(new_vertex);
                ++new_vertex;
            }
        }
    }
}

void
connectivity::connect_elements_2d(const Node& ref_win,
                                  index_t i_lo,
                                  index_t j_lo,
                                  index_t iwidth,
                                  index_t ratio,
                                  index_t& new_vertex,
                                  std::map<index_t, std::vector<index_t> >& elems)
{
    index_t origin_iref = ref_win["origin/i"].to_index_t();
    index_t origin_jref = ref_win["origin/j"].to_index_t();

    index_t ref_size_i = ref_win["dims/i"].to_index_t();
    index_t ref_size_j = ref_win["dims/j"].to_index_t();

    if (ref_size_i == 1)
    {
        index_t jstart = origin_jref - j_lo;
        index_t jend = origin_jref - j_lo + ref_size_j - 1;
        if (origin_iref == i_lo)
        {
            for (index_t jidx = jstart; jidx < jend; ++jidx)
            {
                index_t offset = jidx * (iwidth);
                auto& elem_conn = elems[offset];
                if (ratio > 1)
                {
                    for (index_t nr = ratio-1; nr > 0; --nr)
                    {
                        elem_conn.push_back(new_vertex+nr-1);
                    }
                    new_vertex += ratio - 1;
                }
            }
        }
        else
        {
            for (index_t jidx = jstart; jidx < jend; ++jidx)
            {
                index_t offset = jidx * iwidth + (origin_iref - i_lo - 1);
                auto& elem_conn = elems[offset];
                if (ratio > 1)
                {
                    size_t new_size = elem_conn.size() + ratio - 1;
                    elem_conn.resize(new_size);
                    index_t corner = 1;
                    if (elem_conn[1] - elem_conn[0] != 1)
                    {
                        index_t ioff = offset % iwidth;
                        index_t joff = offset / iwidth;
                        index_t target = (iwidth+1)*joff + ioff + 1;
                        for (index_t nr = 1; nr < 1+ratio; ++nr)
                        {
                            if (elem_conn[nr] == target)
                            {
                                corner = nr;
                                break;
                            }
                        }
                    }
                    for (index_t nr = new_size-1; nr > corner+ratio-1; --nr)
                    {
                        elem_conn[nr] = elem_conn[nr-ratio+1];
                    }
                    for (index_t nr = corner+1; nr < corner+ratio; ++nr)
                    {
                        elem_conn[nr] = new_vertex;
                        ++new_vertex;
                    }
                }
            }
        }
    }
    else if (ref_size_j == 1)
    {
        index_t istart = origin_iref - i_lo;
        index_t iend = origin_iref - i_lo + ref_size_i - 1;
        if (origin_jref == j_lo)
        {
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                auto& elem_conn = elems[iidx];
                if (ratio > 1)
                {
                    size_t new_size = elem_conn.size() + ratio - 1;
                    elem_conn.resize(new_size);
                    for (index_t nr = new_size-1; nr > 1; --nr)
                    {
                        elem_conn[nr] = elem_conn[nr-ratio+1];
                    }
                    for (index_t nr = 1; nr < ratio; ++nr)
                    {
                        elem_conn[nr] = (new_vertex+nr-1);
                    }
                    new_vertex += ratio - 1;
                }
            }
        }
        else
        {
            for (index_t iidx = istart; iidx < iend; ++iidx)
            {
                index_t offset = iidx + ((origin_jref - j_lo - 1) * iwidth);
                auto& elem_conn = elems[offset];
                if (ratio > 1)
                {
                    size_t old_size = elem_conn.size();
                    size_t new_size = old_size + ratio - 1;
                    elem_conn.resize(new_size);
                    index_t corner = 2;
                    if (old_size != 4)
                    {
                        index_t ioff = offset % iwidth;
                        index_t joff = offset / iwidth;
                        index_t target = (iwidth+1)*(joff+1) + ioff + 1;
                        for (index_t nr = 3; nr < 3+ratio; ++nr)
                        {
                            if (elem_conn[nr] == target) {
                                corner = nr;
                                break;
                            }
                        }
                    }
                    for (index_t nr = new_size-1; nr > corner+ratio-1; --nr)
                    {
                        elem_conn[nr] = elem_conn[nr-ratio+1];
                    }
                    for (index_t nr = corner+ratio-1; nr > corner; --nr) {
                        elem_conn[nr] = new_vertex;
                        ++new_vertex;
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::connectivity --
//-----------------------------------------------------------------------------

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
void
coordset::logical_dims(const conduit::Node &n, index_t *d, index_t maxdims)
{
    for(index_t i = 0; i < maxdims; i++)
    {
        d[i] = 1;
    }

    auto info = get_coordset_info(n);
    const std::string cset_type = n["type"].as_string();
    const std::vector<std::string> &cset_axes = info.second;
    if(cset_type == "uniform" || cset_type == "rectilinear")
    {
        const index_t dim = ((index_t)cset_axes.size() > maxdims) ? maxdims
            : (index_t)cset_axes.size();
        for(index_t i = 0; i < dim; i++)
        {
            if(cset_type == "uniform")
            {
                d[i] = n["dims"][LOGICAL_AXES[i]].to_index_t();
            }
            else // if(cset_type == "rectilinear")
            {
                d[i] = n["values"][cset_axes[i]].dtype().number_of_elements();
            }
        }
    }
    else // if(cset_type == "explicit")
    {
        d[0] = n["values"][cset_axes[0]].dtype().number_of_elements();
    }
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
                coordset->fetch_existing("dims")[LOGICAL_AXES[i]].to_index_t() :
                coordset->fetch_existing("values")[csys_axes[i]].dtype().number_of_elements()) - 1;
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
    else if(type == "points")
    {
        const Node *coordset = find_reference_node(n, "coordset");
        if(coordset)
        {
            coordset::logical_dims(*coordset, d, maxdims);
        }
        else
        {
            CONDUIT_ERROR("Unable to find reference node 'coordset'.");
        }
    }
    else // if(type == "unstructured")
    {
        // calc total number of elements

        // polygonal and polyhedral, or otherwise explicit "sizes"
        if(n["elements"].has_child("sizes"))
        {
            d[0] = n["elements/sizes"].dtype().number_of_elements();
        }
        else if( n["elements"].has_child("element_index") )  // stream style (deprecated)
        {
            const Node &elems_idx =  n["elements/element_index"];
            if(elems_idx.has_child("element_counts"))
            {
                index_t_accessor elem_counts_vals  = elems_idx["element_counts"].value();
                index_t total_elems = 0;
                for(index_t i=0;i< elem_counts_vals.number_of_elements();i++)
                {
                    total_elems += elem_counts_vals[i];
                }
                d[0] = total_elems;
            } // else, use size of stream_ids
            else if(elems_idx.has_child("stream_ids"))
            {
                d[0] = elems_idx["stream_ids"].dtype().number_of_elements();
            }
            else
            {
                CONDUIT_ERROR("invalid stream id topology: "
                              "missing elements/element_index/stream_ids");
            }

        }
        else // zoo
        {
            index_t conn_size = n["elements/connectivity"].dtype().number_of_elements();
            std::string shape_type_name = n["elements/shape"].as_string();
            // total number of elements == conn array size / shape size
            ShapeType shape_info(shape_type_name);
            d[0] = conn_size / shape_info.indices;
        }
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
const Node &
topology::coordset(const Node &n)
{
    const Node *res = find_reference_node(n,"coordset");
    return *res;
}

//-----------------------------------------------------------------------------
void
topology::reindex_coords(const Node& topo,
                         const Node& new_coordset,
                         const Node& old_gvids,
                         const Node& new_gvids,
                         Node& out_topo)
{
    if (&out_topo != &topo)
    {
        out_topo.reset();
        out_topo.set(topo);
    }

    // Build a mapping of global vids -> new coordset indices
    // TODO: if we assume clustering of gvids, maybe a vector would be faster?
    std::unordered_map<index_t, index_t> remap_vids;

    index_t_accessor new_gvid_vals = new_gvids["values"].as_index_t_accessor();
    for (index_t idx = 0; idx < new_gvid_vals.number_of_elements(); idx++)
    {
        remap_vids[new_gvid_vals[idx]] = idx;
    }

    std::string node_path = "elements/connectivity";
    if (out_topo["elements/shape"].as_string() == "polyhedral")
    {
        node_path = "subelements/connectivity";
    }

    index_t_accessor old_vids = out_topo[node_path].as_index_t_accessor();
    index_t_accessor old_to_gvids = old_gvids["values"].as_index_t_accessor();
    std::vector<index_t> new_vids(old_vids.number_of_elements());
    for (index_t idx = 0; idx < static_cast<index_t>(new_vids.size()); idx++)
    {
        index_t old_vid = old_vids[idx];
        index_t gvid = old_to_gvids[old_vid];
        new_vids[idx] = remap_vids[gvid];
    }

    // Set the new vertex connectivity
    out_topo[node_path].set(new_vids);

    // Set the new associated coordset name
    out_topo["coordset"] = new_coordset.name();
}

//-----------------------------------------------------------------------------
void
topology::unstructured::generate_offsets_inline(Node &topo)
{
    // check for polyhedral case
    if(topo.has_child("subelements"))
    {
        // if ele or subelee offsets are missing or empty we want to generate
        if( (!topo["elements"].has_child("offsets") || 
             topo["elements/offsets"].dtype().is_empty()) ||
           (!topo["subelements"].has_child("offsets") || 
             topo["subelements/offsets"].dtype().is_empty())
           )
        {
            blueprint::mesh::utils::topology::unstructured::generate_offsets(topo,
                                                                             topo["elements/offsets"],
                                                                             topo["subelements/offsets"]);
        }

    }
    else
    {
        // if ele offsets is missing or empty we want to generate
        if( !topo["elements"].has_child("offsets") || 
            topo["elements/offsets"].dtype().is_empty())
        {
            blueprint::mesh::utils::topology::unstructured::generate_offsets(topo,
                                                                             topo["elements/offsets"]);
        }
    }
}


//-----------------------------------------------------------------------------
void
topology::unstructured::generate_offsets(const Node &topo,
                                         Node &ele_offsets)
{
    Node subele_offsets;
    generate_offsets(topo,ele_offsets,subele_offsets);
}

//-----------------------------------------------------------------------------
void
topology::unstructured::generate_offsets(const Node &topo,
                                         Node &dest_ele_offsets,
                                         Node &dest_subele_offsets)
{
    const ShapeType topo_shape(topo);
    const DataType int_dtype = find_widest_dtype(topo, DEFAULT_INT_DTYPES);
    std::string key("elements/connectivity"), stream_key("elements/stream");

    if(!topo.has_path(key))
        key = stream_key;
    const Node &topo_conn = topo[key];

    const DataType topo_dtype(topo_conn.dtype().id(), 1, 0, 0,
        topo_conn.dtype().element_bytes(), topo_conn.dtype().endianness());

    bool elem_offsets_exist = topo["elements"].has_child("offsets") &&
                              !topo["elements/offsets"].dtype().is_empty();
    bool subelem_offsets_exist = false;

    // if these have already been generate, use set external to copy out results
    if(topo_shape.type == "polyhedral")
    {
        subelem_offsets_exist = topo["subelements"].has_child("offsets") &&
                                !topo["subelements/offsets"].dtype().is_empty();
        if(elem_offsets_exist && subelem_offsets_exist)
        {
            // they are both already here, set external and return
            if(&dest_ele_offsets != &topo["elements/offsets"])
            {
                dest_ele_offsets.set_external(topo["elements/offsets"]);
            }
            if(&dest_subele_offsets != &topo["subelements/offsets"])
            {
                dest_subele_offsets.set_external(topo["subelements/offsets"]);
            }
            // we are done
            return;
        }
    }
    else // non polyhedral
    {
        if(elem_offsets_exist)
        {
            // they are already here, set external and return
            if(&dest_ele_offsets != &topo["elements/offsets"])
            {
                dest_ele_offsets.set_external(topo["elements/offsets"]);
            }
            return;
        }
    }

    // Selectively reset now that early returns have happened. We do the
    // checks for the polyhedral case since some offsets might exist.
    if(!elem_offsets_exist)
        dest_ele_offsets.reset();
    if(!subelem_offsets_exist)
        dest_subele_offsets.reset();

    ///
    /// Generate Cases
    ///
    if(topo.has_path(stream_key))
    {
        ///
        /// TODO STREAM TOPOS ARE DEPRECATED
        ///
        // Mixed element types
        std::map<int,int> stream_id_npts;
        const conduit::Node &n_element_types = topo["elements/element_types"];
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

        const Node &n_stream_ids = topo["elements/element_index/stream_ids"];
        std::vector<index_t> offsets;
        if(topo.has_path("elements/element_index/element_counts"))
        {
            const Node &n_element_counts = topo["elements/element_index/element_counts"];

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
        else if(topo.has_path("elements/element_index/offsets"))
        {
            const Node &n_stream = topo["elements/stream"];
            const Node &n_element_offsets = topo["elements/element_index/offsets"];
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
        off_node.to_data_type(int_dtype.id(), dest_ele_offsets);
    }
    else if(!topo_shape.is_poly())
    {
        // Single element type
        const index_t num_topo_shapes =
            topo_conn.dtype().number_of_elements() / topo_shape.indices;

        Node shape_node(DataType::int64(num_topo_shapes));
        int64_array shape_array = shape_node.as_int64_array();
        for(index_t s = 0; s < num_topo_shapes; s++)
        {
            shape_array[s] = s * topo_shape.indices;
        }
        shape_node.to_data_type(int_dtype.id(), dest_ele_offsets);
    }
    else if(topo_shape.type == "polygonal")
    {
        const Node &topo_size = topo["elements/sizes"];
        int64_accessor topo_sizes = topo_size.as_int64_accessor();
        std::vector<int64> shape_array;
        index_t i = 0;
        index_t s = 0;
        while(i < topo_size.dtype().number_of_elements())
        {
            shape_array.push_back(s);
            s += topo_sizes[i];
            i++;
        }

        Node shape_node;
        shape_node.set_external(shape_array);
        shape_node.to_data_type(int_dtype.id(), dest_ele_offsets);
    }
    else if(topo_shape.type == "polyhedral")
    {
        // Construct any offsets that do not exist.
        if(!elem_offsets_exist)
        {
            index_t_accessor topo_elem_size = topo["elements/sizes"].value();
            index_t es_count = topo_elem_size.number_of_elements();

            dest_ele_offsets.set(DataType::index_t(es_count));
            index_t_array shape_array = dest_ele_offsets.value();

            index_t es = 0;
            for (index_t ei = 0; ei < es_count; ++ei)
            {
                shape_array[ei] = es;
                es += topo_elem_size[ei];
            }
        }
        if(!subelem_offsets_exist)
        {
            index_t_accessor topo_subelem_size = topo["subelements/sizes"].value();
            index_t ses_count = topo_subelem_size.number_of_elements();

            dest_subele_offsets.set(DataType::index_t(ses_count));
            index_t_array subshape_array = dest_subele_offsets.value();

            index_t ses = 0;
            for (index_t ei = 0; ei < ses_count; ++ei)
            {
                subshape_array[ei] = ses;
                ses += topo_subelem_size[ei];
            }
        }
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
    generate_offsets_inline(ntemp);

    const ShapeType topo_shape(ntemp);

    std::set<index_t> pidxs;
    if(!topo_shape.is_poly())
    {
        index_t_accessor poff_vals = ntemp["elements/offsets"].value();
        const index_t eoff = poff_vals[ei];

        index_t_accessor pidxs_vals = ntemp["elements/connectivity"].value();
        for(index_t pi = 0; pi < topo_shape.indices; pi++)
        {
            pidxs.insert(pidxs_vals[eoff + pi]);
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

            index_t_accessor eidxs_vals = ntemp["elements/connectivity"].value();
            o2mrelation::O2MIterator eiter(ntemp["elements"]);
            eiter.to(ei, o2mrelation::ONE);
            eiter.to_front(o2mrelation::MANY);
            while(eiter.has_next(o2mrelation::MANY))
            {
                eiter.next(o2mrelation::MANY);
                const index_t tmp = eidxs_vals[eiter.index(o2mrelation::DATA)];
                eidxs.insert(tmp);
            }
        }

        index_t_accessor pidxs_vals = enode["connectivity"].value();
        o2mrelation::O2MIterator piter(enode);
        for(const index_t eidx : eidxs)
        {
            piter.to(eidx, o2mrelation::ONE);
            piter.to_front(o2mrelation::MANY);
            while(piter.has_next(o2mrelation::MANY))
            {
                piter.next(o2mrelation::MANY);
                const index_t tmp = pidxs_vals[piter.index(o2mrelation::DATA)];
                pidxs.insert(tmp);
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

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::query --
//-----------------------------------------------------------------------------
namespace query
{

const int NullPointQuery::NotFound = -1;

//---------------------------------------------------------------------------
NullPointQuery::NullPointQuery(const conduit::Node &mesh) : m_mesh(mesh),
    m_domInputs(), m_domResults()
{
}

//---------------------------------------------------------------------------
void
NullPointQuery::Reset()
{
    m_domInputs.clear();
    m_domResults.clear();
}

//---------------------------------------------------------------------------
void
NullPointQuery::Add(int dom, const double pt[3])
{
    std::vector<double> &coords = m_domInputs[dom];
    coords.push_back(pt[0]);
    coords.push_back(pt[1]);
    coords.push_back(pt[2]);
}

//---------------------------------------------------------------------------
const std::vector<double> &
NullPointQuery::Inputs(int dom) const
{
    auto it = m_domInputs.find(dom);
    if(it == m_domInputs.end())
    {
        CONDUIT_ERROR("Domain " << dom << " inputs were requested but not found.");
    }
    return it->second;
}

//---------------------------------------------------------------------------
const std::vector<int> &
NullPointQuery::Results(int dom) const
{
    auto it = m_domResults.find(dom);
    if(it == m_domResults.end())
    {
        CONDUIT_ERROR("Domain " << dom << " results were requested but not found.");
    }
    return it->second;
}

//---------------------------------------------------------------------------
void
NullPointQuery::Execute(const std::string & /*coordsetName*/)
{
    for(auto it = m_domInputs.begin(); it != m_domInputs.end(); it++)
    {
        int npts = it->second.size() / 3;
        std::vector<int> &result = m_domResults[it->first];
        result.resize(npts, 0); // success is anything != -1
    }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
PointQuery::PointQuery(const conduit::Node &mesh) : NullPointQuery(mesh)
{
}

//---------------------------------------------------------------------------
void
PointQuery::Execute(const std::string &coordsetName)
{
    CONDUIT_ANNOTATE_MARK_FUNCTION;

    for(auto it = m_domInputs.begin(); it != m_domInputs.end(); it++)
    {
        const conduit::Node *dom = GetDomain(it->first);
        if(dom == nullptr)
        {
            CONDUIT_ERROR("Domain " << it->first << " was requested but not found.");
        }

        const std::vector<double> &input = it->second;
        std::vector<int> &result = m_domResults[it->first];
        FindPointsInDomain(*dom, coordsetName, input, result);
    }
}

//---------------------------------------------------------------------------
const conduit::Node *
PointQuery::GetDomain(int dom)
{
    std::vector<const conduit::Node *> doms = domains(m_mesh);
    for(const auto d : doms)
    {
       int domain_id = 0;
       if(d->has_path("state/domain_id"))
       {
           int domain_id = d->fetch_existing("state/domain_id").value();
           if(domain_id == dom)
               return d;
       }
    }

    return nullptr;
}

//---------------------------------------------------------------------------
void
PointQuery::FindPointsInDomain(const conduit::Node &mesh,
    const std::string &coordsetName,
    const std::vector<double> &input,
    std::vector<int> &result) const
{
    conduit::index_t numInputPts = input.size() / 3;
    result.resize(numInputPts, 0);

    const conduit::Node &cset = mesh.fetch_existing("coordsets/" + coordsetName);
    const conduit::Node *coords[3] = {nullptr, nullptr, nullptr};
    int ci = 0;
    for(const std::string &axis : coordset::axes(cset))
    {
        coords[ci++] = cset.fetch_ptr(axis);
        ci++;
    }
    conduit::index_t numCoordsetPts = coords[0]->dtype().number_of_elements();
    const double *input_ptr = &input[0];

    using policy = conduit::execution::OpenMPExec;
    conduit::execution::for_all<policy>(0, numInputPts, [&](conduit::index_t i)
    {
        constexpr double EPS = 1.e-10;
        auto ptIdx = i * 3;
        int found = -1; // not found
        for(conduit::index_t pi = 0; pi < numCoordsetPts; pi++)
        {
            double p[3];
            for(int ii = 0; ii < ci; ii++)
                p[ii] = coords[ii] ? coords[ii]->as_double_accessor()[pi] : 0.;

            double dx = p[0] - input_ptr[ptIdx];
            double dy = p[1] - input_ptr[ptIdx + 1];
            double dz = p[2] - input_ptr[ptIdx + 2];
            double dSquared = dx * dx + dy * dy + dz * dz;
            if(dSquared < EPS)
            {
                found = static_cast<int>(pi);
                break;
            }
        }
        result[i] = found;
    });
}

//---------------------------------------------------------------------------
std::vector<int>
PointQuery::DomainIds() const
{
    std::vector<const conduit::Node *> doms = domains(m_mesh);
    std::vector<int> domainIds;
    domainIds.reserve(doms.size());
    for(const auto d : doms)
    {
       int domain_id = 0;
       if(d->has_path("state/domain_id"))
           domain_id = d->fetch_existing("state/domain_id").value();
       domainIds.push_back(domain_id); 
    }

    return domainIds;
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::query --
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

