// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_utils_iterate_elements.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_UTILS_ITERATE_ELEMENTS_HPP
#define CONDUIT_BLUEPRINT_MESH_UTILS_ITERATE_ELEMENTS_HPP

// Internal utility header

//-----------------------------------------------------------------------------
// std includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <utility>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_mesh_utils.hpp"

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
// -- begin conduit::blueprint::mesh::utils::topology --
//-----------------------------------------------------------------------------
namespace topology 
{

struct entity
{
    ShapeType                  shape;
    std::vector<index_t>              element_ids;
    std::vector<std::vector<index_t>> subelement_ids;
    index_t                           entity_id; // Local entity id.
};

// Q: Should this exist in conduit_blueprint_mesh_utils.hpp ?
// static const std::vector<std::string> TOPO_SHAPES = {"point", "line", "tri", "quad", "tet", "hex", "polygonal", "polyhedral"};
enum class ShapeId : index_t
{
    Point     = 0,
    Line       = 1,
    Tri        = 2,
    Quad       = 3,
    Tet        = 4,
    Hex        = 5,
    Polygonal  = 6,
    Polyhedral = 7
};

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::utils::topology::impl --
//-----------------------------------------------------------------------------
namespace impl
{

using id_elem_pair =  std::pair<index_t, entity>;

//-----------------------------------------------------------------------------
// NOTE: Not a template, could go in an actual implementation file.
inline void
build_element_vector(const Node &element_types, std::vector<id_elem_pair> &eles)
{
    /*
        element_types: 
        (String name): (Q: Could this also be a list entry?) 
            stream_id: (index_t id)
            shape: (String name of shape)
    */
    eles.clear();
    auto itr = element_types.children();
    while(itr.has_next())
    {
        const Node &n = itr.next();
        const index_t id = n["stream_id"].to_index_t();
        const ShapeType shape(n["shape"].as_string());
        eles.push_back({{id}, {}});
        eles.back().second.shape = shape;
        if(!shape.is_poly())
        {
            eles.back().second.element_ids.resize(shape.indices);
        }
        else
        {
            CONDUIT_ERROR("I cannot handle a stream of polygonal/polyhedral elements!");
            return;
        }
    }
}

//-----------------------------------------------------------------------------
// NOTE: Not a template, could go in an actual implementation file.
inline int
determine_case_number(const Node &topo)
{
/*
Multiple topology formats
0. Single shape topology - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      shape: (String name of shape)
      connectivity: (Integer array of vertex ids)
1. Mixed shape topology 1 - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      -
        shape: (String name of shape)
        connectivity: (Integer array of vertex ids)
2. Mixed shape topology 2 - Fixed celltype, polygonal celltype, polyhedral celltype
  topo:
    coordset: (String name of coordset)
    type: unstructured
    elements:
      (String name):
        shape: (String name of shape)
        connectivity: (Integer array of vertex ids)
3. Stream based toplogy 1
  mesh: 
    type: "unstructured"
    coordset: "coords"
    elements: 
      element_types: 
        (String name): (Q: Could this also be a list entry?) 
          stream_id: (index_t id)
          shape: (String name of shape)
      element_index: 
        stream_ids: (index_t array of stream ids, must be one of the ids listed in element_types)
        element_counts: (index_t array of element counts at a given index (IE 2 would mean 2 of the associated stream ID))
      stream: (index_t array of vertex ids)
4. Stream based topology 2
  topo: 
    type: unstructured
    coordset: (String name of coordset)
    elements: 
      element_types: 
        (String name): (Q: Could this also be a list entry?)
          stream_id: (index_t id)
          shape: (String name of shape)
      element_index: 
        stream_ids: (index_t array of stream ids, must be one of the ids listed in element_types)
        offsets: (index_t array of offsets into stream for each element)
      stream: (index_t array of vertex ids)
*/
    const Node *shape = topo.fetch_ptr("elements/shape");
    int case_num = -1;
    if(shape)
    {
        // This is a single shape topology
        const ShapeType st(shape->as_string());
        if(!st.is_valid())
        {
            CONDUIT_ERROR("Invalid topology passed to iterate_elements.");
        }
        else
        {
            case_num = 0;
        }
    }
    else
    {
        const Node *elements = topo.fetch_ptr("elements");
        if(!elements)
        {
            CONDUIT_ERROR("Invalid topology passed to iterate elements, no \"elements\" node.");
        }

        const Node *etypes = elements->fetch_ptr("element_types");
        const Node *eindex = elements->fetch_ptr("element_index");
        const Node *estream = elements->fetch_ptr("stream");
        if(!etypes && !eindex && !estream)
        {
            // Not a stream based toplogy, either a list or object of element buckets
            if(elements->dtype().is_list())
            {
                case_num = 1;
            }
            else if(elements->dtype().is_object())
            {
                case_num = 2;
            }
        }
        else if(etypes && eindex && estream)
        {
            // Stream based topology, either offsets or counts
            const Node *eoffsets = eindex->fetch_ptr("offsets");
            const Node *ecounts  = eindex->fetch_ptr("element_counts");
            if(ecounts)
            {
                case_num = 3;
            }
            else if(eoffsets)
            {
                case_num = 4;
            }
        }
    }
    return case_num;
}

//-----------------------------------------------------------------------------
template<typename FuncType>
inline void
traverse_fixed_elements(FuncType &&func, const Node &eles, const ShapeType &shape,
                        index_t &ent_id)
{
    // Single celltype
    entity e;
    e.shape = shape;
    const auto ent_size = e.shape.indices;
    e.element_ids.resize(ent_size, 0);

    const Node &conn = eles["connectivity"];
    const auto &conn_dtype = conn.dtype();
    const auto &id_dtype = DataType(conn_dtype.id(), 1);
    const index_t nents = conn_dtype.number_of_elements() / ent_size;
    Node temp;
    index_t ei = 0;
    for(index_t i = 0; i < nents; i++)
    {
        e.entity_id = ent_id;
        for(index_t j = 0; j < ent_size; j++)
        {
            // Pull out vertex id at ei then cast to index_t
            temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
            e.element_ids[j] = temp.to_index_t();
            ei++;
        }

        func(e);
        ent_id++;
    }
}

//-----------------------------------------------------------------------------
template<typename FuncType>
inline void
traverse_polygonal_elements(FuncType &&func, const Node &elements,
                            index_t &ent_id)
{
    entity e;
    e.shape = ShapeType((index_t)ShapeId::Polygonal);
    const Node &conn = elements["connectivity"];
    const Node &sizes = elements["sizes"];
    const auto &sizes_dtype = sizes.dtype();
    const DataType size_dtype(sizes_dtype.id(), 1);
    const DataType id_dtype(sizes.dtype().id(), 1);
    const index_t nents = sizes_dtype.number_of_elements();
    Node temp;
    index_t ei = 0;
    for(index_t i = 0; i < nents; i++)
    {
        e.entity_id = ent_id;
        temp.set_external(size_dtype, const_cast<void*>(sizes.element_ptr(i)));
        const index_t sz = temp.to_index_t();
        e.element_ids.resize(sz);
        for(index_t j = 0; j < sz; j++)
        {
            // Pull out vertex id at ei then cast to index_t
            temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
            e.element_ids[j] = temp.to_index_t();
            ei++;
        }

        func(e);
        ent_id++;
    }
}

//-----------------------------------------------------------------------------
template<typename FuncType>
inline void
traverse_polyhedral_elements(FuncType &&func, const Node &elements,
                             const Node &subelements, index_t &ent_id)
{
    entity e;
    e.shape = ShapeType((index_t)ShapeId::Polyhedral);
    const Node &conn = elements["connectivity"];
    const Node &sizes = elements["sizes"];
    const Node &subconn = subelements["connectivity"];
    const Node &subsizes = subelements["sizes"];
    const Node &suboffsets = subelements["offsets"];
    const auto &sizes_dtype = sizes.dtype();
    const DataType size_dtype(sizes_dtype.id(), 1);
    const DataType id_dtype(sizes.dtype().id(), 1);
    const DataType subid_dtype(subconn.dtype().id(), 1);
    const DataType suboff_dtype(suboffsets.dtype().id(), 1);
    const DataType subsize_dtype(subsizes.dtype().id(), 1);
    const index_t nents = sizes_dtype.number_of_elements();
    Node temp;
    index_t ei = 0;
    for(index_t i = 0; i < nents; i++)
    {
        e.entity_id = ent_id;
        temp.set_external(size_dtype, const_cast<void*>(sizes.element_ptr(i)));
        const index_t sz = temp.to_index_t();
        e.element_ids.resize(sz);
        for(index_t j = 0; j < sz; j++)
        {
            // Pull out vertex id at ei then cast to index_t
            temp.set_external(id_dtype, const_cast<void*>(conn.element_ptr(ei)));
            e.element_ids[j] = temp.to_index_t();
            ei++;
        }

        e.subelement_ids.resize(sz);
        for(index_t j = 0; j < sz; j++)
        {
            // Get the size of the subelement so we can define it in the proper index of subelement_ids
            auto &subele = e.subelement_ids[j];
            temp.set_external(subsize_dtype, const_cast<void*>(subsizes.element_ptr(e.element_ids[j])));
            const index_t subsz = temp.to_index_t();
            subele.resize(subsz);

            // Find the offset of the face definition so we can write the vertex ids
            temp.set_external(suboff_dtype, const_cast<void*>(suboffsets.element_ptr(e.element_ids[j])));
            index_t offset = temp.to_index_t();
            for(index_t k = 0; k < subsz; k++)
            {
                temp.set_external(subid_dtype, const_cast<void*>(subconn.element_ptr(offset)));
                subele[k] = temp.to_index_t();
                offset++;
            }
        }

        func(e);
        ent_id++;
    }
}

//-----------------------------------------------------------------------------
template<typename FuncType>
inline void
traverse_stream(FuncType &&func, const Node &elements)
{
    std::vector<id_elem_pair> etypes;
    build_element_vector(elements["element_types"], etypes);
    const Node &eindex = elements["element_index"];
    const Node &stream = elements["stream"];
    const Node &stream_ids = eindex["stream_ids"];
    const Node *stream_offs = eindex.fetch_ptr("offsets");
    const Node *stream_counts = eindex.fetch_ptr("element_counts");
    const index_t nstream = stream_ids.dtype().number_of_elements();
    const DataType sid_dtype(stream_ids.dtype().id(), 1);
    const DataType stream_dtype(stream.dtype().id(), 1);
    index_t ent_id = 0;
    // For count based this number just keeps rising, for offset based it gets overwritten
    //   by what is stored in the offsets node.
    index_t idx = 0;
    Node temp;
    for(index_t i = 0; i < nstream; i++)
    {
        // Determine which shape we are working with
        temp.set_external(sid_dtype, const_cast<void*>(stream_ids.element_ptr(i)));
        const index_t stream_id = temp.to_index_t();
        auto itr = std::find_if(etypes.begin(), etypes.end(), [=](const id_elem_pair &p){
            return p.first == stream_id;
        });
        entity &e = itr->second;

        // Determine how many elements are in this section of the stream
        index_t start = 0, end = 0;
        if(stream_offs)
        {
            const DataType dt(stream_offs->dtype().id(), 1);
            temp.set_external(dt, const_cast<void*>(stream_offs->element_ptr(i)));
            start = temp.to_index_t();
            if(i == nstream - 1)
            {
                end = stream_offs->dtype().number_of_elements();
            }
            else
            {
                temp.set_external(dt, const_cast<void*>(stream_offs->element_ptr(i+1)));
                end = temp.to_index_t();
            }
        }
        else if(stream_counts)
        {
            const DataType dt(stream_counts->dtype().id(), 1);
            temp.set_external(dt, const_cast<void*>(stream_counts->element_ptr(i)));
            start = idx;
            end   = start + (temp.to_index_t() * e.shape.indices);
        }

        // Iterate the elements in this section
        idx = start;
        while(idx < end)
        {
            const index_t sz = e.shape.indices;
            for(index_t j = 0; j < sz; j++)
            {
                temp.set_external(stream_dtype, const_cast<void*>(stream.element_ptr(idx)));
                e.element_ids[j] = temp.to_index_t();
                idx++;
            }
            e.entity_id = ent_id;
            func(e);
            ent_id++;
        }
        idx = end;
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::topology::impl --
//-----------------------------------------------------------------------------

// NOTE: namespace conduit::blueprint::mesh::utils::topology

//-----------------------------------------------------------------------------
template<typename Func>
inline void
iterate_elements(const Node &topo, Func &&func)
{
    // Determine case number, case numbers documented
    //   at the top of determine_case_number
    int case_num = impl::determine_case_number(topo);
    if(case_num < 0)
    {
        CONDUIT_ERROR("Could not figure out the type of toplogy passed to iterate elements.");
        return;
    }

    using impl::id_elem_pair;
    index_t ent_id = 0;
    switch(case_num)
    {
    case 0:
    {
        ShapeType shape(topo);
        if(shape.is_polyhedral())
        {
            impl::traverse_polyhedral_elements(func, topo["elements"], topo["subelements"], ent_id);
        }
        else if(shape.is_polygonal())
        {
            impl::traverse_polygonal_elements(func, topo["elements"], ent_id);
        }
        else // (known celltype case)
        {
            impl::traverse_fixed_elements(func, topo["elements"], shape, ent_id);
        }
        break;
    }
    case 1: /* Fallthrough */
    case 2:
    {
        // Mixed celltype
        const Node &elements = topo["elements"];
        auto ele_itr = elements.children();
        while(ele_itr.has_next())
        {
            const Node &bucket = ele_itr.next();
            const std::string &shape_name = bucket["shape"].as_string();
            utils::ShapeType shape(shape_name);

            if(shape.is_polyhedral())
            {
                // Need to find corresponding subelements
                const std::string bucket_name = bucket.name();
                if(!topo.has_child("subelements"))
                {
                    CONDUIT_ERROR("Invalid toplogy, shape == polygonal but no subelements node present.");
                    return;
                }
                const Node &subelements = topo["subelements"];
                if(!subelements.has_child(bucket_name))
                {
                    CONDUIT_ERROR("Invalid toplogy, shape == polygonal but no matching subelements node present.");
                    return;
                }
                const Node &subbucket = subelements[bucket_name];
                impl::traverse_polyhedral_elements(func, bucket, subbucket, ent_id);
            }
            else if(shape.is_polygonal())
            {
                impl::traverse_polygonal_elements(func, bucket, ent_id);
            }
            else
            {
                impl::traverse_fixed_elements(func, bucket, shape, ent_id);
            }
        }
        break;
    }
    case 3: /* Fallthrough */
    case 4:
    {
        // Stream with element counts or offsets
        impl::traverse_stream(func, topo["elements"]);
        break;
    }
    default:
        CONDUIT_ERROR("Unsupported topology passed to iterate_elements")
        return;
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::utils::topology --
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
// -- end conduit --
//-----------------------------------------------------------------------------

#endif
