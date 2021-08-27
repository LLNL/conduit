// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_partition.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_partition.hpp"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <deque>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <map>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_log.hpp"

// Uncomment to enable some debugging output from partitioner.
//#define CONDUIT_DEBUG_PARTITIONER

// #define DEBUG_POINT_MERGE
#ifndef DEBUG_POINT_MERGE
#define PM_DEBUG_PRINT(stream)
#else
#define PM_DEBUG_PRINT(stream) do { std::cerr << stream; } while(0)
#endif

using index_t=conduit::index_t;
using std::cout;
using std::endl;

extern void grid_ijk_to_id(const index_t *ijk, const index_t *dims, index_t &grid_id);
extern void grid_id_to_ijk(const index_t id, const index_t *dims, index_t *grid_ijk);

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

//-------------------------------------------------------------------------
static index_t
get_index_t(const conduit::Node &n, bool &ok)
{
    ok = true;
    index_t retval = 0;
    if(n.dtype().is_int8())
        retval = static_cast<index_t>(n.as_int8());
    else if(n.dtype().is_int16())
        retval = static_cast<index_t>(n.as_int16());
    else if(n.dtype().is_int32())
        retval = static_cast<index_t>(n.as_int32());
    else if(n.dtype().is_int64())
        retval = static_cast<index_t>(n.as_int64());
    else if(n.dtype().is_uint8())
        retval = static_cast<index_t>(n.as_uint8());
    else if(n.dtype().is_uint16())
        retval = static_cast<index_t>(n.as_uint16());
    else if(n.dtype().is_uint32())
        retval = static_cast<index_t>(n.as_uint32());
    else if(n.dtype().is_uint64())
        retval = static_cast<index_t>(n.as_uint64());
    else
        ok = false;
    return retval;
}

//---------------------------------------------------------------------------
inline void
as_index_t(const conduit::Node &n, conduit::Node &out)
{
#ifdef CONDUIT_INDEX_32
    n.to_unsigned_int_array(out);
#else
    n.to_unsigned_long_array(out);
#endif
}

//---------------------------------------------------------------------------
#ifdef CONDUIT_INDEX_32
inline conduit::unsigned_int_array
as_index_t_array(const conduit::Node &n)
{
    return n.as_unsigned_int_array();
}
#else
inline conduit::unsigned_long_array
as_index_t_array(const conduit::Node &n)
{
    return n.as_unsigned_long_array();
}
#endif

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
const std::string selection::DOMAIN_KEY("domain_id");
const std::string selection::TOPOLOGY_KEY("topology");
const int selection::FREE_DOMAIN_ID = -1;
const int selection::FREE_RANK_ID = -1;

//---------------------------------------------------------------------------
selection::selection() : whole(selection::WHOLE_UNDETERMINED), domain(0),
    topology(), domain_any(false)
{
}

//---------------------------------------------------------------------------
selection::selection(const selection &obj) : whole(obj.whole), domain(obj.domain),
    topology(obj.topology), domain_any(obj.domain_any)
{
}

//---------------------------------------------------------------------------
selection::~selection()
{
}

//---------------------------------------------------------------------------
index_t
selection::length(const conduit::Node &/*n_mesh*/) const
{
    return 0;
}

//---------------------------------------------------------------------------
bool
selection::get_whole(const conduit::Node &n_mesh)
{
    // If we have not yet determined whether the selection appears whole, 
    // do it now.
    if(whole == WHOLE_UNDETERMINED)
        set_whole(determine_is_whole(n_mesh));

    return whole == WHOLE_DETERMINED_TRUE;
}

//---------------------------------------------------------------------------
void
selection::set_whole(bool value)
{
    whole = value ? WHOLE_DETERMINED_TRUE : WHOLE_DETERMINED_FALSE;
}

//---------------------------------------------------------------------------
bool
selection::requires_initial_partition() const
{
    return false;
}

//---------------------------------------------------------------------------
index_t
selection::get_domain() const
{
    return domain;
}

//---------------------------------------------------------------------------
void
selection::set_domain(index_t value)
{
    domain = value;
}

//---------------------------------------------------------------------------
const std::string &
selection::get_topology() const
{
    return topology;
}

//---------------------------------------------------------------------------
void
selection::set_topology(const std::string &value)
{
    topology = value;
}

//---------------------------------------------------------------------------
int
selection::get_destination_rank() const
{
    return FREE_RANK_ID;
}

//---------------------------------------------------------------------------
int
selection::get_destination_domain() const
{
    return FREE_DOMAIN_ID;
}

//---------------------------------------------------------------------------
bool
selection::init(const conduit::Node &n_options)
{
    bool retval = true;

    try
    {
        if(n_options.has_child(DOMAIN_KEY))
        {
            const conduit::Node &n_dk = n_options[DOMAIN_KEY];
            if(n_dk.dtype().is_string())
            {
                if(n_dk.as_string() == "any" && supports_domain_any())
                {
                    domain_any = true;
                    domain = 0;
                }
                else
                {
                    // domain is not allowed to be "any".
                    retval = false;
                }
            }
            else
            {
                bool ok = false;
                index_t tmp = get_index_t(n_dk, ok);
                if(ok)
                    domain = tmp;
            }
        }

        if(n_options.has_child(TOPOLOGY_KEY))
            topology = n_options[TOPOLOGY_KEY].as_string();
    }
    catch(const conduit::Error &/*e*/)
    {
        //e.print();
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
const conduit::Node &
selection::selected_topology(const conduit::Node &n_mesh) const
{
    if(n_mesh.has_child("topologies"))
    {
        const conduit::Node &n_topos = n_mesh["topologies"];
        if(topology.empty())
            return n_topos[0];
        else if(n_topos.has_child(topology))
            return n_topos[topology];
    }

    std::stringstream oss;
    oss << "The input mesh does not contain a topology with name " << topology;
    CONDUIT_ERROR(oss.str());
    throw conduit::Error(oss.str(), __FILE__, __LINE__);
}

//---------------------------------------------------------------------------
bool
selection::get_domain_any() const
{
    return domain_any;
}

//---------------------------------------------------------------------------
bool
selection::supports_domain_any() const
{
    return false;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
/**
 @brief This class represents a logical IJK selection with start and end
        values. Start begins at 0 and End is the size of the mesh (in terms
        of cells) minus 1.

        A cell with 10x10x10 cells would have the following selection to
        select it all: start={0,0,0}, end={9,9,9}. To select a single cell,
        make start the same as end.
*/
class selection_logical : public selection
{
public:
    selection_logical();
    selection_logical(const selection_logical &);
    virtual ~selection_logical();

    static std::string name() { return "logical"; }

    virtual std::shared_ptr<selection> copy() const override;

    // Initializes the selection from a conduit::Node.
    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length(const conduit::Node &/*n_mesh*/) const override
    {
        return cells_for_axis(0) * 
               cells_for_axis(1) * 
               cells_for_axis(2);
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    void get_start(index_t &s0, index_t &s1, index_t &s2) const
    {
        s0 = start[0];
        s1 = start[1];
        s2 = start[2];
    }

    void set_start(index_t s0, index_t s1, index_t s2)
    {
        start[0] = s0;
        start[1] = s1;
        start[2] = s2;
    }

    void get_end(index_t &e0, index_t &e1, index_t &e2) const
    {
        e0 = end[0];
        e1 = end[1];
        e2 = end[2];
    }

    void set_end(index_t e0, index_t e1, index_t e2)
    {
        end[0] = e0;
        end[1] = e1;
        end[2] = e2;
    }

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    void get_vertex_ids(const conduit::Node &n_mesh,
                        std::vector<index_t> &element_ids) const;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    index_t cells_for_axis(int axis) const
    {
        index_t nc = static_cast<index_t>(end[axis] - start[axis] + 1);
        return (axis >= 0 && axis <= 2) ? nc : 0;
    }

    static const std::string START_KEY;
    static const std::string END_KEY;

    index_t start[3];
    index_t end[3];
};

const std::string selection_logical::START_KEY("start");
const std::string selection_logical::END_KEY("end");

//---------------------------------------------------------------------------
selection_logical::selection_logical() : selection()
{
    start[0] = start[1] = start[2] = 0;
    end[0] = end[1] = end[2] = 0;
}

//---------------------------------------------------------------------------
selection_logical::selection_logical(const selection_logical &obj) : selection(obj)
{
    for(int i = 0; i < 3; i++)
    {
        start[i] = obj.start[i];
        end[i] = obj.end[i];
    }
}

//---------------------------------------------------------------------------
selection_logical::~selection_logical()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
selection_logical::copy() const
{
    return std::make_shared<selection_logical>(*this);
}

//---------------------------------------------------------------------------
bool
selection_logical::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(START_KEY) && n_options.has_child(END_KEY))
        {
            conduit::Node n_s, n_e;
            n_options[START_KEY].to_uint64_array(n_s);
            n_options[END_KEY].to_uint64_array(n_e);

            auto s = n_s.as_uint64_array();
            auto e = n_e.as_uint64_array();
            if(s.number_of_elements() == 3 &&  e.number_of_elements() == 3)
            {
                for(int i = 0; i < 3; i++)
                {
                    start[i] = static_cast<index_t>(s[i]);
                    end[i] = static_cast<index_t>(e[i]);
                }
                retval = true;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the logical selection applies to the input mesh.
 @note The method is not const because it clamps the end values.
 */
bool
selection_logical::applicable(const conduit::Node &n_mesh)
{
    bool retval = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        std::string csname(n_topo["coordset"].as_string());
        const conduit::Node &n_coords = n_mesh["coordsets"][csname];

        bool is_uniform = n_coords["type"].as_string() == "uniform";
        bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
        bool is_structured = n_coords["type"].as_string() == "explicit" && 
                             n_topo["type"].as_string() == "structured";
        if(is_uniform || is_rectilinear || is_structured)
        {
            index_t dims[3] = {1,1,1};
            conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);

            // See that the selection starts inside the dimensions.
            if(start[0] < dims[0] && start[1] < dims[1] && start[2] < dims[2])
            {
                // Clamp the selection to the dimensions of the mesh.
                end[0] = std::min(end[0], dims[0]-1);
                end[1] = std::min(end[1], dims[1]-1);
                end[2] = std::min(end[2], dims[2]-1);

                retval = true;
            }
        }
    }
    catch(conduit::Error &)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
bool
selection_logical::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool retval = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t len = conduit::blueprint::mesh::utils::topology::length(n_topo);
        retval = len == length(n_mesh);
    }
    catch(conduit::Error &)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Partitions along the longest axis and returns a vector containing 2
        logical selections.
 */
std::vector<std::shared_ptr<selection> >
selection_logical::partition(const conduit::Node &n_mesh) const
{
    std::vector<std::shared_ptr<selection> > parts;
    if(length(n_mesh) > 1)
    {
        int la = 0;
        if(cells_for_axis(1) > cells_for_axis(la))
            la = 1;
        if(cells_for_axis(2) > cells_for_axis(la))
            la = 2;
        auto n = cells_for_axis(la);

        auto p0 = std::make_shared<selection_logical>();
        auto p1 = std::make_shared<selection_logical>();
        p0->set_whole(false);
        p1->set_whole(false);
        p0->set_domain(domain);
        p1->set_domain(domain);
        p0->set_topology(topology);
        p1->set_topology(topology);

        if(la == 0)
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(start[0]+n/2-1,   end[1],         end[2]);
            p1->set_start(start[0]+n/2,   start[1],       start[2]);
            p1->set_end(end[0],           end[1],         end[2]);
        }
        else if(la == 1)
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(end[0],           start[1]+n/2-1, end[2]);
            p1->set_start(start[0],       start[1]+n/2,   start[2]);
            p1->set_end(end[0],           end[1],         end[2]);
        }
        else
        {
            p0->set_start(start[0],       start[1],       start[2]);
            p0->set_end(end[0],           end[1],         start[2]+n/2-1);
            p1->set_start(start[0],       start[1],       start[2]+n/2);
            p1->set_end(end[0],           end[1],         end[2]);
        }
#if 0
cout << "****\nselection_logical::partition: la=" << la << endl << "\t";
p0->print(cout);
cout << endl << "\t";
p1->print(cout);
cout << endl << "****" << endl;
#endif
        parts.push_back(p0);
        parts.push_back(p1);
    }
    return parts;
}

//---------------------------------------------------------------------------
void
selection_logical::get_vertex_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t dims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);
        index_t ndims = conduit::blueprint::mesh::utils::topology::dims(n_topo);
        dims[0]++;
        dims[1]++;
        if(ndims > 2)
            dims[2]++;

        ids.clear();
        ids.reserve(dims[0] * dims[1] * dims[2]);
        auto mesh_NXNY = dims[0] * dims[1];
        auto mesh_NX   = dims[0];
        index_t n_end[3];
        n_end[0] = end[0] + 1;
        n_end[1] = end[1] + 1;
        n_end[2] = (ndims > 2) ? (end[2] + 1) : start[2];

        for(index_t k = start[2]; k <= n_end[2]; k++)
        for(index_t j = start[1]; j <= n_end[1]; j++)
        for(index_t i = start[0]; i <= n_end[0]; i++)
        {
            index_t id = k*mesh_NXNY + j*mesh_NX + i;
            ids.push_back(id);
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_logical::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t dims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, dims, 3);

        element_ids.clear();
        element_ids.reserve(length(n_mesh));
        auto mesh_CXCY = dims[0] * dims[1];
        auto mesh_CX   = dims[0];
        for(index_t k = start[2]; k <= end[2]; k++)
        for(index_t j = start[1]; j <= end[1]; j++)
        for(index_t i = start[0]; i <= end[0]; i++)
        {
            auto eid = k*mesh_CXCY + j*mesh_CX + i;
            element_ids.push_back(eid);
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_logical::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"start\":[" << start[0] << ", " << start[1] << ", " << start[2] << "],"
       << "\"end\":[" << end[0] << ", " << end[1] << ", " << end[2] << "]"
       << "}";
}

//---------------------------------------------------------------------------
/**
   @brief This selection explicitly defines which cells we're pulling out from
          a mesh, and in which order.
 */
class selection_explicit : public selection
{
public:
    selection_explicit();
    selection_explicit(const selection_explicit &obj);
    virtual ~selection_explicit();

    static std::string name() { return "explicit"; }

    virtual std::shared_ptr<selection> copy() const override;

    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length(const conduit::Node &/*n_mesh*/) const override
    {
        return ids_storage.dtype().number_of_elements();
    }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    const index_t *get_indices() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ids_storage.data_ptr());
    }

    void set_indices(const conduit::Node &value)
    {
        ids_storage.reset();
        as_index_t(value, ids_storage);
    }

    void set_indices(const std::vector<index_t> &value)
    {
        ids_storage.set(value);
    }

    index_t num_indices() const { return ids_storage.dtype().number_of_elements(); }

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    static const std::string ELEMENTS_KEY;
    conduit::Node ids_storage;
};

const std::string selection_explicit::ELEMENTS_KEY("elements");

//---------------------------------------------------------------------------
selection_explicit::selection_explicit() : selection(), ids_storage()
{
}

//---------------------------------------------------------------------------
selection_explicit::selection_explicit(const selection_explicit &obj) :
    selection(obj), ids_storage(obj.ids_storage)
{
}

//---------------------------------------------------------------------------
selection_explicit::~selection_explicit()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
selection_explicit::copy() const
{
    return std::make_shared<selection_explicit>(*this);
}

//---------------------------------------------------------------------------
bool
selection_explicit::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(ELEMENTS_KEY))
        {
            const conduit::Node &n_elem = n_options[ELEMENTS_KEY];
            if(n_elem.dtype().is_number())
            {
                // Convert to the right type for index_t. This may copy
                // but that's ok since we access through ids_storage since
                // we may later partition that array.
                as_index_t(n_elem, ids_storage);
                retval = true;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the explicit selection applies to the input mesh.
 */
bool
selection_explicit::applicable(const conduit::Node &/*n_mesh*/)
{
    return true;
}

//---------------------------------------------------------------------------
bool
selection_explicit::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool whole = false;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto num_elem_in_mesh = topology::length(n_topo);
        auto n = num_indices();
        if(n == num_elem_in_mesh)
        {
            auto indices = get_indices();
            std::set<index_t> unique;
            for(index_t i = 0; i < n; i++)
                unique.insert(indices[i]);
            whole = static_cast<index_t>(unique.size()) == num_elem_in_mesh;
        }
    }
    catch(conduit::Error &)
    {
        whole = false;
    }

    return whole;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_explicit::partition(const conduit::Node &n_mesh) const
{
    // Get the number of elements in the topology.
    index_t num_elem_in_topo = 0;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        num_elem_in_topo = topology::length(n_topo);
    }
    catch(conduit::Error &)
    {
    }

    // Divide the elements into 2 vectors.
    auto n = num_indices();
    auto n_2 = n/2;
    auto indices = get_indices();
    std::vector<index_t> ids0, ids1;
    ids0.reserve(n_2);
    ids1.reserve(n_2);
    for(index_t i = 0; i < n; i++)
    {
        if(indices[i] < num_elem_in_topo)
        {
            if(i < n_2)
                ids0.push_back(indices[i]);
            else
                ids1.push_back(indices[i]);
        }
    }

    // Make partitioned selections.
    auto p0 = std::make_shared<selection_explicit>();
    auto p1 = std::make_shared<selection_explicit>();
    p0->ids_storage.set(ids0);
    p1->ids_storage.set(ids1);
    p0->set_whole(false);
    p1->set_whole(false);
    p0->set_domain(domain);
    p1->set_domain(domain);
    p0->set_topology(topology);
    p1->set_topology(topology);

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_explicit::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto topolen = topology::length(n_topo);
        
        auto n = ids_storage.dtype().number_of_elements();
        auto indices = get_indices();
        element_ids.reserve(n);
        for(index_t i = 0; i < n; i++)
        {
            auto eid = indices[i];
            if(eid < topolen)
                element_ids.push_back(eid);
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_explicit::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"elements\":[";
    auto n = num_indices();
    auto indices = get_indices();
    for(index_t i = 0; i < n; i++)
    {
        if(i > 0)
            os << ", ";
        os << indices[i];
    }
    os << "]}";
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
class selection_ranges : public selection
{
public:
    selection_ranges();
    selection_ranges(const selection_ranges &obj);
    virtual ~selection_ranges();

    static std::string name() { return "ranges"; }

    virtual std::shared_ptr<selection> copy() const override;

    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length(const conduit::Node &/*n_mesh*/) const override;

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    void set_ranges(const conduit::Node &n)
    {
        ranges_storage.reset();
#ifdef CONDUIT_INDEX_32
        n.to_uint32_array(ranges_storage);
#else
        n.to_uint64_array(ranges_storage);
#endif
    }

    const index_t *get_ranges() const
    {
        // Access the converted data as index_t.
        return reinterpret_cast<const index_t *>(ranges_storage.data_ptr());
    }

    index_t num_ranges() const;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;

    static const std::string RANGES_KEY;
    conduit::Node ranges_storage;
};

const std::string selection_ranges::RANGES_KEY("ranges");

//---------------------------------------------------------------------------
selection_ranges::selection_ranges() : selection(), ranges_storage()
{
}

//---------------------------------------------------------------------------
selection_ranges::selection_ranges(const selection_ranges &obj) : selection(obj),
    ranges_storage(obj.ranges_storage)
{
}

//---------------------------------------------------------------------------
selection_ranges::~selection_ranges()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
selection_ranges::copy() const
{
    return std::make_shared<selection_ranges>(*this);
}

//---------------------------------------------------------------------------
bool
selection_ranges::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(RANGES_KEY))
        {
            const conduit::Node &n_ranges = n_options[RANGES_KEY];
            if(n_ranges.dtype().is_number())
            {
                // Convert to the right type for index_t
#ifdef CONDUIT_INDEX_32
                n_ranges.to_uint32_array(ranges_storage);
#else
                n_ranges.to_uint64_array(ranges_storage);
#endif
                retval = (ranges_storage.dtype().number_of_elements() % 2 == 0);
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
selection_ranges::applicable(const conduit::Node &/*n_mesh*/)
{
    return true;
}

//---------------------------------------------------------------------------
index_t
selection_ranges::length(const conduit::Node &/*n_mesh*/) const
{
    index_t ncells = 0;
    const index_t *ranges = get_ranges();
    auto n = num_ranges();
    for(index_t i = 0; i < n; i++)
    {
        ncells += ranges[2*i+1] - ranges[2*i] + 1;
    }
    return ncells;
}

//---------------------------------------------------------------------------
index_t
selection_ranges::num_ranges() const
{
    return ranges_storage.dtype().number_of_elements() / 2;
}

//---------------------------------------------------------------------------
bool
selection_ranges::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool whole = false;
    index_t num_elem_in_topo = 0;
    try
    {
        // Get the selected topology and coordset.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        num_elem_in_topo = topology::length(n_topo);
    }
    catch(conduit::Error &)
    {
        num_elem_in_topo = 0;
    }

    auto n = num_ranges();
    if(n == 1)
    {
        auto upper_bound = std::min(get_ranges()[1], num_elem_in_topo-1);
        whole = get_ranges()[0] == 0 &&
                upper_bound == num_elem_in_topo-1;
    }
    else
    {
        auto indices = get_ranges();
        std::set<index_t> unique;
        for(index_t i = 0; i < n; i++)
        {
            index_t start = indices[2*i];
            index_t end = std::min(indices[2*i+1], num_elem_in_topo-1);
            for(index_t eid = start; eid <= end; eid++)
                unique.insert(eid);
        }
        whole = static_cast<index_t>(unique.size()) == num_elem_in_topo;
    }

    return whole;
}

//---------------------------------------------------------------------------
std::vector<std::shared_ptr<selection> >
selection_ranges::partition(const conduit::Node &n_mesh) const
{
    index_t ncells = length(n_mesh);
    auto ncells_2 = ncells / 2;
    auto n = num_ranges();
    auto ranges = get_ranges();
    index_t count = 0;
    index_t split_index = 0;
    for(index_t i = 0; i < n; i++)
    {
        auto rc = ranges[2*i+1] - ranges[2*i] + 1;
        if(count + rc > ncells_2)
        {
            split_index = i;
            break;
        }
        else
        {
            count += rc;
        }
    }

    std::vector<index_t> r0, r1;
    for(index_t i = 0; i < n; i++)
    {
        if(i < split_index)
        {
            r0.push_back(ranges[2*i+0]);
            r0.push_back(ranges[2*i+1]);
        }
        else if(i == split_index)
        {
            auto rc = (ranges[2*i+1] - ranges[2*i]) + 1;
            if(rc == 1)
            {
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0]);
            }
            else if(rc == 2)
            {
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0]);

                r1.push_back(ranges[2*i+1]);
                r1.push_back(ranges[2*i+1]);
            }
            else
            {
                auto rc_2 = rc / 2;
                r0.push_back(ranges[2*i+0]);
                r0.push_back(ranges[2*i+0] + rc_2);

                r1.push_back(ranges[2*i+0] + rc_2 + 1);
                r1.push_back(ranges[2*i+1]);
            }
        }
        else //if(i > split_index)
        {
            r1.push_back(ranges[2*i+0]);
            r1.push_back(ranges[2*i+1]);
        }
    }

    auto p0 = std::make_shared<selection_ranges>();
    auto p1 = std::make_shared<selection_ranges>();
    p0->ranges_storage.set(r0);
    p1->ranges_storage.set(r1);
    p0->set_whole(false);
    p1->set_whole(false);
    p0->set_domain(domain);
    p1->set_domain(domain);
    p0->set_topology(topology);
    p1->set_topology(topology);

    std::vector<std::shared_ptr<selection> > parts;
    parts.push_back(p0);
    parts.push_back(p1);

    return parts;
}

//---------------------------------------------------------------------------
void
selection_ranges::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    try
    {
        const conduit::Node &n_topo = selected_topology(n_mesh);
        auto topolen = topology::length(n_topo);
        auto n = num_ranges();
        auto indices = get_ranges();
        for(index_t i = 0; i < n; i++)
        {
            index_t start = indices[2*i];
            index_t end = indices[2*i+1];
            for(index_t eid = start; eid <= end; eid++)
            {
                if(eid < topolen)
                    element_ids.push_back(eid);
            }
        }
    }
    catch(conduit::Error &)
    {
    }
}

//---------------------------------------------------------------------------
void
selection_ranges::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"ranges\":[";
    auto n = num_ranges() * 2;
    auto indices = get_ranges();
    for(index_t i = 0; i < n; i++)
    {
        if(i > 0)
            os << ", ";
        os << indices[i];
    }
    os << "]}";
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
/**
 @brief This class represents a field-based selection where the selection
        uses a field that contains domain numbers to split the mesh into
        chunks.
*/
class selection_field : public selection
{
public:
    selection_field();
    selection_field(const selection_field &obj);
    virtual ~selection_field();

    static std::string name() { return "field"; }

    virtual std::shared_ptr<selection> copy() const override;

    // Initializes the selection from a conduit::Node.
    virtual bool init(const conduit::Node &n_options) override;

    virtual bool applicable(const conduit::Node &n_mesh) override;

    // Computes the number of cells in the selection.
    virtual index_t length(const conduit::Node &n_mesh) const override;

    virtual bool requires_initial_partition() const override { return true; }

    virtual std::vector<std::shared_ptr<selection> > partition(const conduit::Node &n_mesh) const override;

    const std::string & get_field() const
    {
        return field;
    }

    void set_field(const std::string &value)
    {
        field = value;
    }

    void get_selected_value(index_t &value, bool &value_set) const
    {
        value = selected_value;
        value_set = selected_value_set;
    }

    void set_selected_value(index_t value)
    {
        selected_value = value;
        selected_value_set = true;
    }

    /**
     @brief Returns the destination rank for this selection. The default
            version returns -1, indicating that we'll number the domain.
     @return The destination domain for this selection.
     */
    virtual int get_destination_domain() const override;

    virtual void get_element_ids(const conduit::Node &n_mesh,
                                 std::vector<index_t> &element_ids) const override;

    virtual void print(std::ostream &os) const override;

protected:
    virtual bool supports_domain_any() const override { return true; }

    virtual bool determine_is_whole(const conduit::Node &n_mesh) const override;
    bool const_applicable(const conduit::Node &n_mesh) const;

    static const std::string FIELD_KEY;

    std::string field;
    index_t     selected_value;
    bool        selected_value_set;
};

const std::string selection_field::FIELD_KEY("field");

//---------------------------------------------------------------------------
selection_field::selection_field() : selection(), field(), selected_value(0),
    selected_value_set(false)
{
}

//---------------------------------------------------------------------------
selection_field::selection_field(const selection_field &obj) : selection(obj),
    field(obj.field), selected_value(obj.selected_value),
    selected_value_set(obj.selected_value_set)
{
}

//---------------------------------------------------------------------------
selection_field::~selection_field()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
selection_field::copy() const
{
    return std::make_shared<selection_field>(*this);
}

//---------------------------------------------------------------------------
bool
selection_field::init(const conduit::Node &n_options)
{
    bool retval = false;
    if(selection::init(n_options))
    {
        if(n_options.has_child(FIELD_KEY))
        {
            field = n_options[FIELD_KEY].as_string();
            retval = true;
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Returns whether the field selection applies to the input mesh.
 */
bool
selection_field::const_applicable(const conduit::Node &n_mesh) const
{
    bool retval = false;
    const conduit::Node &n_fields = n_mesh["fields"];
    if(n_fields.has_child(field))
    {
        const conduit::Node &n_field = n_fields[field];
        if(n_field.has_child("association") && n_field.has_child("topology"))
        {
            const conduit::Node &n_topo = selected_topology(n_mesh);
            if(n_topo.name() != n_field["topology"].as_string())
            {
                CONDUIT_INFO("Incompatible topology used for field selection.");
            }
            else if(n_field["association"].as_string() != "element")
            {
                CONDUIT_INFO("Field " << field
                             << " has incompatible association for field selection.");
            }
            else
            {
                retval = true;
            }
        }
    }

    return retval;
}

//---------------------------------------------------------------------------
bool
selection_field::applicable(const conduit::Node &n_mesh)
{
    return const_applicable(n_mesh);
}

//---------------------------------------------------------------------------
index_t
selection_field::length(const conduit::Node &n_mesh) const
{
    index_t len = 0;
    const conduit::Node &n_fields = n_mesh["fields"];
    if(const_applicable(n_mesh))
    {
        const conduit::Node &n_field = n_fields[field];
        if(selected_value_set)
        {
            // Count number of occurrances of the selected value.
            conduit::Node n_values;
            as_index_t(n_field["values"], n_values);
            auto iarr = as_index_t_array(n_values);
            index_t N = iarr.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                if(static_cast<index_t>(iarr[i]) == selected_value)
                    len++;
            }
        }
        else
        {
            // No single value has been selected yet so use the field as the length.
            len = n_field["values"].dtype().number_of_elements();
        }
    }
    return len;
}

//---------------------------------------------------------------------------
bool
selection_field::determine_is_whole(const conduit::Node &n_mesh) const
{
    bool retval = false;
    try
    {
        // Get the selected topology.
        const conduit::Node &n_topo = selected_topology(n_mesh);
        index_t len = conduit::blueprint::mesh::utils::topology::length(n_topo);
        retval = len == length(n_mesh);
    }
    catch(conduit::Error &)
    {
        retval = false;
    }

    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Partitions the mesh using the specified field.
 */
std::vector<std::shared_ptr<selection> >
selection_field::partition(const conduit::Node &n_mesh) const
{
    std::vector<std::shared_ptr<selection> > parts;

    if(const_applicable(n_mesh))
    {
        if(!selected_value_set)
        {
            const conduit::Node &n_fields = n_mesh["fields"];
            const conduit::Node &n_field = n_fields[field];

            // Iterate through the field to find the unique values.
            conduit::Node n_values;
            as_index_t(n_field["values"], n_values);
            auto iarr = as_index_t_array(n_values);
            index_t N = iarr.number_of_elements();
            std::set<index_t> unique;
            for(index_t i = 0; i < N; i++)
            {
                unique.insert(iarr[i]);
            }

            // Now, make new selection_field objects that reference only one value.
            for(auto it = unique.begin(); it != unique.end(); it++)
            {
                auto p0 = std::make_shared<selection_field>();
                p0->set_whole(false);
                p0->set_domain(domain);
                p0->set_topology(topology);
                p0->set_field(field);
                p0->set_selected_value(*it);
                parts.push_back(p0);
            }
        }
        else
        {
            // Normally a field selection would be telling us exactly which
            // domain we want to use for the selected cells. We may have set
            // a large target though that means we might still be splitting
            // larger field selections. In that case, make them into explicit
            // selections.
            std::vector<index_t> eids, ids0, ids1;
            get_element_ids(n_mesh, eids);
            size_t n2 = eids.size() / 2;
            for(size_t i = 0; i < eids.size(); i++)
            {
                if(i < n2)
                    ids0.push_back(eids[i]);
                else
                    ids1.push_back(eids[i]);
            }

            // Make partitioned selections.
            auto p0 = std::make_shared<selection_explicit>();
            auto p1 = std::make_shared<selection_explicit>();
            p0->set_indices(ids0);
            p1->set_indices(ids1);
            p0->set_whole(false);
            p1->set_whole(false);
            p0->set_domain(domain);
            p1->set_domain(domain);
            p0->set_topology(topology);
            p1->set_topology(topology);

            parts.push_back(p0);
            parts.push_back(p1);
        }
    }

    return parts;
}

//---------------------------------------------------------------------------
int
selection_field::get_destination_domain() const
{
    return selected_value_set ? static_cast<int>(selected_value) : FREE_DOMAIN_ID;
}

//---------------------------------------------------------------------------
void
selection_field::get_element_ids(const conduit::Node &n_mesh,
    std::vector<index_t> &element_ids) const
{
    if(const_applicable(n_mesh))
    {
        const conduit::Node &n_fields = n_mesh["fields"];
        const conduit::Node &n_field = n_fields[field];

        // Iterate through the field to find the unique values.
        conduit::Node n_values;
        as_index_t(n_field["values"], n_values);
        auto iarr = as_index_t_array(n_values);
        index_t N = iarr.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            if(static_cast<index_t>(iarr[i]) == selected_value)
                element_ids.push_back(i);
        }
    }
}

//---------------------------------------------------------------------------
void
selection_field::print(std::ostream &os) const
{
    os << "{"
       << "\"name\":\"" << name() << "\","
       << "\"domain\":" << get_domain() << ", "
       << "\"topology\":\"" << get_topology() << "\", "
       << "\"field\": " << field << ","
       << "\"selected_value\": " << selected_value << ","
       << "\"selected_value_set\": " << selected_value_set
       << "}";
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
partitioner::chunk::chunk() : mesh(nullptr), owns(false), 
    destination_rank(selection::FREE_RANK_ID),
    destination_domain(selection::FREE_DOMAIN_ID)
{
}

//---------------------------------------------------------------------------
partitioner::chunk::chunk(const Node *m, bool own) : mesh(m), owns(own),
    destination_rank(selection::FREE_RANK_ID),
    destination_domain(selection::FREE_DOMAIN_ID)
{
}

//---------------------------------------------------------------------------
partitioner::chunk::chunk(const Node *m, bool own, int dr, int dd) : 
    mesh(m), owns(own), destination_rank(dr), destination_domain(dd)
{
}

//---------------------------------------------------------------------------
void
partitioner::chunk::free()
{
    if(owns)
    {
        Node *m = const_cast<Node *>(mesh);
        delete m;
        mesh = nullptr;
        owns = false;
    }
}

//---------------------------------------------------------------------------
partitioner::partitioner() : rank(0), size(1), target(1), meshes(), selections(),
    selected_fields(), mapping(true), merge_tolerance(1.e-8)
{
}

//---------------------------------------------------------------------------
partitioner::~partitioner()
{
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
partitioner::create_selection(const std::string &type) const
{
    std::shared_ptr<selection> retval;

    if(type == selection_logical::name())
        retval = std::make_shared<selection_logical>();
    else if(type == selection_explicit::name())
        retval = std::make_shared<selection_explicit>();
    else if(type == selection_ranges::name())
        retval = std::make_shared<selection_ranges>();
    else if(type == selection_field::name())
        retval = std::make_shared<selection_field>();
    else
    {
        CONDUIT_ERROR("Unknown selection type: " << type);
    }
    return retval;
}

//---------------------------------------------------------------------------
std::shared_ptr<selection>
partitioner::create_selection_all_elements(const conduit::Node &n_mesh) const
{
    std::shared_ptr<selection> retval;

    // We're making a selection that includes all elements for the mesh.
    // Take the first topology and its coordset.
    const conduit::Node &n_topo = n_mesh["topologies"][0];
    std::string csname(n_topo["coordset"].as_string());
    const conduit::Node &n_coords = n_mesh["coordsets"][csname];

    // Does the topo+coordset combo look structured?
    bool is_uniform = n_coords["type"].as_string() == "uniform";
    bool is_rectilinear = n_coords["type"].as_string() == "rectilinear";
    bool is_structured = n_coords["type"].as_string() == "explicit" && 
                         n_topo["type"].as_string() == "structured";

    if(is_uniform || is_rectilinear || is_structured)
    {
        index_t edims[3] = {1,1,1};
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, edims, 3);
        retval = create_selection(selection_logical::name());
        retval->set_whole(true);
        retval->set_topology(n_topo.name());
        auto typed_sel = dynamic_cast<selection_logical *>(retval.get());
        if(typed_sel != nullptr)
        {
            typed_sel->set_end(edims[0] > 0 ? edims[0]-1 : 0,
                               edims[1] > 0 ? edims[1]-1 : 0,
                               edims[2] > 0 ? edims[2]-1 : 0);
        }
    }
    else
    {
        index_t nelem = topology::length(n_topo);

        // Create a range that selects the topology.
        retval = create_selection(selection_ranges::name());
        retval->set_whole(true);
        retval->set_topology(n_topo.name());
        auto typed_sel = dynamic_cast<selection_ranges *>(retval.get());
        if(typed_sel != nullptr)
        {
#ifdef CONDUIT_INDEX_32
            std::vector<uint32> range;
            range.push_back(0);
            range.push_back(static_cast<uint32>(nelem - 1));
            conduit::Node n_range;
            n_range.set(range);
            typed_sel->set_ranges(n_range);
#else
            std::vector<uint64> range;
            range.push_back(0);
            range.push_back(static_cast<uint64>(nelem - 1));
            conduit::Node n_range;
            n_range.set(range);
            typed_sel->set_ranges(n_range);
#endif
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
partitioner::options_get_target(const conduit::Node &options, unsigned int &value) const
{
    bool retval = false;
    value = 0;
    if(options.has_child("target"))
    {
        const conduit::Node &n_target = options["target"];
        if(n_target.dtype().is_number())
        {
            // For signed int types, make the min allowable value be 0.
            if(n_target.dtype().is_int8())
            {
                auto v = n_target.as_int8();
                value = static_cast<unsigned int>((v < 0) ? 0 : v);
            }
            else if(n_target.dtype().is_int16())
            {
                auto v = n_target.as_int16();
                value = static_cast<unsigned int>((v < 0) ? 0 : v);
            }
            else if(n_target.dtype().is_int32())
            {
                auto v = n_target.as_int32();
                value = static_cast<unsigned int>((v < 0) ? 0 : v);
            }
            else if(n_target.dtype().is_int64())
            {
                auto v = n_target.as_int64();
                value = static_cast<unsigned int>((v < 0) ? 0 : v);
            }
            else
            {
                value = n_target.to_unsigned_int();
            }
            retval = true;
        }
        else
        {
            CONDUIT_INFO("Nonnumber passed as selection target.");
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
bool
partitioner::initialize(const conduit::Node &n_mesh, const conduit::Node &options)
{
    auto doms = conduit::blueprint::mesh::domains(n_mesh);

    // Iterate over the selections in the options and check them against the
    // domains that were passed in to make a vector of meshes and selections
    // that can be used to partition the meshes.
    selections.clear();
    if(options.has_child("selections"))
    {
        const conduit::Node &n_selections = options["selections"];
        for(index_t i = 0; i < n_selections.number_of_children(); i++)
        {
            const conduit::Node &n_sel = n_selections[i];
            try
            {
                // It has to have type to be a valid selection.
                if(!n_sel.has_child("type"))
                    continue;

                std::string type(n_sel["type"].as_string());
                auto sel = create_selection(type);
                if(sel != nullptr && sel->init(n_sel))
                {
                    // The selection is good. See if it applies to the domains.
                    auto n = static_cast<index_t>(doms.size());
                    for(index_t di = 0; di < n; di++)
                    {
                        // Get the overall index for this domain if it exists.
                        // Otherwise, we use the position in the list.
                        index_t domid = di;
                        if(doms[di]->has_path("state/domain_id"))
                        {
                            bool ok = false;
                            index_t tmp = get_index_t(doms[di]->operator[]("state/domain_id"), ok);
                            if(ok)
                                domid = tmp;
                        }

                        // NOTE: A selection is tied to a single domain at present.
                        //       The field selection could apply to multiple domains
                        //       if we wanted it to.
                        bool domain_compatible = (domid == sel->get_domain()) ||
                                                 sel->get_domain_any();
                        if(domain_compatible && sel->applicable(*doms[di]))
                        {
                            // We may split some selections upfront here so we
                            // can figure these values into the target.
                            if(sel->requires_initial_partition())
                            {
                                if(sel->get_domain_any())
                                    sel->set_domain(domid);
                                auto ps = sel->partition(*doms[di]);
                                for(size_t j = 0; j < ps.size(); j++)
                                {
                                    meshes.push_back(doms[di]);
                                    selections.push_back(ps[j]);
                                }
                            }
                            else
                            {
                                meshes.push_back(doms[di]);
                                if(sel->get_domain_any())
                                {
                                    auto newsel = sel->copy();
                                    newsel->set_domain(domid);
                                    selections.push_back(newsel);
                                }
                                else
                                    selections.push_back(sel);
                            }

                            // If domain_any is true then we want the selection to
                            // apply to the next domain too so do not break.
                            if(!sel->get_domain_any())
                                break;
                        }
                    }
                }
                else
                {
                    CONDUIT_INFO("Could not initialize selection " << i);
                    return false;
                }
            }
            catch(const conduit::Error &e)
            {
                CONDUIT_INFO("Exception thrown handling selection " << i
                     << ": " << e.message());
                return false;
            }
        }
    }
    else
    {
        // Add selections to indicate that we take the whole domain.
        for(size_t i = 0; i < doms.size(); i++)
        {
            auto sel = create_selection_all_elements(*doms[i]);
            // If the mesh has a domain_id then use that as the domain number.
            index_t domid = i;
            if(doms[i]->has_path("state/domain_id"))
            {
                bool ok = false;
                index_t tmp = get_index_t(doms[i]->operator[]("state/domain_id"), ok);
                if(ok)
                    domid = tmp;
            }
            sel->set_domain(domid);
            selections.push_back(sel);
            meshes.push_back(doms[i]);
        }
    }

    // Get the number of target partitions that we're making. We determine
    // whether the options contain a target in a method so we can override
    // it in parallel.
    unsigned int targetval = 1;
    if(options_get_target(options, targetval))
    {
        target = targetval;
    }
    else
    {
        // We are likely using all domains on this rank and on all ranks,
        // so we need to sum the number of domains to arrive at the target.
        // Or, we did not pass target and had valid selections. Or, we 
        // had an invalid target. In any case, we sum the number of valid
        // selections.
        target = count_targets();
    }

    // Get any fields that we're using to limit the selection.
    if(options.has_child("fields"))
    {
        selected_fields.clear();
        const conduit::Node &n_fields = options["fields"];
        for(index_t i = 0; i < n_fields.number_of_children(); i++)
            selected_fields.push_back(n_fields[i].name());
    }

    // Get whether we want to preserve old numbering of vertices, elements.
    if(options.has_child("mapping"))
        mapping = options["mapping"].to_unsigned_int() != 0;

    // Get whether we want to preserve old numbering of vertices, elements.
    if(options.has_child("merge_tolerance"))
        merge_tolerance = options["merge_tolerance"].to_double();

#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << rank << ": partitioner::initialize" << endl;
    cout << "\ttarget=" << target << endl;
    for(size_t i = 0; i < selections.size(); i++)
    {
        cout << "\t";
        selections[i]->print(cout);
        cout << endl;
    }
#endif

    // If we made it to the end then we will have created any applicable
    // selections. Some ranks may have created no selections. That is ok.
    return true; //!selections.empty();
}

//---------------------------------------------------------------------------
unsigned int
partitioner::count_targets() const
{
    // We make a pass over the selections on this rank and determine the
    // number of selections that produce free domains. These are domains that
    // have FREE_DOMAIN_ID for their destination domain. We also figure out
    // the number of named domains with unique domain ids. These are not
    // double-counted. The number of actual targets is the sum.
    unsigned int free_domains = 0;
    std::set<int> named_domains;
    for(size_t i = 0; i < selections.size(); i++)
    {
        int dd = selections[i]->get_destination_domain();
        if(dd == selection::FREE_DOMAIN_ID)
            free_domains++;
        else
            named_domains.insert(dd);
    }

    unsigned int n = free_domains + static_cast<unsigned int>(named_domains.size());
    return n;
}

//---------------------------------------------------------------------------
void
partitioner::get_largest_selection(int &sel_rank, int &sel_index) const
{
    sel_rank = 0;
    sel_index = -1;
    long largest_selection_size = 0;
    for(size_t i = 0; i < selections.size(); i++)
    {
        long ssize = static_cast<long>(selections[i]->length(*meshes[i]));
        if(ssize > largest_selection_size)
        {
            largest_selection_size = ssize;
            sel_index = static_cast<int>(i);
        }
    }
}

//---------------------------------------------------------------------------
long
partitioner::get_total_selections() const
{
    return static_cast<long>(selections.size());
}

//---------------------------------------------------------------------------
void
partitioner::split_selections()
{
    // Splitting based on target.
#ifdef CONDUIT_DEBUG_PARTITIONER
    int iteration = 1;
#endif
    bool splitting = true;
    while(splitting)
    {
        // Get the total number of targets produced by the current selections.
        auto nt = count_targets();
        if(nt == 0)
            splitting = false;
        else
            splitting = target > nt;

        if(splitting)
        {
            // Get the rank with the largest selection and get that local
            // selection index.
            int sel_rank = -1, sel_index = -1;
            get_largest_selection(sel_rank, sel_index);

            if(rank == sel_rank)
            {
                auto ps = selections[sel_index]->partition(*meshes[sel_index]);

                if(!ps.empty())
                {
                    const conduit::Node *m = meshes[sel_index];
                    meshes.insert(meshes.begin()+sel_index, ps.size()-1, m);
                    selections.insert(selections.begin()+sel_index, ps.size()-1, nullptr);
                    for(size_t i = 0; i < ps.size(); i++)
                        selections[sel_index + i] = ps[i];

#ifdef CONDUIT_DEBUG_PARTITIONER
                    if(rank == 0)
                    {
                        cout << "partitioner::split_selections target=" << target
                             << ", nt=" << nt << " (after split "
                             << iteration << ")" << endl;
                        for(size_t i = 0; i < selections.size(); i++)
                        {
                            cout << "\t";
                            selections[i]->print(cout);
                            cout << endl;
                        }
                    }
                    iteration++;
#endif
                }
            }
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::copy_fields(index_t domain, const std::string &topology,
    const std::vector<index_t> &vertex_ids,
    const std::vector<index_t> &element_ids,
    const conduit::Node &n_mesh,
    conduit::Node &n_output) const
{
    if(n_mesh.has_child("fields"))
    {
        const conduit::Node &n_fields = n_mesh["fields"];
        if(!vertex_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(index_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "vertex")
                    {
                        copy_field(n_field, vertex_ids, n_output_fields);
                    }
                }
            }

            if(mapping)
            {
                // Save the vertex_ids as a new MC field.
                conduit::Node &n_field = n_output_fields["original_vertex_ids"];
                n_field["association"] = "vertex";
                if(!topology.empty())
                    n_field["topology"] = topology;
                std::vector<index_t> domain_ids(vertex_ids.size(), domain);
                conduit::Node &n_values = n_field["values"];
                n_values["domains"].set(domain_ids);
                n_values["ids"].set(vertex_ids);
            }
        }

        if(!element_ids.empty())
        {
            conduit::Node &n_output_fields = n_output["fields"];
            for(index_t i = 0; i < n_fields.number_of_children(); i++)
            {
                const conduit::Node &n_field = n_fields[i];
                if(n_field.has_child("association"))
                {
                    auto association = n_field["association"].as_string();
                    if(association == "element")
                    {
                        copy_field(n_field, element_ids, n_output_fields);
                    }
                }
            }

            if(mapping)
            {
                // Save the element_ids as a new MC field.
                conduit::Node &n_field = n_output_fields["original_element_ids"];
                n_field["association"] = "element";
                if(!topology.empty())
                    n_field["topology"] = topology;
                std::vector<index_t> domain_ids(element_ids.size(), domain);
                conduit::Node &n_values = n_field["values"];
                n_values["domains"].set(domain_ids);
                n_values["ids"].set(element_ids);
            }
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::copy_field(const conduit::Node &n_field,
    const std::vector<index_t> &ids, Node &n_output_fields) const
{
    static const std::vector<std::string> keys{"association", "grid_function",
        "volume_dependent", "topology"};

    // If we're subselecting the list of fields, check whether the field we
    // want to copy is in selected_fields.
    if(!selected_fields.empty() &&
       std::find(selected_fields.begin(), selected_fields.end(), n_field.name()) == selected_fields.end())
    {
        return;
    }

// TODO: What about matsets and mixed fields?...
// https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#fields

    // Copy common field attributes from the old field into the new one.
    conduit::Node &n_new_field = n_output_fields[n_field.name()];
    for(const auto &key : keys)
    {
        if(n_field.has_child(key))
            n_new_field[key] = n_field[key];
    }

    const conduit::Node &n_values = n_field["values"];
    conduit::Node &new_values = n_new_field["values"];
    if(n_values.dtype().is_compact()) 
    {
        if(n_values.number_of_children() > 0)
        {

// The vel data must be interleaved. We need to use the DataArray element methods for access.


            // mcarray.
            for(index_t i = 0; i < n_values.number_of_children(); i++)
            {
                const conduit::Node &n_vals = n_values[i];
                slice_array(n_vals, ids, new_values[n_vals.name()]);
            }
        }
        else
            slice_array(n_values, ids, new_values);
    }
    else
    {
        // otherwise, we need to compact our data first
        conduit::Node n;
        n_values.compact_to(n);
        if(n.number_of_children() > 0)
        {
            // mcarray.
            conduit::Node &new_values = n_new_field["values"];
            for(index_t i = 0; i < n.number_of_children(); i++)
            {
                const conduit::Node &n_vals = n[i];
                slice_array(n_vals, ids, new_values[n_vals.name()]);
            }
        }
        else
            slice_array(n, ids, new_values);
    }
}

//---------------------------------------------------------------------------
// @brief Slice the n_src array using the indices stored in ids. We use the
//        array classes for their [] operators that deal with interleaved
//        and non-interleaved arrays.
template <typename T>
inline void
typed_slice_array(const T &src, const std::vector<index_t> &ids, T &dest)
{
    size_t n = ids.size();
    for(size_t i = 0; i < n; i++)
        dest[i] = src[ids[i]];
}

//---------------------------------------------------------------------------
// @note Should this be part of conduit::Node or DataArray somehow. The number
//       of times I've had to slice an array...
void
partitioner::slice_array(const conduit::Node &n_src_values,
    const std::vector<index_t> &ids, Node &n_dest_values) const
{
    // Copy the DataType of the input conduit::Node but override the number of elements
    // before copying it in so assigning to n_dest_values triggers a memory
    // allocation.
    auto dt = n_src_values.dtype();
    n_dest_values = DataType(n_src_values.dtype().id(), ids.size());

    // Do the slice.
    if(dt.is_int8())
    {
        auto dest(n_dest_values.as_int8_array());
        typed_slice_array(n_src_values.as_int8_array(), ids, dest);
    }
    else if(dt.is_int16())
    {
        auto dest(n_dest_values.as_int16_array());
        typed_slice_array(n_src_values.as_int16_array(), ids, dest);
    }
    else if(dt.is_int32())
    {
        auto dest(n_dest_values.as_int32_array());
        typed_slice_array(n_src_values.as_int32_array(), ids, dest);
    }
    else if(dt.is_int64())
    {
        auto dest(n_dest_values.as_int64_array());
        typed_slice_array(n_src_values.as_int64_array(), ids, dest);
    }
    else if(dt.is_uint8())
    {
        auto dest(n_dest_values.as_uint8_array());
        typed_slice_array(n_src_values.as_uint8_array(), ids, dest);
    }
    else if(dt.is_uint16())
    {
        auto dest(n_dest_values.as_uint16_array());
        typed_slice_array(n_src_values.as_uint16_array(), ids, dest);
    }
    else if(dt.is_uint32())
    {
        auto dest(n_dest_values.as_uint32_array());
        typed_slice_array(n_src_values.as_uint32_array(), ids, dest);
    }
    else if(dt.is_uint64())
    {
        auto dest(n_dest_values.as_uint64_array());
        typed_slice_array(n_src_values.as_uint64_array(), ids, dest);
    }
    else if(dt.is_char())
    {
        auto dest(n_dest_values.as_char_array());
        typed_slice_array(n_src_values.as_char_array(), ids, dest);
    }
    else if(dt.is_short())
    {
        auto dest(n_dest_values.as_short_array());
        typed_slice_array(n_src_values.as_short_array(), ids, dest);
    }
    else if(dt.is_int())
    {
        auto dest(n_dest_values.as_int_array());
        typed_slice_array(n_src_values.as_int_array(), ids, dest);
    }
    else if(dt.is_long())
    {
        auto dest(n_dest_values.as_long_array());
        typed_slice_array(n_src_values.as_long_array(), ids, dest);
    }
    else if(dt.is_unsigned_char())
    {
        auto dest(n_dest_values.as_unsigned_char_array());
        typed_slice_array(n_src_values.as_unsigned_char_array(), ids, dest);
    }
    else if(dt.is_unsigned_short())
    {
        auto dest(n_dest_values.as_unsigned_short_array());
        typed_slice_array(n_src_values.as_unsigned_short_array(), ids, dest);
    }
    else if(dt.is_unsigned_int())
    {
        auto dest(n_dest_values.as_unsigned_int_array());
        typed_slice_array(n_src_values.as_unsigned_int_array(), ids, dest);
    }
    else if(dt.is_unsigned_long())
    {
        auto dest(n_dest_values.as_unsigned_long_array());
        typed_slice_array(n_src_values.as_unsigned_long_array(), ids, dest);
    }
    else if(dt.is_float())
    {
        auto dest(n_dest_values.as_float_array());
        typed_slice_array(n_src_values.as_float_array(), ids, dest);
    }
    else if(dt.is_double())
    {
        auto dest(n_dest_values.as_double_array());
        typed_slice_array(n_src_values.as_double_array(), ids, dest);
    }
}

//---------------------------------------------------------------------------
/**
 @brief Iterates over the cells in the topo that are specified in element_ids
        and adds their vertex ids into vertex_ids so we can build up a set of
        vertices that will need to be pulled from the coordset.
 */
void
partitioner::get_vertex_ids_for_element_ids(const conduit::Node &n_topo,
    const std::vector<index_t> &element_ids,
    std::set<index_t> &vertex_ids) const
{
    bool is_base_rectilinear = n_topo["type"].as_string() == "rectilinear";
    bool is_base_structured = n_topo["type"].as_string() == "structured";
    bool is_base_uniform = n_topo["type"].as_string() == "uniform";

    if(is_base_rectilinear || is_base_structured || is_base_uniform)
    {
        index_t edims[3] = {1,1,1}, dims[3] = {0,0,0};
        auto ndims = topology::dims(n_topo);
        conduit::blueprint::mesh::utils::topology::logical_dims(n_topo, edims, 3);
//cout << "get_vertex_ids_for_element_ids: edims=" << edims[0] << ", " << edims[1] << ", " << edims[2] << endl;
        dims[0] = edims[0] + 1;
        dims[1] = edims[1] + (ndims > 1 ? 1 : 0);
        dims[2] = edims[2] + (ndims > 2 ? 1 : 0);
//cout << "get_vertex_ids_for_element_ids: dims=" << dims[0] << ", " << dims[1] << ", " << dims[2] << endl;

        index_t cell_ijk[3]={0,0,0}, pt_ijk[3] = {0,0,0}, ptid = 0;
        static const index_t offsets[8][3] = {
            {0,0,0},
            {1,0,0},
            {0,1,0},
            {1,1,0},
            {0,0,1},
            {1,0,1},
            {0,1,1},
            {1,1,1}
        };
        int np = (ndims == 2) ? 4 : 8;
        auto n = element_ids.size();
        for(size_t i = 0; i < n; i++)
        {
            // Get the IJK coordinate of the element.
            grid_id_to_ijk(element_ids[i], edims, cell_ijk);
//cout << element_ids[i] << "-> " << cell_ijk[0] << "," << cell_ijk[1] << "," << cell_ijk[2] << endl;
            // Turn the IJK into vertex ids.
            for(int i = 0; i < np; i++)
            {
                pt_ijk[0] = cell_ijk[0] + offsets[i][0];
                pt_ijk[1] = cell_ijk[1] + offsets[i][1];
                pt_ijk[2] = cell_ijk[2] + offsets[i][2];
                grid_ijk_to_id(pt_ijk, dims, ptid);

//cout << "\t" << pt_ijk[0] << "," << pt_ijk[1] << "," << pt_ijk[2] << "-> " << ptid << endl;

                vertex_ids.insert(ptid);
            }
        }
    }
    else
    {
        conduit::blueprint::mesh::utils::ShapeType shape(n_topo);
        if(shape.is_polygonal())
        {
            conduit::Node n_indices;
            as_index_t(n_topo["elements/connectivity"], n_indices);
            auto iptr = as_index_t_array(n_indices);

            // Make sure we have offsets we can use.
            conduit::Node n_offsets, n_sizes;
#ifdef CONDUIT_INDEX_32
            if(n_topo.has_path("elements/offsets"))
                n_topo["elements/offsets"].to_unsigned_int_array(n_offsets);
            else
                conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(n_topo, n_offsets);
            auto offsets = n_offsets.as_unsigned_int_array();
#else
            if(n_topo.has_path("elements/offsets"))
                n_topo["elements/offsets"].to_unsigned_long_array(n_offsets);
            else
                conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(n_topo, n_offsets);
            auto offsets = n_offsets.as_unsigned_long_array();
#endif
            n_topo["elements/sizes"].to_unsigned_int_array(n_sizes);
            auto sizes = n_sizes.as_unsigned_int_array();
            for(size_t i = 0; i < element_ids.size(); i++)
            {
                auto offset = offsets[element_ids[i]];
                auto sz = sizes[element_ids[i]];
                for(unsigned int ptid = 0; ptid < sz; ptid++)
                    vertex_ids.insert(iptr[offset + ptid]);
            }
        }
        else if(shape.is_polyhedral())
        {
            conduit::Node n_indices;
            as_index_t(n_topo["elements/connectivity"], n_indices);
            auto iptr = as_index_t_array(n_indices);

            // NOTE: I'm assuming that offsets are built already.
            conduit::Node n_offsets, n_sizes;
            as_index_t(n_topo["elements/offsets"], n_offsets);
            as_index_t(n_topo["elements/sizes"], n_sizes);
            auto offsets = as_index_t_array(n_offsets);
            auto sizes = as_index_t_array(n_sizes);

            conduit::Node n_se_conn, n_se_offsets, n_se_sizes;
            as_index_t(n_topo["subelements/connectivity"], n_se_conn);
            as_index_t(n_topo["subelements/offsets"], n_se_offsets);
            as_index_t(n_topo["subelements/sizes"], n_se_sizes);
            auto se_conn = as_index_t_array(n_se_conn);
            auto se_offsets = as_index_t_array(n_se_offsets);
            auto se_sizes = as_index_t_array(n_se_sizes);

            for(auto eid : element_ids)
            {
                auto offset = static_cast<index_t>(offsets[eid]);
                auto nfaces = static_cast<index_t>(sizes[eid]);
                for(index_t fi = 0; fi < nfaces; fi++)
                {
                    // Now, look up the vertices in this face.
                    auto face_id = iptr[offset + fi];
                    auto face_offset = static_cast<index_t>(se_offsets[face_id]);
                    auto face_nfaces = static_cast<index_t>(se_sizes[face_id]);
                    for(index_t vi = 0; vi < face_nfaces; vi++)
                    {
                        vertex_ids.insert(se_conn[face_offset + vi]);
                    }
                }
            }
        }
        else if(n_topo.has_path("elements/element_types"))
        {
            // It looks like the topology is unstructured but contains multiple
            // element types.

            // Determine the number of points for each stream id and shape.
            std::map<int,int> stream_id_npts;
            const conduit::Node &n_element_types = n_topo["elements/element_types"];
            for(index_t i = 0; i < n_element_types.number_of_children(); i++)
            {
                const conduit::Node &n = n_element_types[i];
                auto stream_id = n["stream_id"].to_int();
                std::string shape(n["shape"].as_string());
                for(size_t j = 0; j < utils::TOPO_SHAPES.size(); j++)
                {
                    if(shape == utils::TOPO_SHAPES[j])
                    {
                        stream_id_npts[stream_id] = utils::TOPO_SHAPE_INDEX_COUNTS[j];
                        break;
                    }
                }
            }

            conduit::Node n_stream_ids, n_element_counts, n_stream;
            as_index_t(n_topo["elements/element_index/stream_ids"], n_stream_ids);
            as_index_t(n_topo["elements/element_index/element_counts"], n_element_counts);
            as_index_t(n_topo["elements/stream"], n_stream);
            auto stream_ids = as_index_t_array(n_stream_ids);
            auto element_counts = as_index_t_array(n_element_counts);
            auto stream = as_index_t_array(n_stream);

            // Compute some size,offsets into the stream to help
            std::vector<index_t> sizes, offsets;
            index_t offset = 0, elemid = 0;
            for(index_t j = 0; j < stream_ids.number_of_elements(); j++)
            {
                auto n = static_cast<index_t>(element_counts[j]);
                auto npts = stream_id_npts[stream_ids[j]];
                for(index_t i = 0; i < n; i++)
                {
                    sizes.push_back(npts);
                    offsets.push_back(offset);
                    offset += npts;
                    elemid++;
                }
            }

            // Now, for each element, insert its vertices.
            for(size_t i = 0; i < element_ids.size(); i++)
            {
                auto npts = sizes[element_ids[i]];
                for(index_t j = 0; j < npts; j++)
                    vertex_ids.insert(stream[offsets[element_ids[i]] + j]);
            }
        }
        else
        {
            // Shapes are single types one after the next in the connectivity.
            conduit::Node n_indices;
            as_index_t(n_topo["elements/connectivity"], n_indices);
            auto iptr = as_index_t_array(n_indices);
            auto nverts_in_shape = conduit::blueprint::mesh::utils::TOPO_SHAPE_INDEX_COUNTS[shape.id];
            for(size_t i = 0; i < element_ids.size(); i++)
            {
                auto offset = element_ids[i] * nverts_in_shape;
                for(index_t j = 0; j < nverts_in_shape; j++)
                    vertex_ids.insert(iptr[offset + j]);
            }
        }
    }
}

//---------------------------------------------------------------------------
inline void
index_t_set_to_vector(const std::set<index_t> &src, std::vector<index_t> &dest)
{
    dest.reserve(src.size());
    for(auto it = src.begin(); it != src.end(); it++)
        dest.push_back(*it);
}

//---------------------------------------------------------------------------
conduit::Node *
partitioner::extract(size_t idx, const conduit::Node &n_mesh) const
{
    if(idx >= selections.size())
        return nullptr;

    conduit::Node *retval = nullptr;
    try
    {
        // Get the appropriate topology and coordset nodes.
        const conduit::Node &n_topo = selections[idx]->selected_topology(n_mesh);
        const conduit::Node &n_coordsets = n_mesh["coordsets"];
        std::string csname(n_topo["coordset"].as_string());
        const conduit::Node &n_coordset = n_coordsets[csname];

        // Make output.
        retval = new conduit::Node;
        conduit::Node &n_output = *retval;
        conduit::Node &n_new_coordsets = n_output["coordsets"];
        conduit::Node &n_new_topos = n_output["topologies"];

        // Copy state.
        if(n_mesh.has_child("state"))
        {
            const conduit::Node &n_state = n_mesh["state"];
            n_output["state"].set(n_state);
            // Override the dd if we have set it.
            int dd = selections[idx]->get_destination_domain();
            if(dd != selection::FREE_DOMAIN_ID)
                n_output["state/domain_id"] = dd;
        }

        // Get the indices of the selected elements.
        std::vector<index_t> element_ids, vertex_ids;
        selections[idx]->get_element_ids(n_mesh, element_ids);

        // Try special case for logical selections. We do this so we can make
        // logically structured outputs.
        auto log_sel = dynamic_cast<selection_logical *>(selections[idx].get());
        if(log_sel != nullptr)
        {
            index_t start[3], end[3];
            log_sel->get_start(start[0], start[1], start[2]);
            log_sel->get_end(end[0], end[1], end[2]);

            // Get the vertex ids of the selected elements. We can do it like
            // this faster than using a set. (needed for field extraction)
            log_sel->get_vertex_ids(n_mesh, vertex_ids);

            if(n_coordset["type"].as_string() == "uniform")
                create_new_uniform_coordset(n_coordset, start, end, n_new_coordsets[csname]);
            else if(n_coordset["type"].as_string() == "rectilinear")
                create_new_rectilinear_coordset(n_coordset, start, end, n_new_coordsets[csname]);
            else
                create_new_explicit_coordset(n_coordset, vertex_ids, n_new_coordsets[csname]);

            // Now, create new topologies.
            if(n_topo["type"].as_string() == "uniform")
                create_new_uniform_topo(n_topo, csname, start, n_new_topos[n_topo.name()]);
            else if(n_topo["type"].as_string() == "rectilinear")
                create_new_rectilinear_topo(n_topo, csname, start, n_new_topos[n_topo.name()]);
            else if(n_topo["type"].as_string() == "structured")
                create_new_structured_topo(n_topo, csname, start, end, n_new_topos[n_topo.name()]);
        }
        else
        {
            // Get the unique set of vertex ids used by the elements.
            std::set<index_t> vertex_ids_set;
            get_vertex_ids_for_element_ids(n_topo, element_ids, vertex_ids_set);
            index_t_set_to_vector(vertex_ids_set, vertex_ids);

            // Create a new coordset consisting of the selected vertex ids.
            create_new_explicit_coordset(n_coordset, vertex_ids, n_new_coordsets[csname]);

            // Create a new topology consisting of the selected element ids.
            create_new_unstructured_topo(n_topo, csname, 
                element_ids, vertex_ids, n_new_topos[n_topo.name()]);
        }

        // Create new fields.
        copy_fields(selections[idx]->get_domain(), n_topo.name(),
            vertex_ids, element_ids, n_mesh, n_output);
    }
    catch(conduit::Error &)
    {
        delete retval;
        retval = nullptr;
    }

    return retval;
}

//---------------------------------------------------------------------------
void
partitioner::create_new_uniform_coordset(const conduit::Node &n_coordset,
    const index_t start[3], const index_t end[3], conduit::Node &n_new_coordset) const
{
    // Set coordset dimensions from element start/end.
    index_t ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coordset);
    n_new_coordset["type"] = "uniform";
    n_new_coordset["dims/i"] = end[0] - start[0] + 2;
    n_new_coordset["dims/j"] = end[1] - start[1] + 2;
    if(ndims > 2)
        n_new_coordset["dims/k"] = end[2] - start[2] + 2;

    // Adjust the origins to the right start values.
    auto axes(conduit::blueprint::mesh::utils::coordset::axes(n_coordset));
    const conduit::Node &n_origin = n_coordset["origin"];
    const conduit::Node &n_spacing = n_coordset["spacing"];
    conduit::Node &n_new_origin = n_new_coordset["origin"];
    for(index_t i = 0; i < ndims; i++)
    {
        conduit::Node &origin_i = n_new_origin[n_origin[i].name()];
        double org = n_origin[i].to_double() + n_spacing[i].to_double() * start[i];
        origin_i.set(org);
    }
    // Copy all the spacings
    n_new_coordset["spacing"].set(n_coordset["spacing"]);
}

//---------------------------------------------------------------------------
void
partitioner::create_new_rectilinear_coordset(const conduit::Node &n_coordset,
    const index_t start[3], const index_t end[3], conduit::Node &n_new_coordset) const
{
    const conduit::Node &n_values = n_coordset["values"];
    conduit::Node &n_new_values = n_new_coordset["values"];
    n_new_coordset["type"] = "rectilinear";
    // Slice each axis logically.
    index_t ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coordset);
    for(index_t d = 0; d < ndims; d++)
    {
        std::vector<index_t> indices;
        auto nend = end[d] + 1;
        for(index_t i = start[d]; i <= nend; i++)
            indices.push_back(i);

        const conduit::Node &src = n_values[d];
        slice_array(src, indices, n_new_values[src.name()]);
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_explicit_coordset(const conduit::Node &n_coordset,
    const std::vector<index_t> &vertex_ids, conduit::Node &n_new_coordset) const
{
    conduit::Node n_explicit;
    n_new_coordset["type"] = "explicit";
    if(n_coordset["type"].as_string() == "uniform")
    {
        conduit::blueprint::mesh::coordset::uniform::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"].as_string() == "rectilinear")
    {
        conduit::blueprint::mesh::coordset::rectilinear::to_explicit(n_coordset, n_explicit);

        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_explicit);
        const conduit::Node &n_values = n_explicit["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
    else if(n_coordset["type"].as_string() == "explicit")
    {
        auto axes = conduit::blueprint::mesh::utils::coordset::axes(n_coordset);
        const conduit::Node &n_values = n_coordset["values"];
        conduit::Node &n_new_values = n_new_coordset["values"];
        for(size_t i = 0; i < axes.size(); i++)
        {
            const conduit::Node &n_axis_values = n_values[axes[i]];
            conduit::Node &n_new_axis_values = n_new_values[axes[i]];
            slice_array(n_axis_values, vertex_ids, n_new_axis_values);
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_uniform_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "uniform";
    n_new_topo["coordset"] = csname;
    const char *keys[] = {"elements/origin/i",
        "elements/origin/j",
        "elements/origin/k"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_path(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_rectilinear_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "rectilinear";
    n_new_topo["coordset"] = csname;
    const char *keys[] = {"elements/origin/i",
        "elements/origin/j",
        "elements/origin/k"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_path(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
void
partitioner::create_new_structured_topo(const conduit::Node &n_topo,
    const std::string &csname, const index_t start[3], const index_t end[3],
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"] = "structured";
    n_new_topo["coordset"] = csname;
    conduit::Node &n_dims = n_new_topo["elements/dims"];
    n_dims["i"].set(static_cast<conduit::int64>(end[0] - start[0] + 1));
    n_dims["j"].set(static_cast<conduit::int64>(end[1] - start[1] + 1));
    if(n_topo.has_path("elements/dims/k"))
        n_dims["k"].set(static_cast<conduit::int64>(end[2] - start[2] + 1));
    const char *keys[] = {"elements/origin/i0",
        "elements/origin/j0",
        "elements/origin/k0"};
    for(int i = 0; i < 3; i++)
    {
        if(n_topo.has_child(keys[i]))
        {
            const conduit::Node &value = n_topo[keys[i]];
            n_new_topo[keys[i]].set(static_cast<conduit::int64>(value.to_uint64() + start[i]));
        }
    }
}

//---------------------------------------------------------------------------
/**
 @note We pass in the coordset name because the conversion routines in here
       result in topologies that do not set a coordset name.
 */
void
partitioner::create_new_unstructured_topo(const conduit::Node &n_topo,
    const std::string &csname,
    const std::vector<index_t> &element_ids,
    const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    if(n_topo["type"].as_string() == "uniform")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::uniform::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "rectilinear")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::rectilinear::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "structured")
    {
        conduit::Node n_uns, cdest; // what is cdest?
        conduit::blueprint::mesh::topology::structured::to_unstructured(n_topo, n_uns, cdest);
        unstructured_topo_from_unstructured(n_uns, csname, element_ids, vertex_ids, n_new_topo);
    }
    else if(n_topo["type"].as_string() == "unstructured")
    {
        unstructured_topo_from_unstructured(n_topo, csname, element_ids, vertex_ids, n_new_topo);
    }
}

//---------------------------------------------------------------------------
void
partitioner::unstructured_topo_from_unstructured(const conduit::Node &n_topo,
    const std::string &csname,
    const std::vector<index_t> &element_ids,
    const std::vector<index_t> &vertex_ids,
    conduit::Node &n_new_topo) const
{
    n_new_topo["type"].set("unstructured");
    n_new_topo["coordset"].set(csname);

    // vertex_ids contains the list of old vertex ids that our selection uses
    // from the old coordset. It can serve as a new to old map.
    std::map<index_t,index_t> old2new;
    for(size_t i = 0; i < vertex_ids.size(); i++)
        old2new[vertex_ids[i]] = static_cast<index_t>(i);

    conduit::blueprint::mesh::utils::ShapeType shape(n_topo);
    std::vector<index_t> new_conn;
    if(shape.is_polygonal())
    {
        conduit::Node n_indices;
        as_index_t(n_topo["elements/connectivity"], n_indices);
        auto iptr = as_index_t_array(n_indices);

        // Make sure we have offsets we can use.
        conduit::Node n_offsets, n_sizes;
#ifdef CONDUIT_INDEX_32
        if(n_topo.has_path("elements/offsets"))
            n_topo["elements/offsets"].to_unsigned_int_array(n_offsets);
        else
            conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(n_topo, n_offsets);
        auto offsets = n_offsets.as_unsigned_int_array();
#else
        if(n_topo.has_path("elements/offsets"))
            n_topo["elements/offsets"].to_unsigned_long_array(n_offsets);
        else
            conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(n_topo, n_offsets);
        auto offsets = n_offsets.as_unsigned_long_array();
#endif
        n_topo["elements/sizes"].to_unsigned_int_array(n_sizes);
        auto sizes = n_sizes.as_unsigned_int_array();

        std::vector<index_t> new_offsets, new_sizes;
        index_t next_offset = 0;
        for(size_t i = 0; i < element_ids.size(); i++)
        {
            auto offset = offsets[element_ids[i]];
            auto sz = sizes[element_ids[i]];
            for(unsigned int ptid = 0; ptid < sz; ptid++)
                new_conn.push_back(old2new[iptr[offset + ptid]]);
            new_sizes.push_back(sz);
            new_offsets.push_back(next_offset);
            next_offset += sz;
        }
        n_new_topo["elements/shape"].set(n_topo["elements/shape"]);
        n_new_topo["elements/connectivity"].set(new_conn);
        n_new_topo["elements/sizes"] = new_sizes;
        n_new_topo["elements/offsets"] = new_offsets;
    }
    else if(shape.is_polyhedral())
    {
        conduit::Node n_indices;
        as_index_t(n_topo["elements/connectivity"], n_indices);
        auto iptr = as_index_t_array(n_indices);

        // NOTE: I'm assuming that offsets are built already.
        conduit::Node n_offsets, n_sizes;
        as_index_t(n_topo["elements/offsets"], n_offsets);
        as_index_t(n_topo["elements/sizes"], n_sizes);
        auto offsets = as_index_t_array(n_offsets);
        auto sizes = as_index_t_array(n_sizes);

        conduit::Node n_se_conn, n_se_offsets, n_se_sizes;
        as_index_t(n_topo["subelements/connectivity"], n_se_conn);
        as_index_t(n_topo["subelements/offsets"], n_se_offsets);
        as_index_t(n_topo["subelements/sizes"], n_se_sizes);
        auto se_conn = as_index_t_array(n_se_conn);
        auto se_offsets = as_index_t_array(n_se_offsets);
        auto se_sizes = as_index_t_array(n_se_sizes);

        std::map<index_t,index_t> old2new_faces;
        std::vector<index_t> new_sizes, new_offsets, new_se_conn, new_se_sizes, new_se_offsets;

        index_t new_offset = 0, new_se_offset = 0;
        for(auto eid : element_ids)
        {
            auto offset = static_cast<index_t>(offsets[eid]);
            auto nfaces = static_cast<index_t>(sizes[eid]);
            for(index_t fi = 0; fi < nfaces; fi++)
            {
                auto face_id = iptr[offset + fi];
                auto it = old2new_faces.find(face_id);
                if(it == old2new_faces.end())
                {
                    // We have not seen the face before. Add it.
                    auto new_face_id = static_cast<index_t>(old2new_faces.size());
                    old2new_faces[face_id] = new_face_id;
                    new_conn.push_back(new_face_id);

                    auto face_offset = se_offsets[face_id];
                    auto face_nverts = static_cast<index_t>(se_sizes[face_id]);
                    for(index_t vi = 0; vi < face_nverts; vi++)
                    {
                        auto vid = se_conn[face_offset + vi];
#if 1
                        if(old2new.find(vid) == old2new.end())
                            cout << " ERROR - no vertex " << vid << " in old2new." << endl;
#endif
                        auto nvid = old2new[vid];
                        new_se_conn.push_back(nvid);
                    }

                    new_se_sizes.push_back(face_nverts);
                    new_se_offsets.push_back(new_se_offset);
                    new_se_offset += face_nverts;
                }
                else
                {
                    // We've seen the face before. Reference it.
                    new_conn.push_back(it->second);
                }
            }

            new_sizes.push_back(nfaces);
            new_offsets.push_back(new_offset);
            new_offset += nfaces;
        }

        n_new_topo["elements/sizes"].set(new_sizes);
        n_new_topo["elements/offsets"].set(new_offsets);
        n_new_topo["elements/shape"].set(n_topo["elements/shape"]);
        n_new_topo["elements/connectivity"].set(new_conn);
        n_new_topo["subelements/shape"].set(n_topo["subelements/shape"].as_string());
        n_new_topo["subelements/connectivity"].set(new_se_conn);
        n_new_topo["subelements/sizes"].set(new_se_sizes);
        n_new_topo["subelements/offsets"].set(new_se_offsets);
    }
    else if(n_topo.has_path("elements/element_types"))
    {
        const conduit::Node &n_element_types = n_topo["elements/element_types"];

        // Determine the number of points for each stream id and shape.
        std::map<int,int> stream_id_npts;
        std::set<int> unique_shape_types;
        std::map<std::string, int> shape_stream_id;
        std::map<int, std::string> stream_id_shape;

        for(index_t i = 0; i < n_element_types.number_of_children(); i++)
        {
            const conduit::Node &n = n_element_types[i];
            auto stream_id = n["stream_id"].to_int();
            std::string shape(n["shape"].as_string());
            for(size_t j = 0; j < utils::TOPO_SHAPES.size(); j++)
            {
                if(shape == utils::TOPO_SHAPES[j])
                {
                    stream_id_npts[stream_id] = utils::TOPO_SHAPE_INDEX_COUNTS[j];
                    unique_shape_types.insert(utils::TOPO_SHAPE_INDEX_COUNTS[j]);
                    shape_stream_id[shape] = stream_id;
                    stream_id_shape[stream_id] = shape;
                    break;
                }
            }
        }

        if(unique_shape_types.size() > 1)
        {
            // Elements are different types.

            conduit::Node n_stream_ids, n_element_counts, n_stream;
            as_index_t(n_topo["elements/element_index/stream_ids"], n_stream_ids);
            as_index_t(n_topo["elements/element_index/element_counts"], n_element_counts);
            as_index_t(n_topo["elements/stream"], n_stream);
            auto stream_ids = as_index_t_array(n_stream_ids);
            auto element_counts = as_index_t_array(n_element_counts);
            auto stream = as_index_t_array(n_stream);

            // Compute some size,offsets into the stream to help
            std::vector<index_t> stream_ids_expanded, offsets;
            index_t offset = 0, elemid = 0;
            for(index_t j = 0; j < stream_ids.number_of_elements(); j++)
            {
                auto n = static_cast<index_t>(element_counts[j]);
                auto npts = stream_id_npts[stream_ids[j]];
                for(index_t i = 0; i < n; i++)
                {
                    stream_ids_expanded.push_back(stream_ids[j]);
                    offsets.push_back(offset);
                    offset += npts;
                    elemid++;
                }
            }

            // Now, for each element, add its topology to the new stream.
            int current_stream_id = -1;
            index_t current_stream_count = 0;
            std::vector<index_t> new_stream_ids, new_element_counts;
            for(size_t i = 0; i < element_ids.size(); i++)
            {
                int sid = stream_ids_expanded[element_ids[i]];
                auto npts = stream_id_npts[sid];

                // Save the element's vertices into the new stream.
                for(index_t j = 0; j < npts; j++)
                {
                    auto vid = stream[offsets[element_ids[i]] + j];
                    new_conn.push_back(old2new[vid]);
                }

                if(current_stream_id == -1)
                {
                    current_stream_id = sid;
                    current_stream_count = 1;
                }
                else if(current_stream_id != sid)
                {
                    new_stream_ids.push_back(current_stream_id);
                    new_element_counts.push_back(current_stream_count);

                    current_stream_id = sid;
                    current_stream_count = 1;
                }
                else
                {
                    current_stream_count++;
                }
            }
            if(current_stream_count > 0)
            {
                new_stream_ids.push_back(current_stream_id);
                new_element_counts.push_back(current_stream_count);
            }

            n_new_topo["elements/element_types"].set(n_topo["elements/element_types"]);
            n_new_topo["elements/element_index/stream_ids"].set(new_stream_ids);
            n_new_topo["elements/element_index/element_counts"].set(new_element_counts);
            n_new_topo["elements/stream"].set(new_conn);            
        }
        else
        {
            // All elements are actually a single type.
            int nverts_in_shape = -1;
            const conduit::Node &n_et = n_topo["elements/element_types"][0];
            std::string shape(n_et["shape"].as_string());
            for(size_t j = 0; j < utils::TOPO_SHAPES.size(); j++)
            {
                if(shape == utils::TOPO_SHAPES[j])
                {
                    nverts_in_shape = utils::TOPO_SHAPE_INDEX_COUNTS[j];
                    break;
                }
            }
            if(nverts_in_shape == -1)
            {
                CONDUIT_ERROR("Invalid shape");
            }
            else
            {
                conduit::Node n_indices;
                as_index_t(n_topo["elements/stream"], n_indices);
                auto iptr = as_index_t_array(n_indices);
                for(size_t i = 0; i < element_ids.size(); i++)
                {
                    auto offset = element_ids[i] * nverts_in_shape;
                    for(index_t j = 0; j < nverts_in_shape; j++)
                        new_conn.push_back(old2new[iptr[offset + j]]);
                }

                n_new_topo["elements/shape"].set(n_topo["elements/shape"]);
                n_new_topo["elements/connectivity"].set(new_conn);
            }
        }
    }
    else
    {
        // Shapes are single types one after the next in the connectivity.
        conduit::Node n_indices;
        as_index_t(n_topo["elements/connectivity"], n_indices);
        auto iptr = as_index_t_array(n_indices);

        auto nverts_in_shape = conduit::blueprint::mesh::utils::TOPO_SHAPE_INDEX_COUNTS[shape.id];
        for(size_t i = 0; i < element_ids.size(); i++)
        {
            auto offset = element_ids[i] * nverts_in_shape;
#if 0
            cout << "cell " << element_ids[i] << ":  old(";
            for(index_t j = 0; j < nverts_in_shape; j++)
                cout << iptr[offset + j] << ", ";
            cout << "), new(";
            for(index_t j = 0; j < nverts_in_shape; j++)
            {
                auto it = old2new.find(iptr[offset + j]);
                if(it == old2new.end())
                    cout << "ERROR" << ", ";
                else
                    cout << it->second << ", ";
            }
            cout << ")" << endl;
#endif
            for(index_t j = 0; j < nverts_in_shape; j++)
                new_conn.push_back(old2new[iptr[offset + j]]);
        }

        n_new_topo["elements/shape"].set(n_topo["elements/shape"]);
        n_new_topo["elements/connectivity"].set(new_conn);
    }
}

//---------------------------------------------------------------------------
/**
 @brief This method wraps the input node and adds original elements/vertices
        to it since downstream it is best that these fields exist.
 */
conduit::Node *
partitioner::wrap(size_t idx, const conduit::Node &n_mesh) const
{
    conduit::Node *n_wrap = new conduit::Node;

    // Add the members from n_mesh into n_wrap but treat fields specially.
    for(index_t i = 0; i < n_mesh.number_of_children(); i++)
    {
        const conduit::Node &f = n_mesh[i];
        if(f.name() != "fields")
            (*n_wrap)[f.name()].set_external_node(f);
    }
    conduit::Node &n_new_fields = (*n_wrap)["fields"];
    // Make references to any fields from the input into the n_wrap/fields node.
    if(n_mesh.has_path("fields"))
    {
        const conduit::Node &n_fields = n_mesh["fields"];
        for(index_t i = 0; i < n_fields.number_of_children(); i++)
        {
            const conduit::Node &f = n_fields[i];
            n_new_fields[f.name()].set_external_node(f);
        }
    }

    const conduit::Node &n_topo = selections[idx]->selected_topology(n_mesh);
    std::string toponame(n_topo.name());
    index_t nelements = conduit::blueprint::mesh::topology::length(n_topo);
    index_t domain = selections[idx]->get_domain();
    std::string csname(n_topo["coordset"].as_string());
    index_t nvertices = 0;
    if(n_mesh.has_child("coordsets"))
    {
        const conduit::Node &n_cs = n_mesh["coordsets"];
        nvertices = mesh::utils::coordset::length(n_cs[csname]);
    }

    // Save the vertex_ids as a new MC field.
    if(nvertices > 0)
    {
        conduit::Node &n_field = n_new_fields["original_vertex_ids"];
        n_field["association"] = "vertex";
        if(!toponame.empty())
            n_field["topology"] = toponame;
        std::vector<index_t> vertex_ids(nvertices);
        for(index_t i = 0; i < nvertices; i++)
            vertex_ids[i] = i;
        std::vector<index_t> domain_ids(nvertices, domain);
        n_field["values/domains"].set(domain_ids);
        n_field["values/ids"].set(vertex_ids);
    }

    // Save the element_ids as a new MC field.
    if(nelements > 0)
    {
        conduit::Node &n_field = n_new_fields["original_element_ids"];
        n_field["association"] = "element";
        if(!toponame.empty())
            n_field["topology"] = toponame;
        std::vector<index_t> elem_ids(nelements);
        for(index_t i = 0; i < nelements; i++)
            elem_ids[i] = i;
        std::vector<index_t> domain_ids(nelements, domain);
        n_field["values/domains"].set(domain_ids);
        n_field["values/ids"].set(elem_ids);
    }

    return n_wrap;
}

//---------------------------------------------------------------------------
void
partitioner::execute(conduit::Node &output)
{
    // By this stage, we will have at least target selections spread across
    // the participating ranks. Now, we need to process the selections to
    // make chunks.
    std::vector<chunk> chunks;
    for(size_t i = 0; i < selections.size(); i++)
    {
        // Get destination rank, domain if the selection has any. If not, it
        // will return -1,-1 so we have some flexibility in how data are moved.
        int dr = selections[i]->get_destination_rank();
        int dd = selections[i]->get_destination_domain();

        if(selections[i]->get_whole(*meshes[i]))
        {
            // We had a selection that spanned the entire mesh so we'll take
            // the whole mesh rather than extracting. If we are using "mapping"
            // then we will be wrapping the mesh so we can add vertex and element
            // maps to it without changing the input mesh.
            if(mapping)
            {
                conduit::Node *c = wrap(i, *meshes[i]);
                chunks.push_back(chunk(c, true, dr, dd));
            }
            else
                chunks.push_back(chunk(meshes[i], false, dr, dd));
        }
        else
        {
            conduit::Node *c = extract(i, *meshes[i]);
            chunks.push_back(chunk(c, true, dr, dd));
        }
    }

    // Compute the destination rank and destination domain of each input
    // chunk present on this rank.
    std::vector<int> dest_rank, dest_domain, offsets;
    map_chunks(chunks, dest_rank, dest_domain, offsets);

    // Communicate chunks to the right destination ranks
    std::vector<chunk> chunks_to_assemble;
    std::vector<int> chunks_to_assemble_domains;
    communicate_chunks(chunks, dest_rank, dest_domain, offsets,
        chunks_to_assemble, chunks_to_assemble_domains);

    // Now that we have all the parts we need in chunks_to_assemble, combine
    // the chunks.
    std::set<int> unique_doms;
    for(size_t i = 0; i < chunks_to_assemble_domains.size(); i++)
        unique_doms.insert(chunks_to_assemble_domains[i]);

    if(!chunks_to_assemble.empty())
    {
        output.reset();
        for(auto dom = unique_doms.begin(); dom != unique_doms.end(); dom++)
        {
            // Get the chunks for this output domain.
            std::vector<const Node *> this_dom_chunks;
            for(size_t i = 0; i < chunks_to_assemble_domains.size(); i++)
            {
                if(chunks_to_assemble_domains[i] == *dom)
                    this_dom_chunks.push_back(chunks_to_assemble[i].mesh);
            }

            if(this_dom_chunks.size() == 1)
            {
                if(unique_doms.size() > 1)
                {
                    // There are multiple domains in the output.
                    conduit::Node &n = output.append();
                    n.set(*this_dom_chunks[0]); // Could we transfer ownership if we own the chunk?
                }
                else
                {
                    // There is one domain in the output.
                    output.set(*this_dom_chunks[0]); // Could we transfer ownership if we own the chunk?
                }
            }
            else if(this_dom_chunks.size() > 1)
            {
                // Combine the chunks for this domain and add to output or to
                // a list in output.
                if(unique_doms.size() > 1)
                {
                    // There are multiple domains in the output.
                    combine(*dom, this_dom_chunks, output.append());
                }
                else
                {
                    // There is one domain in the output.
                    combine(*dom, this_dom_chunks, output);
                }
            }
        }
    }

    // Clean up
    for(size_t i = 0; i < chunks.size(); i++)
        chunks[i].free();
    for(size_t i = 0; i < chunks_to_assemble.size(); i++)
        chunks_to_assemble[i].free();
}

//-------------------------------------------------------------------------
unsigned int
partitioner::starting_index(const std::vector<partitioner::chunk> &/*chunks*/)
{
    return 0;
}

//-------------------------------------------------------------------------
void
partitioner::map_chunks(const std::vector<partitioner::chunk> &chunks,
    std::vector<int> &dest_ranks,
    std::vector<int> &dest_domain, 
    std::vector<int> &offsets)
{
    // All data for this rank begins at offset 0.
    offsets.push_back(0);

    // All data stays on this rank in serial.
    dest_ranks.resize(chunks.size());
    for(size_t i = 0; i < chunks.size(); i++)
        dest_ranks[i] = rank;
#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << "map_chunks:" << endl;
    for(size_t i = 0; i < chunks.size(); i++)
        chunks[i].mesh->print();
#endif

    // Determine average chunk size.
    std::vector<index_t> chunk_sizes;
    index_t total_len = 0;
    for(size_t i =0 ; i < chunks.size(); i++)
    {
        const conduit::Node &n_topos = chunks[i].mesh->operator[]("topologies");
        index_t len = 0;
        for(index_t j = 0; j < n_topos.number_of_children(); j++)
            len += conduit::blueprint::mesh::topology::length(n_topos[j]);
        total_len += len;
        chunk_sizes.push_back(len);
    }
    index_t len_per_target = total_len / std::max(1u,target);
#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << "map_chunks: chunks.size=" << chunks.size()
         << ", total_len = " << total_len
         << ", target=" << target
         << ", len_per_target=" << len_per_target << endl;
#endif
    // Come up with a list of domain ids to avoid in our numbering.
    std::set<int> reserved_dd;
    for(size_t i = 0; i < chunks.size(); i++)
    {
        if(chunks[i].destination_domain != selection::FREE_DOMAIN_ID)
            reserved_dd.insert(chunks[i].destination_domain);
    }
#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << "map_chunks: reserved_dd={";
    for(auto dd : reserved_dd)
        cout << dd << ", ";
    cout << "}" << endl;
#endif
    int start_index = starting_index(chunks);
    // Get a domain id for the first free domain.
    int domid = start_index;
    while(reserved_dd.find(domid) != reserved_dd.end())
        domid++;
    reserved_dd.insert(domid);
#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << "map_chunks: domid=" << domid << endl;
#endif

    // We have a certain number of chunks but determine how many targets
    // that makes. It ought to be equal to target.
    auto targets_from_chunks = static_cast<unsigned int>(count_targets());

    if(targets_from_chunks == target)
    {
        // The number of targets we'd make from the chunks is the same
        // as the target.
        for(size_t i =0 ; i < chunks.size(); i++)
        {
            int dd = chunks[i].destination_domain;
            if(dd == selection::FREE_DOMAIN_ID)
            {
                dest_domain.push_back(domid);

                // Get the next domain id.
                while(reserved_dd.find(domid) != reserved_dd.end())
                    domid++;
                reserved_dd.insert(domid);
            }
            else
            {
                // We know it goes in this domain.
                dest_domain.push_back(dd);
            }
        }
    }
    else if(targets_from_chunks > target)
    {
        // This may happen when we need to combine domains to a smaller
        // target count.

        // NOTE: For domains that do not already have a destination domain,
        //       we are just grouping adjacent chunks in the overall list
        //       while trying to target a certain number of cells per domain.
        //       We may someday also want to consider the bounding boxes so
        //       we group chunks that are close spatially.

        index_t running_len = 0;
        for(size_t i = 0; i < chunks.size(); i++)
        {
            int dd = chunks[i].destination_domain;
            if(dd == selection::FREE_DOMAIN_ID)
            {
                running_len += chunk_sizes[i];
                if(running_len > len_per_target)
                {
                    running_len = 0;
                    // Get the next domain id.
                    while(reserved_dd.find(domid) != reserved_dd.end())
                        domid++;
                    reserved_dd.insert(domid);
                }

                dest_domain.push_back(domid);
            }
            else
            {
                // We know it goes in this domain.
                dest_domain.push_back(dd);
            }
        }
    }
    else
    {
        // The number of chunks is less than the target. Something is wrong!
        CONDUIT_ERROR("The number of chunks (" << chunks.size()
                      << ") is smaller than requested (" << target << ").");
    }
#ifdef CONDUIT_DEBUG_PARTITIONER
    cout << "dest_ranks={";
    for(size_t i = 0; i < dest_ranks.size(); i++)
        cout << dest_ranks[i] << ", ";
    cout << "}" << endl;
    cout << "dest_domain={";
    for(size_t i = 0; i < dest_domain.size(); i++)
        cout << dest_domain[i] << ", ";
    cout << "}" << endl;
#endif
}

//-------------------------------------------------------------------------
void
partitioner::communicate_chunks(const std::vector<partitioner::chunk> &chunks,
    const std::vector<int> &/*dest_rank*/,
    const std::vector<int> &dest_domain,
    const std::vector<int> &/*offsets*/,
    std::vector<partitioner::chunk> &chunks_to_assemble,
    std::vector<int> &chunks_to_assemble_domains)
{
    // In serial, communicating the chunks among ranks means passing them
    // back in the output arguments. We mark them as not-owned so we do not
    // double-free.
    for(size_t i = 0; i < chunks.size(); i++)
    {
        chunks_to_assemble.push_back(chunk(chunks[i].mesh, false));
        chunks_to_assemble_domains.push_back(dest_domain[i]);
    }
}

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------
namespace coordset
{

/**
 @brief The implmentation of conduit::blueprint::mesh::coordset::merge
*/
class point_merge
{
public:
    void execute(const std::vector<const conduit::Node *> &coordsets, 
                 double tolerance,
                 Node &output);

private:
    enum class coord_system
    {
        cartesian,
        cylindrical,
        spherical,
        logical
    };

    int examine_extents(std::vector<std::vector<float64>> &extents) const;

    /**
    @brief Useful when none of the coordsets overlap. Combines all coordinates
        to one array.
    */
    void append_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension);

    /**
    @brief Used when coordsets overlap. Combines all coordinates into
        one array; merging points within tolerance.
    */
    void merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t, double tolerance);

    void create_output(index_t dimension, Node &output) const;

    /**
    @brief Iterates the coordinates in the given coordinate set,
        invoking the given lambda function with the signature
        void (float64 *point, index_t dim)
    NOTE: It will always be valid to index the "point" as if its
        "dim" was 3. For example if the original data was only 2D then
        point[2] will be 0 and dim will be 2.
    */
    template<typename Func>
    void iterate_coordinates(const Node &coordset, Func &&func);

    /**
    @brief Determines how many points there are and reserves space in member vectors
        new_points and old_to_new_ids
    @return npoints*dimension
    */
    index_t reserve_vectors(const std::vector<Node> &coordsets, index_t dimension);

    /**
    @brief The simple (slow) approach to merging the data based off distance.
    */
    void simple_merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance);

    void spatial_search_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance);

    void truncate_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance);

    static void xyz_to_rtp(double x, double y, double z, double &out_r, double &out_t, double &out_p);
    // TODO
    static void xyz_to_rz (double x, double y, double z, double &out_r, double &out_z);

    // TODO
    static void rz_to_xyz (double r, double z, double &out_x, double &out_y, double &out_z);
    // TODO
    static void rz_to_rtp (double r, double z, double &out_r, double &out_t, double &out_p);

    static void rtp_to_xyz(double r, double t, double p, double &out_x, double &out_y, double &out_z);
    // TODO
    static void rtp_to_rz (double r, double t, double p, double &out_r, double &out_z);

    static void translate_system(coord_system in_system, coord_system out_system,
        float64 p0, float64 p1, float64 p2, float64 &out_p1, float64 &out_p2, float64 &out_p3);

    /**
    @brief Returns the axis names for the given coordinate system
    */
    static const std::vector<std::string> &get_axes_for_system(coord_system);

    coord_system out_system;

    // Outputs
    std::vector<std::vector<index_t>> old_to_new_ids;
    std::vector<float64> new_coords;
};

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::coordset::utils --
//-----------------------------------------------------------------------------
namespace utils
{

/**
 @brief A simple vector struct to be used by the kdtree
*/
template<typename T, size_t Size>
struct vector
{
    using this_type = vector<T, Size>;
    using data_type = std::array<T, Size>;
    using value_type = T;
private:
    // Used to alias vector data
    template<size_t Index>
    struct accessor
    {
        data_type data;

        constexpr operator T() const
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index];
        }

        T operator=(T v)
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index] = v;
        }
    };

public:
    // Possible to access vector data with x/y/z
    union
    {
        data_type    v;
        accessor<0>  x;
        accessor<1>  y;
        accessor<2>  z;
    };

    constexpr size_t size() const
    {
        return Size;
    }

    T operator[](size_t index) const
    {
        return v[index];
    }

    T &operator[](size_t index)
    {
        return v[index];
    }

    void zero()
    {
        set_all(0);
    }

    void set_all(T val)
    {
        for(size_t i = 0u; i < size(); i++)
        {
            v[i] = val;
        }
    }

    // NOTE: Defining operator= makes this non-POD type
    // this_type operator=(const this_type &other)
    // {
    //     for(size_t i = 0u; i < size(); i++)
    //     {
    //         v[i] = other[i];
    //     }
    //     return *this;
    // }

    void copy(const this_type &other)
    {
        for(auto i = 0u; i < size(); i++)
            other.v[i] = v[i];
    }

    bool operator<=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] <= other[i];
        return retval;
    }

    bool operator>=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] >= other[i];
        return retval;
    }

    this_type operator+(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] + scalar;
        }
        return retval;
    }

    this_type operator-(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] - scalar;
        }
        return retval;
    }

    double distance2(const this_type &other) const
    {
        double d2 = 0.;
        for(size_t i = 0u; i < size(); i++)
        {
            const auto diff = other[i] - v[i];
            d2 += (diff*diff);
        }
        return d2;
    }

    double distance(const this_type &other) const
    {
        return std::sqrt(distance2(other));
    }
};

/**
 @brief A simple bounding box struct to be used by the kdtree
*/
template<typename VectorType>
struct bounding_box
{
    using value_type = typename VectorType::value_type;
    using data_type = typename VectorType::data_type;
    using T = value_type;
    VectorType min;
    VectorType max;

    bool contains(const VectorType &point) const
    {
        return (point >= min && point <= max);
    }

    bool contains(const VectorType &point, double tolerance) const
    {
        return (point >= (min - tolerance) && point <= (max + tolerance));
    }

    void expand(const VectorType &point)
    {
        for(size_t i = 0u; i < min.size(); i++)
        {
            min[i] = std::min(min[i], point[i]);
            max[i] = std::max(max[i], point[i]);
        }
    }
};

using vec2f = vector<float,2>;
using vec3f = vector<float,3>;
using vec2  = vector<double,2>;
using vec3  = vector<double,3>;

/**
 @brief A spatial search structure used to merge points within a given tolerance
*/
template<typename VectorType, typename DataType>
class kdtree
{
private:
    using Float = typename VectorType::value_type;
    // using IndexType = conduit_index_t;
public:
    constexpr static auto dimension = std::tuple_size<typename VectorType::data_type>::value;
    using vector_type = VectorType;
    using data_type = DataType;
    using IndexType = size_t;

    template<typename Precision, size_t D>
    struct kdnode
    {
        using node = kdnode<Precision, D>;
        std::vector<VectorType> points;
        std::vector<DataType> data;
        bounding_box<VectorType> bb;
        node *left{nullptr};
        node *right{nullptr};
        Float split{0.0};
        unsigned int dim;
        bool has_split{false};
    };
    using node = kdnode<Float, dimension>;
    using pair = std::pair<std::pair<node*,DataType&>, bool>;

    kdtree() = default;
    ~kdtree()
    {
        const auto lambda = [](node *node, unsigned int)
        {
            delete node;
        };
        if(root) { traverse_lrn(lambda, root); }
    }

    /**
    @brief Searches the tree for the given point using
            tolerance as an acceptable distance to merge like-points.
    */
    DataType *find_point(const VectorType &point, Float tolerance)
    {
        DataType *retval = nullptr;
        if(!root)
        {
            retval = nullptr;
        }
        else if(root->bb.contains(point, tolerance))
        {
            retval = find_point(root, 0, point, tolerance);
        }
        return retval;
    }

    void insert(const VectorType &point, const DataType &r)
    {
        scratch.reserve(point_vec_size*2);
        if(!root)
        {
            root = create_node(point, r);
            npoints++;
        }
        else
        {
            insert(root, 0, point, r);
        }
    }

    IndexType size()  const { return npoints; }
    IndexType nodes() const { return nnodes; }
    IndexType depth() const { return tree_depth; }

    void set_bucket_size(IndexType n) { point_vec_size = n; }
    IndexType get_bucket_size() const { return point_vec_size; }

    template<typename Func>
    void iterate_points(Func &&func)
    {
        IndexType point_id = 0u;
        const auto lambda = [&](node *n, unsigned int) {
            for(IndexType i = 0u; i < n->points.size(); i++)
            {
                func(point_id, n->points[i], n->data[i]);
                point_id++;
            }
        };
        if(root) { traverse_lnr(lambda, root); }
    }

    template<typename Func>
    void traverse_nodes(Func &&func)
    {
        if(root) { traverse_lnr(func, root); }
    }

private:
    // Create an empty node
    node *create_node()
    {
        node *newnode = new node;
        newnode->points.reserve(point_vec_size);
        newnode->data.reserve(point_vec_size);
        newnode->bb.min[0] = std::numeric_limits<Float>::max();
        newnode->bb.min[1] = std::numeric_limits<Float>::max();
        newnode->bb.min[2] = std::numeric_limits<Float>::max();
        newnode->bb.max[0] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[1] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[2] = std::numeric_limits<Float>::lowest();
        newnode->left = nullptr;
        newnode->right = nullptr;
        newnode->split = 0;
        newnode->dim = 0;
        newnode->has_split = false;
        nnodes++;
        return newnode;
    }

    // Create a node with initial values inserted
    node *create_node(VectorType loc, const DataType &r)
    {
        node *newnode = create_node();
        node_add_data(newnode, loc, r);
        return newnode;
    }

    static void node_add_data(node *n, const VectorType &p, const DataType &d)
    {
        n->bb.expand(p);
        n->points.push_back(p);
        n->data.push_back(d);
    }

    /**
    @brief Splits the given node and inserts point/data into the proper child
    */
    void node_split(node *n, const VectorType &point, const DataType &data)
    {
        // Determine which dim to split on
        IndexType dim = 0;
        {
            Float longest_dim = std::numeric_limits<Float>::lowest();
            for(IndexType i = 0; i < n->bb.min.size(); i++)
            {
                const Float dim_len = n->bb.max[i] - n->bb.min[i];
                if(longest_dim < dim_len)
                {
                    dim = i;
                    longest_dim = dim_len;
                }
            }
            n->dim = dim;
        }

        // Determine what value on the dim to split on
        {
            scratch.clear();
            for(IndexType i = 0; i < point_vec_size; i++)
            {
                scratch.push_back(i);
            }
            std::sort(scratch.begin(), scratch.end(), [=](IndexType i0, IndexType i1) {
                return n->points[i0][dim] < n->points[i1][dim];
            });

            // If the index stored in scratch is point_vec_size
            const IndexType scratch_idx = scratch.size() / 2;
            const IndexType median_idx = scratch[scratch_idx];
            Float median = n->points[median_idx][dim];
            // Check if the new point is our actual median
            if(point[dim] > n->points[scratch[scratch_idx-1]][dim] && point[dim] < median)
            {
                median = point[dim];
            }

            n->split = median;
            n->left = create_node();
            n->right = create_node();
            n->has_split = true;

            for(IndexType i = 0; i < point_vec_size; i++)
            {
                const Float temp = n->points[i][dim];
                if(temp < median)
                {
                    node_add_data(n->left, n->points[i], n->data[i]);
                }
                else
                {
                    node_add_data(n->right, n->points[i], n->data[i]);
                }
            }

            if(point[dim] < median)
            {
                node_add_data(n->left, point, data);
            }
            else
            {
                node_add_data(n->right, point, data);
            }

            // Clear the data from the parent node
            std::vector<VectorType>{}.swap(n->points);
            std::vector<DataType>{}.swap(n->data);
        }
    }

    DataType *find_point(node *current, unsigned int depth, const VectorType &point, Float tolerance)
    {
        // If we got here we know that the point was in this node's bounding box
        DataType *retval = nullptr;

        // This node has children
        if(current->has_split)
        {
            const bool left_contains = current->left->bb.contains(point, tolerance);
            const bool right_contains = current->right->bb.contains(point, tolerance);
            if(!left_contains && !right_contains)
            {
                // ERROR! This shouldn't happen, the tree must've been built improperly
                retval = nullptr;
            }
            else if(left_contains)
            {
                // Traverse left
                retval = find_point(current->left, depth+1, point, tolerance);
            }
            else if(right_contains)
            {
                // Traverse right
                retval = find_point(current->right, depth+1, point, tolerance);
            }
            else // (left_contains && right_contains)
            {
                // Rare, but possible due to tolerance.
                // Check if the left side has the point without tolerance
                const bool pref_left = current->left->bb.contains(point);
                retval = (pref_left)
                    ? find_point(current->left, depth+1, point, tolerance)
                    : find_point(current->right, depth+1, point, tolerance);
                // We tried the preferred side but it didn't contain the point
                if(retval == nullptr)
                {
                    retval = (pref_left)
                        ? find_point(current->right, depth+1, point, tolerance)
                        : find_point(current->left, depth+1, point, tolerance);
                }
            }
        }
        else
        {
            // This is a leaf node.
            const auto t2 = tolerance * tolerance;
            const IndexType N = current->points.size();
            IndexType idx = 0;
            for(idx = 0; idx < N; idx++)
            {
                const auto &p = current->points[idx];
                const auto dist2 = point.distance2(p);
                if(dist2 <= t2)
                {
                    break;
                }
            }

            // Did not find point
            if(idx == N)
            {
                retval = nullptr;
            }
            else
            {
                retval = &current->data[idx];
            }
        }
        return retval;
    }

    void insert(node *current, unsigned int depth, const VectorType &loc, const DataType &r)
    {
        // No matter what we need to add this point to the current bounding box
        current->bb.expand(loc);

        // This node has children
        if(current->has_split)
        {
            const auto dim = current->dim;
            if(loc[dim] < current->split)
            {
                // Go left
                insert(current->left, depth+1, loc, r);
            }
            else // (loc[dim] >= current->split)
            {
                // Go right
                insert(current->right, depth+1, loc, r);
            }
        }
        else
        {
            // This is a leaf node
            // Determine if the node needs to be split
            if((current->points.size()) == point_vec_size)
            {
                // This will add the point and data to the correct child
                node_split(current, loc, r);
                tree_depth = std::max(tree_depth, (IndexType)depth+1);
                npoints++;
            }
            else
            {
                // This node does not need to be split
                node_add_data(current, loc, r);
                npoints++;
            }
        }
    }

    template<typename Func>
    void traverse_lnr(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lnr(func, node->left, depth + 1); }
        func(node, depth);
        if(node->right) { traverse_lnr(func, node->right, depth + 1); }
    }

    template<typename Func>
    void traverse_lrn(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lrn(func, node->left, depth + 1); }
        if(node->right) { traverse_lrn(func, node->right, depth + 1); }
        func(node, depth);
    }

    // Keep track of tree performance
    IndexType npoints{0u};
    IndexType nnodes{0u};
    IndexType tree_depth{0u};

    node *root{nullptr};
    IndexType point_vec_size{32};
    std::vector<IndexType> scratch;
};

using combine_implicit_data_t = std::pair<const Node*, bounding_box<vec3>>;

template<typename InDataArray, typename OutDataArray>
static index_t
copy_node_data_impl2(const InDataArray &in, OutDataArray &out, index_t offset)
{
    index_t out_idx = offset;
    for(index_t i = 0; i < in.number_of_elements(); i++, out_idx++)
    {
        out[out_idx] = in[i];
    }
    return out_idx;
}

template<typename OutDataArray>
static index_t
copy_node_data_impl(const Node &in, OutDataArray &out, index_t offset)
{
    const auto id = in.dtype().id();
    index_t retval = offset;
    switch(id)
    {
    case conduit::DataType::INT8_ID:
    {
        DataArray<int8> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::INT16_ID:
    {
        DataArray<int16> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::INT32_ID:
    {
        DataArray<int32> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::INT64_ID:
    {
        DataArray<int64> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::UINT8_ID:
    {
        DataArray<uint8> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::UINT16_ID:
    {
        DataArray<uint16> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::UINT32_ID:
    {
        DataArray<uint32> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::UINT64_ID:
    {
        DataArray<uint64> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::FLOAT32_ID:
    {
        DataArray<float32> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    case conduit::DataType::FLOAT64_ID:
    {
        DataArray<float64> da = in.value();
        retval = copy_node_data_impl2(da, out, offset);
        break;
    }
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
    return retval;
}

static index_t
copy_node_data(const Node &in, Node &out, index_t offset = 0)
{
    const auto id = out.dtype().id();
    index_t retval = offset;
    switch(id)
    {
    case conduit::DataType::INT8_ID:
    {
        DataArray<int8> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::INT16_ID:
    {
        DataArray<int16> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::INT32_ID:
    {
        DataArray<int32> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::INT64_ID:
    {
        DataArray<int64> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::UINT8_ID:
    {
        DataArray<uint8> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::UINT16_ID:
    {
        DataArray<uint16> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::UINT32_ID:
    {
        DataArray<uint32> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::UINT64_ID:
    {
        DataArray<uint64> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::FLOAT32_ID:
    {
        DataArray<float32> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    case conduit::DataType::FLOAT64_ID:
    {
        DataArray<float64> da = out.value();
        retval = copy_node_data_impl(in, da, offset);
        break;
    }
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
    return retval;
}

template<typename LhsDataArray, typename RhsDataArray>
static bool
node_value_compare_impl2(const LhsDataArray &lhs, const RhsDataArray &rhs, double epsilon)
{
    const index_t nele = lhs.number_of_elements();
    if(nele != rhs.number_of_elements())
    {
        return false;
    }

    bool retval = true;
    for(index_t i = 0; i < nele; i++)
    {
        const double diff = std::abs(lhs[i] - rhs[i]);
        if(!(diff <= epsilon))
        {
            retval = false;
            break;
        }
    }
    return retval;
}

template<typename RhsDataArray>
static bool
node_value_compare_impl(const Node &lhs, const RhsDataArray &rhs,  double epsilon)
{
    const auto id = lhs.dtype().id();
    bool retval = true;
    switch(id)
    {
    case conduit::DataType::INT8_ID:
    {
        DataArray<int8> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::INT16_ID:
    {
        DataArray<int16> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::INT32_ID:
    {
        DataArray<int32> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::INT64_ID:
    {
        DataArray<int64> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::UINT8_ID:
    {
        DataArray<uint8> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::UINT16_ID:
    {
        DataArray<uint16> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::UINT32_ID:
    {
        DataArray<uint32> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::UINT64_ID:
    {
        DataArray<uint64> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::FLOAT32_ID:
    {
        DataArray<float32> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    case conduit::DataType::FLOAT64_ID:
    {
        DataArray<float64> da = lhs.value();
        retval = copy_node_data_impl2(da, rhs, epsilon);
        break;
    }
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
    return retval;
}

static bool
node_value_compare(const Node &lhs, const Node &rhs, double epsilon = CONDUIT_EPSILON)
{
    const auto id = rhs.dtype().id();
    bool retval = true;
    switch(id)
    {
    case conduit::DataType::INT8_ID:
    {
        DataArray<int8> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::INT16_ID:
    {
        DataArray<int16> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::INT32_ID:
    {
        DataArray<int32> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::INT64_ID:
    {
        DataArray<int64> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::UINT8_ID:
    {
        DataArray<uint8> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::UINT16_ID:
    {
        DataArray<uint16> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::UINT32_ID:
    {
        DataArray<uint32> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::UINT64_ID:
    {
        DataArray<uint64> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::FLOAT32_ID:
    {
        DataArray<float32> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    case conduit::DataType::FLOAT64_ID:
    {
        DataArray<float64> da = rhs.value();
        retval = node_value_compare_impl(lhs, da, epsilon);
        break;
    }
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
    return retval;
}

template<typename DataArray_t, typename T>
static index_t
find_rectilinear_offset(const DataArray_t &da, T val, double tolerance = CONDUIT_EPSILON)
{
    // TODO: Binary search? Rectilinear values should be sorted.
    index_t retval = -1;
    for(index_t i = 0; i < da.number_of_elements(); i++)
    {
        const auto diff = val - da[i];
        if(diff <= tolerance)
        {
            retval = i;
            break;
        }
    }
    return retval;
}

static std::vector<index_t>
find_implicit_coordset_offsets(const Node &whole_cset, const Node &sub_cset, double tolerance = CONDUIT_EPSILON)
{
    std::vector<index_t> offsets;
    const std::string wtype = whole_cset["type"].as_string();
    if(wtype == "uniform")
    {
        const auto worigin = mesh::utils::coordset::uniform::origin(whole_cset);
        const auto sorigin = mesh::utils::coordset::uniform::origin(sub_cset);
        const auto spacing = mesh::utils::coordset::uniform::spacing(whole_cset);
        for(size_t i = 0; i < worigin.size(); i++)
        {
            const auto difference = sorigin[i] - worigin[i];
            offsets.push_back(difference / spacing[i]);
        }
    }
    else if(wtype == "rectilinear")
    {
        const auto exts = mesh::utils::coordset::extents(sub_cset);
        const Node &n_values = whole_cset["values"];
        const auto cset_axes = mesh::utils::coordset::axes(whole_cset);
        for(size_t i = 0; i < cset_axes.size(); i++)
        {
            const Node &n_value = n_values[cset_axes[i]];
            if(n_value.dtype().is_float32())
            {
                DataArray<float32> da = n_value.value();
                offsets.push_back(find_rectilinear_offset(da, exts[i*2], tolerance));
            }
            else if(n_value.dtype().is_float64())
            {
                DataArray<float64> da = n_value.value();
                offsets.push_back(find_rectilinear_offset(da, exts[i*2], tolerance));
            }
            else
            {
                CONDUIT_ERROR("Unknown value type for recilinear coordset. " << n_value.dtype().name());
            }
        }
    }
    else
    {
        CONDUIT_ERROR("Non implicit coordset passed to find_implicit_coordset_offsets");
    }
    return offsets;
}

static void
build_implicit_maps(const std::vector<const Node *> &n_coordsets, 
        const Node &final_cset, 
        Node &out_pointmaps,
        Node &out_element_map)
{
    const std::vector<index_t> final_dim_lengths = mesh::utils::coordset::dim_lengths(final_cset);
    auto elem_dims = final_dim_lengths;
    index_t Nelem = 1;
    for(auto &elem_dim : elem_dims)
    {
        elem_dim = elem_dim - 1;
        Nelem = Nelem * elem_dim;
    }

    // Allocate the output_elem_map
    out_element_map.set(DataType::index_t(Nelem*2));
    DataArray<index_t> emap_da = out_element_map.value();

    index_t dom_idx = 0;
    for(const Node *n_cset : n_coordsets)
    {
        const auto this_dim_lengths = mesh::utils::coordset::dim_lengths(*n_cset);
        const index_t N = mesh::utils::coordset::length(*n_cset);
        const DataType dt(DataType::index_t(N));
        Node &pointmap = out_pointmaps.append();
        pointmap.set(dt);
        DataArray<index_t> pmap_da = pointmap.value();
        const auto offsets = find_implicit_coordset_offsets(final_cset, *n_cset);
        if(final_dim_lengths.size() == 3)
        {
            // Do pointmap
            {
                const index_t nx   = final_dim_lengths[0];
                const index_t nxny = nx * final_dim_lengths[1];
                const index_t ioff = offsets[0];
                const index_t joff = offsets[1] * nx;
                const index_t koff = offsets[2] * nxny;
                index_t idx = 0;
                for(index_t k = 0; k < this_dim_lengths[2]; k++)
                {
                    const index_t knxny = koff + k * nxny;
                    for(index_t j = 0; j < this_dim_lengths[1]; j++)
                    {
                        const index_t jnx = joff + j * nx;
                        for(index_t i = 0; i < this_dim_lengths[0]; i++, idx++)
                        {
                            pmap_da[idx] = knxny + jnx + ioff + i;
                        }
                    }
                }
            }

            // Do element_map
            {

                const index_t nx   = elem_dims[0];
                const index_t nxny = nx*elem_dims[1];
                const index_t ioff = offsets[0];
                const index_t joff = offsets[1] * nx;
                const index_t koff = offsets[2] * nxny;
                const index_t this_nx = this_dim_lengths[0]-1;
                const index_t this_nxny = this_nx * (this_dim_lengths[1] - 1);
                for(index_t k = 0; k < this_dim_lengths[2]-1; k++)
                {
                    const index_t this_knxny = k * this_nxny;
                    const index_t knxny = koff + k * nxny;
                    for(index_t j = 0; j < this_dim_lengths[1]-1; j++)
                    {
                        const index_t this_jnx = this_knxny + j * this_nx;
                        const index_t jnx = joff + j * nx;
                        for(index_t i = 0; i < this_dim_lengths[0]-1; i++)
                        {
                            const index_t id  = knxny + jnx + ioff + i;
                            const index_t idx = id * 2;
                            emap_da[idx]   = dom_idx;
                            emap_da[idx+1] = this_jnx + i;
                        }
                    }
                }
            }
        }
        else if(final_dim_lengths.size() == 2)
        {
            // Do pointmap
            {
                const index_t ioff = offsets[0];
                const index_t joff = offsets[1] * final_dim_lengths[0];
                index_t idx = 0;
                for(index_t j = 0; j < this_dim_lengths[1]; j++)
                {
                    const auto jnx = joff + j * final_dim_lengths[0];
                    for(index_t i = 0; i < this_dim_lengths[0]; i++, idx++)
                    {
                        pmap_da[idx] = jnx + ioff + i;
                    }
                }
            }

            // Do element_map
            {
                const index_t nx   = elem_dims[0];
                const index_t ioff = offsets[0];
                const index_t joff = offsets[1] * nx;
                const index_t this_nx = this_dim_lengths[0]-1;
                for(index_t j = 0; j < this_dim_lengths[1]-1; j++)
                {
                    const index_t this_jnx = j * this_nx;
                    const index_t jnx = joff + j * nx;
                    for(index_t i = 0; i < this_dim_lengths[0]-1; i++)
                    {
                        const index_t id  = jnx + ioff + i;
                        const index_t idx = id * 2;
                        emap_da[idx]   = dom_idx;
                        emap_da[idx+1] = this_jnx + i;
                    }
                }
            }
        }
        else // if(dim_lengths.size() == 1)
        {
            // Do pointmap
            {
                const index_t ioff = offsets[0];
                for(index_t i = 0; i < this_dim_lengths[0]; i++)
                {
                    pmap_da[i] = ioff + i;
                }
            }

            // Do element_map
            {
                const index_t ioff = offsets[0];
                for(index_t i = 0; i < this_dim_lengths[0]-1; i++)
                {
                    const index_t id  = ioff + i;
                    const index_t idx = id * 2;
                    emap_da[idx] = dom_idx;
                    emap_da[idx] = i;
                }
            }
        }

        dom_idx++;
    }
}

static const std::vector<std::string> &
figure_out_implicit_axes(const std::vector<const Node *> &n_inputs)
{
    // n_inputs was already checked to be > 1
    // It's possible that n_inputs[0] claims to be logical
    //  but the rest of the inputs exist in a different coordsys.
    // For example if the first input has no explicit origin or spacing
    //  it will report as "logical" but the next guy may report an origin in xyz.
    const std::string &csys0 = mesh::utils::coordset::coordsys(*n_inputs[0]);
    const std::string &csys1 = mesh::utils::coordset::coordsys(*n_inputs[1]);
    return (csys0 == "logical"
        // if csys0 was logical, check csys1 for a non-logical coordsys
        ? (csys1 == "cartesian" ? mesh::utils::CARTESIAN_AXES
            : (csys1 == "cylindrical" ? mesh::utils::CYLINDRICAL_AXES
                : csys1 == "spherical" ? mesh::utils::SPHERICAL_AXES : mesh::utils::LOGICAL_AXES))
        // if csys0 wasn't logical, lookup the proper axes
        : (csys0 == "cartesian" ? mesh::utils::CARTESIAN_AXES
            : (csys0 == "cylindrical" ? mesh::utils::CYLINDRICAL_AXES
                : csys0 == "spherical" ? mesh::utils::SPHERICAL_AXES : mesh::utils::LOGICAL_AXES))
    );
}

static bool
combine_implicit(const std::vector<const Node *> &n_inputs, 
                 double tolerance, Node &output)
{
    output.reset();
    if(n_inputs.size() == 1)
    {
        output = *n_inputs[0];
        return true;
    }

    // Which type of coordset we will be using
    std::string type = "uniform";
    for(size_t i = 0; i < n_inputs.size(); i++)
    {
        const Node &n_input = *n_inputs[i];
        std::string cset_type = n_input["type"].as_string();
        if(cset_type == "explicit")
        {
            type = "explicit";
        }
        else if(type != "explicit" && cset_type == "rectilinear")
        {
            type = "rectilinear";
        }
        // We defaulted to "uniform", so do nothing.
    }

    // Determine which axes labels to use
    const std::vector<std::string> &axes = figure_out_implicit_axes(n_inputs);

    std::vector<Node> temp_nodes;
    std::vector<const Node*> n_coordsets;
    index_t dimension = dims(*n_inputs[0]);
    if(type == "uniform")
    {
        // Inspect input[0] for baseline spacing/dimension
        const auto baseline_spacing = mesh::utils::coordset::uniform::spacing(*n_inputs[0]);
        n_coordsets.push_back(n_inputs[0]);
        for(size_t i = 1; i < n_inputs.size(); i++)
        {
            const Node &n_input = *n_inputs[i];
            if(dimension != dims(n_input))
            {
                type = "explicit";
                break;
            }

            // Get spacing for this domain
            const auto spacing = mesh::utils::coordset::uniform::spacing(n_input);
            // Check that spacing matches
            for(index_t d = 0; d < dimension; d++)
            {
                // If spacing doesn't match try to do rectilinear
                if(spacing[d] != baseline_spacing[d])
                {
                    type = "rectilinear";
                    break;
                }
            }

            if(type != "uniform")
            {
                break;
            }

            n_coordsets.push_back(n_inputs[i]);
        }

        // If we are able to continue merging these as uniform
        if(type == "uniform")
        {
            bool needs_spacing = false;
            for(const auto s : baseline_spacing)
            {
                if(s != 1.) needs_spacing = true;
            }
            if(needs_spacing)
            {
                Schema s;
                for(index_t d = 0; d < dimension; d++)
                {
                    s["d"+axes[d]].set(DataType::c_double(1, d*sizeof(double), dimension*sizeof(double)));
                }
                output["spacing"].set(s);
                for(index_t d = 0; d < dimension; d++)
                {
                    output["spacing/d"+axes[d]].set(baseline_spacing[d]);
                }
            }
        }
    }

    // Must convert uniform to rectilinear before continuing
    if(type == "rectilinear")
    {
        n_coordsets.clear();
        temp_nodes.reserve(n_inputs.size());
        for(size_t i = 0; i < n_inputs.size(); i++)
        {
            const Node &n_input = *n_inputs[i];
            if(dimension != dims(n_input))
            {
                type = "explicit";
                break;
            }

            std::string cset_type = n_input["type"].as_string();
            if(cset_type == "uniform")
            {
                temp_nodes.emplace_back();
                mesh::coordset::uniform::to_rectilinear(n_input, temp_nodes.back());
                n_coordsets.push_back(&temp_nodes.back());
            }
            else
            {
                n_coordsets.push_back(n_inputs[i]);
            }
        }
    }

    // No support for structured grids yet
    if(type == "explicit")
    {
        return false;
    }
    
    // std::cout << "Passed preliminary check! In mode " << type << std::endl;

    // Make sure extents matchup in the correct way
    std::vector<combine_implicit_data_t> csets_and_bbs;
    for(const Node *n_cset : n_coordsets)
    {
        auto extents = mesh::utils::coordset::extents(*n_cset);
        bounding_box<vec3> bb;
        for(index_t d = 0; d < dimension; d++)
        {
            const index_t ext_idx = d*2;
            bb.min[d] = extents[ext_idx];
            bb.max[d] = extents[ext_idx+1];
        }
        csets_and_bbs.push_back({{n_cset}, {bb}});
    }

    // Match and combine edges/planes on coordest boundaries until we have 1 left
    Node n_temporary_csets;
    index_t iteration = 0;
    while(csets_and_bbs.size() > 1)
    {
        // Print the work in progress
        // std::cout << "iteration " << iteration << "\n";
    #if 0
        for(size_t ei = 0; ei < csets_and_bbs.size(); ei++)
        {
            // const Node *n = csets_and_bbs[ei].first;
            const auto &bb = csets_and_bbs[ei].second;
            std::cout << "  " << ei << ": min[";
            for(index_t d = 0; d < dimension; d++)
            {
                std::cout << bb.min[d] << (d == (dimension - 1) ? "] " : ", ");
            }
            std::cout << "  " << ei << ": max[";
            for(index_t d = 0; d < dimension; d++)
            {
                std::cout << bb.max[d] << (d == (dimension - 1) ? "] " : ", ");
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    #elif 0
        output.print();
    #elif 0
        for(size_t ei = 0; ei < csets_and_bbs.size(); ei++)
        {
            std::cout << "[" << ei << "]" << std::endl;
            csets_and_bbs[ei].first->print();        
        }
        std::cout << std::endl;
    #else
    #endif
        iteration++;

        // Get the first extents
        bool any_matches = false;
        for(size_t ei = 0; ei < csets_and_bbs.size(); ei++)
        {
            const Node *n_cseti = csets_and_bbs[ei].first;
            auto &exti = csets_and_bbs[ei].second;

            // Find a match
            const index_t NOT_FOUND = csets_and_bbs.size();
            index_t matched_extents = NOT_FOUND;
            for(size_t ej = ei+1; ej < csets_and_bbs.size(); ej++)
            {
                const Node *n_csetj = csets_and_bbs[ej].first;
                const auto &extj = csets_and_bbs[ej].second;

                for(index_t di = 0; di < dimension; di++)
                {
                    // First check if the end of one domain touch the start of another
                    const bool check1 = std::abs(exti.max[di] - extj.min[di]) <= tolerance;
                    const bool check2 = std::abs(exti.min[di] - extj.max[di]) <= tolerance;
                    if(!check1 && !check2)
                    {
                        continue;
                    }
                    // Now check that the extents of the touching domains match in the other dimensions
                    bool corners_match = true;
                    for(index_t dj = 0; dj < dimension; dj++)
                    {
                        if(dj == di) { continue; }
                        // All other axis extents should be equal
                        const bool check3 = std::abs(exti.min[dj] - extj.min[dj]) <= tolerance;
                        const bool check4 = std::abs(exti.max[dj] - extj.max[dj]) <= tolerance;
                        if(!check3 || !check4)
                        {
                            corners_match = false;
                            break;
                        }
                    }
                    // If the corners match combine them
                    if(corners_match)
                    {
                        Node &new_cset = n_temporary_csets.append();
                        new_cset["type"] = type;
                        if(type == "uniform")
                        {
                            // std::cout << "Handling uniform combine" << std::endl;
                            std::vector<double> spacing = {1, 1, 1};
                            if(output.has_child("spacing"))
                            {
                                new_cset["spacing"] = output["spacing"];
                                for(index_t dj = 0; dj < dimension; dj++)
                                {
                                    spacing[dj] = output["spacing"][dj].to_double();
                                }
                            }

                            Schema s_origin;
                            Schema s_dims;
                            for(index_t dj = 0; dj < dimension; dj++)
                            {
                                s_origin[axes[dj]].set(DataType::c_double(1, dj*sizeof(double), dimension*sizeof(double)));
                                s_dims[mesh::utils::LOGICAL_AXES[dj]].set(DataType::index_t(1, dj*sizeof(index_t), dimension*sizeof(index_t)));
                            }
                            new_cset["origin"].set(s_origin);
                            new_cset["dims"].set(s_dims);

                            // Update the extents
                            exti.min[di] = std::min(exti.min[di], extj.min[di]);
                            exti.max[di] = std::max(exti.max[di], extj.max[di]);

                            for(index_t dj = 0; dj < dimension; dj++)
                            {
                                const std::string dims_path = "dims/"+mesh::utils::LOGICAL_AXES[dj];
                                new_cset["origin/"+axes[dj]] = exti.min[dj];
                                // Extents are inclusive so we need to add 1 to the difference
                                new_cset[dims_path] = (index_t)(((exti.max[dj] - exti.min[dj]) / spacing[dj]) + 1.5);
                            }
                            csets_and_bbs[ei].first = &new_cset;
                            csets_and_bbs.erase(csets_and_bbs.begin() + ej);
                            matched_extents = ej;
                            break;
                        }
                        else if(type == "rectilinear")
                        {
                            // std::cout << "Handling rectilinear combine" << std::endl;
                            // We need to further check that the spacing along the matched edge/plane is okay
                            bool ok = true;
                            index_t dim_sizes[3];
                            index_t max_bytes = 0;
                            for(index_t dj = 0; dj < dimension; dj++)
                            {
                                const Node &n_vals0 = (*n_cseti)["values/"+axes[dj]];
                                const Node &n_vals1 = (*n_csetj)["values/"+axes[dj]];
                                max_bytes = std::max(max_bytes, n_vals0.dtype().element_bytes());
                                max_bytes = std::max(max_bytes, n_vals1.dtype().element_bytes());
                                if(di == dj)
                                {
                                    dim_sizes[dj] = n_vals0.dtype().number_of_elements() + n_vals1.dtype().number_of_elements() - 1;
                                }
                                else
                                {
                                    dim_sizes[dj] = n_vals0.dtype().number_of_elements();
                                    ok = node_value_compare(n_vals0, n_vals1);
                                    if(!ok)
                                    {
                                        std::cout << "Incompatible rectilinear domains" << std::endl;
                                        break;
                                    }
                                }
                            }

                            // std::cout << "ok? " << ok << std::endl;

                            if(!ok)
                            {
                                n_temporary_csets.remove(n_temporary_csets.number_of_children() - 1);
                                matched_extents = NOT_FOUND;
                                break;
                            }

                            // std::cout << "Made it to the heavy lifting!" << std::endl;

                            // Update the extents
                            exti.min[di] = std::min(exti.min[di], extj.min[di]);
                            exti.max[di] = std::max(exti.max[di], extj.max[di]);

                            // Allocate the output arrays
                            const DataType out_dtype((max_bytes < 8) ? DataType::c_float() : DataType::c_double());
                            index_t offset = 0;
                            Schema s;
                            for(index_t dj = 0; dj < dimension; dj++)
                            {
                                s[axes[dj]].set(DataType(out_dtype.id(), dim_sizes[dj], offset, 
                                    out_dtype.element_bytes(), out_dtype.element_bytes(), out_dtype.endianness()));
                                offset += out_dtype.element_bytes() * dim_sizes[dj];
                            }
                            new_cset["values"].set(s);
                            std::array<const Node*, 2> n_in_di{
                                ((check1) ? n_cseti->fetch_ptr("values/"+axes[di]) : n_csetj->fetch_ptr("values/"+axes[di])),
                                ((check1) ? n_csetj->fetch_ptr("values/"+axes[di]) : n_cseti->fetch_ptr("values/"+axes[di]))
                            };
                            for(index_t dj = 0; dj < dimension; dj++)
                            {
                                Node &n_out_values = new_cset["values/"+axes[dj]];
                                if(di == dj)
                                {
                                    index_t out_idx = 0;
                                    for(size_t i = 0; i < n_in_di.size(); i++, out_idx--)
                                    {
                                        const Node &n_in_values = *n_in_di[i];
                                        out_idx = copy_node_data(n_in_values, n_out_values, out_idx);
                                    }
                                }
                                else
                                {
                                    const Node &n_vals0 = (*n_cseti)["values/"+axes[dj]];
                                    copy_node_data(n_vals0, n_out_values);
                                }
                            }

                            csets_and_bbs[ei].first = &new_cset;
                            csets_and_bbs.erase(csets_and_bbs.begin() + ej);
                            matched_extents = ej;
                            break;
                        }
                    }
                }

                if(matched_extents != NOT_FOUND)
                {
                    any_matches = true;
                    break;
                }
            }
        }

        if(any_matches == false)
        {
            break;
        }
    }
    // std::cout << "REMAINING DOMAINS " << csets_and_bbs.size() << std::endl;
    bool retval = false;
    if(csets_and_bbs.size() == 1)
    {
        // TODO: Figure out how to move the data out of the temporary node
        //  instead of deep copying.
        output = *csets_and_bbs[0].first;
        build_implicit_maps(n_coordsets, output, output["pointmaps"], output["element_map"]);
        retval = true;
    }
    return retval;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset::utils --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
point_merge::execute(const std::vector<const Node *> &coordsets,
                     double tolerance,
                     Node &output)
{
    if(coordsets.empty())
        return;

    if(coordsets.size() == 1)
    {
        if(coordsets[0] != nullptr)
        {
            output.reset();
            output["coordsets/coords"] = *coordsets[0];
        }
        return;
    }

    // What do we need to know about the coordsets before we execute the algorithm
    //  - Systems
    //  - Types
    //  - Dimension
    std::vector<Node> working_sets;
    std::vector<coord_system> systems;
    std::vector<std::vector<float64>> extents;
    index_t ncartesian = 0, ncylindrical = 0, nspherical = 0, nlogical = 0;
    index_t dimension = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const Node *cset = coordsets[i];
        if(!cset)
        {
            // ERROR! You passed me a nullptr for a coordset!
            continue;
        }

        if(!cset->has_child("type"))
        {
            // Error! Invalid coordset!
            continue;
        }
        const std::string type = cset->child("type").as_string();

        // Track some information about each set
        dimension = std::max(dimension, mesh::utils::coordset::dims(*cset));
        extents.push_back(mesh::utils::coordset::extents(*cset));

        // Translate coordsystem string to enum
        std::string system = mesh::utils::coordset::coordsys(*cset);
        if(system == "cylindrical")
        {
            ncylindrical++;
            systems.push_back(coord_system::cylindrical);
        }
        else if(system == "spherical")
        {
            nspherical++;
            systems.push_back(coord_system::spherical);
        }
        else if(system == "logical")
        {
            nlogical++;
            systems.push_back(coord_system::logical);
        }
        else // system == cartesian
        {
            ncartesian++;
            systems.push_back(coord_system::cartesian);
        }

        // We only work on explicit sets
        working_sets.emplace_back();
        if(type == "uniform")
        {
            mesh::coordset::uniform::to_explicit(*cset, working_sets.back());
        }
        else if(type == "rectilinear")
        {
            mesh::coordset::rectilinear::to_explicit(*cset, working_sets.back());
        }
        else // type == "explicit"
        {
            working_sets.back() = *cset;
        }
    }

    // Determine best output coordinate system
    // Prefer cartesian, if they are all the same use that coordinate system
    if(ncartesian > 0 || (ncylindrical > 0 && nspherical > 0))
    {
        out_system = coord_system::cartesian;
    }
    else if(nspherical > 0)
    {
        out_system = coord_system::spherical;
    }
    else if(ncylindrical > 0)
    {
        out_system = coord_system::cylindrical;
    }
    else
    {
        // Error! Unhandled case!
        std::cerr << "UNHANDLED CASE " << ncartesian << " " << ncylindrical << " " << nspherical << std::endl;
        return;
    }

    int noverlapping_sets = examine_extents(extents);
    PM_DEBUG_PRINT("I was given " << coordsets.size() << " I am combining " << working_sets.size() << std::endl);
    PM_DEBUG_PRINT("noverlapping sets: " << noverlapping_sets << std::endl);
    if(noverlapping_sets == 0)
    {
        // We don't need to do any kind of merging
        append_data(working_sets, systems, dimension);
    }
    else
    {
        merge_data(working_sets, systems, dimension, tolerance);
    }

    create_output(dimension, output);
}

//-----------------------------------------------------------------------------
int
point_merge::examine_extents(std::vector<std::vector<float64>> &extents) const
{
    const auto overlap = [](const float64 box1[6], const float64 box2[6]) -> bool {
        bool retval = true;
        for(auto i = 0u; i < 3u; i++)
        {
            const auto idx  = i * 2;
            const auto min1 = box1[idx];
            const auto max1 = box1[idx+1];
            const auto min2 = box2[idx];
            const auto max2 = box2[idx+1];
            retval &= (max1 >= min2 && max2 >= min1);
        }
        return retval;
    };

    int retval = 0;
    for(auto i = 0u; i < extents.size(); i++)
    {
        float64 box1[6] = {0.,0.,0.,0.,0.,0.};
        const auto &ext1 = extents[i];
        for(auto j = 0u; j < ext1.size(); j++)
        {
            box1[j] = ext1[j];
        }

        for(auto j = 0u; j < extents.size(); j++)
        {
            if(i == j) { continue; }
            const auto ext2 = extents[j];
            float64 box2[6] = {0.,0.,0.,0.,0.,0.};
            for(auto k = 0u; k < ext2.size(); k++)
            {
                box2[k] = ext2[k];
            }

            retval += overlap(box1, box2);
        }
    }
    return retval;
}

//-----------------------------------------------------------------------------
void
point_merge::append_data(const std::vector<Node> &coordsets,
    const std::vector<coord_system> &systems, index_t dimension)
{
    reserve_vectors(coordsets, dimension);

    index_t newid = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto append = [&](float64 *p, index_t)
        {
            old_to_new_ids[i].push_back(newid);
            for(auto i = 0; i < dimension; i++)
            {
                new_coords.push_back(p[i]);
            }
            newid++;
        };

        const auto translate_append = [&](float64 *p, index_t d) {
            translate_system(systems[i], out_system,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            append(p, d);
        };


        // Invoke the proper lambda on each coordinate
        if(systems[i] != out_system
            && systems[i] != coord_system::logical)
        {
            iterate_coordinates(coordsets[i], translate_append);
        }
        else
        {
            iterate_coordinates(coordsets[i], append);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
#define USE_SPATIAL_SEARCH_MERGE
#if   defined(USE_TRUNCATE_PRECISION_MERGE)
    truncate_merge(coordsets, systems, dimension, tolerance);
#elif defined(USE_SPATIAL_SEARCH_MERGE)
    spatial_search_merge(coordsets, systems, dimension, tolerance);
#else
    simple_merge_data(coordsets, systems, dimension, tolerance);
#endif
#undef USE_SPATIAL_SEARCH_MERGE
}

//-----------------------------------------------------------------------------
void
point_merge::create_output(index_t dimension, Node &output) const
{
    if(dimension < 0 || dimension > 3)
    {
        // ERROR! Invalid dimension!
        return;
    }

    output.reset();

    // Add the new coordset
    {
        output["type"] = "explicit";
        auto &values = output.add_child("values");

        // Define the node schema for coords
        Schema s;
        const auto npoints = new_coords.size() / dimension;
        const index_t stride = sizeof(float64) * dimension;
        const index_t size = sizeof(float64);
        const auto &axes = get_axes_for_system(out_system);
        for(auto i = 0; i < dimension; i++)
        {
            s[axes[i]].set(DataType::float64(npoints,i*size,stride));
        }

        // Copy out coordinate values
        values.set(s);
        float64_array coord_arrays[3];
        for(auto i = 0; i < dimension; i++)
        {
            coord_arrays[i] = values[axes[i]].value();
        }

        index_t point_id = 0;
        for(auto itr = new_coords.begin(); itr != new_coords.end();)
        {
            for(auto d = 0; d < dimension; d++)
            {
                coord_arrays[d][point_id] = *itr++;
            }
            point_id++;
        }
    }

    // Add the pointmaps
    {
    #ifdef DEBUG_POINT_MERGE
        std::cout << "ID MAP SIZES:" << std::endl;
        for(const auto &idmap : old_to_new_ids)
        {
            std::cout << "  " << idmap.size() << std::endl;
        }
    #endif
        auto &pointmaps = output["pointmaps"];
        for(const auto &idmap : old_to_new_ids)
        {
            const auto size = idmap.size();
            // Create the list entry
            auto &ids = pointmaps.append();
            ids.set(DataType::index_t(size));
            // Copy the contents into the node
            DataArray<index_t> ids_data = ids.value();
            for(size_t i = 0u; i < size; i++)
            {
                ids_data[i] = idmap[i];
            }
        }
    }
}

//-----------------------------------------------------------------------------
template<typename Func>
void
point_merge::iterate_coordinates(const Node &coordset, Func &&func)
{
    if(!coordset.has_child("type"))
    {
        CONDUIT_ERROR("Coordset does not have a type");
        return;
    }

    if(coordset["type"].as_string() != "explicit")
    {
        CONDUIT_ERROR("Coordset is not explicit");
        return;
    }

    if(!coordset.has_child("values"))
    {
        CONDUIT_ERROR("Coordset does not have values");
        return;
    }

    const Node &coords = coordset["values"];

    // Fetch the nodes for the coordinate values
    const Node *xnode = coords.fetch_ptr("x");
    const Node *ynode = nullptr, *znode = nullptr;
    if(xnode)
    {
        // Cartesian
        ynode = coords.fetch_ptr("y");
        znode = coords.fetch_ptr("z");
    }
    else if((xnode = coords.fetch_ptr("r")))
    {
        if((ynode = coords.fetch_ptr("z")))
        {
            // Cylindrical
        }
        else if((ynode = coords.fetch_ptr("theta")))
        {
            // Spherical
            znode = coords.fetch_ptr("phi");
        }
    }
    else if(((xnode = coords.fetch_ptr("i"))))
    {
        // Logical
        ynode = coords.fetch_ptr("j");
        znode = coords.fetch_ptr("k");
    }

    // Iterate accordingly
    float64 p[3] {0., 0., 0.};
    if(xnode && ynode && znode)
    {
        // 3D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        const auto ztype = znode->dtype();
        if(xtype.is_float32() && ytype.is_float32() && ztype.is_float32())
        {
            auto xarray = xnode->as_float32_array();
            auto yarray = ynode->as_float32_array();
            auto zarray = znode->as_float32_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = yarray[i]; p[2] = zarray[i];
                func(p, 3);
            }
        }
        else if(xtype.is_float64() && ytype.is_float64() && ztype.is_float64())
        {
            auto xarray = xnode->as_float64_array();
            auto yarray = ynode->as_float64_array();
            auto zarray = znode->as_float64_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = yarray[i]; p[2] = zarray[i];
                func(p, 3);
            }
        }
        else
        {
            Node xtemp, ytemp, ztemp;
            const DataType xdt = DataType(xtype.id(), 1);
            const DataType ydt = DataType(ytype.id(), 1);
            const DataType zdt = DataType(ztype.id(), 1);
            const index_t N = xtype.number_of_elements();
            for(index_t  i = 0; i < N; i++)
            {
                xtemp.set_external(xdt, const_cast<void*>(xnode->element_ptr(i)));
                ytemp.set_external(ydt, const_cast<void*>(ynode->element_ptr(i)));
                ztemp.set_external(zdt, const_cast<void*>(znode->element_ptr(i)));
                p[0] = xtemp.to_float64();
                p[1] = ytemp.to_float64();
                p[2] = ztemp.to_float64();
                func(p, 3);
            }
        }
    }
    else if(xnode && ynode)
    {
        // 2D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        if(xtype.is_float32() && ytype.is_float32())
        {
            auto xarray = xnode->as_float32_array();
            auto yarray = ynode->as_float32_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = yarray[i]; p[2] = 0.;
                func(p, 3);
            }
        }
        else if(xtype.is_float64() && ytype.is_float64())
        {
            auto xarray = xnode->as_float64_array();
            auto yarray = ynode->as_float64_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = yarray[i]; p[2] = 0.;
                func(p, 2);
            }
        }
        else
        {
            Node xtemp, ytemp;
            const DataType xdt = DataType(xtype.id(), 1);
            const DataType ydt = DataType(ytype.id(), 1);
            const index_t N = xtype.number_of_elements();
            for(index_t  i = 0; i < N; i++)
            {
                xtemp.set_external(xdt, const_cast<void*>(xnode->element_ptr(i)));
                ytemp.set_external(ydt, const_cast<void*>(ynode->element_ptr(i)));
                p[0] = xtemp.to_float64();
                p[1] = ytemp.to_float64();
                p[2] = 0.;
                func(p, 2);
            }
        }
    }
    else if(xnode)
    {
        // 1D
        const auto xtype = xnode->dtype();
        if(xtype.is_float32())
        {
            auto xarray = xnode->as_float32_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = 0.; p[2] = 0.;
                func(p, 1);
            }
        }
        else if(xtype.is_float64())
        {
            auto xarray = xnode->as_float64_array();
            const index_t N = xarray.number_of_elements();
            for(index_t i = 0; i < N; i++)
            {
                p[0] = xarray[i]; p[1] = 0.; p[2] = 0.;
                func(p, 1);
            }
        }
        else
        {
            Node xtemp;
            const DataType xdt = DataType(xtype.id(), 1);
            const index_t N = xtype.number_of_elements();
            for(index_t  i = 0; i < N; i++)
            {
                xtemp.set_external(xdt, const_cast<void*>(xnode->element_ptr(i)));
                p[0] = xtemp.to_float64();
                p[1] = 0.;
                p[2] = 0.;
                func(p, 1);
            }
        }
    }
    else
    {
        CONDUIT_ERROR("No valid node values found.");
    }
}

//-----------------------------------------------------------------------------
index_t
point_merge::reserve_vectors(const std::vector<Node> &coordsets, index_t dimension)
{
    old_to_new_ids.reserve(coordsets.size());
    index_t new_size = 0;
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const Node *values = coordsets[i].fetch_ptr("values");
        index_t npts = 0;
        if(values)
        {
            const Node *xnode = values->fetch_ptr("x");
            if(!xnode)
            {
                xnode = values->fetch_ptr("r");
            }
            if(!xnode)
            {
                xnode = values->fetch_ptr("i");
            }

            if(xnode)
            {
                npts = xnode->dtype().number_of_elements();
            }
        #ifdef DEBUG_POINT_MERGE
            std::cout << "coordset " << i << " ";
            std::cout << npts << std::endl;
            coordsets[i].print();
        #endif
        }

        old_to_new_ids.push_back({});
        old_to_new_ids.back().reserve(npts);
        new_size += npts*dimension;
    }
    new_coords.reserve(new_size);
    return new_size;
}

//-----------------------------------------------------------------------------
void
point_merge::simple_merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
    PM_DEBUG_PRINT("Simple merging!" << std::endl);
    reserve_vectors(coordsets, dimension);

    const auto t2 = (tolerance*tolerance);
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const index_t end_check = (index_t)new_coords.size();
        const Node &coordset = coordsets[i];
        auto &idmap = old_to_new_ids[i];

        // To be invoked on each coordinate
        const auto merge = [&](float64 *p, index_t) {
            for(index_t idx = 0; idx < end_check; idx += dimension)
            {
                float64 dist2 = 0.;
                for(index_t d = 0; d < dimension; d++)
                {
                    const auto diff = p[d] - new_coords[idx+d];
                    dist2 += (diff*diff);
                }

                // Within tolerance!
                if(dist2 < t2)
                {
                    // idx / dim should truncate to the proper id
                    idmap.push_back(idx / dimension);
                    return;
                }
            }

            // Did not find another point within tolerance
            const auto newid = (index_t)(new_coords.size() / dimension);
            idmap.push_back(newid);
            for(index_t d = 0; d < dimension; d++)
            {
                new_coords.push_back(p[d]);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], coord_system::cartesian,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != coord_system::cartesian
            && systems[i] != coord_system::logical)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::spatial_search_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance)
{
    PM_DEBUG_PRINT("Spatial search merging!" << std::endl);
    reserve_vectors(coordsets, dimension);

    using namespace utils;
    kdtree<vec3, index_t> point_records;
    point_records.set_bucket_size(32);
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto &coordset = coordsets[i];

        // To be invoked on every coordinate
        const auto merge = [&](float64 *p, index_t) {
            vec3 key;
            key.v[0] = p[0]; key.v[1] = p[1]; key.v[2] = p[2];
            const auto potential_id = new_coords.size() / dimension;
            const index_t *existing_id = point_records.find_point(key, tolerance);
            // Point wasn't already in the tree, insert it
            if(!existing_id)
            {
                old_to_new_ids[i].push_back(potential_id);
                for(index_t d = 0; d < dimension; d++)
                {
                    new_coords.push_back(p[d]);
                }
                // Store the point in the tree so we can look it up later
                point_records.insert(key, potential_id);
            }
            else
            {
                // Point already existed in the tree, reference the known id
                old_to_new_ids[i].push_back(*existing_id);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], coord_system::cartesian,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != coord_system::cartesian
            && systems[i] != coord_system::logical)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }

    PM_DEBUG_PRINT("Number of points in tree " << point_records.size()
        << ", depth of tree " << point_records.depth()
        << ", nodes in tree " << point_records.nodes() << std::endl);
}

//-----------------------------------------------------------------------------
void
point_merge::truncate_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
    PM_DEBUG_PRINT("Truncate merging!" << std::endl);
    // Determine what to scale each value by
    // TODO: Be dynamic
    (void)tolerance;
    double scale = 0.;
    {
        auto decimal_places = 4u;
        static const std::array<double, 7u> lookup = {
            1.,
            (2u << 4),
            (2u << 7),
            (2u << 10),
            (2u << 14),
            (2u << 17),
            (2u << 20)
        };
        if(decimal_places < lookup.size())
        {
            scale = lookup[decimal_places];
        }
        else
        {
            scale = lookup[6];
        }
    }

    /*index_t size = */reserve_vectors(coordsets, dimension);

    // Iterate each of the coordinate sets
    using fp_type = int64;
    using tup = std::tuple<fp_type, fp_type, fp_type>;
    std::map<tup, index_t> point_records;

    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto &coordset = coordsets[i];

        // To be invoked on every coordinate
        const auto merge = [&](float64 *p, index_t) {
            tup key = std::make_tuple(
                static_cast<fp_type>(std::round(p[0] * scale)),
                static_cast<fp_type>(std::round(p[1] * scale)),
                static_cast<fp_type>(std::round(p[2] * scale)));
            auto res = point_records.insert({key, {}});
            if(res.second)
            {
                const index_t id = (index_t)(new_coords.size() / dimension);
                res.first->second = id;
                old_to_new_ids[i].push_back(id);
                for(index_t j = 0; j < dimension; j++)
                {
                    new_coords.push_back(p[j]);
                }
            }
            else
            {
                old_to_new_ids[i].push_back(res.first->second);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], out_system,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != out_system
            && systems[i] != coord_system::logical)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}

//-----------------------------------------------------------------------------
void
point_merge::xyz_to_rtp(double x, double y, double z, double &out_r, double &out_t, double &out_p)
{
    const auto r = std::sqrt(x*x + y*y + z*z);
    out_r = r;
    out_t = std::acos(r / z);
    out_p = std::atan(y / x);
}

//-----------------------------------------------------------------------------
void
point_merge::rtp_to_xyz(double r, double t, double p, double &out_x, double &out_y, double &out_z)
{
    out_x = r * std::cos(p) * std::sin(t);
    out_y = r * std::sin(p) * std::sin(t);
    out_z = r * std::cos(t);
}

//-----------------------------------------------------------------------------
void
point_merge::translate_system(coord_system in_system, coord_system out_system,
        float64 p0, float64 p1, float64 p2,
        float64 &out_p0, float64 &out_p1, float64 &out_p2)
{
    // TODO: Handle rz
    switch(out_system)
    {
    case coord_system::cartesian:
        switch(in_system)
        {
        case coord_system::cylindrical:
            // TODO
            break;
        case coord_system::spherical:
            rtp_to_xyz(p0,p1,p2, out_p0,out_p1,out_p2);
            break;
        default:    // case coord_system:: cartesian Can't happen
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    case coord_system::cylindrical:
        switch(in_system)
        {
        case coord_system::cartesian:
            // TODO
            break;
        case coord_system::spherical:
            // TODO
            break;
        default:
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    case coord_system::spherical:
        switch(in_system)
        {
        case coord_system::cartesian:
            xyz_to_rtp(p0,p1,p2, out_p0,out_p1,out_p2);
            break;
        case coord_system::cylindrical:
            // TODO
            break;
        default:
            out_p0 = p0;
            out_p1 = p1;
            out_p2 = p2;
            break;
        }
        break;
    default:
        out_p0 = p0;
        out_p1 = p1;
        out_p2 = p2;
        break;
    }
}

//-----------------------------------------------------------------------------
const std::vector<std::string> &
point_merge::get_axes_for_system(coord_system cs)
{
    const std::vector<std::string> *retval = &mesh::utils::CARTESIAN_AXES;
    switch(cs)
    {
    case coord_system::cylindrical:
        retval = &mesh::utils::CYLINDRICAL_AXES;
        break;
    case coord_system::spherical:
        retval = &mesh::utils::SPHERICAL_AXES;
        break;
    case coord_system::cartesian: // Do nothing
        break;
    default:
        break;
    }
    return *retval;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::coordset --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------
namespace topology
{

struct entity
{
    utils::ShapeType                  shape;
    // utils::ShapeCascade               cascade; // User can make the cascade if they want using the shape
    std::vector<index_t>              element_ids;
    std::vector<std::vector<index_t>> subelement_ids;
    index_t                           entity_id; // Local entity id.
};

// static const std::vector<std::string> TOPO_SHAPES = {"point", "line", "tri", "quad", "tet", "hex", "polygonal", "polyhedral"};
// Q: Why doesn't this exist in conduit_blueprint_mesh_utils.hpp ?
enum class ShapeId : index_t
{
    Vertex     = 0,
    Line       = 1,
    Tri        = 2,
    Quad       = 3,
    Tet        = 4,
    Hex        = 5,
    Polygonal  = 6,
    Polyhedral = 7
};

template<typename Func>
static void iterate_elements(const Node &topo, Func &&func)
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

    int case_num = -1;
    // Determine case number
    {
        const Node *shape = topo.fetch_ptr("elements/shape");
        if(shape)
        {
            // This is a single shape topology
            const utils::ShapeType st(shape->as_string());
            if(!st.is_valid())
            {
                CONDUIT_ERROR("Invalid topology passed to iterate_elements.");
                return;
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
                return;
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
    }
    if(case_num < 0)
    {
        CONDUIT_ERROR("Could not figure out the type of toplogy passed to iterate elements.");
        return;
    }

    // Define the lambda functions to be invoked for each topology type
    const auto traverse_fixed_elements = [&func](const Node &eles, const utils::ShapeType &shape, index_t &ent_id) {
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
    };

    const auto traverse_polygonal_elements = [&func](const Node &elements, index_t &ent_id) {
        entity e;
        e.shape = utils::ShapeType((index_t)ShapeId::Polygonal);
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
    };

    const auto traverse_polyhedral_elements = [&func](const Node &elements, const Node &subelements, index_t &ent_id) {
        entity e;
        e.shape = utils::ShapeType((index_t)ShapeId::Polyhedral);
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
    };

    using id_elem_pair =  std::pair<index_t, entity>;
    const auto build_element_vector = [](const Node &element_types, std::vector<id_elem_pair> &eles)
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
            const utils::ShapeType shape(n["shape"].as_string());
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
    };

    index_t ent_id = 0;
    switch(case_num)
    {
    case 0:
    {
        utils::ShapeType shape(topo);
        if(shape.is_polyhedral())
        {
            traverse_polyhedral_elements(topo["elements"], topo["subelements"], ent_id);
        }
        else if(shape.is_polygonal())
        {
            traverse_polygonal_elements(topo["elements"], ent_id);
        }
        else // (known celltype case)
        {
            traverse_fixed_elements(topo["elements"], shape, ent_id);
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
                traverse_polyhedral_elements(bucket, subbucket, ent_id);
            }
            else if(shape.is_polygonal())
            {
                traverse_polygonal_elements(bucket, ent_id);
            }
            else
            {
                traverse_fixed_elements(bucket, shape, ent_id);
            }
        }
        break;
    }
    case 3: /* Fallthrough */
    case 4:
    {
        // Stream with element counts or offsets
        const Node &elements = topo["elements"];
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
        break;
    }
    default:
        CONDUIT_ERROR("Unsupported topology passed to iterate_elements")
        return;
    }
}

//-----------------------------------------------------------------------------
template<typename T, typename Func>
static void iterate_int_data_impl(const conduit::Node &node, Func &&func)
{
    conduit::DataArray<T> int_da = node.value();
    const index_t nele = int_da.number_of_elements();
    for(index_t i = 0; i < nele; i++)
    {
        func((index_t)int_da[i]);
    }
}

//-----------------------------------------------------------------------------
template<typename Func>
static void iterate_int_data(const conduit::Node &node, Func &&func)
{
    const auto id = node.dtype().id();
    switch(id)
    {
    case conduit::DataType::INT8_ID:
        iterate_int_data_impl<int8>(node, func);
        break;
    case conduit::DataType::INT16_ID:
        iterate_int_data_impl<int16>(node, func);
        break;
    case conduit::DataType::INT32_ID:
        iterate_int_data_impl<int32>(node, func);
        break;
    case conduit::DataType::INT64_ID:
        iterate_int_data_impl<int64>(node, func);
        break;
    case conduit::DataType::UINT8_ID:
        iterate_int_data_impl<uint8>(node, func);
        break;
    case conduit::DataType::UINT16_ID:
        iterate_int_data_impl<uint16>(node, func);
        break;
    case conduit::DataType::UINT32_ID:
        iterate_int_data_impl<uint32>(node, func);
        break;
    case conduit::DataType::UINT64_ID:
        iterate_int_data_impl<uint64>(node, func);
        break;
    default:
        CONDUIT_ERROR("Tried to iterate " << conduit::DataType::id_to_name(id) << " as integer data!");
        break;
    }
}

//-----------------------------------------------------------------------------
static void
build_unstructured_output(const std::vector<const Node*> &topologies,
                          const Node &pointmaps,
                          const std::string &cset_name,
                          Node &output)
{
    // std::cout << "Building unstructured output!" << std::endl;
    output.reset();
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    std::vector<std::string>          shape_types;
    std::vector<std::vector<index_t>> out_connectivity;
    std::vector<std::vector<index_t>> out_elem_map;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node *topo = topologies[i];
        const Node *elements = topo->fetch_ptr("elements");
        if(!elements)
        {
            // ERROR
            continue;
        }

        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            // ERROR
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();

        iterate_elements(*topo, [&](const entity &e) {
            // See if we already have a bucket for this shape in our output
            const std::string &shape_string = e.shape.type;
            // Find the index for this shape's bucket
            const auto itr = std::find(shape_types.begin(), shape_types.end(), shape_string);
            index_t idx = (index_t)(itr - shape_types.begin());
            if(itr == shape_types.end())
            {
                idx = shape_types.size();
                shape_types.push_back(shape_string);
                out_connectivity.emplace_back();
                out_elem_map.emplace_back();
            }

            out_elem_map[idx].push_back(i);
            out_elem_map[idx].push_back(e.entity_id);

            // Translate the point ids using the pointmap.
            std::vector<index_t> &out_conn = out_connectivity[idx];
            for(const index_t id : e.element_ids)
            {
                out_conn.push_back(pmap_da[id]);
            }
        });
    }

    if(shape_types.size() == 1)
    {
        output["element_map"].set(out_elem_map[0]);
        output["elements/shape"].set(shape_types[0]);
        output["elements/connectivity"].set(out_connectivity[0]);
    }
    else if(shape_types.size() > 1)
    {
        std::vector<index_t> elem_map;
        const index_t nshapes = (index_t)shape_types.size();
        for(index_t i = 0; i < nshapes; i++)
        {
            const std::string name = shape_types[i] + "s";
            Node &bucket = output["elements"].add_child(name);
            bucket["shape"].set(shape_types[i]);
            bucket["connectivity"].set(out_connectivity[i]);
            for(index_t id : out_elem_map[i])
            {
                elem_map.push_back(id);
            }
            std::vector<index_t>().swap(out_elem_map[i]);
        }
        output["element_map"].set(elem_map);
    }
}

//-----------------------------------------------------------------------------
static void
build_polygonal_output(const std::vector<const Node*> &topologies,
                       const Node &pointmaps,
                       const std::string &cset_name,
                       Node &output)
{
    // std::cout << "Building polygonal output!" << std::endl;
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    output["elements/shape"].set("polygonal");

    std::vector<index_t> out_offsets;
    std::vector<index_t> out_conn;
    std::vector<index_t> out_sizes;
    std::vector<index_t> out_elem_map;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node &topo = *topologies[i];
        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            CONDUIT_WARN("Could not merge topology " << i <<  ", no associated pointmap.");
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();

        iterate_elements(topo, [&out_offsets, &out_conn, &out_sizes, &pmap_da, &out_elem_map, i](const entity &e) {
            // If it is a polygon or 2D/1D the remapping is trivial
            if(e.shape.is_polygonal() || e.shape.dim == 2 || e.shape.dim == 1)
            {
                out_elem_map.push_back(i);
                out_elem_map.push_back(e.entity_id);
                const index_t sz = (index_t)e.element_ids.size();
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(sz);
                for(index_t j = 0; j < sz; j++)
                {
                    out_conn.push_back(pmap_da[e.element_ids[j]]);
                }
            }
            else if(e.shape.is_polyhedral())
            {
                // Q: Will this even be a case? I'd imagine if polyhedra are present then
                //  everything would need to be promoted to polyhedra
            }
            else if(e.shape.id == (index_t)ShapeId::Tet || e.shape.id == (index_t)ShapeId::Hex)
            {
                // Q: Will this even be a case? I'd imagine if 3D elements exist then the output
                //  would have to be a 3D topology
                const index_t embeded_sz = mesh::utils::TOPO_SHAPE_INDEX_COUNTS[e.shape.embed_id];
                index_t ei = 0;
                for(index_t j = 0; j < e.shape.embed_count; j++)
                {
                    out_elem_map.push_back(i);
                    out_elem_map.push_back(e.entity_id);
                    out_offsets.push_back(out_conn.size());
                    out_sizes.push_back(embeded_sz);
                    for(index_t k = 0; k < embeded_sz; k++)
                    {
                        // Use the embedding table to find the correct index into the element_ids
                        //  array. Then map the element_id through the pointmap to get the new id.
                        out_conn.push_back(pmap_da[e.element_ids[e.shape.embedding[ei]]]);
                        ei++;
                    }
                }
            }
            else // (e.shape.dim == 0 || !e.shape.is_valid()) Q: Any other cases caught in here?
            {
                CONDUIT_ERROR("Encountered invalid element! At element " << e.entity_id);
                return;
            }
        });
    }
    output["element_map"].set(out_elem_map);
    output["elements/sizes"].set(out_sizes);
    output["elements/offsets"].set(out_offsets);
    output["elements/connectivity"].set(out_conn);
}

//-----------------------------------------------------------------------------
static void
build_polyhedral_output(const std::vector<const Node*> &topologies,
                       const Node &pointmaps,
                       const std::string &cset_name,
                       Node &output)
{
    // std::cout << "Building polyhedral output!" << std::endl;
    output.reset();
    output["type"].set("unstructured");
    output["coordset"].set(cset_name);
    output["elements/shape"].set("polyhedral");
    output["subelements/shape"].set("polygonal");
    std::vector<index_t> out_conn;
    std::vector<index_t> out_sizes;
    std::vector<index_t> out_offsets;
    std::vector<index_t> out_subconn;
    std::vector<index_t> out_subsizes;
    std::vector<index_t> out_suboffsets;
    std::vector<index_t> out_elem_map;
    const index_t ntopos = (index_t)topologies.size();
    for(index_t i = 0; i < ntopos; i++)
    {
        const Node &topo = *topologies[i];
        const Node *pointmap = pointmaps.child_ptr(i);
        if(!pointmap)
        {
            CONDUIT_WARN("Could not merge topology " << i <<  ", no associated pointmap.");
            continue;
        }
        DataArray<index_t> pmap_da = pointmap->value();

        iterate_elements(topo, [&](const entity &e) {
            out_elem_map.push_back(i);
            out_elem_map.push_back(e.entity_id);
            if(e.shape.is_polyhedral())
            {
                // Copy the subelements to their new offsets
                const index_t sz = e.element_ids.size();
                const index_t offset = out_conn.size();
                out_offsets.push_back(offset);
                out_sizes.push_back(sz);
                for(index_t i = 0; i < sz; i++)
                {
                    const index_t subidx = out_subsizes.size();
                    const index_t subsz = e.subelement_ids[i].size();
                    const index_t suboffset = out_subconn.size();
                    out_conn.push_back(subidx);
                    out_suboffsets.push_back(suboffset);
                    out_subsizes.push_back(subsz);
                    for(index_t j = 0; j < subsz; j++)
                    {
                        out_subconn.push_back(pmap_da[e.subelement_ids[i][j]]);
                    }
                }
            }
            else if(utils::TOPO_SHAPE_IDS[e.shape.id] == "f" || utils::TOPO_SHAPE_IDS[e.shape.id] == "l")
            {
                // 2D faces / 1D lines: Line, Tri, Quad
                // Add the shape to the sub elements
                const index_t subidx = out_subsizes.size();
                const index_t suboffset = out_subconn.size();
                out_suboffsets.push_back(suboffset);
                const index_t subsz = e.element_ids.size();
                out_subsizes.push_back(subsz);
                for(index_t j = 0; j < subsz; j++)
                {
                    out_subconn.push_back(pmap_da[e.element_ids[j]]);
                }
                // Then create a polyhedron with only 1 subelement
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(1);
                out_conn.push_back(subidx);
            }
            else if(utils::TOPO_SHAPE_IDS[e.shape.id] == "c")
            {
                // 3D cells: Tet, Hex
                const index_t nfaces = e.shape.embed_count;
                const index_t embeded_sz = mesh::utils::TOPO_SHAPE_INDEX_COUNTS[e.shape.embed_id];
                out_offsets.push_back(out_conn.size());
                out_sizes.push_back(nfaces);
                index_t ei = 0;
                for(index_t j = 0; j < nfaces; j++)
                {
                    out_conn.push_back(out_subsizes.size());
                    out_suboffsets.push_back(out_subconn.size());
                    out_subsizes.push_back(embeded_sz);
                    for(index_t k = 0; k < embeded_sz; k++)
                    {
                        // Use the embedding table to find the correct index into the element_ids
                        //  array. Then map the element_id through the pointmap to get the new id.
                        out_subconn.push_back(pmap_da[e.element_ids[e.shape.embedding[ei]]]);
                        ei++;
                    }
                }
            }
            else // (utils::TOPO_SHAPE_IDS[e.shape.id] == "p" || !e.shape.is_valid()) Q: Any other cases caught in here?
            {
                CONDUIT_ERROR("Encountered invalid element! At element " << e.entity_id);
                return;
            }
        });
    }
    output["element_map"].set(out_elem_map);
    output["elements/sizes"].set(out_sizes);
    output["elements/offsets"].set(out_offsets);
    output["elements/connectivity"].set(out_conn);
    output["subelements/sizes"].set(out_subsizes);
    output["subelements/offsets"].set(out_suboffsets);
    output["subelements/connectivity"].set(out_subconn);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::topology --
//-----------------------------------------------------------------------------

namespace fields
{

static void
determine_schema(const Node &in,
        const index_t ntuples, index_t &out_ncomps,
        Schema &out_schema)
{
    out_ncomps = 0;
    out_schema.reset();

    const index_t num_children = in.number_of_children();
    if(num_children)
    {
        out_ncomps = num_children;
        index_t offset = 0;
        // TODO: Keep track of whether the original field was interleaved
        //  and preserve the interleaved-ness
        for(index_t i = 0; i < num_children; i++)
        {
            const DataType dt(in[i].dtype().id(), ntuples, offset,
                in[i].dtype().element_bytes(), in[i].dtype().element_bytes(),
                in[i].dtype().endianness());
            out_schema[in[i].name()].set(dt);
            offset += dt.number_of_elements() * dt.element_bytes();
        }
    }
    else
    {
        out_ncomps = 1;
        out_schema.set(DataType(in.dtype().id(), ntuples));
    }
}

//-------------------------------------------------------------------------
static void
map_vertex_field(const std::vector<const Node*> &in_nodes,
        const std::vector<DataArray<index_t>> &pointmaps,
        const index_t num_verticies,
        Node &out_node)
{
    out_node.reset();
    if(in_nodes.empty() || pointmaps.empty())
    {
        return;
    }

    if(in_nodes.size() != pointmaps.size())
    {
        CONDUIT_WARN("Number of input fields and number of pointmaps should be equal!");
    }

    // Figure out num components and out dtype
    index_t ncomps = 0;
    Schema out_schema;
    determine_schema((*in_nodes[0])["values"], num_verticies, ncomps, out_schema);
    out_node.set(out_schema);

    // out_schema.print();
    // out_node.print();

    const index_t npmaps = (index_t)pointmaps.size();
    if(ncomps > 1)
    {
        for(index_t fi = 0; fi < npmaps; fi++)
        {
            const auto &pmap = pointmaps[fi];
            const Node &in_values = in_nodes[fi]->child("values");
            for(index_t idx = 0; idx < pmap.number_of_elements(); idx++)
            {
                const index_t out_idx = pmap[idx];
                for(index_t ci = 0; ci < ncomps; ci++)
                {
                    const auto bytes = out_node[ci].dtype().element_bytes();
                    void *out_data = out_node[ci].element_ptr(out_idx);
                    const void *in_data = in_values[ci].element_ptr(idx);
                    memcpy(out_data, in_data, bytes);
                }
            }
        }
    }
    else
    {
        const auto bytes = out_node.dtype().element_bytes();
        for(index_t fi = 0; fi < npmaps; fi++)
        {
            const auto &pmap = pointmaps[fi];
            const Node &in_values = in_nodes[fi]->child("values");
            for(index_t idx = 0; idx < pmap.number_of_elements(); idx++)
            {
                const index_t out_idx = pmap[idx];
                void *out_data = out_node.element_ptr(out_idx);
                const void *in_data = in_values.element_ptr(idx);
                memcpy(out_data, in_data, bytes);
            }
        }
    }
}

//-------------------------------------------------------------------------
static void
map_element_field(const std::vector<const Node*> &in_nodes,
        const DataArray<index_t> &elemmap,
        Node &out_node)
{
    out_node.reset();
    if(in_nodes.empty())
    {
        return;
    }

    const index_t nelements = elemmap.number_of_elements() / 2;

    index_t ncomps = 0;
    Schema out_schema;
    determine_schema(in_nodes[0]->child("values"), nelements, ncomps, out_schema);
    out_node.set(out_schema);
    if(ncomps > 1)
    {
        for(index_t out_idx = 0; out_idx < nelements; out_idx++)
        {
            const index_t idx      = out_idx * 2;
            const index_t dom_idx  = elemmap[idx];
            const index_t elem_idx = elemmap[idx+1];
            const Node &data = in_nodes[dom_idx]->child("values");
            for(index_t ci = 0; ci < ncomps; ci++)
            {
                const auto bytes = out_node[ci].dtype().element_bytes();
                void *out_data = out_node[ci].element_ptr(out_idx);
                const void *in_data = data[ci].element_ptr(elem_idx);
                memcpy(out_data, in_data, bytes);
            }
        }
    }
    else
    {
        const auto bytes = out_node.dtype().element_bytes();
        for(index_t out_idx = 0; out_idx < nelements; out_idx++)
        {
            const index_t idx      = out_idx * 2;
            const index_t dom_idx  = elemmap[idx];
            const index_t elem_idx = elemmap[idx+1];
            const Node &data = in_nodes[dom_idx]->child("values");
            void *out_data = out_node.element_ptr(out_idx);
            const void *in_data = data.element_ptr(elem_idx);
            memcpy(out_data, in_data, bytes);
        }
    }
}

//-----------------------------------------------------------------------------
static void
combine(const std::vector<const Node*> &in_fields,
        const Node &assoc_topology, const Node &assoc_coordset, Node &output)
{
    const std::string &assoc = in_fields[0]->child("association").as_string();
    const std::string &assoc_topo = in_fields[0]->child("topology").as_string();
    output["topology"] = assoc_topo;
    output["association"] = assoc;
    // Determine if we are vertex or element associated
    if(assoc == utils::ASSOCIATIONS[0])
    {
        // Vertex association
        // Need to use pointmaps to map this field
        // Get the point map
        std::vector<DataArray<index_t>> pmaps;
        {
            const Node *pointmaps = assoc_coordset.fetch_ptr("pointmaps");
            if(!pointmaps) { CONDUIT_ERROR("No pointmap for coordset"); return; }
            for(index_t pi = 0; pi < pointmaps->number_of_children(); pi++)
            {
                pmaps.emplace_back(pointmaps->child(pi).value());
            }
        }

        const index_t nt = coordset::length(assoc_coordset);
        fields::map_vertex_field(in_fields, pmaps, nt, output["values"]);
    }
    else if(assoc == utils::ASSOCIATIONS[1])
    {
        // Element association
        // Need to use element maps to map this field
        const Node &out_topo_map = assoc_topology["element_map"];
        const DataArray<index_t> tmap = out_topo_map.value();
        fields::map_element_field(in_fields, tmap, output["values"]);
    }
    else
    {
        CONDUIT_WARN("Unsupported association for field " << assoc);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::fields --
//-----------------------------------------------------------------------------

//-------------------------------------------------------------------------
std::string
partitioner::recommended_topology(const std::vector<const Node *> &inputs) const
{
    // TODO: See if the inputs are uniform, rectilinear, etc and could be combined
    //       to form an output of one of those types. For example, uniform meshes
    //       can be combined if they abut and combine into larger bricks that
    //       cover space.
    // Meshes:
    //   Uniform
    //   Rectilinear
    //   Structured
    //   Unstructured

    // Coordsets:
    //   Uniform
    //   Rectilinear
    //   Explicit

    // Topologies:
    //   Points
    //   Uniform
    //   Rectilinear
    //   Structured
    //   Unstructured

    // Redefine these here because the order matters
    static const std::array<std::string, 3> coordset_types = {
        "uniform", 
        "rectilinear", 
        "explicit"
    };

    static const std::array<std::string, 5> topology_types = {
        "points",
        "uniform",
        "rectilinear",
        "structured",
        "unstructured"
    };

    index_t worst_coordset = 0;
    index_t worst_topology = 0;
    for(const Node *input : inputs)
    {
        const Node *n_topologies = input->fetch_ptr("topologies");
        if(n_topologies)
        {
            for(index_t i = 0; i < n_topologies->number_of_children(); i++)
            {
                const std::string &type = n_topologies->child(i)["type"].as_string();
                const index_t idx = std::find(topology_types.begin(), 
                    topology_types.end(), type) - topology_types.begin();
                worst_topology = std::max(worst_topology, idx);
            }
        }
        const Node *n_coordsets = input->fetch_ptr("coordsets");
        if(n_coordsets)
        {
            for(index_t i = 0; i < n_coordsets->number_of_children(); i++)
            {
                const std::string &type = n_coordsets->child(i)["type"].as_string();
                const index_t idx = std::find(coordset_types.begin(), 
                    coordset_types.end(), type) - coordset_types.begin();
                worst_coordset = std::max(worst_coordset, idx);
            }
        }
    }

    std::string retval;
    if(worst_topology < 2 && worst_coordset < 1)
    {
        retval = "uniform";
    }
    else if(worst_topology < 3 && worst_coordset < 2)
    {
        retval = "rectilinear";
    }
    else
    {
        retval = "unstructured";
    }
    return retval;
}

//-------------------------------------------------------------------------
void
partitioner::combine(int domain,
    const std::vector<const Node *> &inputs,
    Node &output)
{
    // NOTE: Some decisions upstream, for the time being, make all the chunks
    //       unstructured. We will try to relax that so we might end up
    //       trying to combine multiple uniform,rectilinear,structured
    //       topologies.
    // Handle trivial cases
    // std::cout << "domain " << domain << " size " << inputs.size() << std::endl;
    // std::cout << "INPUTS:";
    // for(const Node *in : inputs)
    // {
    //     in->print();
    // }
    output.reset();
    const auto sz = inputs.size();
    if(sz == 0)
    {
        CONDUIT_ERROR("partitioner::combine called with 0 inputs, cannot combine 0 inputs.")
        return;
    }
    else if(sz == 1)
    {
        output = *inputs[0];
        output["state/domain_id"] = domain;
        return;
    }

    // Combine state - take state from inputs[0], overwrite domain_id
    // Q: Should this be more involved?
    if(inputs[0]->has_child("state"))
    {
        output["state"] = inputs[0]->child("state");
    }
    output["state/domain_id"] = domain;

    // Determine the combine approach
    std::string rt(recommended_topology(inputs));
    // std::cout << "Recommended approach: " << rt << std::endl;

    // Combine the coordsets
    // Determine names of all coordsets
    std::vector<std::string> coordset_names;
    Node &output_coordsets = output.add_child("coordsets");
    {
        // Group all the like-named coordsets
        std::vector<std::vector<const Node*>> coordset_groups;
        const index_t ninputs = (index_t)inputs.size();
        for(index_t i = 0; i < ninputs; i++)
        {
            const Node *input = inputs[i];
            if(!input) { continue; }

            const Node *csets = input->fetch_ptr("coordsets");
            if(!csets) { continue; }

            const auto &cset_names = csets->child_names();
            const index_t ncset_names = (index_t)cset_names.size();
            for(index_t j = 0; j < ncset_names; j++)
            {
                const auto &cset_name = cset_names[j];
                const Node *cset = csets->fetch_ptr(cset_name);
                if(!cset) { continue; }

                auto itr = std::find(coordset_names.begin(), coordset_names.end(), cset_name);
                if(itr != coordset_names.end())
                {
                    const auto idx = itr - coordset_names.begin();
                    coordset_groups[idx].push_back(cset);
                }
                else // (itr == names.end())
                {
                    const auto idx = coordset_groups.size();
                    coordset_names.push_back(cset_name);
                    coordset_groups.emplace_back();
                    coordset_groups[idx].push_back(cset);
                }
            }
        }

        const index_t ngroups = (index_t)coordset_groups.size();
#if 0
        // Some debug output to see how coordsets were grouped
        std::cout << "Coordsets:\n";
        for(index_t i = 0; i < ngroups; i++)
        {
            std::cout << "  -\n"
                << "    name: " << coordset_names[i] << "\n"
                << "    size: " << coordset_groups[i].size() << "\n";
        }
        std::cout << std::endl;
#endif
        conduit::Node opts;
        opts["type"] = ((rt == "rectilinear" || rt == "uniform") ? "implicit" : "explicit");
        opts["merge_tolerance"] = merge_tolerance;
        for(index_t i = 0; i < ngroups; i++)
        {
            const auto &coordset_group = coordset_groups[i];
            coordset::combine(coordset_group, output_coordsets.add_child(coordset_names[i]), &opts);
            // pointmaps / element_map are correct!
            // std::cout << "COMBINED CSET" << std::endl;
            // output_coordsets[coordset_names[i]].print();
        }
    }

    // Iterate over all topology names and combine like-named topologies
    // as new unstructured topology.
    Node &output_topologies = output.add_child("topologies");
    using topo_group_t = std::pair<std::string, std::vector<const Node*>>;
    std::vector<topo_group_t> topo_groups;
    {
        // Group all the like-named toplogies
        const index_t ninputs = (index_t)inputs.size();
        for(index_t i = 0; i < ninputs; i++)
        {
            const Node *input = inputs[i];
            if(!input) { continue; }

            const Node *topos = input->fetch_ptr("topologies");
            if(!topos) { continue; }

            const auto &topo_names = topos->child_names();
            const index_t ntopo_names = (index_t)topo_names.size();
            for(index_t j = 0; j < ntopo_names; j++)
            {
                const auto &topo_name = topo_names[j];
                const Node *topo = topos->fetch_ptr(topo_name);
                if(!topo) { continue; }

                auto itr = std::find_if(topo_groups.begin(), topo_groups.end(), [&](const topo_group_t &g) {
                    return g.first == topo_name;
                });
                if(itr != topo_groups.end())
                {
                    const auto idx = itr - topo_groups.begin();
                    // Need to check if the topologies in this group have the same coordset
                    const Node *group_cset = topo_groups[idx].second[0]->fetch_ptr("coordset");
                    if(!group_cset) { continue; }
                    const Node *this_cset = topo->fetch_ptr("coordset");
                    if(!this_cset) { continue; }
                    if(group_cset->as_string() != this_cset->as_string())
                    {
                        // Error! Cannot merge two topologies that reference different named coordsets
                        continue;
                    }
                    topo_groups[idx].second.push_back(topo);
                }
                else // (itr == names.end())
                {
                    const auto idx = topo_groups.size();
                    topo_groups.emplace_back();
                    topo_groups[idx].first = topo_name;
                    topo_groups[idx].second.push_back(topo);
                }
            }
        }

        const index_t ngroups = topo_groups.size();
        for(index_t i = 0; i < ngroups; i++)
        {
            const auto &topo_group = topo_groups[i];
            // All topologies in the same group must reference a coordset with the same name
            const Node *group_cset = topo_group.second[0]->fetch_ptr("coordset");
            if(!group_cset) { continue; }

            auto itr = std::find(coordset_names.begin(), coordset_names.end(), group_cset->as_string());
            if(itr == coordset_names.end())
            {
                CONDUIT_ERROR("Topology " << topo_group.first << " references unknown coordset " << group_cset->as_string());
                continue;
            }

            const Node *pointmaps = output_coordsets[*itr].fetch_ptr("pointmaps");
            if(!pointmaps) { continue; }

            Node opts;
            opts["type"] = rt;
            // std::cout << "Resulting pointmaps: " << std::endl;
            // pointmaps->print();
            topology::combine(topo_group.second, *pointmaps, output_coordsets[*itr], 
                output_topologies.add_child(topo_group.first), &opts);
            if(output_coordsets[*itr].has_child("element_map"))
            {
                output_coordsets[*itr].remove_child("element_map");
            }
        }
    }

    // All inputs must have a fields object to merge fields
    bool have_fields = true;
    for(const Node *n : inputs)
    {
        if(!n->has_child("fields"))
        {
            have_fields = false;
            break;
        }
    }

    Node &output_fields = output["fields"];
    if(have_fields)
    {
        // Note: It should already be verified that they have a "fields" child
        using field_group_t = std::pair<std::string, std::vector<const Node*>>;
        std::vector<field_group_t> field_groups;
        for(const Node *n : inputs)
        {
            const Node &fields = n->child("fields");
            auto itr = fields.children();
            while(itr.has_next())
            {
                const Node &n = itr.next();
                auto itr = std::find_if(field_groups.begin(), field_groups.end(), [&](const field_group_t &g) {
                    return g.first == n.name();
                });
                if(itr != field_groups.end())
                {
                    itr->second.push_back(&n);
                }
                else
                {
                    field_groups.emplace_back();
                    field_groups.back().first = n.name();
                    field_groups.back().second.push_back(&n);
                }
            }
        }

        for(size_t fgi = 0; fgi < field_groups.size(); fgi++)
        {
            const auto &field_name = field_groups[fgi].first;
            const auto &field_group = field_groups[fgi].second;
            // Figure out field association and topology
            if(!field_group[0]->has_child("topology") || !field_group[0]->has_child("values"))
            {
                CONDUIT_WARN("Field " << field_name << " is not topology based, TODO: Implement material based field combinations.");
                continue;
            }
            const std::string &assoc_topo_name = field_group[0]->child("topology").as_string();

            // Make sure we have an output toplogy for the given name
            if(!output_topologies.has_child(assoc_topo_name))
            {
                CONDUIT_ERROR("Field " << field_name << " references " << assoc_topo_name
                    << " which doesn't exist.");
                continue;
            }
            Node &out_topo = output_topologies[assoc_topo_name];

            // Make sure there were as many input topologies as there are fields
            //  for this topology name
            auto topo_itr = std::find_if(topo_groups.begin(), topo_groups.end(),
                    [&assoc_topo_name](const topo_group_t &g) {
                return g.first == assoc_topo_name;
            });

            if(topo_itr->second.size() != field_group.size())
            {
                CONDUIT_INFO("Field " << field_name << " is not present on all input domains, skipping...");
                continue;
            }

            // Make sure we have an output coordset for the given name
            const std::string &assoc_cset_name = out_topo["coordset"].as_string();
            if(!output_coordsets.has_child(assoc_cset_name))
            {
                CONDUIT_ERROR("Topology " << assoc_topo_name << " references coordset "
                    << assoc_cset_name << " which doesn't exist. This error was found when building output fields.");
                continue;
            }
            Node &out_coordset = output_coordsets[assoc_cset_name];
            fields::combine(field_group, out_topo, out_coordset, output_fields[field_name]);
        }
    }

    // Cleanup the output node, add original cells/verticies in needed
    if(!output_fields.has_child("original_element_ids"))
    {
        // Q: What happens in the case of multiple topologies?
        const Node &n_elem_map = output_topologies[0]["element_map"];
        Node &out_field = output_fields["original_element_ids"];
        out_field["topology"].set(output_topologies[0].name());
        // utils::ASSOCIATIONS[1]
        out_field["association"].set("element");
        Schema s;
        const DataType &dt = n_elem_map.dtype();
        const index_t sz = dt.number_of_elements() / 2;
        s["domains"].set(DataType(dt.id(), sz, 0, 
            2*dt.element_bytes(), dt.element_bytes(), dt.endianness()));
        s["values"].set(DataType(dt.id(), sz, 1*dt.element_bytes(),
            2*dt.element_bytes(), dt.element_bytes(), dt.endianness()));
        out_field["values"].set(s);

        std::memcpy(
            out_field["values/domains"].element_ptr(0),
            n_elem_map.element_ptr(0),
            dt.element_bytes()*dt.number_of_elements());

        std::vector<index_t> domain_map;
        bool has_domain_ids = false;
        for(index_t i = 0; i < (index_t)inputs.size(); i++)
        {
            if(inputs[i]->has_path("state/domain_id"))
            {
                const index_t did = (*inputs[i])["state/domain_id"].to_index_t();
                if(did != i)
                {
                    domain_map.push_back(did);
                    has_domain_ids = true;
                    continue;
                }
            }
            domain_map.push_back(i);
        }

        if(has_domain_ids)
        {
            DataArray<index_t> out_domains = out_field["values/domains"].value();
            for(index_t i = 0; i < out_domains.number_of_elements(); i++)
            {
                out_domains[i] = domain_map[out_domains[i]];
            }
        }

    }
    // Remove the element_maps from the output
    for(index_t i = 0; i < output_topologies.number_of_children(); i++)
    {
        output_topologies[i].remove("element_map");
    }

    if(!output_fields.has_child("original_vertex_ids"))
    {
        // Get the pointmaps
        const std::string &coordset_name = output_topologies[0]["coordset"].as_string();
        if(!output_coordsets.has_child(coordset_name))
        {
            CONDUIT_ERROR("Output topology 0 references coordset " << coordset_name << " which doesn't exist.");
            return;
        }
        const Node &n_pointmaps = output_coordsets[coordset_name]["pointmaps"];
        std::vector<DataArray<index_t>> pointmaps;
        for(index_t i = 0; i < n_pointmaps.number_of_children(); i++)
        {
            pointmaps.emplace_back(n_pointmaps[i].value());
        }

        Node &out_field = output_fields["original_vertex_ids"];
        out_field["topology"].set(output_topologies[0].name());
        // utils::ASSOCIATIONS[0]
        out_field["association"].set("vertex");

        const index_t sz = mesh::coordset::length(output_coordsets[coordset_name]);
        const DataType dt(pointmaps[0].dtype().id(), sz);
        out_field["values"]["domains"].set(dt);
        out_field["values"]["ids"].set(dt);
        DataArray<index_t> out_domains = out_field["values/domains"].value();
        DataArray<index_t> out_ids     = out_field["values/ids"].value();

        for(index_t pi = 0; pi < (index_t)pointmaps.size(); pi++)
        {
            index_t dom_id = pi;
            if(inputs[pi]->has_path("state/domain_id"))
            {
                dom_id = (*inputs[pi])["state/domain_id"].to_index_t();
            }
            const auto &pmap = pointmaps[pi];
            for(index_t vi = 0; vi < pmap.number_of_elements(); vi++)
            {
                const auto out_idx = pmap[vi];
                out_domains[out_idx] = dom_id;
                out_ids[out_idx]     = vi;
            }
        }

    }
    // Remove the pointmaps from the output
    for(index_t i = 0; i < output_coordsets.number_of_children(); i++)
    {
        output_coordsets[i].remove("pointmaps");
    }
}

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
namespace coordset
{
// Q: Why is this exposed? A: For tests
void CONDUIT_BLUEPRINT_API combine(const std::vector<const conduit::Node *> &coordsets,
                                 conduit::Node &output,
                                 const conduit::Node *options)
{
    double tolerance = CONDUIT_EPSILON;
    std::string approach = "explicit";
    if(options)
    {
        const Node *n_tolerance = options->fetch_ptr("merge_tolerance");
        if(n_tolerance)
        {
            tolerance = n_tolerance->to_double();
        }
        const Node *n_type = options->fetch_ptr("type");
        if(n_type)
        {
            const std::string type = n_type->as_string();
            if(type == "explicit" || type == "implicit")
            {
                approach = type;
            }
            else
            {
                CONDUIT_WARN("Invalid \"type\" passed to coordset::combine, expected \"implicit\" or \"explicit\", got " << type
                    << ". Continuing as explicit.");
            }
        }
    }

    bool do_explicit = true;
    if(approach == "implicit")
    {
        const bool success = utils::combine_implicit(coordsets, tolerance, output);
        do_explicit = !success;
    }

    // If implicit combination failed fall back on explicit combination,
    //  or we were told explicit from the start.
    if(do_explicit)
    {
        point_merge pm;
        pm.execute(coordsets, tolerance, output);
    }
}

}

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
namespace topology
{

// Q: Why is this exposed?
void CONDUIT_BLUEPRINT_API combine(const std::vector<const conduit::Node *> &topologies,
                                   const conduit::Node &pointmaps,
                                   const conduit::Node &coordset,
                                   conduit::Node &output,
                                   conduit::Node *options)
{
    if(topologies.size() == 0)
    {
        return;
    }

    const static std::array<std::string,3> VALID_TYPES = {
        "unstructured", "uniform", "rectilinear"
    };
    std::string type = "unstructured";
    bool force_polyhedral = false;
    bool force_polygonal = false;
    if(options)
    {
        if(options->has_child("force_polyhedral"))
        {
            force_polyhedral = true;
        }

        if(options->has_child("force_polygonal"))
        {
            force_polygonal = true;
        }

        if(options->has_child("type"))
        {
            type = options->child("type").as_string();
            bool valid = false;

            // No support for structured yet.
            if(type == "structured")
            {
                type = "unstructured";
            }

            // Validate.
            for(const std::string &t : VALID_TYPES)
            {
                if(type == t) { valid = true; break; }
            }

            if(!valid)
            {
                CONDUIT_ERROR("Invalid type passed to topology::combine - " << type);
                return;
            }
        }
    }
    
    // std::cout << "combining topology as type " << type << std::endl;
    if(type == "rectilinear" || type == "uniform")
    {
        if(!(coordset["type"].as_string() == "rectilinear" || 
            coordset["type"].as_string() == "uniform"))
        {
            type = "unstructured";
        }
        else
        {
            output["type"] = type;
            output["coordset"] = coordset.name();
            output["element_map"] = coordset["element_map"];
        }
    }
    
    if(type == "unstructured")
    {
        const std::string &cset_name = coordset.name();
        std::vector<const Node*> working_topologies;
        std::vector<Node> temporary_nodes;
        temporary_nodes.reserve(topologies.size());

        // Validate / translate inputs
        {
            const index_t ntopos = topologies.size();
            for(index_t i = 0; i < ntopos; i++)
            {
                const Node *topo = topologies[i];
                Node temp;
                if(!topo) { continue; }

                const Node *type = topo->fetch_ptr("type");
                if(!type) { continue; }

                const Node *cset = topo->fetch_ptr("coordset");
                if(!cset) { continue; }

                const std::string &t = type->as_string();
                if(t == "points")
                {
                    temporary_nodes.emplace_back();
                    // topology::points::to_explicit(topo, temporary_nodes.back());
                    working_topologies.push_back(&temporary_nodes.back());
                }
                else if(t == "uniform")
                {
                    temporary_nodes.emplace_back();
                    topology::uniform::to_unstructured(*topo, temporary_nodes.back(), temp);
                    working_topologies.push_back(&temporary_nodes.back());
                }
                else if(t == "rectilinear")
                {
                    temporary_nodes.emplace_back();
                    topology::rectilinear::to_unstructured(*topo, temporary_nodes.back(), temp);
                    working_topologies.push_back(&temporary_nodes.back());
                }
                else if(t == "structured")
                {
                    temporary_nodes.emplace_back();
                    topology::structured::to_unstructured(*topo, temporary_nodes.back(), temp);
                    working_topologies.push_back(&temporary_nodes.back());
                }
                else if(t == "unstructured")
                {
                    working_topologies.push_back(topo);
                }
                else
                {
                    //  ERROR!
                    continue;
                }
            }

            // Make sure we have the correct number of point maps
            if(pointmaps.number_of_children() != (index_t)working_topologies.size())
            {
                CONDUIT_ERROR("Number of input pointmaps and number of input topologies do not match! " 
                    << pointmaps.number_of_children() << " != " << working_topologies.size());
                return;
            }
        }

        // Start building the output
        output.reset();
        if(working_topologies.size())
        {
            bool do_polyhedral = force_polyhedral;
            bool do_polygonal  = force_polygonal;
            index_t dim = 0;
            for(const Node *topo : topologies)
            {
                // TODO: Write a dim function that knows about all the different topologies
                const Node *shape = topo->fetch_ptr("elements/shape");
                if(!shape) { continue; }
                const utils::ShapeType s(shape->as_string());
                if(s.is_polyhedral())
                {
                    do_polyhedral = true;
                }
                else if(s.is_polygonal())
                {
                    do_polygonal = true;
                }
                dim = std::max(dim, s.dim);
                if(do_polyhedral) break;
            }

            if(do_polyhedral || (do_polygonal && dim > 2))
            {
                build_polyhedral_output(working_topologies, pointmaps, cset_name, output);
            }
            else if(do_polygonal)
            {
                build_polygonal_output(working_topologies, pointmaps, cset_name, output);
            }
            else // Single shape & multi shape topologies
            {
                build_unstructured_output(working_topologies, pointmaps, cset_name, output);
            }
        }
    }
}

}

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
