#ifndef POINT_MERGE_HPP
#define POINT_MERGE_HPP

// NOTE: THIS CLASS WILL BE MOVED

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

class CONDUIT_BLUEPRINT_API point_merge
{
public:
    void execute(const std::vector<conduit::Node *> coordsets, double tolerance,
                Node &output);

private:
    struct record
    {
        using index_t = conduit_index_t;
        index_t orig_domain;
        index_t orig_id;
    };

    enum class coord_system
    {
        cartesian,
        cylindrical,
        spherical
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
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

// cpp
#include <conduit_node.hpp>
#include <conduit_blueprint_mesh.hpp>
#include <conduit_blueprint_mesh_utils.hpp>
#include <cmath>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <map>

#define DEBUG_POINT_MERGE
#ifndef DEBUG_POINT_MERGE
#define PM_DEBUG_PRINT(stream)
#else
#define PM_DEBUG_PRINT(stream) do { std::cerr << stream; } while(0)
#endif

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
inline void
point_merge::xyz_to_rtp(double x, double y, double z, double &out_r, double &out_t, double &out_p)
{
    const auto r = std::sqrt(x*x + y*y + z*z);
    out_r = r;
    out_t = std::acos(r / z);
    out_p = std::atan(y / x);
}

//-----------------------------------------------------------------------------
inline void
point_merge::rtp_to_xyz(double r, double t, double p, double &out_x, double &out_y, double &out_z)
{
    out_x = r * std::cos(p) * std::sin(t);
    out_y = r * std::sin(p) * std::sin(t);
    out_z = r * std::cos(t);
}

//-----------------------------------------------------------------------------
inline void
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
    }
}

//-----------------------------------------------------------------------------
inline const std::vector<std::string> &
point_merge::get_axes_for_system(coord_system cs)
{
    switch(cs)
    {
    case coord_system::cartesian:
        return mesh::utils::CARTESIAN_AXES;
    case coord_system::cylindrical:
        return mesh::utils::CYLINDRICAL_AXES;
    case coord_system::spherical:
        return mesh::utils::SPHERICAL_AXES;
    }
}

//-----------------------------------------------------------------------------
inline void
point_merge::execute(const std::vector<Node *> coordsets, double tolerance,
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
    index_t ncartesian = 0, ncylindrical = 0, nspherical = 0;
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
inline int
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
inline void
point_merge::append_data(const std::vector<Node> &coordsets,
    const std::vector<coord_system> &systems, index_t dimension)
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
            if(xnode == nullptr)
            {
                xnode = xnode = values->fetch_ptr("r");
            }

            if(xnode)
            {
                npts = xnode->dtype().number_of_elements();
            }
        }        

        old_to_new_ids.push_back({});
        old_to_new_ids.back().reserve(npts);
        new_size += npts*dimension;
    }
    new_coords.reserve(new_size);

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


        if(systems[i] == out_system)
        {
            iterate_coordinates(coordsets[i], append);
        }
        else
        {
            iterate_coordinates(coordsets[i], translate_append);
        }
    }
}

//-----------------------------------------------------------------------------
inline void
point_merge::merge_data(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
// #define USE_SPATIAL_SEARCH_MERGE
#if   defined(USE_TRUNCATE_PRECISION_MERGE)
    truncate_merge(coordsets, systems, dimension, tolerance);
#elif defined(USE_SPATIAL_SEARCH_MERGE)
    spatial_search_merge(coordsets, systems, dimension, tolerance);
#else
    simple_merge_data(coordsets, systems, dimension, tolerance);
#endif
}

//-----------------------------------------------------------------------------
inline void
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
        auto &coordset = output.add_child("coordsets");
        auto &coords = coordset.add_child("coords");
        coords["type"] = "explicit";
        auto &values = coords.add_child("values");
        
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
inline void
point_merge::iterate_coordinates(const Node &coordset, Func &&func)
{
    if(!coordset.has_child("type"))
        return;

    if(coordset["type"].as_string() != "explicit")
        return;
    
    if(!coordset.has_child("values"))
        return;
    
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

    // Iterate accordingly
    float64 p[3] {0., 0., 0.};
    if(xnode && ynode && znode)
    {
        // 3D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        const auto ztype = znode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        auto yarray = ynode->as_double_array();
        auto zarray = znode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = yarray[i]; p[2] = zarray[i];
            func(p, 3);
        }
    }
    else if(xnode && ynode)
    {
        // 2D
        const auto xtype = xnode->dtype();
        const auto ytype = ynode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        auto yarray = ynode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = yarray[i]; p[2] = 0.;
            func(p, 2);
        }
    }
    else if(xnode)
    {
        // 1D
        const auto xtype = xnode->dtype();
        // TODO: Handle different types
        auto xarray = xnode->as_double_array();
        const index_t N = xarray.number_of_elements();
        for(index_t i = 0; i < N; i++)
        {
            p[0] = xarray[i]; p[1] = 0.; p[2] = 0.;
            func(p, 1);
        }
    }
    else
    {
        // ERROR! No valid nodes passed.
    }
}

//-----------------------------------------------------------------------------
inline index_t
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
            if(xnode == nullptr)
            {
                xnode = xnode = values->fetch_ptr("r");
            }

            if(xnode)
            {
                npts = xnode->dtype().number_of_elements();
            }
        }        

        old_to_new_ids.push_back({});
        old_to_new_ids.back().reserve(npts);
        new_size += npts*dimension;
    }
    new_coords.reserve(new_size);
    return new_size;
}

//-----------------------------------------------------------------------------
inline void
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
        if(systems[i] != coord_system::cartesian)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}

#if 0
//-----------------------------------------------------------------------------
inline void
point_merge::spatial_search_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension,
        double tolerance)
{
    PM_DEBUG_PRINT("Spatial search merging!" << std::endl);
    reserve_vectors(coordsets, dimension);

    kdtree<vec3, index_t> point_records;
    point_records.set_tolerance(tolerance);
    point_records.set_bucket_size(32);
    for(size_t i = 0u; i < coordsets.size(); i++)
    {
        const auto &coordset = coordsets[i];

        // To be invoked on every coordinate
        const auto merge = [&](float64 *p, index_t) {
            vec3 key;
            key.v[0] = p[0]; key.v[1] = p[1]; key.v[2] = p[2];
            const auto potential_id = new_coords.size() / dimension;
            auto res = point_records.insert(key, potential_id);
            if(res.second)
            {
                // Potential id was inserted successfully
                old_to_new_ids[i].push_back(potential_id);
                for(index_t j = 0; j < dimension; j++)
                {
                    new_coords.push_back(p[j]);
                }
            }
            else
            {
                PM_DEBUG_PRINT("Found point (" << p[0] << "," << p[1] << ") at " << res.first.second << std::endl);
                PM_DEBUG_PRINT("[");
                for(auto v : res.first.first->points)
                {
                    PM_DEBUG_PRINT("(" << v.x << "," << v.y << "),");
                }
                PM_DEBUG_PRINT("]" << std::endl);
                PM_DEBUG_PRINT("[");
                for(auto idx : res.first.first->data)
                {
                    PM_DEBUG_PRINT(idx << ",");
                }
                PM_DEBUG_PRINT("]" << std::endl);
                old_to_new_ids[i].push_back(res.first.second);
            }
        };

        const auto translate_merge = [&](float64 *p, index_t d) {
            translate_system(systems[i], coord_system::cartesian,
                p[0], p[1], p[2], p[0], p[1], p[2]);
            merge(p, d);
        };

        // Invoke the proper lambda on each coordinate
        if(systems[i] != coord_system::cartesian)
        {
            iterate_coordinates(coordset, translate_merge);
        }
        else
        {
            iterate_coordinates(coordset, merge);
        }
    }
}
#endif

//-----------------------------------------------------------------------------
inline void
point_merge::truncate_merge(const std::vector<Node> &coordsets,
        const std::vector<coord_system> &systems, index_t dimension, double tolerance)
{
    PM_DEBUG_PRINT("Truncate merging!" << std::endl);
    // Determine what to scale each value by
    // TODO: Be dynamic
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
        if(systems[i] != out_system)
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
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------




#endif