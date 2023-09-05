// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_tiled.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_TILED_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_TILED_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_exports.h"
#include "conduit_blueprint_mesh_kdtree.hpp"
#include "conduit_blueprint_mesh_utils.hpp"

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

//-----------------------------------------------------------------------------
/// Detail
//-----------------------------------------------------------------------------
namespace detail
{

std::vector<int> spatial_reordering(const conduit::Node &topo)
{
    // Make a new centroid topo and coordset. The coordset will contain the
    // element centers.
    Node topo_dest, coords_dest, s2dmap, d2smap;
    mesh::topology::unstructured::generate_centroids(topo,
                                                     topo_dest,
                                                     coords_dest,
                                                     s2dmap,
                                                     d2smap);
    // Bundle the coordset components into a vector.
    std::vector<conduit::double_accessor> coords;
    const conduit::Node &values = coords_dest.fetch_existing("values");
    for(conduit::index_t i = 0; i < values.number_of_children(); i++)
    {
        coords.push_back(values[i].as_double_accessor());
    }

    // Sort the coordinates spatially
    std::vector<int> reorder;
    if(coords.size() == 2)
    {
        conduit::blueprint::mesh::utils::kdtree<conduit::double_accessor, double, 2> spatial_sort;
        spatial_sort.initialize(&coords[0], coords[0].number_of_elements());
        reorder = std::move(spatial_sort.getIndices());
    }
    else if(coords.size() == 3)
    {
        conduit::blueprint::mesh::utils::kdtree<conduit::double_accessor, double, 3> spatial_sort;
        spatial_sort.initialize(&coords[0], coords[0].number_of_elements());
        reorder = std::move(spatial_sort.getIndices());
    }
    return reorder;
}
//---------------------------------------------------------------------------
// @brief Slice the n_src array using the indices stored in ids. We use the
//        array classes for their [] operators that deal with interleaved
//        and non-interleaved arrays.
template <typename T, typename IndexType>
inline void
typed_slice_array(const T &src, const std::vector<IndexType> &ids, T &dest)
{
    size_t n = ids.size();
    for(size_t i = 0; i < n; i++)
        dest[i] = src[ids[i]];
}

//---------------------------------------------------------------------------
// @note Should this be part of conduit::Node or DataArray somehow. The number
//       of times I've had to slice an array...
template <typename IndexType>
void
slice_array(const conduit::Node &n_src_values,
            const std::vector<IndexType> &ids,
            Node &n_dest_values)
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

void
slice_field(const conduit::Node &src,
            const std::vector<int> &ids,
            conduit::Node &dest)
{
    if(src.number_of_children() > 0)
    {
        // Reorder an mcarray
        for(conduit::index_t ci = 0; ci < src.number_of_children(); ci++)
        {
            const conduit::Node &comp = src[ci];
            slice_array(comp, ids, dest[comp.name()]);
        }
    }
    else
    {
        slice_array(src, ids, dest);
    }
}

void
reorder_topo(const conduit::Node &topo, const conduit::Node &coordset, const conduit::Node &fields,
             conduit::Node &dest_topo, conduit::Node &dest_coordset, conduit::Node &dest_fields,
             const std::vector<int> &reorder)
{
    conduit::blueprint::mesh::utils::ShapeType shape(topo);

    // Handle unstructured meshes (but not polyhedral meshes yet)
    if(topo.fetch_existing("type").as_string() == "unstructured" && !shape.is_polyhedral())
    {
        // Input connectivity information.
        const auto &n_conn = topo.fetch_existing("elements/connectivity");
        const auto &n_sizes = topo.fetch_existing("elements/sizes");
        const auto &n_offsets = topo.fetch_existing("elements/offsets");
        const auto conn = n_conn.as_index_t_accessor();
        const auto sizes = n_sizes.as_index_t_accessor();
        const auto offsets = n_offsets.as_index_t_accessor();

        // Temp vectors to store reordered connectivity.
        std::vector<conduit::index_t> newconn, newoffsets, newsizes;
        newconn.reserve(conn.number_of_elements());
        newsizes.reserve(sizes.number_of_elements());
        newoffsets.reserve(offsets.number_of_elements());

        // Mapping information for the points.
        auto npts = conduit::blueprint::mesh::coordset::length(coordset);
        std::vector<int> old2NewPoints(npts, -1), ptReorder(npts, -1);
        int newPointIndex = 0;

        // We iterate over elements in the specified order. We iterate over the
        // points in each element and renumber the points.
        conduit::index_t newoffset = 0;
        for(const int cellIndex : reorder)
        {
            for(conduit::index_t i = 0; i < sizes[cellIndex]; i++)
            {
                auto id = conn[offsets[cellIndex] + i];
#define REORDER_POINTS
#ifdef REORDER_POINTS
                if(old2NewPoints[id] == -1)
                {
                    ptReorder[newPointIndex] = id;
                    old2NewPoints[id] = newPointIndex++;
                }
                newconn.push_back(old2NewPoints[id]);
#else
                newconn.push_back(id);
#endif
            }
            newsizes.push_back(sizes[cellIndex]);
            newoffsets.push_back(newoffset);
            newoffset += sizes[cellIndex];
        }

        // Store the new connectivity.
        dest_topo["type"] = topo["type"];
        dest_topo["coordset"] = dest_coordset.name(); //topo["coordset"];
        dest_topo["elements/shape"] = topo["elements/shape"];
        conduit::Node tmp;
        tmp.set_external(newconn.data(), newconn.size());
        tmp.to_data_type(n_conn.dtype().id(), dest_topo["elements/connectivity"]);
        tmp.set_external(newsizes.data(), newsizes.size());
        tmp.to_data_type(n_sizes.dtype().id(), dest_topo["elements/sizes"]);
        tmp.set_external(newoffsets.data(), newoffsets.size());
        tmp.to_data_type(n_offsets.dtype().id(), dest_topo["elements/offsets"]);

#ifdef REORDER_POINTS
        // Reorder the coordset now, making it explicit if needed.
        dest_coordset["type"] = "explicit";
        conduit::Node coordset_explicit;
        if(coordset["type"].as_string() == "rectilinear")
            conduit::blueprint::mesh::coordset::rectilinear::to_explicit(coordset, coordset_explicit);
        else if(coordset["type"].as_string() == "uniform")
            conduit::blueprint::mesh::coordset::uniform::to_explicit(coordset, coordset_explicit);
        else
            coordset_explicit.set_external(coordset);
        slice_field(coordset_explicit["values"], ptReorder, dest_coordset["values"]);
#else
        dest_coordset["type"] = coordset["type"];
        dest_coordset["values"].set(coordset["values"]);
#endif
        // Reorder fields that match this topo
        for(conduit::index_t fi = 0; fi < fields.number_of_children(); fi++)
        {
            const conduit::Node &src = fields[fi];
            if(src["topology"].as_string() == topo.name())
            {
                auto &newfields = dest_topo["fields"];
                conduit::Node &dest = newfields[src.name()];
                dest["association"] = src["association"];
                dest["topology"] = dest_topo.name(); //src["topology"];
                if(dest["association"].as_string() == "element")
                {
                    slice_field(src["values"], reorder, dest["values"]);
                }
                else
                {
                    slice_field(src["values"], ptReorder, dest["values"]);
                }
            }
        }
    }
}

/**
 \brief Keep track of some tile information.
 */
class Tile
{
public:
    Tile() : ptids()
    {
    }

    /// Reset the tile. 
    void reset(size_t npts)
    {
        ptids = std::vector<int>(npts, -1);
    }

    /// Return the point ids.
          std::vector<int> &getPointIds() { return ptids; }
    const std::vector<int> &getPointIds() const { return ptids; }

    /// Get the specified point ids for this tile using the supplied indices.
    std::vector<int> getPointIds(const std::vector<int> &indices) const
    {
        std::vector<int> ids;
        ids.reserve(indices.size());
        for(const auto &idx : indices)
           ids.push_back(ptids[idx]);
        return ids;
    }

    // Set the point ids
    void setPointIds(const std::vector<int> &indices, const std::vector<int> &ids)
    {
        for(size_t i = 0; i < indices.size(); i++)
        {
            ptids[indices[i]] = ids[i];
        }
    }

private:
    std::vector<int> ptids;  //!< This tile's point ids.
};

/**
 \brief Build a mesh from tiles. There is a default tile pattern, although it can
        be replaced using an options Node containing new tile information.
 */
class Tiler
{
public:
    Tiler();

    /// Generate the tiled mesh.
    void generate(int nx, int ny, int nz,
                  conduit::Node &res,
                  const conduit::Node &options);
protected:
    /// Fill a default tile pattern into the filter.
    void initialize();

    /// Fill in the tile pattern from a Node.
    void initialize(const conduit::Node &t);

    /// Return point indices of points along left edge.
    const std::vector<int> &left() const { return m_left; }

    /// Return point indices of points along right edge.
    const std::vector<int> &right() const { return m_right; }

    /// Return point indices of points along bottom edge.
    const std::vector<int> &bottom() const { return m_bottom; }

    /// Return point indices of points along top edge.
    const std::vector<int> &top() const { return m_top; }

    /// Return tile width
    double width() const { return m_width; }

    /// Return tile height
    double height() const { return m_height; }

    /// Creates the points for the tile (if they need to be created).
    void addPoints(double origin[2],
                   std::vector<int> &ptids,
                   std::vector<double> &x,
                   std::vector<double> &y)
    {
        // Iterate through points in the template and add them if they have
        // not been created yet.
        for(size_t i = 0; i < m_xpts.size(); i++)
        {
            if(ptids[i] == -1)
            {
                ptids[i] = static_cast<int>(x.size());
                x.push_back(origin[0] + m_xpts[i]);
                y.push_back(origin[1] + m_ypts[i]);
            }
        }
    }

    /// Emit the quad cells using this tile's point ids.
    void addFaces(const std::vector<int> &ptids,
                  std::vector<int> &conn, std::vector<int> &sizes,
                  int offset = 0, bool reverse = false) const
    {
        const size_t nquads = m_quads.size() / 4;
        int order[] = {reverse ? 3 : 0, reverse ? 2 : 1, reverse ? 1 : 2, reverse ? 0 : 3};
        for(size_t i = 0; i < nquads; i++)
        {
            conn.push_back(offset + ptids[m_quads[4*i + order[0]]]);
            conn.push_back(offset + ptids[m_quads[4*i + order[1]]]);
            conn.push_back(offset + ptids[m_quads[4*i + order[2]]]);
            conn.push_back(offset + ptids[m_quads[4*i + order[3]]]);
            sizes.push_back(4);
        }
    }

    /// Emit the hex cells using this tile's point ids.
    void addHexs(const std::vector<int> &ptids, int plane1Offset, int plane2Offset, std::vector<int> &conn, std::vector<int> &sizes) const
    {
        const size_t nquads = m_quads.size() / 4;
        for(size_t i = 0; i < nquads; i++)
        {
            conn.push_back(plane1Offset + ptids[m_quads[4*i + 0]]);
            conn.push_back(plane1Offset + ptids[m_quads[4*i + 1]]);
            conn.push_back(plane1Offset + ptids[m_quads[4*i + 2]]);
            conn.push_back(plane1Offset + ptids[m_quads[4*i + 3]]);

            conn.push_back(plane2Offset + ptids[m_quads[4*i + 0]]);
            conn.push_back(plane2Offset + ptids[m_quads[4*i + 1]]);
            conn.push_back(plane2Offset + ptids[m_quads[4*i + 2]]);
            conn.push_back(plane2Offset + ptids[m_quads[4*i + 3]]);

            sizes.push_back(8);
        }
    }

    /// Compute the extents of the supplied values.
    double computeExtents(const std::vector<double> &values) const
    {
        double ext[2] = {values[0], values[0]};
        for(auto val : values)
        {
            ext[0] = std::min(ext[0], val);
            ext[1] = std::max(ext[1], val);
        }
        return ext[1] - ext[0];
    }

    /// Turn a node into a double vector.
    std::vector<double> toDoubleVector(const conduit::Node &n) const
    {
        auto acc = n.as_double_accessor();
        std::vector<double> vec;
        vec.reserve(acc.number_of_elements());
        for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
            vec.push_back(acc[i]);
        return vec;
    }

    /// Turn a node into an int vector.
    std::vector<int> toIntVector(const conduit::Node &n) const
    {
        auto acc = n.as_int_accessor();
        std::vector<int> vec;
        vec.reserve(acc.number_of_elements());
        for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
            vec.push_back(acc[i]);
        return vec;
    }

    /// Make 2D boundaries.
    void makeBoundaries2D(const std::vector<Tile> &tiles, int nx, int ny,
                          std::vector<int> &bconn, std::vector<int> &bsizes,
                          std::vector<int> &btype, const conduit::Node &options) const;

    /// Make 3D boundaries.
    void makeBoundaries3D(const std::vector<Tile> &tiles, int nx, int ny, int nz,
                          size_t nPtsPerPlane,
                          std::vector<int> &bconn, std::vector<int> &bsizes,
                          std::vector<int> &btype, const conduit::Node &options) const;
private:
    std::vector<double> m_xpts, m_ypts;
    double m_width, m_height;
    std::vector<int> m_left, m_right, m_bottom, m_top, m_quads;
};

Tiler::Tiler() : m_xpts(), m_ypts(),  m_width(0.), m_height(0.),
                 m_left(), m_right(), m_bottom(), m_top(), m_quads()
{
    initialize();
}

void
Tiler::initialize()
{
    // Default pattern
    m_xpts = std::vector<double>{
        0., 3., 10., 17., 20.,
        0., 3., 17., 20.,
        5., 15.,
        7., 10., 13.,
        0., 7., 10., 13., 20.,
        7., 10., 13.,
        5., 15.,
        0., 3., 17., 20.,
        0., 3., 10., 17., 20.,
    };

    m_ypts = std::vector<double>{
        0., 0., 0., 0., 0.,
        3., 3., 3., 3.,
        5., 5.,
        7., 7., 7.,
        10., 10., 10., 10., 10.,
        13., 13., 13.,
        15., 15.,
        17., 17., 17., 17.,
        20., 20., 20., 20., 20.,
    };

    m_quads = std::vector<int>{
        // lower-left quadrant
        0,1,6,5,
        1,2,9,6,
        2,12,11,9,
        5,6,9,14,
        9,11,15,14,
        11,12,16,15,
        // lower-right quadrant
        2,3,7,10,
        3,4,8,7,
        7,8,18,10,
        2,10,13,12,
        12,13,17,16,
        10,18,17,13,
        // upper-left quadrant
        14,22,25,24,
        14,15,19,22,
        15,16,20,19,
        24,25,29,28,
        22,30,29,25,
        19,20,30,22,
        // upper-right quadrant
        16,17,21,20,
        17,18,23,21,
        18,27,26,23,
        20,21,23,30,
        23,26,31,30,
        26,27,32,31
    };

    m_left = std::vector<int>{0,5,14,24,28};
    m_right = std::vector<int>{4,8,18,27,32};
    m_bottom = std::vector<int>{0,1,2,3,4};
    m_top = std::vector<int>{28,29,30,31,32};

    m_width = computeExtents(m_xpts);
    m_height = computeExtents(m_ypts);
}

void
Tiler::initialize(const conduit::Node &t)
{
    m_xpts = toDoubleVector(t.fetch_existing("x"));
    m_ypts = toDoubleVector(t.fetch_existing("y"));
    m_quads = toIntVector(t.fetch_existing("quads"));
    m_left = toIntVector(t.fetch_existing("left"));
    m_right = toIntVector(t.fetch_existing("right"));
    m_bottom = toIntVector(t.fetch_existing("bottom"));
    m_top = toIntVector(t.fetch_existing("top"));

    m_width = computeExtents(m_xpts);
    m_height = computeExtents(m_ypts);
}

/**
 \brief Generate coordinate and connectivity arrays using a tiled mesh pattern,
        given by the Tile class.

 \param nx The number of tiles in the X dimension.
 \param ny The number of tiles in the Y dimension.
 \param nz The number of tiles in the Z dimension.
 \param[out] res The output node.
 \param options A node that may contain additional control options.
 */
void
Tiler::generate(int nx, int ny, int nz,
    conduit::Node &res,
    const conduit::Node &options)
{
    double origin[] = {0., 0., 0.};
    std::vector<double> x, y, z;
    std::vector<int> conn, sizes, bconn, bsizes, btype;

    // Process any options.
    if(options.has_path("origin/x"))
        origin[0] = options.fetch_existing("origin/x").to_double();
    if(options.has_path("origin/y"))
        origin[1] = options.fetch_existing("origin/y").to_double();
    if(options.has_path("origin/z"))
        origin[2] = options.fetch_existing("origin/z").to_double();
    if(options.has_path("tile"))
        initialize(options.fetch_existing("tile"));
#if 1
    // Make cell centers for each tile and record that no tiles have been visited.
    int ncells = nx * ny;
    std::vector<int> visited(ncells, 0);
    std::vector<double> cx(ncells), cy(ncells);
    for(int j = 0, idx = 0; j < ny; j++)
    {
        for(int i = 0; i < nx; i++, idx++)
        {
            cx[idx] = origin[0] + (i + 0.5f) * width();
            cy[idx] = origin[1] + (j + 0.5f) * height();
        }
    }
    double *tileCenters[2] = {&cx[0], &cy[0]};
    conduit::blueprint::mesh::utils::kdtree<double*, double, 2> spatial_sort;
    spatial_sort.initialize(tileCenters, ncells);

    // Traverse the cells in the desired spatial order.
    std::vector<Tile> tiles(nx * ny);
    for(const int idx : spatial_sort.getIndices())
    {
        // The first time we've used the tile, set its size.
        Tile &current = tiles[idx];
        current.reset(m_xpts.size());

        // Copy neighbor points to the current tile if we can.
        int i = idx % nx;
        int j = idx / nx;
        int left_idx = (i > 0) ? (idx - 1) : -1;
        int right_idx = (i < nx-1) ? (idx + 1) : -1;
        int bottom_idx = (j > 0) ? (idx - nx) : -1;
        int top_idx = (j < ny-1) ? (idx + nx) : -1;
        if(left_idx != -1 && visited[left_idx])
        {
            current.setPointIds(left(), tiles[left_idx].getPointIds(right()));
        }
        if(right_idx != -1 && visited[right_idx])
        {
            current.setPointIds(right(), tiles[right_idx].getPointIds(left()));
        }
        if(bottom_idx != -1 && visited[bottom_idx])
        {
            current.setPointIds(bottom(), tiles[bottom_idx].getPointIds(top()));
        }
        if(top_idx != -1 && visited[top_idx])
        {
            current.setPointIds(top(), tiles[top_idx].getPointIds(bottom()));
        }

        // Make this tile's points
        double newOrigin[] = {origin[0] + i * width(), origin[1] + j * height(), origin[2]};
        addPoints(newOrigin, current.getPointIds(), x, y);

        visited[idx] = 1;
    }
    
#else
    // Make a pass where we make nx*ny tiles so we can generate their points.
    std::vector<Tile> tiles(nx * ny);
    double newOrigin[] = {origin[0], origin[1], origin[2]};
    for(int j = 0; j < ny; j++)
    {
        newOrigin[0] = origin[0];
        for(int i = 0; i < nx; i++)
        {
            Tile &current = tiles[(j*nx + i)];

            // The first time we've used the tile, set its size.
            current.reset(m_xpts.size());

            // Copy some previous points over so they can be shared.
            if(i > 0)
            {
               Tile &prevX = tiles[(j*nx + i - 1)];
               current.setPointIds(left(), prevX.getPointIds(right()));
            }
            if(j > 0)
            {
               Tile &prevY = tiles[((j-1)*nx + i)];
               current.setPointIds(bottom(), prevY.getPointIds(top()));
            }

            addPoints(newOrigin, current.getPointIds(), x, y);
            newOrigin[0] += width();
        }
        newOrigin[1] += height();
    }
#endif
    if(nz < 1)
    {
        // Iterate over the tiles and add their quads.
        // TODO: reserve size for conn, sizes
#if 1
        // Add the cells in spatial sort order.
        for(const int idx : spatial_sort.getIndices())
        {
            addFaces(tiles[idx].getPointIds(), conn, sizes);
        }
#else
        for(int j = 0; j < ny; j++)
        {
            for(int i = 0; i < nx; i++)
            {
                Tile &current = tiles[(j*nx + i)];
                addFaces(current.getPointIds(), conn, sizes);
            }
        }
#endif
        // NOTE: z coords in output will be empty.

        // Boundaries
        makeBoundaries2D(tiles, nx, ny, bconn, bsizes, btype, options);
        if(!bconn.empty())
        {
            res["topologies/boundary/type"] = "unstructured";
            res["topologies/boundary/coordset"] = "coords";
            res["topologies/boundary/elements/shape"] = "line";
            res["topologies/boundary/elements/connectivity"].set(bconn);
            res["topologies/boundary/elements/sizes"].set(bsizes);

            res["fields/boundary_type/topology"] = "boundary";
            res["fields/boundary_type/association"] = "element";
            res["fields/boundary_type/values"].set(btype);
        }
    }
    else
    {
        // We have x,y points now. We need to replicate them to make multiple planes.
        // We make z coordinates too.
        size_t ptsPerPlane = x.size();
        int nplanes = nz + 1;
        x.reserve(ptsPerPlane * nplanes);
        y.reserve(ptsPerPlane * nplanes);
        z.reserve(ptsPerPlane * nplanes);
        for(size_t i = 0; i < ptsPerPlane; i++)
            z.push_back(origin[2]);
        for(int p = 1; p < nplanes; p++)
        {
            double zvalue = origin[2] + static_cast<double>(p) * std::max(width(), height());
            for(size_t i = 0; i < ptsPerPlane; i++)
            {
                x.push_back(x[i]);
                y.push_back(y[i]);
                z.push_back(zvalue);
            }
        }

        // Iterate over the tiles and add their hexs.
        // TODO: reserve size for conn, sizes
        for(int k = 0; k < nz; k++)
        {
            int offset1 = k * ptsPerPlane;
            int offset2 = offset1 + ptsPerPlane;

            for(int j = 0; j < ny; j++)
            {
                for(int i = 0; i < nx; i++)
                {
                    Tile &current = tiles[(j*nx + i)];
                    addHexs(current.getPointIds(), offset1, offset2, conn, sizes);
                }
            }
        }

        // Boundaries
        makeBoundaries3D(tiles, nx, ny, nz, ptsPerPlane, bconn, bsizes, btype, options);
        if(!bconn.empty())
        {
            res["topologies/boundary/type"] = "unstructured";
            res["topologies/boundary/coordset"] = "coords";
            res["topologies/boundary/elements/shape"] = "quad";
            res["topologies/boundary/elements/connectivity"].set(bconn);
            res["topologies/boundary/elements/sizes"].set(bsizes);

            res["fields/boundary_type/topology"] = "boundary";
            res["fields/boundary_type/association"] = "element";
            res["fields/boundary_type/values"].set(btype);
        }
    }

    res["coordsets/coords/type"] = "explicit";
    res["coordsets/coords/values"] = "explicit";
    res["coordsets/coords/values/x"].set(x);
    res["coordsets/coords/values/y"].set(y);
    if(!z.empty())
        res["coordsets/coords/values/z"].set(z);

    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = z.empty() ? "quad" : "hex";
    res["topologies/mesh/elements/connectivity"].set(conn);
    res["topologies/mesh/elements/sizes"].set(sizes);

#if 1
    if(nz > 0)
    {
        // We need offsets.
        conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(res["topologies/mesh"], res["topologies/mesh/elements/offsets"]);

        // Reorder the mesh in 3D. NOTE: boundaries would have to be fixed because
        // of the changes to node ordering, which we'd have to pass out the node ordering.
        const auto reorder = spatial_reordering(res["topologies/mesh"]);
        reorder_topo(res["topologies/mesh"], res["coordsets/coords"], res["fields"],
                     res["topologies/rmesh"], res["coordsets/rcoords"], res["rfields"],
                     reorder);

conduit::Node opts;
opts["num_children_threshold"] = 100000;
opts["num_elements_threshold"] = 500;
std::cout << res.to_summary_string(opts) << std::endl;
    }
#endif
}

void
Tiler::makeBoundaries2D(const std::vector<Tile> &tiles, int nx, int ny,
    std::vector<int> &bconn, std::vector<int> &bsizes, std::vector<int> &btype,
    const conduit::Node &options) const
{
    if(options.has_path("boundaries/left") && options.fetch_existing("boundaries/left").to_int() > 0)
    {
        for(int i = 0, j = ny-1; j >= 0; j--)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(left());
            for(size_t bi = ids.size() - 1; bi > 0; bi--)
            {
                bconn.push_back(ids[bi]);
                bconn.push_back(ids[bi - 1]);
                bsizes.push_back(2);
                btype.push_back(0);
            }
        }
    }
    if(options.has_path("boundaries/bottom") && options.fetch_existing("boundaries/bottom").to_int() > 0)
    {
        for(int i = 0, j = 0; i < nx; i++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(bottom());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                bconn.push_back(ids[bi]);
                bconn.push_back(ids[bi + 1]);
                bsizes.push_back(2);
                btype.push_back(2);
            }
        }
    }
    if(options.has_path("boundaries/right") && options.fetch_existing("boundaries/right").to_int() > 0)
    {
        for(int i = nx - 1, j = 0; j < ny; j++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(right());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                bconn.push_back(ids[bi]);
                bconn.push_back(ids[bi + 1]);
                bsizes.push_back(2);
                btype.push_back(1);
            }
        }
    }
    if(options.has_path("boundaries/top") && options.fetch_existing("boundaries/top").to_int() > 0)
    {
        for(int i = nx - 1, j = ny - 1; i >= 0; i--)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(top());
            for(size_t bi = ids.size() - 1; bi > 0; bi--)
            {
                bconn.push_back(ids[bi]);
                bconn.push_back(ids[bi - 1]);
                bsizes.push_back(2);
                btype.push_back(3);
            }
        }
    }
}

void
Tiler::makeBoundaries3D(const std::vector<Tile> &tiles, int nx, int ny, int nz,
    size_t nPtsPerPlane,
    std::vector<int> &bconn, std::vector<int> &bsizes, std::vector<int> &btype,
    const conduit::Node &options) const
{
    if(options.has_path("boundaries/left") && options.fetch_existing("boundaries/left").to_int() > 0)
    {
        for(int k = 0; k < nz; k++)
        {
            int offset1 = k * nPtsPerPlane;
            int offset2 = (k + 1) * nPtsPerPlane;
            for(int i = 0, j = ny-1; j >= 0; j--)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(left());
                for(size_t bi = ids.size() - 1; bi > 0; bi--)
                {
                    bconn.push_back(offset1 + ids[bi]);
                    bconn.push_back(offset1 + ids[bi - 1]);
                    bconn.push_back(offset2 + ids[bi - 1]);
                    bconn.push_back(offset2 + ids[bi]);
                    bsizes.push_back(4);
                    btype.push_back(0);
                }
            }
        }
    }
    if(options.has_path("boundaries/right") && options.fetch_existing("boundaries/right").to_int() > 0)
    {
        for(int k = 0; k < nz; k++)
        {
            int offset1 = k * nPtsPerPlane;
            int offset2 = (k + 1) * nPtsPerPlane;
            for(int i = nx - 1, j = 0; j < ny; j++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(right());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    bconn.push_back(offset1 + ids[bi]);
                    bconn.push_back(offset1 + ids[bi + 1]);
                    bconn.push_back(offset2 + ids[bi + 1]);
                    bconn.push_back(offset2 + ids[bi]);
                    bsizes.push_back(4);
                    btype.push_back(1);
                }
            }
        }
    }
    if(options.has_path("boundaries/bottom") && options.fetch_existing("boundaries/bottom").to_int() > 0)
    {
        for(int k = 0; k < nz; k++)
        {
            int offset1 = k * nPtsPerPlane;
            int offset2 = (k + 1) * nPtsPerPlane;
            for(int i = 0, j = 0; i < nx; i++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(bottom());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    bconn.push_back(offset1 + ids[bi]);
                    bconn.push_back(offset1 + ids[bi + 1]);
                    bconn.push_back(offset2 + ids[bi + 1]);
                    bconn.push_back(offset2 + ids[bi]);
                    bsizes.push_back(4);
                    btype.push_back(2);
                }
            }
        }
    }
    if(options.has_path("boundaries/top") && options.fetch_existing("boundaries/top").to_int() > 0)
    {
        for(int k = 0; k < nz; k++)
        {
            int offset1 = k * nPtsPerPlane;
            int offset2 = (k + 1) * nPtsPerPlane;
            for(int i = nx - 1, j = ny - 1; i >= 0; i--)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(top());
                for(size_t bi = ids.size() - 1; bi > 0; bi--)
                {
                    bconn.push_back(offset1 + ids[bi]);
                    bconn.push_back(offset1 + ids[bi - 1]);
                    bconn.push_back(offset2 + ids[bi - 1]);
                    bconn.push_back(offset2 + ids[bi]);
                    bsizes.push_back(4);
                    btype.push_back(3);
                }
            }
        }
    }
    if(options.has_path("boundaries/back") && options.fetch_existing("boundaries/back").to_int() > 0)
    {
        for(int j = 0; j < ny; j++)
        for(int i = nx - 1; i >= 0; i--)
        {
           const Tile &current = tiles[(j*nx + i)];
           size_t s0 = bsizes.size();
           addFaces(current.getPointIds(), bconn, bsizes, 0, true);
           for( ; s0 < bsizes.size(); s0++)
               btype.push_back(4);
        }
    }
    if(options.has_path("boundaries/front") && options.fetch_existing("boundaries/front").to_int() > 0)
    {
        for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
        {
           const Tile &current = tiles[(j*nx + i)];
           size_t s0 = bsizes.size();
           addFaces(current.getPointIds(), bconn, bsizes, nz * nPtsPerPlane);
           for( ; s0 < bsizes.size(); s0++)
               btype.push_back(5);
        }
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples::detail --
//-----------------------------------------------------------------------------

void
tiled(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
      conduit::Node &res, const conduit::Node &options)
{
    detail::Tiler T;
    T.generate(nx, ny, nz, res, options);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
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



