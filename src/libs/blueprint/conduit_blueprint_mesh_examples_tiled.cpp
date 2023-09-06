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
#include "conduit_blueprint_mesh_examples_tiled.hpp"
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

void
reorder_topo(const conduit::Node &topo, const conduit::Node &coordset, const conduit::Node &fields,
             conduit::Node &dest_topo, conduit::Node &dest_coordset, conduit::Node &dest_fields,
             const std::vector<conduit::index_t> &reorder)
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
        constexpr conduit::index_t invalidNode = -1;
        auto npts = conduit::blueprint::mesh::coordset::length(coordset);
        std::vector<conduit::index_t> old2NewPoints(npts, invalidNode), ptReorder(npts, invalidNode);
        conduit::index_t newPointIndex = 0;

        // We iterate over elements in the specified order. We iterate over the
        // points in each element and renumber the points.
        conduit::index_t newoffset = 0;
        for(const auto cellIndex : reorder)
        {
            for(conduit::index_t i = 0; i < sizes[cellIndex]; i++)
            {
                auto id = conn[offsets[cellIndex] + i];
                if(old2NewPoints[id] == invalidNode)
                {
                    ptReorder[newPointIndex] = id;
                    old2NewPoints[id] = newPointIndex++;
                }
                newconn.push_back(old2NewPoints[id]);
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

        // Reorder the coordset now, making it explicit if needed.
        dest_coordset["type"] = "explicit";
        conduit::Node coordset_explicit;
        if(coordset["type"].as_string() == "rectilinear")
            conduit::blueprint::mesh::coordset::rectilinear::to_explicit(coordset, coordset_explicit);
        else if(coordset["type"].as_string() == "uniform")
            conduit::blueprint::mesh::coordset::uniform::to_explicit(coordset, coordset_explicit);
        else
            coordset_explicit.set_external(coordset);
        conduit::blueprint::mesh::utils::slice_field(coordset_explicit["values"], ptReorder, dest_coordset["values"]);

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
                    conduit::blueprint::mesh::utils::slice_field(src["values"], reorder, dest["values"]);
                }
                else
                {
                    conduit::blueprint::mesh::utils::slice_field(src["values"], ptReorder, dest["values"]);
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
    static const conduit::index_t INVALID_POINT;

    Tile() : ptids()
    {
    }

    /// Reset the tile. 
    void reset(size_t npts)
    {
        ptids = std::vector<conduit::index_t>(npts, -1);
    }

    /// Return the point ids.
          std::vector<conduit::index_t> &getPointIds() { return ptids; }
    const std::vector<conduit::index_t> &getPointIds() const { return ptids; }

    /// Get the specified point ids for this tile using the supplied indices.
    std::vector<conduit::index_t> getPointIds(const std::vector<conduit::index_t> &indices) const
    {
        std::vector<conduit::index_t> ids;
        ids.reserve(indices.size());
        for(const auto &idx : indices)
           ids.push_back(ptids[idx]);
        return ids;
    }

    // Set the point ids
    void setPointIds(const std::vector<conduit::index_t> &indices, const std::vector<conduit::index_t> &ids)
    {
        for(size_t i = 0; i < indices.size(); i++)
        {
            ptids[indices[i]] = ids[i];
        }
    }

private:
    std::vector<conduit::index_t> ptids;  //!< This tile's point ids.
};

const conduit::index_t Tile::INVALID_POINT = -1;

/**
 \brief Build a mesh from tiles. There is a default tile pattern, although it can
        be replaced using an options Node containing new tile information.
 */
class Tiler
{
public:
    Tiler();

    /// Generate the tiled mesh.
    void generate(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
                  conduit::Node &res,
                  const conduit::Node &options);
protected:
    /// Fill a default tile pattern into the filter.
    void initialize();

    /// Fill in the tile pattern from a Node.
    void initialize(const conduit::Node &t);

    /// Return point indices of points along left edge.
    const std::vector<conduit::index_t> &left() const { return m_left; }

    /// Return point indices of points along right edge.
    const std::vector<conduit::index_t> &right() const { return m_right; }

    /// Return point indices of points along bottom edge.
    const std::vector<conduit::index_t> &bottom() const { return m_bottom; }

    /// Return point indices of points along top edge.
    const std::vector<conduit::index_t> &top() const { return m_top; }

    /// Return tile width
    double width() const { return m_width; }

    /// Return tile height
    double height() const { return m_height; }

    /// Creates the points for the tile (if they need to be created).
    void addPoints(double origin[2],
                   std::vector<conduit::index_t> &ptids,
                   std::vector<double> &x,
                   std::vector<double> &y)
    {
        // Iterate through points in the template and add them if they have
        // not been created yet.
        for(size_t i = 0; i < m_xpts.size(); i++)
        {
            if(ptids[i] == Tile::INVALID_POINT)
            {
                ptids[i] = static_cast<int>(x.size());
                x.push_back(origin[0] + m_xpts[i]);
                y.push_back(origin[1] + m_ypts[i]);
            }
        }
    }

    /// Emit the quad cells using this tile's point ids.
    void addFaces(const std::vector<conduit::index_t> &ptids,
                  std::vector<conduit::index_t> &conn,
                  std::vector<conduit::index_t> &sizes,
                  conduit::index_t offset = 0,
                  bool reverse = false) const
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
    void addHexs(const std::vector<conduit::index_t> &ptids,
                 conduit::index_t plane1Offset,
                 conduit::index_t plane2Offset,
                 std::vector<conduit::index_t> &conn,
                 std::vector<conduit::index_t> &sizes) const
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
    std::vector<conduit::index_t> toIndexVector(const conduit::Node &n) const
    {
        auto acc = n.as_index_t_accessor();
        std::vector<index_t> vec;
        vec.reserve(acc.number_of_elements());
        for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
            vec.push_back(acc[i]);
        return vec;
    }

    /// Make 2D boundaries.
    void makeBoundaries2D(const std::vector<Tile> &tiles,
                          conduit::index_t nx,
                          conduit::index_t ny,
                          std::vector<conduit::index_t> &bconn,
                          std::vector<conduit::index_t> &bsizes,
                          std::vector<int> &btype,
                          const conduit::Node &options) const;

    /// Make 3D boundaries.
    void makeBoundaries3D(const std::vector<Tile> &tiles,
                          conduit::index_t nx,
                          conduit::index_t ny,
                          conduit::index_t nz,
                          conduit::index_t nPtsPerPlane,
                          std::vector<conduit::index_t> &bconn,
                          std::vector<conduit::index_t> &bsizes,
                          std::vector<int> &btype,
                          const conduit::Node &options) const;
private:
    std::vector<double> m_xpts, m_ypts;
    double m_width, m_height;
    std::vector<conduit::index_t> m_left, m_right, m_bottom, m_top, m_quads;
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

    m_quads = std::vector<conduit::index_t>{
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

    m_left = std::vector<conduit::index_t>{0,5,14,24,28};
    m_right = std::vector<conduit::index_t>{4,8,18,27,32};
    m_bottom = std::vector<conduit::index_t>{0,1,2,3,4};
    m_top = std::vector<conduit::index_t>{28,29,30,31,32};

    m_width = computeExtents(m_xpts);
    m_height = computeExtents(m_ypts);
}

void
Tiler::initialize(const conduit::Node &t)
{
    m_xpts = toDoubleVector(t.fetch_existing("x"));
    m_ypts = toDoubleVector(t.fetch_existing("y"));
    m_quads = toIndexVector(t.fetch_existing("quads"));
    m_left = toIndexVector(t.fetch_existing("left"));
    m_right = toIndexVector(t.fetch_existing("right"));
    m_bottom = toIndexVector(t.fetch_existing("bottom"));
    m_top = toIndexVector(t.fetch_existing("top"));

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
Tiler::generate(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
    conduit::Node &res,
    const conduit::Node &options)
{
    double origin[] = {0., 0., 0.};
    std::vector<double> x, y, z;
    std::vector<conduit::index_t> conn, sizes, bconn, bsizes;
    std::vector<int> btype;

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
    conduit::index_t ncells = nx * ny;
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
    constexpr conduit::index_t invalidIndex = -1;
    for(const auto idx : spatial_sort.getIndices())
    {
        // The first time we've used the tile, set its size.
        Tile &current = tiles[idx];
        current.reset(m_xpts.size());

        // Copy neighbor points to the current tile if we can.
        conduit::index_t i = idx % nx;
        conduit::index_t j = idx / nx;
        conduit::index_t left_idx = (i > 0) ? (idx - 1) : invalidIndex;
        conduit::index_t right_idx = (i < nx - 1) ? (idx + 1) : invalidIndex;
        conduit::index_t bottom_idx = (j > 0) ? (idx - nx) : invalidIndex;
        conduit::index_t top_idx = (j < ny - 1) ? (idx + nx) : invalidIndex;
        if(left_idx != invalidIndex && visited[left_idx])
        {
            current.setPointIds(left(), tiles[left_idx].getPointIds(right()));
        }
        if(right_idx != invalidIndex && visited[right_idx])
        {
            current.setPointIds(right(), tiles[right_idx].getPointIds(left()));
        }
        if(bottom_idx != invalidIndex && visited[bottom_idx])
        {
            current.setPointIds(bottom(), tiles[bottom_idx].getPointIds(top()));
        }
        if(top_idx != invalidIndex && visited[top_idx])
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
    for(conduit::index_t j = 0; j < ny; j++)
    {
        newOrigin[0] = origin[0];
        for(conduit::index_t i = 0; i < nx; i++)
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
        for(const auto idx : spatial_sort.getIndices())
        {
            addFaces(tiles[idx].getPointIds(), conn, sizes);
        }
#else
        for(conduit::index_t j = 0; j < ny; j++)
        {
            for(conduit::index_t i = 0; i < nx; i++)
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
        conduit::index_t ptsPerPlane = static_cast<conduit::index_t>(x.size());
        conduit::index_t nplanes = nz + 1;
        x.reserve(ptsPerPlane * nplanes);
        y.reserve(ptsPerPlane * nplanes);
        z.reserve(ptsPerPlane * nplanes);
        for(conduit::index_t i = 0; i < ptsPerPlane; i++)
            z.push_back(origin[2]);
        for(conduit::index_t p = 1; p < nplanes; p++)
        {
            double zvalue = origin[2] + static_cast<double>(p) * std::max(width(), height());
            for(conduit::index_t i = 0; i < ptsPerPlane; i++)
            {
                x.push_back(x[i]);
                y.push_back(y[i]);
                z.push_back(zvalue);
            }
        }

        // Iterate over the tiles and add their hexs.
        // TODO: reserve size for conn, sizes
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * ptsPerPlane;
            conduit::index_t offset2 = offset1 + ptsPerPlane;

            for(conduit::index_t j = 0; j < ny; j++)
            {
                for(conduit::index_t i = 0; i < nx; i++)
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
        const auto reorder = conduit::blueprint::mesh::utils::topology::spatial_ordering(res["topologies/mesh"]);
        reorder_topo(res["topologies/mesh"], res["coordsets/coords"], res["fields"],
                     res["topologies/rmesh"], res["coordsets/rcoords"], res["fields"],
                     reorder);

conduit::Node opts;
opts["num_children_threshold"] = 100000;
opts["num_elements_threshold"] = 500;
std::cout << res.to_summary_string(opts) << std::endl;
    }
#endif
}

void
Tiler::makeBoundaries2D(const std::vector<Tile> &tiles,
    conduit::index_t nx,
    conduit::index_t ny,
    std::vector<conduit::index_t> &bconn,
    std::vector<conduit::index_t> &bsizes,
    std::vector<int> &btype,
    const conduit::Node &options) const
{
    if(options.has_path("boundaries/left") && options.fetch_existing("boundaries/left").to_int() > 0)
    {
        for(conduit::index_t i = 0, j = ny-1; j >= 0; j--)
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
        for(conduit::index_t i = 0, j = 0; i < nx; i++)
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
        for(conduit::index_t i = nx - 1, j = 0; j < ny; j++)
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
        for(conduit::index_t i = nx - 1, j = ny - 1; i >= 0; i--)
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
Tiler::makeBoundaries3D(const std::vector<Tile> &tiles,
    conduit::index_t nx,
    conduit::index_t ny,
    conduit::index_t nz,
    conduit::index_t nPtsPerPlane,
    std::vector<conduit::index_t> &bconn,
    std::vector<conduit::index_t> &bsizes,
    std::vector<int> &btype,
    const conduit::Node &options) const
{
    if(options.has_path("boundaries/left") && options.fetch_existing("boundaries/left").to_int() > 0)
    {
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = 0, j = ny-1; j >= 0; j--)
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
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = nx - 1, j = 0; j < ny; j++)
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
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = 0, j = 0; i < nx; i++)
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
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = nx - 1, j = ny - 1; i >= 0; i--)
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
        for(conduit::index_t j = 0; j < ny; j++)
        for(conduit::index_t i = nx - 1; i >= 0; i--)
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
        for(conduit::index_t j = 0; j < ny; j++)
        for(conduit::index_t i = 0; i < nx; i++)
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



