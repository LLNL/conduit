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

    if(nz < 1)
    {
        // Iterate over the tiles and add their quads.
        // TODO: reserve size for conn, sizes
        for(int j = 0; j < ny; j++)
        {
            for(int i = 0; i < nx; i++)
            {
                Tile &current = tiles[(j*nx + i)];
                addFaces(current.getPointIds(), conn, sizes);
            }
        }
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



